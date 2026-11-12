"""Hand-written MLP neurons, delivered as rank-1 LoRA additions (CPU, closed form, no training).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: "Compile a rule into weights". In the MLP of each chosen layer one existing, nearly silent neuron j is
  overwritten so that it writes a chosen residual direction d (a steering write), either always (const) or only when a
  read direction is present (cond). Gemma 4 has no biases, so the constant comes from the mean direction of the MLP
  input: a ridge read-out q with q . xh ~= 1 at every position (fit on train positions, checked on heldout).
  MLP of layer l (v2/tii/trainer/aligned_forward.py `_mlp_block`):
      xh = RMSNorm(x; pre_ff)                       # captured by `collect acts --mlp-in` as xin
      a_j = gelu_tanh(gate_j . xh) * (up_j . xh)
      y = sum_j down[:, j] a_j                      # rms(y) captured as yrms
      h_out = layer_scalar * (x + w_post * y / rms(y) [+ PLE])
      const   gate_j = G q, up_j = q                      -> a_j ~= gelu(G) at every position
      cond    gate_j = (g / dl) (rh - th q), up_j = q     -> a_j ~= gelu(g (rh.xh - th) / dl); rh = unit
              difference of class means of xh, th = midpoint of the class means of rh.xh, dl = half their gap
      down[:, j] = beta * rho * sqrt(n) * uh / gelu(G or g), uh = unit(dh / w_post), rho = mean rms(y)
  so the neuron adds beta times the norm of the natural MLP output, in the y-direction whose post-norm image is exactly
  dh. Predicted shift of the layer output along dh (y_nat taken orthogonal to uh):
      layer_scalar * beta / sqrt(1 + beta^2) * sqrt(n) * |w_post * uh|, reported in units of the class gap |mu1 - mu0|.
  Side effect, also predicted: the natural MLP write shrinks by 1 / sqrt(1 + beta^2) (shared post-norm).
  d (per layer) = difference of class means of the layer output over the train split at --tags; AUC on heldout.
  Controls: --control rand uses the same neuron, gate, up and |down| with a random unit uh; --control orth the same
  with a random residual write made exactly orthogonal to dh (norm-matched in weight space); --control match
  writes dh with random signs per dimension, orthogonalised (matched in weight space and in residual space).
  LoRA packing (source scaling alpha / r must be 2; output rank r + 1, scaling 2, exact up to bf16 storage):
      gate / up:  B' = [B, e_j],                 A' = [A; (w_new - W_eff[j]) / 2]
      down:       B' = [B, v - W_eff[:, j]],     A' = [A; e_j^T / 2]
  every other module is zero-padded to rank r + 1. W_eff = dequantized INT4 base + 2 B A.
  Reads: base checkpoint, source adapter, acts .pt with h / xin / yrms / meta. Writes <out>/adapter_config.json,
  adapter_model.safetensors, neuron_report.json.
  Usage: python -m v2.steer.neuron --base M --adapter A --acts acts_w5.pt --tags edit@0,edit@32 --layers 14-20
             --beta 0.3 --mode const --out DIR [--control rand] [--sign -1]
SRP/DRY check: Pass - reuses dequant / parse_layers / PFX of v2.steer.fold; fold edits existing writes along a
  direction, this adds a new write.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from v2.steer.fold import PFX, dequant_int4_bf16, parse_layers


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def main(argv: list[str] | None = None) -> int:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser(prog="python -m v2.steer.neuron")
    p.add_argument("--base", type=Path, default=Path("/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--tags", required=True, help="comma list of position tags defining the class contrast")
    p.add_argument("--layers", required=True)
    p.add_argument("--beta", type=float, required=True, help="neuron output norm / natural MLP output norm")
    p.add_argument("--mode", choices=("const", "cond"), default="const")
    p.add_argument("--control", choices=("none", "rand", "orth", "match"), default="none")
    p.add_argument("--sign", type=float, default=1.0, help="-1 writes the opposite direction")
    p.add_argument("--read-tags", default="", help="cond only: tags of the class contrast the gate reads (default: --tags)")
    p.add_argument("--read-sign", type=float, default=0.0, help="cond only: -1 fires on class 0 of the read contrast (default: --sign)")
    p.add_argument("--gain", type=float, default=4.0, help="gate pre-activation G (const) or g at the class-1 mean (cond)")
    p.add_argument("--target", type=float, default=0.0,
                   help=">0: choose beta per layer so the predicted shift is this many class gaps (--beta is the cap)")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge of the constant read-out, relative to mean |xh|^2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    gen = torch.Generator().manual_seed(a.seed)

    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r = int(cfg["r"])
    if abs(float(cfg["lora_alpha"]) / r - 2.0) > 1e-9:
        raise ValueError("source scaling alpha/r must be 2")
    sd = {k: v.float() for k, v in load_file(str(a.adapter / "adapter_model.safetensors")).items()}
    base = safe_open(str(a.base / "model.safetensors"), "pt")
    acts = torch.load(a.acts, map_location="cpu")
    meta = acts["meta"]
    tags = set(a.tags.split(","))
    sel = lambda sp, lab: torch.tensor([i for i, m in enumerate(meta)  # noqa: E731
                                        if m["tag"] in tags and m["split"] == sp and m["label"] == lab])
    tr1, tr0, ho1, ho0 = sel("train", 1), sel("train", 0), sel("heldout", 1), sel("heldout", 0)
    rsign = a.read_sign or a.sign
    if a.read_tags:  # IF <read contrast> THEN write <--tags direction>: the gate reads another class contrast
        tags = set(a.read_tags.split(","))
        r1, r0, rh1, rh0 = sel("train", 1), sel("train", 0), sel("heldout", 1), sel("heldout", 0)
        if rsign < 0:
            r1, r0, rh1, rh0 = r0, r1, rh0, rh1
    else:
        r1, r0, rh1, rh0 = tr1, tr0, ho1, ho0
    layers = parse_layers(a.layers)
    is_tr = torch.tensor([m["split"] == "train" for m in meta])
    n = acts["h"].shape[-1]
    new: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rep: dict[str, Any] = {"args": {k: str(v) for k, v in vars(a).items()}, "n_train": [len(tr1), len(tr0)],
                           "n_heldout": [len(ho1), len(ho0)], "layers": {}}
    for li in layers:
        bp = f"model.language_model.layers.{li}."
        H = acts["h"][:, li].float()
        X = acts["xin"][:, li].float()
        rho = float(acts["yrms"][:, li].float().mean())
        w_post = base.get_tensor(bp + "post_feedforward_layernorm.weight").float()
        ls = float(base.get_tensor(bp + "layer_scalar").float())
        d = H[tr1].mean(0) - H[tr0].mean(0)
        gap = float(d.norm())
        dh = a.sign * d / d.norm()
        W = {}
        for site in ("gate_proj", "up_proj", "down_proj"):
            W0 = dequant_int4_bf16(base.get_tensor(f"{bp}mlp.{site}.weight_packed"),
                                   base.get_tensor(f"{bp}mlp.{site}.weight_scale")).float()
            mod = f"{PFX}{li}.mlp.{site}"
            W[site] = W0 + 2.0 * sd[mod + ".lora_B.weight"] @ sd[mod + ".lora_A.weight"]
        # the neuron to overwrite: smallest mean |a_j| * |down[:, j]| over the captured positions
        act = F.gelu(X @ W["gate_proj"].T, approximate="tanh") * (X @ W["up_proj"].T)
        score = act.abs().mean(0) * W["down_proj"].norm(dim=0)
        j = int(score.argmin())
        # constant read-out q (q . xh ~= 1 at every position): ridge fit on train-split positions, dual form
        Zt = X[is_tr].double()
        K = Zt @ Zt.T
        lam = a.ridge * float(K.diagonal().mean()) * K.shape[0]
        q = (Zt.T @ torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=torch.double),
                                       torch.ones(K.shape[0], dtype=torch.double))).float()
        pm = X[~is_tr] @ q
        c = float(pm.mean())
        up_new = q
        if a.mode == "const":
            gate_new = a.gain * q
        else:
            Xd = X[r1].mean(0) - X[r0].mean(0)
            rh = (1.0 if a.read_tags else a.sign) * Xd / Xd.norm()
            p1, p0 = float((X[r1] @ rh).mean()), float((X[r0] @ rh).mean())
            th, dl = (p1 + p0) / 2.0, (p1 - p0) / 2.0
            gate_new = (a.gain / dl) * (rh - th * q)
        fire = F.gelu(X @ gate_new, approximate="tanh") * (X @ up_new)  # realised activation of the new neuron
        ref = float(F.gelu(torch.tensor(a.gain), approximate="tanh"))
        uh = dh / w_post
        uh = uh / uh.norm()
        eff = float((w_post * uh).norm())
        if a.control == "rand":
            uh = torch.randn(n, generator=gen)
            uh = uh / uh.norm()
        elif a.control == "orth":  # same |down| column; residual write exactly orthogonal to the fitted direction
            rv = torch.randn(n, generator=gen)
            rv = rv - (rv @ dh) * dh / (dh @ dh)
            uh = rv / w_post
            uh = uh / uh.norm()
        elif a.control == "match":  # dh with random signs per dimension, orthogonalised: same |down| column AND
            rv = dh * (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()  # same residual write norm
            rv = rv - (rv @ dh) * dh / (dh @ dh)
            uh = rv / w_post
            uh = uh / uh.norm()
        beta = a.beta
        if a.target > 0:  # per-layer dose: predicted shift = target * class gap (beta capped at --beta)
            kappa = a.target * gap / (ls * math.sqrt(n) * eff)
            beta = min(a.beta, kappa / math.sqrt(1 - kappa ** 2)) if kappa < 1 else a.beta
        v = beta * rho * math.sqrt(n) * uh / ref
        pred = ls * beta / math.sqrt(1 + beta ** 2) * math.sqrt(n) * eff
        is_bg = torch.tensor([mm["tag"].startswith("bg@") for mm in meta])
        info = {"neuron": j, "neuron_score_over_median": float(score[j] / score.median()), "rho": rho,
                "layer_scalar": ls, "gap": gap, "heldout_auc": auc(H[ho1] @ dh, H[ho0] @ dh) if len(ho1) else None,
                "const_heldout_mean": c, "const_heldout_sd": float(pm.std()), "const_heldout_min": float(pm.min()),
                "w_post_absmin": float(w_post.abs().min()), "write_eff": eff, "write_eff_realised": float((w_post * uh).norm()),
                "cos_write_d": float(F.cosine_similarity(w_post * uh, dh, dim=0)),
                "pred_shift": pred, "pred_shift_over_gap": pred / gap,
                "beta": beta, "natural_write_scale": 1 / math.sqrt(1 + beta ** 2),
                "fire_over_ref": {"bg": float(fire[is_bg].mean() / ref), "class1": float(fire[r1].mean() / ref),
                                  "class0": float(fire[r0].mean() / ref), "ho1": float(fire[rh1].mean() / ref),
                                  "ho0": float(fire[rh0].mean() / ref), "bg_sd": float(fire[is_bg].std() / ref),
                                  "write1": float(fire[tr1].mean() / ref), "write0": float(fire[tr0].mean() / ref)}}
        if a.mode == "cond":
            info["read_heldout_auc"] = auc(X[rh1] @ rh, X[rh0] @ rh) if len(rh1) else None
        ej_in = torch.zeros(1, W["down_proj"].shape[1])
        ej_in[0, j] = 0.5
        ej_out = torch.zeros(W["gate_proj"].shape[0], 1)
        ej_out[j, 0] = 1.0
        for site, tgt in (("gate_proj", gate_new), ("up_proj", up_new)):
            arow = ((tgt - W[site][j]) / 2.0)[None]
            new[f"{PFX}{li}.mlp.{site}"] = (arow, ej_out)
            got = W[site][j] + 2.0 * arow[0].bfloat16().float()
            info[f"{site}_row_rel_err"] = float((got - tgt).norm() / tgt.norm())
        bcol = (v - W["down_proj"][:, j])[:, None]
        new[f"{PFX}{li}.mlp.down_proj"] = (ej_in, bcol)
        got = W["down_proj"][:, j] + bcol[:, 0].bfloat16().float()
        info["down_col_rel_err"] = float((got - v).norm() / v.norm())
        rep["layers"][str(li)] = info
        print(li, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in info.items()
                              if k != "fire_over_ref"}), {k: round(x, 3) for k, x in info["fire_over_ref"].items()},
              flush=True)
    tensors: dict[str, torch.Tensor] = {}
    for key, t in sd.items():
        mod, ab = key.rsplit(".lora_", 1)
        add = new.get(mod)
        if ab.startswith("A"):
            t2 = torch.cat([t, add[0] if add else t.new_zeros(1, t.shape[1])], 0)
        else:
            t2 = torch.cat([t, add[1] if add else t.new_zeros(t.shape[0], 1)], 1)
        tensors[key] = t2.to(torch.bfloat16).contiguous()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "adapter_config.json").write_text(json.dumps(dict(cfg, r=r + 1, lora_alpha=2 * (r + 1)), indent=2))
    save_file(tensors, str(a.out / "adapter_model.safetensors"))
    rep["pred_shift_over_gap_sum"] = sum(x["pred_shift_over_gap"] for x in rep["layers"].values())
    (a.out / "neuron_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"out": str(a.out), "rank": r + 1, "pred_shift_over_gap_sum": rep["pred_shift_over_gap_sum"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
