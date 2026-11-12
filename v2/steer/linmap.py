"""Compile a prompt rule into k hand-written linear MLP neurons (rank-k LoRA addition; CPU, closed form, no training).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: A system-prompt rule shifts the residual stream at each decision by an amount that depends on the context;
  a constant write (v2.steer.neuron --mode const) reproduces only the mean of that shift. Here the shift is fitted as
  a low-rank linear map of the MLP input and written into k nearly silent neurons of one layer:
      T_i = h_rule_i - h_plain_i                 layer output with / without the rule, same call, same position
      T ~= xh W, W = ridge(xh_plain -> T)        dual form, train pairs only; heldout R^2 reported
      W ~= sum_j U_j S_j V_j^T                   top-k SVD
      neuron j:  gate = G q, up = U_j            a_j = gelu(G q.xh) (U_j.xh) ~= gelu(G) (U_j.xh): linear in xh
                 down = t rho S_j V_j / (layer_scalar w_post gelu(G))      (up / c, down * c: see packing)
  q is the constant read-out of v2.steer.neuron (q.xh ~= 1), rho the mean rms of the natural MLP output, t the dose.
  Through the post-norm (h_out = layer_scalar (x + w_post y / rms(y))) the k neurons shift the layer output by
  ~= t xh U S V^T / sqrt(1 + beta^2), beta = |added y| / |natural y|; the realised heldout R^2 at t = 1 (actual gelu,
  per-position rms) is reported next to the ridge and rank-k values, as is beta.
  --control match: same neurons, gate, up and |down| columns; every write direction gets the same random sign per
  dimension and is made orthogonal to the span of the fitted write directions.
  LoRA packing as in v2.steer.neuron, k rows / columns at a time (source scaling alpha / r must be 2).
  Reads: base checkpoint, source adapter, acts .pt (h / xin / yrms / meta with tag, label, pair, split).
  Writes <out>/adapter_config.json, adapter_model.safetensors, linmap_report.json.
  Usage: python -m v2.steer.linmap --adapter ZERO --acts acts_base.pt --tags pedit@0 --layer 18 --rank 8 --scale 1
             --out DIR [--control match] [--lam 1.0]
SRP/DRY check: Pass - neuron.py writes one direction per layer; this writes a rank-k map. Shares fold's helpers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from v2.steer.fold import PFX, dequant_int4_bf16


def main(argv: list[str] | None = None) -> int:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser(prog="python -m v2.steer.linmap")
    p.add_argument("--base", type=Path, default=Path("/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--tags", required=True, help="comma list of rule tags; their shifts are summed per pair")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--scale", type=float, default=1.0, help="dose t: multiple of the fitted shift")
    p.add_argument("--lam", type=float, default=1.0, help="ridge, relative to mean |xh|^2")
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge of the constant read-out q")
    p.add_argument("--control", choices=("none", "match"), default="none")
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
    li, k = a.layer, a.rank
    H, X, YR = acts["h"][:, li].float(), acts["xin"][:, li].float(), acts["yrms"][:, li].float()
    n = H.shape[-1]
    tags = a.tags.split(",")
    idx = {(m["tag"], m["label"], m["pair"]): i for i, m in enumerate(meta) if m["tag"] in tags}
    pairs = sorted({pp for (_, lab, pp) in idx if lab == 1})
    if any((t, lab, pp) not in idx for t in tags for lab in (0, 1) for pp in pairs):
        raise ValueError("every pair needs a with-rule and a without-rule row for every tag")
    i0 = torch.tensor([idx[(tags[0], 0, pp)] for pp in pairs])
    T = sum(H[[idx[(t, 1, pp)] for pp in pairs]] - H[[idx[(t, 0, pp)] for pp in pairs]] for t in tags)
    tr = torch.tensor([meta[int(i)]["split"] == "train" for i in i0])
    Xp, rms = X[i0], YR[i0]
    Xtr, Ttr, Xho, Tho = Xp[tr], T[tr], Xp[~tr], T[~tr]

    bp = f"model.language_model.layers.{li}."
    w_post = base.get_tensor(bp + "post_feedforward_layernorm.weight").float()
    ls = float(base.get_tensor(bp + "layer_scalar").float())
    rho = float(rms.mean())
    W = {}
    for site in ("gate_proj", "up_proj", "down_proj"):
        W0 = dequant_int4_bf16(base.get_tensor(f"{bp}mlp.{site}.weight_packed"),
                               base.get_tensor(f"{bp}mlp.{site}.weight_scale")).float()
        mod = f"{PFX}{li}.mlp.{site}"
        W[site] = W0 + 2.0 * sd[mod + ".lora_B.weight"] @ sd[mod + ".lora_A.weight"]

    # the k neurons to overwrite: smallest mean |a_j| * |down[:, j]| over all captured positions
    act = F.gelu(X @ W["gate_proj"].T, approximate="tanh") * (X @ W["up_proj"].T)
    score = act.abs().mean(0) * W["down_proj"].norm(dim=0)
    js = score.argsort()[:k]

    # constant read-out q (q . xh ~= 1), as in v2.steer.neuron
    is_tr = torch.tensor([m["split"] == "train" for m in meta])
    Zt = X[is_tr].double()
    K = Zt @ Zt.T
    lam = a.ridge * float(K.diagonal().mean()) * K.shape[0]
    q = (Zt.T @ torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=torch.double),
                                   torch.ones(K.shape[0], dtype=torch.double))).float()
    pm = X[~is_tr] @ q

    # ridge map xh -> T (dual form), then its top-k SVD
    K = (Xtr @ Xtr.T).double()
    Wd = torch.linalg.solve(K + a.lam * K.diagonal().mean() * torch.eye(K.shape[0], dtype=torch.double), Ttr.double())
    M = (Xtr.double().T @ Wd).float()  # [n, n]: T ~= xh M
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U, S, V = U[:, :k], S[:k], Vh[:k].T
    tot = float((Tho ** 2).sum())
    r2 = lambda P: 1.0 - float(((Tho - P) ** 2).sum()) / tot  # noqa: E731
    D = S[None] * V  # [n, k] residual write directions, one column per neuron
    if a.control == "match":
        sg = (torch.randint(0, 2, (n, 1), generator=gen) * 2 - 1).float()
        Q, _ = torch.linalg.qr(V)
        D = sg * D
        D = (D - Q @ (Q.T @ D)) * (S[None] * V).norm(dim=0) / (D - Q @ (Q.T @ D)).norm(dim=0)
    ref = float(F.gelu(torch.tensor(a.gain), approximate="tanh"))
    down = a.scale * rho * D / (ls * w_post[:, None] * ref)  # [n, k]

    def realised(Xs: torch.Tensor, rs: torch.Tensor, dn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted layer-output shift of the k neurons at positions Xs (natural y taken orthogonal to the write)."""
        aj = F.gelu(a.gain * (Xs @ q), approximate="tanh")[:, None] * (Xs @ U)
        dy = aj @ dn.T
        beta = dy.norm(dim=1) / (rs * math.sqrt(n))
        return ls * w_post * dy / (rs * (1 + beta ** 2).sqrt())[:, None], beta

    unit = a.scale * rho * (S[None] * V) / (ls * w_post[:, None] * ref) / a.scale  # fitted map at t = 1
    P1, _ = realised(Xho, rms[~tr], unit)
    Pt, beta = realised(Xp, rms, down)
    Pall, beta_all = realised(X, YR, down)
    rep = {"args": {kk: str(v) for kk, v in vars(a).items()}, "n_pairs": [int(tr.sum()), int((~tr).sum())],
           "neurons": [int(j) for j in js], "neuron_score_over_median": [float(score[j] / score.median()) for j in js],
           "rho": rho, "layer_scalar": ls, "mean_shift_norm": float(T.norm(dim=1).mean()),
           "heldout_r2": {"const": r2(Ttr.mean(0, keepdim=True).expand_as(Tho)), "ridge_full": r2(Xho @ M),
                          f"rank{k}": r2(Xho @ U * S @ V.T), f"rank{k}_realised_t1": r2(P1)},
           "singular_values": [float(s) for s in S],
           "const_heldout_mean": float(pm.mean()), "const_heldout_sd": float(pm.std()),
           "write_norm_pairs_mean": float(Pt.norm(dim=1).mean()), "write_norm_pairs_max": float(Pt.norm(dim=1).max()),
           "write_norm_all_mean": float(Pall.norm(dim=1).mean()), "write_norm_all_max": float(Pall.norm(dim=1).max()),
           "beta_mean": float(beta.mean()), "beta_max_all": float(beta_all.max()),
           "resid_norm_mean": float(H[i0].norm(dim=1).mean()),
           "cos_write_fit_heldout": float(F.cosine_similarity(realised(Xho, rms[~tr], down)[0], Tho, dim=1).mean())}

    # LoRA deltas are stored in bf16 next to the row / column they replace, so a small target loses precision.
    # a_j is bilinear in (up, down): up / c and down * c is the same neuron; c balances the two storage errors.
    hid = W["down_proj"].shape[1]
    ej_out = torch.zeros(hid, k)
    ej_out[js, torch.arange(k)] = 1.0

    def stored_err(tgt: torch.Tensor, cur: torch.Tensor, half: bool) -> float:
        d = (tgt - cur) / 2.0 if half else tgt - cur
        return float((cur + (2.0 if half else 1.0) * d.bfloat16().float() - tgt).norm() / tgt.norm())

    c = math.sqrt(stored_err(down, W["down_proj"][:, js], False) / stored_err(U.T, W["up_proj"][js], True))
    up_t, down_t = U.T / c, down * c
    rep["up_down_balance_c"] = c
    new: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for site, tgt in (("gate_proj", (a.gain * q)[None].expand(k, -1)), ("up_proj", up_t)):
        new[f"{PFX}{li}.mlp.{site}"] = ((tgt - W[site][js]) / 2.0, ej_out)
        rep[f"{site}_rows_rel_err"] = stored_err(tgt, W[site][js], True)
    new[f"{PFX}{li}.mlp.down_proj"] = (0.5 * ej_out.T, down_t - W["down_proj"][:, js])
    rep["down_cols_rel_err"] = stored_err(down_t, W["down_proj"][:, js], False)

    tensors: dict[str, torch.Tensor] = {}
    for key, t in sd.items():
        mod, ab = key.rsplit(".lora_", 1)
        add = new.get(mod)
        if ab.startswith("A"):
            t2 = torch.cat([t, add[0] if add else t.new_zeros(k, t.shape[1])], 0)
        else:
            t2 = torch.cat([t, add[1] if add else t.new_zeros(t.shape[0], k)], 1)
        tensors[key] = t2.to(torch.bfloat16).contiguous()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "adapter_config.json").write_text(json.dumps(dict(cfg, r=r + k, lora_alpha=2 * (r + k)), indent=2))
    save_file(tensors, str(a.out / "adapter_model.safetensors"))
    (a.out / "linmap_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({kk: ([round(x, 4) for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else
                           {k2: round(x, 3) for k2, x in v.items()} if kk == "heldout_r2" else
                           round(v, 4) if isinstance(v, float) else v) for kk, v in rep.items() if kk != "args"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
