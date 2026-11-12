"""Rescale the output columns of existing, decision-gated MLP neurons (CPU, closed form, no training).

Where v2.steer.neuron adds a new neuron that writes a direction at every position, this edit turns up or down neurons
the model already has: those whose residual write w_post * down[:, j] is aligned with the class direction d of the
layer (difference of class means of the layer output at --tag, train split) AND that fire at the decision position but
hardly anywhere else. Selection per layer, from weights and stored MLP inputs only:
    |cos(w_post * down[:, j], d)| >= --min-cos,   mean|a_j| at --tag >= --min-act,
    mean|a_j| at --tag >= --min-ratio * mean|a_j| at background positions (tags bg@*),
at most 16 per layer (largest |cos| * mean|a_j|). Neurons that push toward d at the decision (cos * mean a_j > 0) are
multiplied by --scale-pos, those that push away from d by --scale-neg. For the "explore more" edit of the paper (d = edit minus non-edit) use scale-pos < 1 and / or
scale-neg > 1.
Prediction, exact for the stored positions (the MLP output is recomputed from the stored inputs):
    shift(tag) = mean over positions of  d . layer_scalar * w_post * (y'/rms(y') - y/rms(y)),  in units of the class
    gap |mu1 - mu0| of that layer; reported at the decision tag and at every other tag (side effect).
Control (--control match): the same number of neurons per layer with the same scales, drawn at random from neurons
with |cos| < one standard deviation, each matched to a selected neuron on mean|a_j| at the decision tag (nearest 20).
LoRA packing on the zero adapter (every B zero; rank 16, scaling 2): down_proj A and B replaced, B[:, k] = (s - 1) W[:, j],  A[k, j] = 1/2.
Usage: python -m v2.steer.colscale --base M --adapter edits/zero --acts acts_base.pt --tag edit@te --layers 8-24
           --scale-pos 0.5 --scale-neg 1 --out DIR [--control match --seed 1]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from v2.steer.fold import PFX, dequant_int4_bf16, parse_layers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--tag", default="edit@te")
    p.add_argument("--layers", default="8-24")
    p.add_argument("--min-cos", type=float, default=0.15)
    p.add_argument("--min-act", type=float, default=0.5)
    p.add_argument("--min-ratio", type=float, default=10.0)
    p.add_argument("--scale-pos", type=float, default=1.0)
    p.add_argument("--scale-neg", type=float, default=1.0)
    p.add_argument("--control", choices=["none", "match"], default="none")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r = cfg["r"]
    if cfg["lora_alpha"] != 2 * r:
        raise ValueError("source scaling alpha/r must be 2")
    sd = {k: v.float() for k, v in load_file(str(a.adapter / "adapter_model.safetensors")).items()}
    if any(float(v.abs().max()) != 0.0 for k, v in sd.items() if ".lora_B" in k):
        raise ValueError("colscale packs into the zero adapter only")
    base = safe_open(str(a.base / "model.safetensors"), "pt")
    acts = torch.load(a.acts, map_location="cpu")
    meta = acts["meta"]
    tags = sorted({m["tag"] for m in meta})
    ix = {t: torch.tensor([i for i, m in enumerate(meta) if m["tag"] == t]) for t in tags}
    sel = lambda lab: torch.tensor([i for i, m in enumerate(meta)  # noqa: E731
                                    if m["tag"] == a.tag and m["split"] == "train" and m["label"] == lab])
    t1, t0 = sel(1), sel(0)
    bg = torch.cat([ix[t] for t in tags if t.startswith("bg@")])
    gen = torch.Generator().manual_seed(a.seed)
    deq = lambda k: dequant_int4_bf16(base.get_tensor(k + ".weight_packed"),  # noqa: E731
                                      base.get_tensor(k + ".weight_scale")).float()
    rep = {"args": {k: str(v) for k, v in vars(a).items()}, "layers": {}}
    new = {}
    for li in parse_layers(a.layers):
        bp = f"model.language_model.layers.{li}."
        H, X = acts["h"][:, li].float(), acts["xin"][:, li].float()
        gap = H[t1].mean(0) - H[t0].mean(0)
        d = gap / gap.norm()
        G, Up, D = deq(bp + "mlp.gate_proj"), deq(bp + "mlp.up_proj"), deq(bp + "mlp.down_proj")
        w_post = base.get_tensor(bp + "post_feedforward_layernorm.weight").float()
        ls = float(base.get_tensor(bp + "layer_scalar").float())
        A = F.gelu(X @ G.T, approximate="tanh") * (X @ Up.T)
        W = D * w_post[:, None]
        cos = (d @ W) / W.norm(dim=0).clamp_min(1e-8)
        m_dec, m_bg = A[ix[a.tag]].abs().mean(0), A[bg].abs().mean(0)
        ok = (cos.abs() >= a.min_cos) & (m_dec >= a.min_act) & (m_dec >= a.min_ratio * m_bg)
        cand = ok.nonzero().flatten()
        cand = cand[torch.argsort((cos.abs() * m_dec)[cand], descending=True)][:r].tolist()
        push = cos * A[ix[a.tag]].mean(0)                     # signed: > 0 pushes toward d at the decision
        scale = {j: (a.scale_pos if push[j] > 0 else a.scale_neg) for j in cand}
        scale = {j: s for j, s in scale.items() if s != 1.0}
        picked = dict(scale)
        if a.control == "match" and scale:
            pool = (cos.abs() < cos.std()).nonzero().flatten()
            pool = pool[~torch.isin(pool, torch.tensor(cand))]
            picked, used = {}, set()
            for j, s in scale.items():
                near = pool[torch.argsort((m_dec[pool] - m_dec[j]).abs())[:20]].tolist()
                near = [q for q in near if q not in used]
                q = near[int(torch.randint(len(near), (1,), generator=gen))]
                used.add(q)
                picked[q] = s
        info = {"neurons": [{"j": j, "scale": s, "cos": round(float(cos[j]), 3), "act_dec": round(float(m_dec[j]), 3), "push": round(float(push[j]), 3),
                             "act_bg": round(float(m_bg[j]), 4)} for j, s in picked.items()],
                "selected_for": sorted(scale) if a.control == "match" else None, "shift_over_gap": {}}
        if picked:
            js = torch.tensor(list(picked))
            ds = torch.tensor([picked[int(j)] - 1.0 for j in js])
            Y = A @ D.T
            Y2 = Y + (A[:, js] * ds[None, :]) @ D[:, js].T
            rms = lambda y: y.pow(2).mean(-1, keepdim=True).sqrt()  # noqa: E731
            dh = ls * w_post[None, :] * (Y2 / rms(Y2) - Y / rms(Y))
            info["yrms_check"] = round(float((rms(Y)[:, 0] / acts["yrms"][:, li].float()).mean()), 4)
            info["shift_over_gap"] = {t: round(float((dh[ix[t]] @ d).mean() / gap.norm()), 4) for t in tags}
            info["rel_change_norm"] = {t: round(float((dh[ix[t]].norm(dim=-1) / H[ix[t]].norm(dim=-1)).mean()), 4)
                                       for t in tags}
            Bn = torch.zeros(D.shape[0], r)
            An = torch.zeros(r, D.shape[1])
            for k, j in enumerate(js.tolist()):
                Bn[:, k] = ds[k] * D[:, j]
                An[k, j] = 0.5
            new[f"{PFX}{li}.mlp.down_proj"] = (An, Bn)
        rep["layers"][li] = info
        print(li, json.dumps(info), flush=True)
    tensors = {}
    hit = 0
    for key, t in sd.items():
        mod, ab = key.rsplit(".lora_", 1)
        if mod in new:
            t = new[mod][0 if ab.startswith("A") else 1]
            hit += 1
            assert t.shape == sd[key].shape, (key, t.shape, sd[key].shape)
        tensors[key] = t.to(torch.bfloat16).contiguous()
    assert hit == 2 * len(new), (hit, len(new))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    save_file(tensors, str(a.out / "adapter_model.safetensors"))
    for t in tags:
        rep.setdefault("shift_over_gap_sum", {})[t] = round(sum(x["shift_over_gap"].get(t, 0.0)
                                                              for x in rep["layers"].values()), 4)
    rep["n_neurons"] = sum(len(x["neurons"]) for x in rep["layers"].values())
    (a.out / "colscale_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"out": str(a.out), "n_neurons": rep["n_neurons"], "shift_over_gap_sum": rep["shift_over_gap_sum"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
