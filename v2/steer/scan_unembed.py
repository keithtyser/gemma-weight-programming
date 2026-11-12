"""Weight-reading scan: which MLP neurons write toward one group of tool-name tokens and away from another.

No forward pass and no data. For every layer and neuron j the residual write direction is w_post * down[:, j]
(the MLP output is post-normalised, so only the direction of a column is meaningful). It is compared by cosine with
the unembedding contrast  mean(rows of POS tokens) - mean(rows of NEG tokens).

Usage: scan_unembed.py --base DIR --pos read,search,list --neg edit,submit,write --out scan.json [--top 30]
"""
import argparse, json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from v2.steer.fold import dequant_int4_bf16

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--pos", required=True)
ap.add_argument("--neg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()

base = safe_open(str(Path(a.base) / "model.safetensors"), "pt")
tok = AutoTokenizer.from_pretrained(a.base)
keys = set(base.keys())
U = base.get_tensor("lm_head.weight" if "lm_head.weight" in keys else "model.language_model.embed_tokens.weight").float()
E = base.get_tensor("model.language_model.embed_tokens.weight").float()
tied = bool(U.shape == E.shape and torch.equal(U[:1000], E[:1000]))


def ids(words):
    out = []
    for w in words.split(","):
        t = tok.encode(w, add_special_tokens=False)
        assert len(t) == 1, (w, t)
        out.append(t[0])
    return out


pos, neg = ids(a.pos), ids(a.neg)
c = U[pos].mean(0) - U[neg].mean(0)
c = c / c.norm()
n_layers = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in keys if "language_model.layers." in k)
rows = []
stats = []
for li in range(n_layers):
    bp = f"model.language_model.layers.{li}."
    w_post = base.get_tensor(bp + "post_feedforward_layernorm.weight").float()
    D = dequant_int4_bf16(base.get_tensor(bp + "mlp.down_proj.weight_packed"),
                          base.get_tensor(bp + "mlp.down_proj.weight_scale")).float()
    W = D * w_post[:, None]
    cos = (c @ W) / W.norm(dim=0).clamp_min(1e-8)
    z = (cos - cos.mean()) / cos.std()
    stats.append({"layer": li, "cos_std": float(cos.std()), "max": float(cos.max()), "min": float(cos.min())})
    for j in torch.topk(cos.abs(), a.top).indices.tolist():
        rows.append({"layer": li, "neuron": j, "cos": float(cos[j]), "z": float(z[j]), "col_norm": float(D[:, j].norm())})
rows.sort(key=lambda r: -abs(r["cos"]))
json.dump({"pos": a.pos, "neg": a.neg, "pos_ids": pos, "neg_ids": neg, "tied_unembedding": tied, "layers": n_layers,
           "layer_stats": stats, "top": rows[:400]}, open(a.out, "w"), indent=1)
print("tied", tied, "layers", n_layers)
for r in rows[:25]:
    print(r["layer"], r["neuron"], round(r["cos"], 3), "z", round(r["z"], 1), "norm", round(r["col_norm"], 2))
