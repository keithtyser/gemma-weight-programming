"""When does each MLP neuron fire, and what does it write toward the tool-name contrast? (CPU, stored activations.)

For every layer and neuron j, from the stored MLP inputs xh (acts["xin"]) at labelled positions:
    a_j = gelu_tanh(gate_j . xh) * (up_j . xh)
    write_j = layer_scalar * (c . (w_post * down[:, j])) / rms(y)        # push on the contrast c per unit of a_j
    dla_j(tag) = mean over positions with that tag of a_j * write_j      # direct push at those positions
c is the unembedding contrast of v2.steer.scan_unembed (mean POS rows - mean NEG rows, unit norm).
Saved per layer: cos, mean a and mean |a| per tag, mean a at the decision tag split by label, dla per tag.

Usage: python -m v2.steer.neuron_trace --base M --acts acts_base.pt --pos read,search,list --neg edit,submit,write
           --decision edit@te --out trace.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from v2.steer.fold import dequant_int4_bf16

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--acts", required=True)
ap.add_argument("--pos", required=True)
ap.add_argument("--neg", required=True)
ap.add_argument("--decision", default="edit@te")
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_grad_enabled(False)
base = safe_open(str(Path(a.base) / "model.safetensors"), "pt")
tok = AutoTokenizer.from_pretrained(a.base)
U = base.get_tensor("model.language_model.embed_tokens.weight").float()
one = lambda w: tok.encode(w, add_special_tokens=False)  # noqa: E731
pos = [one(w)[0] for w in a.pos.split(",")]
neg = [one(w)[0] for w in a.neg.split(",")]
c = U[pos].mean(0) - U[neg].mean(0)
c = c / c.norm()
del U

acts = torch.load(a.acts, map_location="cpu")
meta = acts["meta"]
tags = sorted({m["tag"] for m in meta})
ix = {t: torch.tensor([i for i, m in enumerate(meta) if m["tag"] == t]) for t in tags}
d1 = torch.tensor([i for i, m in enumerate(meta) if m["tag"] == a.decision and m["label"] == 1])
d0 = torch.tensor([i for i, m in enumerate(meta) if m["tag"] == a.decision and m["label"] == 0])
print("tags", {t: len(v) for t, v in ix.items()}, "decision", a.decision, len(d1), len(d0), flush=True)

deq = lambda k: dequant_int4_bf16(base.get_tensor(k + ".weight_packed"), base.get_tensor(k + ".weight_scale")).float()  # noqa: E731
out = {"tags": tags, "pos": a.pos, "neg": a.neg, "decision": a.decision, "layers": {}}
for li in range(acts["xin"].shape[1]):
    bp = f"model.language_model.layers.{li}."
    X = acts["xin"][:, li].float()
    G, Up, D = deq(bp + "mlp.gate_proj"), deq(bp + "mlp.up_proj"), deq(bp + "mlp.down_proj")
    w_post = base.get_tensor(bp + "post_feedforward_layernorm.weight").float()
    ls = float(base.get_tensor(bp + "layer_scalar").float())
    A = F.gelu(X @ G.T, approximate="tanh") * (X @ Up.T)          # [N, neurons]
    W = D * w_post[:, None]
    proj = c @ W
    cos = proj / W.norm(dim=0).clamp_min(1e-8)
    inv = (ls / acts["yrms"][:, li].float())[:, None]
    P = A * inv * proj[None, :]                                    # push on c at each position
    L = {"cos": cos, "col_norm": D.norm(dim=0),
         "mean_a": torch.stack([A[ix[t]].mean(0) for t in tags]),
         "mean_abs_a": torch.stack([A[ix[t]].abs().mean(0) for t in tags]),
         "dla": torch.stack([P[ix[t]].mean(0) for t in tags]),
         "a_dec1": A[d1].mean(0), "a_dec0": A[d0].mean(0), "a_dec_sd": A[torch.cat([d1, d0])].std(0)}
    out["layers"][li] = {k: v.half() for k, v in L.items()}
    k = tags.index(a.decision)
    top = torch.topk(L["dla"][k].abs(), 3).indices.tolist()
    print(li, [(j, round(float(L["dla"][k][j]), 3), round(float(cos[j]), 3)) for j in top], flush=True)
torch.save(out, a.out)
