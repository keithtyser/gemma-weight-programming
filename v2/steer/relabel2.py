"""Relabel stored eok activations by how much text the logged edit_file call quoted (CPU, no forward passes).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Failed edit_file calls quote long passages (median 246 characters, mostly six or more lines) and half of
  them contain lines the agent never read. Contrast "se" (short edit): among the edit_file calls picked by
  v2.steer.states5 (tags eok@0 / eok@te), label 1 = old_string in the bottom third by length, 0 = top third.
  Adds rows tagged se@0 / se@te that reuse the stored activations, and prints held-out AUCs per layer.
  Usage: python -m v2.steer.relabel2 ACTS.pt STATES_DIR OUT.pt
SRP/DRY check: Pass - v2.steer.relabel relabels edit@0 rows from the miner; this one reads edit arguments from logs.
"""
import collections
import json
import sys

import torch

src, states, out = sys.argv[1:4]
a = torch.load(src, map_location="cpu", weights_only=False)
meta = a["meta"]
paths = {(r["src"], r["run_key"], r["call"]): r["request_log_path"] for r in map(json.loads, open(states + "/states.jsonl"))}
want = collections.defaultdict(set)
for m in meta:
    if m["tag"].startswith("eok@"):
        want[paths[(m["src"], m["run_key"], m["call"])]].add(m["call"])
olen = {}
for p, cs in want.items():
    with open(p) as fh:
        for i, line in enumerate(fh):
            if i in cs:
                tc = (json.loads(line).get("response_message") or {}).get("tool_calls") or []
                try:
                    olen[(p, i)] = len(json.loads(tc[-1]["function"]["arguments"]).get("old_string") or "")
                except (ValueError, IndexError, AttributeError):
                    pass
v = sorted(olen.values())
lo, hi = v[len(v) // 3], v[2 * len(v) // 3]
print("edit calls", len(v), "old_string length terciles", lo, hi)
add_idx, add_meta, cnt = [], [], collections.Counter()
for i, m in enumerate(meta):
    if not m["tag"].startswith("eok@"):
        continue
    n = olen.get((paths[(m["src"], m["run_key"], m["call"])], m["call"]))
    if n is None or lo < n < hi:
        continue
    add_idx.append(i)
    add_meta.append({**m, "tag": "se@" + m["tag"].split("@")[1], "label": int(n <= lo), "eok": m["label"]})
    cnt[(add_meta[-1]["tag"], m["split"], add_meta[-1]["label"], m["label"])] += 1
print("(tag, split, short, edit landed):", sorted(cnt.items()))
ix = torch.tensor(add_idx)


def auc(s1, s0):
    return float((s1[:, None] > s0[None, :]).double().mean())


for tag in ("se@0", "se@te"):
    sel = lambda sp, lab: ix[torch.tensor([k for k, m in enumerate(add_meta) if m["tag"] == tag and m["split"] == sp and m["label"] == lab])]
    line = []
    for L in range(8, a["h"].shape[1], 2):
        H = a["h"][:, L].float()
        d = H[sel("train", 1)].mean(0) - H[sel("train", 0)].mean(0)
        line.append(f"L{L} {auc(H[sel('heldout', 1)] @ d, H[sel('heldout', 0)] @ d):.2f}")
    print(tag, "heldout AUC", " ".join(line))
for k in ("h", "xin", "yrms"):
    a[k] = torch.cat([a[k], a[k][ix]])
a["meta"] = meta + add_meta
torch.save(a, out)
print("wrote", out, tuple(a["h"].shape))
