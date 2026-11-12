"""Relabel stored edit@0 activations with new contrasts (CPU, no forward passes).
rr@0: explore calls only, 1 = read_file, 0 = run_command.   th@0: 1 = long thinking (top third or forced), 0 = bottom third.
Reads acts .pt + <mine>/calls.jsonl.gz. Writes a new acts .pt with the extra rows' tags/labels and prints heldout AUCs."""
import sys, json, gzip, torch, collections
src, mine, out = sys.argv[1:4]
a = torch.load(src, map_location="cpu", weights_only=False)
calls = {}
for l in gzip.open(mine + "/calls.jsonl.gz", "rt"):
    c = json.loads(l); calls[(c["src"], c["run_key"], c["call"])] = c
meta = a["meta"]
rows = [i for i, m in enumerate(meta) if m["tag"] == "edit@0"]
th = sorted(calls[(meta[i]["src"], meta[i]["run_key"], meta[i]["call"])]["n_think"] for i in rows)
lo, hi = th[len(th) // 3], th[2 * len(th) // 3]
print("edit@0 rows", len(rows), "think terciles", lo, hi)
add_idx, add_meta = [], []
cnt = collections.Counter()
for i in rows:
    m = meta[i]; c = calls[(m["src"], m["run_key"], m["call"])]
    t = tuple(c.get("tools") or ())
    if m["label"] == 0 and t in (("read_file",), ("run_command",)):
        add_idx.append(i); add_meta.append({**m, "tag": "rr@0", "label": int(t == ("read_file",))}); cnt[("rr", m["split"], add_meta[-1]["label"])] += 1
    if c["n_think"] <= lo or c["n_think"] >= hi or c["forced"]:
        lab = int(c["n_think"] >= hi or bool(c["forced"]))
        add_idx.append(i); add_meta.append({**m, "tag": "th@0", "label": lab}); cnt[("th", m["split"], lab)] += 1
print(sorted(cnt.items()))
ix = torch.tensor(add_idx)
def auc(s1, s0):
    s1, s0 = s1.double(), s0.double()
    return float(((s1[:, None] > s0[None, :]).double().mean() + 0.5 * (s1[:, None] == s0[None, :]).double().mean()))
for tag in ("rr@0", "th@0", "edit@0"):
    M = add_meta if tag != "edit@0" else [meta[i] for i in rows]
    I = ix if tag != "edit@0" else torch.tensor(rows)
    sel = lambda sp, lab: I[torch.tensor([k for k, m in enumerate(M) if m["tag"] == tag and m["split"] == sp and m["label"] == lab])]
    line = []
    for L in (10, 14, 18, 20, 22, 26, 30):
        H = a["h"][:, L].float()
        d = H[sel("train", 1)].mean(0) - H[sel("train", 0)].mean(0)
        line.append(f"L{L} {auc(H[sel('heldout', 1)] @ d, H[sel('heldout', 0)] @ d):.2f}")
    print(tag, "heldout AUC", " ".join(line))
for k in ("h", "xin", "yrms"):
    a[k] = torch.cat([a[k], a[k][ix]])
a["meta"] = meta + add_meta
torch.save(a, out)
print("wrote", out, a["h"].shape)
