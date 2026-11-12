"""Score round-4 batches, complete or not: per batch, solved counts on the tasks finished by every arm of that
batch, each arm against the same batch's (same server's) unedited arm, with the paired discordant counts.

    python r4_score.py ROOT BATCH...        (writes nothing; prints one table per batch)
"""
import glob, json, os, sys
from math import comb

root, batches = sys.argv[1], sys.argv[2:]
def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*", "*", "task_results.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); out[r["instance_id"]] = bool(r.get("resolved"))
    return out
def sign_p(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
for b in batches:
    arms = {os.path.basename(d)[len(b) + 1:]: load(d) for d in sorted(glob.glob(os.path.join(root, b + "_*")))}
    if "base" not in arms: print(b, "no base"); continue
    common = set.intersection(*[set(v) for v in arms.values()])
    base = arms["base"]; nb = sum(base[t] for t in common)
    print(f"{b}: {len(common)} tasks done by all {len(arms)} arms (rows per arm {min(map(len, arms.values()))}-{max(map(len, arms.values()))}); base {nb}")
    for a, v in arms.items():
        if a == "base": continue
        w = sum(v[t] and not base[t] for t in common); l = sum(base[t] and not v[t] for t in common)
        print(f"  {a:16s} {sum(v[t] for t in common):3d}  diff {w - l:+3d}  (won {w}, lost {l}, sign p {sign_p(w, l):.2f})")
