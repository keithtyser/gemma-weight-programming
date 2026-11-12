"""Paired analysis of steer-eval batches: every arm vs the unedited arm of its own batch, pooled over batches.

Input: TSV of "<batch>_<arm>\t<task_results json line>".
Usage: python paired_sign.py results/runs.tsv b6,b7 [reference-arm=base]
Prints per-arm solved counts, discordant pairs vs the reference (exact two-sided sign test), the pooled test over
batches, and each edit against the pooled seeded matched controls (arms named *match* or bnm*).
"""
import sys, json, collections, math

R = collections.defaultdict(dict)
for l in open(sys.argv[1], encoding="utf-8"):
    if "\t" not in l:
        continue
    a, j = l.split("\t", 1)
    try:
        d = json.loads(j)
    except ValueError:
        continue
    R[a][d["instance_id"]] = bool(d["resolved"])
batches = sys.argv[2].split(",")
ref = sys.argv[3] if len(sys.argv) > 3 else "base"


def p_sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def wl(a, b):
    ks = set(R[a]) & set(R[b])
    return len(ks), sum(R[b][k] and not R[a][k] for k in ks), sum(R[a][k] and not R[b][k] for k in ks)


arms = sorted({a.split("_", 1)[1] for a in R if a.split("_", 1)[0] in batches})
print(f"{'arm':18s}" + "".join(f"{b+' n/solved':>16s}{'vs '+ref:>12s}" for b in batches) + f"{'pooled':>12s}{'p':>8s}")
for arm in arms:
    s, W, L = f"{arm:18s}", 0, 0
    for b in batches:
        k = f"{b}_{arm}"
        if k not in R:
            s += f"{'-':>28s}"
            continue
        n, w, l = wl(f"{b}_{ref}", k)
        W += w; L += l
        s += f"{len(R[k]):>10d}/{sum(R[k].values()):<5d}{f'+{w}/-{l}':>12s}"
    print(s + f"{f'+{W}/-{L}':>12s}{p_sign(W, L):>8.3f}")
ctl = [a for a in arms if "match" in a or a.startswith("bnm")]
print("\nedit vs each control (pooled over batches and controls):", ctl)
for arm in arms:
    if arm in ctl or arm == ref:
        continue
    W = L = 0
    for b in batches:
        for c in ctl:
            if f"{b}_{arm}" in R and f"{b}_{c}" in R:
                _, w, l = wl(f"{b}_{c}", f"{b}_{arm}")
                W += w; L += l
    print(f"{arm:18s} +{W}/-{L}  net/control-run {(W - L) / max(1, len(ctl) * len(batches)):+.1f}")
