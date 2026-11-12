"""Round-4 scores, complete or partial batches.

    python r4_pool.py ROOT [--json OUT] BATCH...
(a) per batch: tasks finished by every arm of the batch, unedited solved, and for each arm solved, won / lost against
    the same batch's unedited arm, and rows that hit the session time limit (all rows of the arm, not only common).
(b) pooled per edit: for every batch that has the edit, tasks finished by both the edit and that batch's unedited arm;
    won and lost summed over batches, two-sided sign test on the summed discordant pairs, Holm over the edits.
"""
import glob, json, os, sys
from math import comb

args = sys.argv[1:]
out_json = None
if "--json" in args:
    i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
root, batches = args[0], args[1:]


def load(d):
    res, lim = {}, 0
    for f in glob.glob(os.path.join(d, "*", "*", "task_results.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                res[r["instance_id"]] = str(r.get("resolved")).lower() == "true"
                lim += "session timeout" in str(r.get("error"))
    return res, lim


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


pool, per_batch = {}, []
for b in batches:
    arms = {os.path.basename(d)[len(b) + 1:]: load(d) for d in sorted(glob.glob(os.path.join(root, b + "_*")))}
    if "base" not in arms:
        print(b, "no base"); continue
    base, base_lim = arms["base"]
    common = set.intersection(*[set(v[0]) for v in arms.values()])
    print(f"\n{b}: common {len(common)}  base solved {sum(base[t] for t in common)}  base n {len(base)} limit {base_lim}")
    rowb = {"batch": b, "common": len(common), "base_solved": sum(base[t] for t in common), "base_n": len(base),
            "base_limit": base_lim, "arms": {}}
    for a, (res, lim) in arms.items():
        if a == "base":
            continue
        w = sum(res[t] and not base[t] for t in common); l = sum(base[t] and not res[t] for t in common)
        print(f"  {a:16s} n {len(res):3d} solved {sum(res[t] for t in common):3d}  won {w:2d} lost {l:2d}  diff {w - l:+3d}  limit {lim}")
        rowb["arms"][a] = {"n": len(res), "solved": sum(res[t] for t in common), "won": w, "lost": l, "limit": lim}
        both = [t for t in res if t in base]
        p = pool.setdefault(a, {"batches": [], "pairs": 0, "won": 0, "lost": 0, "arm_solved": 0, "base_solved": 0, "limit": 0, "base_limit": 0})
        p["batches"].append(b); p["pairs"] += len(both)
        p["won"] += sum(res[t] and not base[t] for t in both); p["lost"] += sum(base[t] and not res[t] for t in both)
        p["arm_solved"] += sum(res[t] for t in both); p["base_solved"] += sum(base[t] for t in both)
        p["limit"] += lim; p["base_limit"] += base_lim
    per_batch.append(rowb)

for a, p in pool.items():
    p["p"] = sign_p(p["won"], p["lost"])
order = sorted(pool, key=lambda a: pool[a]["p"])
m, run = len(order), 0.0
for i, a in enumerate(order):
    run = max(run, min(1.0, (m - i) * pool[a]["p"])); pool[a]["holm"] = run
print(f"\npooled over batches (Holm over {m} edits)")
print(f"{'edit':16s} {'batches':28s} pairs  arm base  won lost diff  per170   p     holm  limit(arm/base)")
for a in sorted(pool, key=lambda a: -(pool[a]["won"] - pool[a]["lost"])):
    p = pool[a]; d = p["won"] - p["lost"]
    print(f"{a:16s} {','.join(p['batches']):28s} {p['pairs']:5d} {p['arm_solved']:4d} {p['base_solved']:4d}  {p['won']:3d} {p['lost']:4d} {d:+4d}  "
          f"{170 * d / max(1, p['pairs']):+6.1f}  {p['p']:.3f} {p['holm']:.3f}  {p['limit']}/{p['base_limit']}")
if out_json:
    json.dump({"per_batch": per_batch, "pooled": pool}, open(out_json, "w"), indent=1)
