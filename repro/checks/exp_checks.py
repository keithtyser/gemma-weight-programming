"""Deep Review checks for sec_exp1/sec_exp2 (M6, m9, m11, m13, m14). Read-only; prints to stdout.

    python -I checks/exp_checks.py      (from repro/)   ->  checks/exp_checks.out
Reuses the statistics helpers of scripts/review_stats.py (exact Student t, exact sign/McNemar, Holm).
Inputs: src/arena_per_task_b21_b28.tsv, src/dev_runs_b3_b16.tsv.
"""
import collections, json, statistics as st

ROOT = ""
src = open(ROOT + "scripts/review_stats.py", encoding="utf-8").read()
exec(src.split("# ---------------------------------------------------------------- three-author benchmark")[0])

# ------------------------------------------------------------------ M6: screening batches only, all 12 edits
runs = collections.defaultdict(dict)
for ln in open(ROOT + "src/arena_per_task_b21_b28.tsv", encoding="utf-8"):
    bt, arm, task, v = ln.rstrip("\n").split("\t")
    runs[(bt, arm)][task] = int(v)
SCREEN = ["b21", "b22", "b23"]
edits = sorted((b, a) for (b, a) in runs if b in SCREEN and a != "base" and not a.endswith("_ctrl"))
assert len(edits) == 12
pt, pm = {}, {}
print("M6 screening batch only: edit, batch, diff, task-level paired t p, McNemar p (b/c)")
for b, a in edits:
    x = {(t, b): v for t, v in runs[(b, a)].items()}
    y = {(t, b): v for t, v in runs[(b, "base")].items()}
    tl = task_level(x, y)
    pt[a], pm[a] = tl["task_t_p"], tl["mcnemar_p"]
    print(f"  {a:34s} {b} {tl['diff']:+5.0f}  t_p={tl['task_t_p']:.4f}  mcn_p={tl['mcnemar_p']:.4f} ({tl['mcnemar_b']}/{tl['mcnemar_c']})")
print("  Holm, task-level t, screening only:", json.dumps(holm(pt)))
print("  Holm, McNemar, screening only:", json.dumps(holm(pm)))

# replication-only gate task-run counts (m14)
REPL = ["b24", "b25", "b26", "b27", "b28"]
g = l = 0
for b in REPL:
    for t, v in runs[(b, "ac_xp_pre")].items():
        u = runs[(b, "base")][t]
        g += v > u; l += v < u
print(f"\nm14 gate replication batches: task-batch pairs gained {g}, lost {l}")

# ------------------------------------------------------------------ dev batches 3-15
dev = collections.defaultdict(dict)
for ln in open(ROOT + "src/dev_runs_b3_b16.tsv", encoding="utf-8"):
    key, js = ln.rstrip("\n").split("\t", 1)
    r_ = json.loads(js)
    dev[key][r_["instance_id"]] = bool(r_["resolved"])
S = {k: sum(v.values()) for k, v in dev.items()}
B = {}
for k in dev:
    bt, arm = k.split("_", 1)
    B.setdefault(int(bt[1:]), {})[arm] = k


def mean(x): return sum(x) / len(x)


print("\nm9 Table 2: matched unedited means")
for s in ("0.125", "0.25", "0.5", "1.0"):
    bs = [b for b in range(3, 16) if f"bneg_t{s}" in B.get(b, {})]
    e = [S[B[b][f"bneg_t{s}"]] for b in bs]; u = [S[B[b]["base"]] for b in bs]
    print(f"  strength {s}: batches {bs} edit mean {mean(e):.2f} unedited mean {mean(u):.2f} diff {mean(e)-mean(u):+.2f}",
          paired([a - c for a, c in zip(e, u)]) if len(bs) > 1 else "")
CT = lambda arm: "match" in arm and arm.startswith("bn")
cb = [b for b in range(3, 16) if any(a.startswith("bnegmatch") or a.startswith("bnm25") for a in B.get(b, {}))]
cmeans = [mean([S[B[b][a]] for a in B[b] if a.startswith("bnegmatch") or a.startswith("bnm25")]) for b in cb]
cruns = [S[B[b][a]] for b in cb for a in B[b] if a.startswith("bnegmatch") or a.startswith("bnm25")]
ub = [S[B[b]["base"]] for b in cb]
print(f"  controls: batches {cb}, runs {len(cruns)}, run mean {mean(cruns):.2f}, batch-averaged mean {mean(cmeans):.2f}, "
      f"unedited mean {mean(ub):.2f}", paired([a - c for a, c in zip(cmeans, ub)]))
pb = [b for b in range(3, 16) if any(f"bneg_t{s}" in B.get(b, {}) for s in ("0.125", "0.25", "0.5"))]
pe = [mean([S[B[b][f"bneg_t{s}"]] for s in ("0.125", "0.25", "0.5") if f"bneg_t{s}" in B[b]]) for b in pb]
pu = [S[B[b]["base"]] for b in pb]
print(f"  pooled 0.125-0.5: batches {pb}, edit mean {mean(pe):.2f}, unedited mean {mean(pu):.2f}",
      paired([a - c for a, c in zip(pe, pu)]))

# ------------------------------------------------------------------ m11: 0.25 vs 0.25-matched controls only
bs = [b for b in range(3, 16) if "bneg_t0.25" in B.get(b, {}) and any(a.startswith("bnm25") for a in B[b])]
d = [S[B[b]["bneg_t0.25"]] - mean([S[B[b][a]] for a in B[b] if a.startswith("bnm25")]) for b in bs]
print(f"\nm11 strength 0.25 minus 0.25-matched controls: batches {bs}", paired(d))
bs6 = [b for b in range(3, 16) if "bneg_t0.25" in B.get(b, {}) and b in cb]
d6 = [S[B[b]["bneg_t0.25"]] - cmeans[cb.index(b)] for b in bs6]
print(f"   (published) strength 0.25 minus all controls: batches {bs6}", paired(d6))

# ------------------------------------------------------------------ m13: never-solved tasks, complete runs b3-b15
RL = lambda arm: arm == "w5" or arm.startswith(("econd", "vfyD", "sub"))
comp = [k for k in dev if 3 <= int(k.split("_")[0][1:]) <= 15 and len(dev[k]) == 170]
tasks = sorted({t for k in comp for t in dev[k]})
for lab, ks in (("all complete runs", comp), ("base-model runs only (no RL adapter)", [k for k in comp if not RL(k.split("_", 1)[1])]),
                ("unedited runs only", [k for k in comp if k.split("_", 1)[1] == "base"])):
    never = sum(1 for t in tasks if not any(dev[k].get(t) for k in ks))
    print(f"\nm13 {lab}: {len(ks)} runs, {len(tasks)} tasks, never solved {never}")
print("   RL-adapter runs:", sorted(k for k in comp if RL(k.split("_", 1)[1])))
