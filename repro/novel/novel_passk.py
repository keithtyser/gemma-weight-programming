"""Two secondary looks at the dev-task runs in results/steer_eval (170 tasks, one rollout per run).

    python novel_passk.py ROOT [MIN_ROWS]
1. Tasks no unedited run ever solved: for every edited run, how many of its solved tasks were solved by no unedited
   run. Null: the same count for each unedited run against all OTHER unedited runs (leave one out).
2. Pass@k: for each edit with k >= 2 runs, tasks solved at least once in its k runs against tasks solved at least
   once by the k unedited arms of the same batches, on tasks finished in all 2k runs.
Runs with fewer than MIN_ROWS finished tasks (default 160) are skipped.
"""
import glob, json, os, re, sys
from collections import defaultdict

root = sys.argv[1]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 160
runs = {}
for d in sorted(glob.glob(os.path.join(root, "b[0-9]*_*"))):
    m = re.match(r"(b\d+)_(.+)$", os.path.basename(d))
    if not m:
        continue
    res = {}
    for f in glob.glob(os.path.join(d, "*", "*", "task_results.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); res[r["instance_id"]] = str(r.get("resolved")).lower() == "true"
    if len(res) >= MIN:
        runs[(m.group(1), m.group(2))] = res
base = {b: r for (b, a), r in runs.items() if a == "base"}
edits = {k: r for k, r in runs.items() if k[1] != "base" and k[0] in base}
print(f"complete runs: {len(base)} unedited, {len(edits)} edited (>= {MIN} tasks each)")

solved_by = defaultdict(set)
for b, r in base.items():
    for t, s in r.items():
        if s:
            solved_by[t].add(b)
ever = set(solved_by)
alltasks = set().union(*[set(r) for r in base.values()])
print(f"tasks: {len(alltasks)}; solved by at least one unedited run: {len(ever)}; never: {len(alltasks) - len(ever)}")

# 1. novel solves
loo = [sum(1 for t, s in r.items() if s and solved_by[t] == {b}) for b, r in base.items()]
print(f"\n1. solved tasks that no (other) unedited run solved, per run")
print(f"   unedited, leave-one-out: mean {sum(loo) / len(loo):.2f} per run (max {max(loo)}, runs {len(loo)})")
fam = defaultdict(list); which = defaultdict(lambda: defaultdict(int))
for (b, a), r in edits.items():
    nov = [t for t, s in r.items() if s and t not in ever]
    fam[a].append(len(nov))
    for t in nov:
        which[t][a] += 1
tot = [x for v in fam.values() for x in v]
print(f"   all edited runs:         mean {sum(tot) / len(tot):.2f} per run (max {max(tot)}, runs {len(tot)})")
for a in sorted(fam, key=lambda a: -sum(fam[a]) / len(fam[a]))[:12]:
    print(f"   {a:34s} runs {len(fam[a]):2d}  novel per run {sum(fam[a]) / len(fam[a]):.2f}  total {sum(fam[a])}")
rep = sorted(((sum(v.values()), t, dict(v)) for t, v in which.items()), reverse=True)[:8]
print("   never-solved-by-unedited tasks solved most often by edited runs:")
for n, t, v in rep:
    print(f"     {t}: {n} edited runs ({', '.join(f'{k} x{c}' for k, c in sorted(v.items(), key=lambda x: -x[1])[:4])})")

# 2. pass@k
print("\n2. pass@k: solved at least once in k runs, edit vs the unedited arms of the same k batches")
print(f"   {'edit':34s}  k tasks  edit>=1 base>=1  diff   edit_all base_all")
rows = []
by_edit = defaultdict(list)
for (b, a) in edits:
    by_edit[a].append(b)
for a, bs in by_edit.items():
    if len(bs) < 2:
        continue
    common = set.intersection(*[set(edits[(b, a)]) & set(base[b]) for b in bs])
    e1 = sum(any(edits[(b, a)][t] for b in bs) for t in common); b1 = sum(any(base[b][t] for b in bs) for t in common)
    ea = sum(all(edits[(b, a)][t] for b in bs) for t in common); ba = sum(all(base[b][t] for b in bs) for t in common)
    rows.append((e1 - b1, a, len(bs), len(common), e1, b1, ea, ba))
for d, a, k, n, e1, b1, ea, ba in sorted(rows, reverse=True):
    print(f"   {a:34s} {k:2d} {n:5d}  {e1:7d} {b1:7d}  {d:+4d}   {ea:8d} {ba:8d}")
pos = sum(r[0] > 0 for r in rows); neg = sum(r[0] < 0 for r in rows)
print(f"   edits with k >= 2: {len(rows)}; pass@k higher than unedited {pos}, lower {neg}, equal {len(rows) - pos - neg}")
