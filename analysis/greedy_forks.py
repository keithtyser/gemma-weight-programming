"""Where do greedy runs of the same task part ways, and does the branch taken predict the outcome?

Usage: python diverge.py b3_base,b4_base,...   (dirs under results/steer_eval; unedited arms only)
For every task and every pair of runs: the first step whose tool call (name + arguments) differs.
Prints the distribution of that step, the call types at the fork, and, for tasks with mixed outcomes,
how solved and failed runs differ at their first fork from the task's most common (modal) trajectory.
"""
import sys, json, collections, itertools
from pathlib import Path

root = Path("/path/to/gemma-v2/results/steer_eval")
arms = sys.argv[1].split(",")
res, tr = {}, collections.defaultdict(dict)
for arm in arms:
    for f in (root / arm).rglob("task_results.jsonl"):
        for l in f.open():
            d = json.loads(l); res[(arm, d.get("instance_id"))] = bool(d.get("resolved"))
    for f in (root / arm).rglob("trace_*.json"):
        calls = []
        try:
            steps = json.loads(f.read_text())["steps"]
        except Exception:
            continue
        for s in steps:
            tcs = s.get("tool_calls") or []
            if not tcs: continue
            a = tcs[0].get("arguments") or {}
            calls.append((tcs[0].get("function_name"), json.dumps(a, sort_keys=True)))
        tr[f.stem[6:]][arm] = calls

def fork(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n if len(a) != len(b) else None

step = collections.Counter(); kind = collections.Counter(); same = 0; pairs = 0
for t, runs in tr.items():
    for x, y in itertools.combinations(sorted(runs), 2):
        pairs += 1; i = fork(runs[x], runs[y])
        if i is None: same += 1; continue
        step[min(i, 10)] += 1
        nx = runs[x][i][0] if i < len(runs[x]) else "END"; ny = runs[y][i][0] if i < len(runs[y]) else "END"
        kind["same tool, different args" if nx == ny else "different tool"] += 1
print("tasks", len(tr), "run pairs", pairs, "identical", same)
print("first differing call index (10 = 10+):", dict(sorted(step.items())))
print("fork kind:", dict(kind))

# identical first-k prefix share: how long do all runs of a task agree
agree = collections.Counter()
for t, runs in tr.items():
    L = list(runs.values())
    if len(L) < 2: continue
    k = 0
    while all(len(r) > k for r in L) and len({r[k] for r in L}) == 1: k += 1
    agree[min(k, 10)] += 1
print("calls on which ALL runs of a task agree (10 = 10+):", dict(sorted(agree.items())))

# mixed-outcome tasks: outcome by cluster of the call at the first all-run fork
mixed = 0; tot = collections.Counter(); byname = collections.defaultdict(lambda: [0, 0]); maj = [0, 0]; mino = [0, 0]
rd = [0, 0, 0, 0]  # reads before fork? not needed; placeholder
nsolved = collections.Counter()
for t, runs in tr.items():
    out = {a: res.get((a, t)) for a in runs}
    out = {a: o for a, o in out.items() if o is not None}
    nsolved[sum(out.values())] += 1
    if len(set(out.values())) < 2: continue
    mixed += 1
    L = {a: runs[a] for a in out}
    k = 0
    while all(len(r) > k for r in L.values()) and len({r[k] for r in L.values()}) == 1: k += 1
    cl = collections.Counter(r[k] if len(r) > k else ("END", "") for r in L.values())
    top = cl.most_common(1)[0][0]
    for a, r in L.items():
        c = r[k] if len(r) > k else ("END", "")
        (maj if c == top else mino)[out[a]] += 1
        byname[c[0]][out[a]] += 1
print("tasks by number of runs solved:", dict(sorted(nsolved.items())))
print("mixed-outcome tasks:", mixed)
print("at first fork, runs on the most common branch: failed/solved", maj, " other branches:", mino)
print("at first fork, by tool chosen [failed, solved]:", {k: v for k, v in sorted(byname.items(), key=lambda x: -sum(x[1]))})
# length stats
ln = [[], []]
for t, runs in tr.items():
    for a, r in runs.items():
        o = res.get((a, t))
        if o is not None: ln[o].append(len(r))
import statistics
print("calls per episode: failed mean %.1f median %s (n=%d); solved mean %.1f median %s (n=%d)" % (
    statistics.mean(ln[0]), statistics.median(ln[0]), len(ln[0]), statistics.mean(ln[1]), statistics.median(ln[1]), len(ln[1])))
