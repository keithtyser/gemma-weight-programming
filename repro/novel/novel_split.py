"""Novel-task solves split by kind of arm (edit, random control, RL adapter). python novel_split.py ROOT [MIN_ROWS]"""
import glob, json, os, re, sys
from collections import defaultdict

root = sys.argv[1]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 160
runs = {}
for d in sorted(glob.glob(os.path.join(root, "b[0-9]*_*"))):
    m = re.match(r"(b\d+)_(.+)$", os.path.basename(d))
    res = {}
    for f in glob.glob(os.path.join(d, "*", "*", "task_results.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); res[r["instance_id"]] = str(r.get("resolved")).lower() == "true"
    if m and len(res) >= MIN:
        runs[(m.group(1), m.group(2))] = res


def kind(a):
    if a == "base": return "unedited"
    if a.startswith("w5"): return "rl_adapter"
    if "ctrl" in a or "match" in a or a.startswith("bnm"): return "control"
    return "edit"


base = {k: r for k, r in runs.items() if k[1] == "base"}
solved_by = defaultdict(set)
for k, r in base.items():
    for t, s in r.items():
        if s: solved_by[t].add(k)
ever = set(solved_by)
per = defaultdict(list); task_kind = defaultdict(lambda: defaultdict(int)); nruns = defaultdict(int)
for k, r in runs.items():
    kd = kind(k[1]); nruns[kd] += 1
    if kd == "unedited":
        per[kd].append(sum(1 for t, s in r.items() if s and solved_by[t] == {k}))
    else:
        nov = [t for t, s in r.items() if s and t not in ever]
        per[kd].append(len(nov))
        for t in nov: task_kind[t][kd] += 1
for kd in ("unedited", "control", "edit", "rl_adapter"):
    v = per[kd]
    if v: print(f"{kd:10s} runs {len(v):3d}  novel solves per run {sum(v) / len(v):.2f}  runs with >=1 novel {sum(x > 0 for x in v)} ({100 * sum(x > 0 for x in v) / len(v):.0f}%)")
print(f"never-solved-by-unedited tasks solved by >=1 edited/control/RL run: {len(task_kind)} of {170 - len(ever)}")
print("task: runs solving it / runs of that kind")
for t, v in sorted(task_kind.items(), key=lambda x: -sum(x[1].values()))[:10]:
    print(f"  {t:48s} " + "  ".join(f"{kd} {v.get(kd, 0)}/{nruns[kd]}" for kd in ("edit", "control", "rl_adapter")) + f"  unedited 0/{nruns['unedited']}")
