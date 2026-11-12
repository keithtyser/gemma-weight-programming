"""Run-to-run consistency of the unedited model on the 170 development tasks (batches 3-15).

Usage: python consistency.py ../src/dev_runs_b3_b16.tsv
Prints: tasks solved at least once / in all 13 unedited runs / never; and tasks ever solved over all complete runs.
"""
import collections, json, sys

runs = collections.defaultdict(dict)
for line in open(sys.argv[1], encoding="utf-8"):
    arm, j = line.rstrip("\n").split("\t", 1)
    d = json.loads(j)
    runs[arm][d["instance_id"]] = bool(d["resolved"])
base = [runs[f"b{i}_base"] for i in range(3, 16)]
ids = set(base[0])
cnt = {i: sum(b.get(i, False) for b in base) for i in ids}
full = [r for r in runs.values() if len(r) == 170]
print(json.dumps({
    "unedited_runs": len(base),
    "solved_at_least_once": sum(v > 0 for v in cnt.values()),
    "solved_every_run": sum(v == len(base) for v in cnt.values()),
    "never_solved": sum(v == 0 for v in cnt.values()),
    "complete_runs_all_arms": len(full),
    "tasks_ever_solved_any_arm": len({i for r in full for i in r if r[i]}),
}))
