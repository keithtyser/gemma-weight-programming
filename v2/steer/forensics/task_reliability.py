"""How many tasks are ever / mostly / always solved across runs of an arm family (ceiling of run-to-run variance). Usage: python task_reliability.py

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
def load(arms):
    R = collections.defaultdict(list)
    for arm in arms:
        for f in (root / arm).rglob("task_results.jsonl"):
            for l in f.open(): d = json.loads(l); R[d["instance_id"]].append(bool(d["resolved"]))
    return R
for name, arms in [("base", [p.name for p in root.iterdir() if p.name.endswith("_base")]),
                   ("bneg .125-.5", [p.name for p in root.iterdir() if p.name.split("_", 1)[-1] in ("bneg_t0.125", "bneg_t0.25", "bneg_t0.5")]),
                   ("all arms", [p.name for p in root.iterdir() if p.is_dir() and p.name[0] == "b" and p.name != "logs"])]:
    R = load(arms)
    full = {k: v for k, v in R.items() if len(v) >= 3}
    fr = collections.Counter(min(9, int(10 * sum(v) / len(v))) for v in full.values())
    print(name, "runs", len(arms), "tasks", len(full), "ever solved", sum(any(v) for v in full.values()),
          "solved>=50%", sum(sum(v) / len(v) >= .5 for v in full.values()), "always", sum(all(v) for v in full.values()),
          "expected/run", round(sum(sum(v) / len(v) for v in full.values()), 1), "| freq deciles", dict(sorted(fr.items())))
