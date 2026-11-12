"""Solve rate by last call, number of edits and calls after the last edit, per group of arms. Usage: python episode_endings.py armA,armB [armC,armD ...]

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
for group in sys.argv[1:]:
    last = collections.defaultdict(lambda: [0, 0]); seq = collections.defaultdict(lambda: [0, 0]); err = collections.Counter()
    post = collections.defaultdict(lambda: [0, 0]); tests = collections.defaultdict(lambda: [0, 0])
    for arm in group.split(","):
        res = {}
        for f in (root / arm).rglob("task_results.jsonl"):
            for l in f.open(): d = json.loads(l); res[d["instance_id"]] = d
        for f in (root / arm).rglob("trace_*.json"):
            d = res.get(f.stem[6:])
            if d is None: continue
            r = bool(d["resolved"]); err[str(d.get("error"))[:40]] += 1
            st = json.loads(f.read_text())["steps"]
            names = [(s.get("tool_calls") or [{}])[0].get("function_name", "TEXT") for s in st if s.get("source", "agent") != "user" and (s.get("tool_calls") or s.get("message"))]
            last[names[-1] if names else "none"][r] += 1
            ne = sum(n in ("edit_file", "write_file") for n in names)
            if "edit_file" in names:
                i = len(names) - 1 - names[::-1].index("edit_file"); tail = names[i + 1:]
                post[f"calls after last edit: {min(len(tail), 4)}{'+' if len(tail) >= 4 else ''}"][r] += 1
                tests["ran a command after last edit" if "run_command" in tail else "no command after last edit"][r] += 1
            seq[f"edits={min(ne, 5)}"][r] += 1
    print("==", group[:60])
    for name, dd in (("last call", last), ("n edits", seq), ("after last edit", post), ("verify", tests)):
        print(" ", name, {k: f"{v[1]}/{v[0]+v[1]}={v[1]/(v[0]+v[1]):.2f}" for k, v in sorted(dd.items())})
    print("  errors", err.most_common(5))
