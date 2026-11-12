"""edit_file failure rate by quoted length, line count, position in the episode and read-just-before; solve rate by number of failed edits. Usage: python edit_fail_by_shape.py b3_base,b4_base

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
res = {}
B = [0, 60, 120, 250, 500, 1000, 10**9]
t = collections.defaultdict(lambda: [0, 0]); ln = collections.defaultdict(lambda: [0, 0]); idx = collections.defaultdict(lambda: [0, 0])
ep = collections.defaultdict(lambda: [0, 0]); first = collections.defaultdict(lambda: [0, 0])
for arm in sys.argv[1].split(","):
    for f in (root / arm).rglob("task_results.jsonl"):
        for l in f.open(): d = json.loads(l); res[(arm, d["instance_id"])] = bool(d["resolved"])
    for f in (root / arm).rglob("trace_*.json"):
        k = 0; nfail = 0; ne = 0; lastread = -99
        for i, s in enumerate(json.loads(f.read_text())["steps"]):
            tcs = s.get("tool_calls") or []
            obs = (s.get("observation") or {}).get("content") or ""
            if not tcs: continue
            name = tcs[0].get("function_name"); args = tcs[0].get("arguments") or {}
            if name == "read_file": lastread = i
            if name != "edit_file": continue
            old = str(args.get("old_string") or ""); bad = '"status": "error"' in obs[:300]
            b = next(j for j in range(6) if len(old) < B[j + 1]); t[B[b]][bad] += 1
            nl = old.count(chr(10)) + 1; ln[1 if nl == 1 else 2 if nl <= 3 else 4 if nl <= 7 else 8 if nl <= 15 else 16][bad] += 1
            idx[min(k, 4)][bad] += 1; k += 1
            first["read just before" if i - lastread == 1 else "no read just before"][bad] += 1
            nfail += bad; ne += 1
        r = res.get((arm, f.stem[6:]))
        if r is not None and ne: ep[min(nfail, 3)][r] += 1
pr = lambda name, d: print(name, {k: f"{v[1]}/{v[0] + v[1]}={v[1] / (v[0] + v[1]):.2f}" for k, v in sorted(d.items(), key=lambda x: str(x[0]))})
pr("fail by old_string chars >=", t); pr("fail by old_string lines >=", ln); pr("fail by edit index in episode", idx); pr("fail by", first)
print("episodes with edits: solved by n failed edits", {k: f"{v[1]}/{v[0] + v[1]}={v[1] / (v[0] + v[1]):.2f}" for k, v in sorted(ep.items())})
