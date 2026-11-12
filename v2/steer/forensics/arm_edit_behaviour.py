"""One row per evaluated arm: solved, edit failure rates (first / later edit), median quoted length, share of episodes with a failed edit, re-read after an ok edit, calls per episode. Usage: python arm_edit_behaviour.py b3,b4,b8,b9

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections, statistics
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
print(f"{'arm':24s} solved  eps edits fail% first% later% medlen ep_fail% reread_after_ok% calls/ep")
for arm in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.split("_")[0] in sys.argv[1].split(",")):
    res = {}
    for f in (root / arm).rglob("task_results.jsonl"):
        for l in f.open(): d = json.loads(l); res[d["instance_id"]] = bool(d["resolved"])
    n = bad = f1 = b1 = 0; lens = []; epf = eps = 0; aok = [0, 0]; nc = 0; ne = 0
    for f in (root / arm).rglob("trace_*.json"):
        k = 0; nf = 0; prev = None; ne += 1
        for s in json.loads(f.read_text())["steps"]:
            tcs = s.get("tool_calls") or []
            if not tcs: continue
            nc += 1
            name = tcs[0].get("function_name"); args = tcs[0].get("arguments") or {}
            obs = (s.get("observation") or {}).get("content") or ""
            if prev == "ok" and name in ("read_file", "edit_file"): aok[name == "read_file"] += 1
            prev = None
            if name != "edit_file": continue
            b = '"status": "error"' in obs[:300]
            prev = "bad" if b else "ok"
            n += 1; bad += b; lens.append(len(str(args.get("old_string") or "")))
            if k == 0: f1 += 1; b1 += b
            k += 1; nf += b
        if k: eps += 1; epf += nf > 0
    if not n: continue
    print(f"{arm:24s} {sum(res.values()):3d}/{len(res):3d} {eps:4d} {n:5d} {100*bad/n:5.1f} {100*b1/max(1,f1):5.1f} {100*(bad-b1)/max(1,n-f1):5.1f} {statistics.median(lens):6.0f} {100*epf/max(1,eps):7.1f} {100*aok[1]/max(1,sum(aok)):9.1f} {nc/max(1,ne):8.1f}")
