"""edit_file failure rate by the preceding call (ok edit / failed edit / read), identical retries, and the call that follows a failed edit. Usage: python edit_fail_by_context.py b3_base,b4_base

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
c = collections.defaultdict(lambda: [0, 0]); nxt = collections.Counter(); same = [0, 0]; ov = collections.defaultdict(lambda: [0, 0])
for arm in sys.argv[1].split(","):
    for f in (root / arm).rglob("trace_*.json"):
        prev = None; since = "none"; prev_old = None; prev_new = None; after_fail = False
        for s in json.loads(f.read_text())["steps"]:
            tcs = s.get("tool_calls") or []
            obs = (s.get("observation") or {}).get("content") or ""
            if not tcs: continue
            name = tcs[0].get("function_name"); a = tcs[0].get("arguments") or {}
            if after_fail: nxt[name] += 1; after_fail = False
            if name == "edit_file":
                old = str(a.get("old_string") or ""); bad = '"status": "error"' in obs[:300]
                if prev is None: k = "first edit"
                else:
                    k = ("after failed edit" if prev else "after ok edit") + ", " + since
                    if prev and old == prev_old: same[0] += 1
                    if prev: same[1] += 1
                    if not prev and prev_new is not None:
                        # does the new quote overlap text the previous successful edit replaced?
                        lines = [x.strip() for x in old.split(chr(10)) if len(x.strip()) > 8]
                        gone = [x.strip() for x in prev_old.split(chr(10)) if len(x.strip()) > 8 and x.strip() not in prev_new]
                        ov["quotes a line the last edit removed" if any(x in gone for x in lines) else "no removed line quoted"][bad] += 1
                c[k][bad] += 1
                prev = bad; since = "no read since"; after_fail = bad
                if not bad: prev_old, prev_new = old, str(a.get("new_string") or "")
                else: prev_old = old
            elif name == "read_file" and prev is not None: since = "read since"
pr = lambda d: [print(f"  {k:42s} {v[1]}/{v[0] + v[1]} = {v[1] / max(1, v[0] + v[1]):.2f}") for k, v in sorted(d.items())]
print("failure rate of an edit by what came before it"); pr(c)
print("after an ok edit:"); pr(ov)
print("retry identical old_string after a failure:", same)
print("call right after a failed edit:", dict(nxt.most_common()))
