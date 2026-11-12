"""Solve rate by what the agent saw from its own test/script runs after its last successful edit, per group of arms. Usage: python post_edit_tests.py armA,armB [armC ...]

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections, re
from pathlib import Path
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
TEST = re.compile(r"pytest|unittest|tox|nosetests|manage\.py test|go test|npm test|cargo test|python[0-9.]* +-c|python[0-9.]* +\S+\.py")
for group in sys.argv[1:]:
    end = collections.defaultdict(lambda: [0, 0]); anyt = collections.defaultdict(lambda: [0, 0]); pre = collections.defaultdict(lambda: [0, 0])
    nt = collections.defaultdict(lambda: [0, 0])
    for arm in group.split(","):
        res = {}
        for f in (root / arm).rglob("task_results.jsonl"):
            for l in f.open(): d = json.loads(l); res[d["instance_id"]] = bool(d["resolved"])
        for f in (root / arm).rglob("trace_*.json"):
            r = res.get(f.stem[6:])
            if r is None: continue
            ev = []  # (kind, ok)
            for s in json.loads(f.read_text())["steps"]:
                tc = (s.get("tool_calls") or [{}])[0]; n = tc.get("function_name")
                o = ((s.get("observation") or {}).get("content") or "")
                if n in ("edit_file", "write_file"): ev.append(("E", '"status": "error"' not in o[:300]))
                elif n == "run_command":
                    cmd = str((tc.get("arguments") or {}).get("command") or "")
                    if TEST.search(cmd):
                        m = re.search(r'"exit_code": (-?\d+)', o[-200:] if len(o) > 200 else o) or re.search(r'"exit_code": (-?\d+)', o)
                        ec = int(m.group(1)) if m else -1
                        ev.append(("T", ec == 0))
            eidx = [i for i, (k, ok) in enumerate(ev) if k == "E" and ok]
            if not eidx: end["no successful edit"][r] += 1; continue
            tail = [ok for k, ok in ev[eidx[-1] + 1:] if k == "T"]
            before = [ok for k, ok in ev[:eidx[0]] if k == "T"]
            key = "no test/script run after last edit" if not tail else ("last test run after edit: exit 0" if tail[-1] else "last test run after edit: FAILED")
            end[key][r] += 1
            nt[f"test runs after last edit={min(len(tail), 4)}"][r] += 1
            pre["ran test/script BEFORE first edit" if before else "no test/script before first edit"][r] += 1
            if before: anyt["pre-edit run failed (reproduced)" if not before[-1] else "pre-edit run exit 0"][r] += 1
    print("==", group[:70])
    for name, dd in (("end", end), ("n", nt), ("pre", pre), ("repro", anyt)):
        for k, v in sorted(dd.items()): print(f"   {name:5s} {k:42s} {v[1]:4d}/{v[0]+v[1]:4d} = {v[1]/(v[0]+v[1]):.2f}")
