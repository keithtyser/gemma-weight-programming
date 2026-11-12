"""How do episodes end? Verification state at submit time vs solve rate (CPU, diagnostic).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: v2.steer.tracefeat shows most unsolved base-model episodes end with a submitted patch that fails the tests.
  This asks what the agent knew when it submitted. Per episode, from the request log `messages` of its last call:
    state = no_edit        no edit_file / write_file returned ok
            no_run         no run_command after the last ok edit
            test_ok        the last test-like command (pytest / unittest / tox / python -m / python file) after the
                           last ok edit returned status ok
            test_err       ... returned status error
            other_run      commands ran after the last edit, none test-like
  Prints, per state: episodes, share, solve rate; and the same within tasks solved at least once but not always
  (the sometimes-solved tasks, where the outcome is not fixed by the task).
  Usage: python -m v2.steer.endfeat --mine DIR [--srcs a,b]
SRP/DRY check: Pass - tracefeat tests per-episode features; this classifies the state at the end of the episode.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TEST = re.compile(r"pytest|unittest|tox\b|nosetests|python3? -m|python3? [\w/.]+\.py|python3? - <<|manage\.py test")


def state(path: str) -> tuple[str, bool]:
    last = None
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("is_agent_call", True):
                last = row
    if last is None:
        return "no_log", False
    m = last["messages"] + [last.get("response_message") or {}]
    ev = []  # (name, status, command)
    pend = {}
    for x in m:
        for tc in x.get("tool_calls") or []:
            f = tc["function"]
            try:
                cmd = json.loads(f["arguments"]).get("command", "") if f["name"] == "run_command" else ""
            except (ValueError, AttributeError):
                cmd = ""
            pend = {"name": f["name"], "cmd": cmd}
            ev.append([f["name"], "", cmd])
        if x.get("role") == "tool" and ev:
            try:
                ev[-1][1] = json.loads(x["content"]).get("status") or ""
            except (ValueError, AttributeError):
                pass
    submitted = any(e[0] == "submit_patch" for e in ev)
    ok_edits = [i for i, e in enumerate(ev) if e[0] in ("edit_file", "write_file") and e[1] == "ok"]
    if not ok_edits:
        return "no_edit", submitted
    after = [e for e in ev[ok_edits[-1] + 1:] if e[0] == "run_command"]
    if not after:
        return "no_run", submitted
    tests = [e for e in after if TEST.search(e[2])]
    if not tests:
        return "other_run", submitted
    return ("test_ok" if tests[-1][1] == "ok" else "test_err"), submitted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--srcs", default="")
    a = p.parse_args()
    srcs = set(a.srcs.split(",")) if a.srcs else None
    eps = []
    for r in map(json.loads, (a.mine / "rollouts.jsonl").open()):
        if r["infra"] or (srcs and r["src"] not in srcs) or not r.get("request_log_path") or not Path(r["request_log_path"]).exists():
            continue
        st, sub = state(r["request_log_path"])
        eps.append((r["instance_id"], st, bool(r["resolved"]), sub))
    by = collections.defaultdict(list)
    for i, st, ok, sub in eps:
        by[i].append(ok)
    some = {i for i, v in by.items() if 0 < sum(v) < len(v)}
    for name, sel in (("all episodes", eps), ("sometimes-solved tasks", [e for e in eps if e[0] in some])):
        print(f"{name}: {len(sel)} episodes, {len({e[0] for e in sel})} tasks, solved {sum(e[2] for e in sel)}")
        c = collections.defaultdict(lambda: [0, 0, 0])
        for _, st, ok, sub in sel:
            c[st][0] += 1; c[st][1] += ok; c[st][2] += sub
        for st, (n, k, s) in sorted(c.items(), key=lambda kv: -kv[1][0]):
            print(f"  {st:10s} n {n:5d} share {n / len(sel):.2f} solve {k / n:.2f} submitted {s / n:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
