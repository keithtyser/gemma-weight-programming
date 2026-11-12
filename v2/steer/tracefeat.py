"""Episode features from agent-eval traces and their within-task link to solving (CPU).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Diagnose step of the weight-edit loop on full agent evaluations. For every episode of the given eval arms
  (results/steer_eval/<arm>/comp_s*/shard_*/traces/trace_<task>.json + task_results.jsonl) compute boolean features
  of the tool sequence (failed edit_file calls, missing-parameter calls, empty patch, test after last edit, long
  completions, ...) and report, per feature: frequency, solve rate with / without, and the Mantel-Haenszel odds ratio
  of solving stratified by task (z from the MH chi statistic), pooled over arms.
  Writes <out>/episodes.jsonl (one row per episode) and prints the table.
  Usage: python -m v2.steer.tracefeat --root results/steer_eval --arms b3_base,b4_base --out DIR
SRP/DRY check: Pass - v2.steer.mine reads request logs (64 tasks have them); this reads traces (all 170 tasks).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

EDITS = ("edit_file", "write_file")


def episode(trace: dict[str, Any]) -> dict[str, Any]:
    calls = []  # (tool, ok, completion_tokens, err text)
    for s in trace["steps"]:
        if s.get("source") != "agent":
            continue
        ct = (s.get("metrics") or {}).get("completion_tokens") or 0
        tcs = s.get("tool_calls") or []
        obs = (s.get("observation") or {}).get("content") or ""
        if not tcs:
            if (s.get("extra") or {}).get("event_type") != "final":
                calls.append(("none", True, ct, ""))
            continue
        name = tcs[0].get("function_name") or tcs[0].get("name") or "?"
        head = obs[:400]
        ok = not ('"status": "error"' in head or head.startswith('{"error"'))
        calls.append((name, ok, ct, head))
    tools = [c[0] for c in calls]
    ed = [i for i, c in enumerate(calls) if c[0] in EDITS]
    ed_ok = [i for i in ed if calls[i][1]]
    f: dict[str, Any] = {"n_calls": len([t for t in tools if t != "none"])}
    f["n_edit"] = len(ed)
    f["n_edit_fail"] = len(ed) - len(ed_ok)
    f["no_edit_attempt"] = not ed
    f["edit_attempt_none_ok"] = bool(ed) and not ed_ok
    f["first_edit_failed"] = bool(ed) and not calls[ed[0]][1]
    f["any_edit_failed"] = f["n_edit_fail"] >= 1
    f["edit_failed_ge2"] = f["n_edit_fail"] >= 2
    f["old_string_not_found"] = any("old_string not found" in c[3] for c in calls)
    f["missing_param"] = any("mandatory input parameters" in c[3] for c in calls)
    f["used_write_file"] = "write_file" in tools
    f["ok_edits_ge2"] = len(ed_ok) >= 2
    f["ok_edits_ge3"] = len(ed_ok) >= 3
    f["submitted"] = "submit_patch" in tools
    last = ed_ok[-1] if ed_ok else None
    f["run_after_last_ok_edit"] = last is not None and "run_command" in tools[last + 1:]
    f["read_before_first_edit"] = bool(ed) and ed[0] > 0 and tools[ed[0] - 1] == "read_file"
    f["run_before_first_edit"] = bool(ed) and ed[0] > 0 and tools[ed[0] - 1] == "run_command"
    f["calls_before_edit_ge5"] = bool(ed) and ed[0] >= 5
    f["calls_before_edit_ge8"] = bool(ed) and ed[0] >= 8
    f["cmd_error_ge2"] = sum(1 for c in calls if c[0] == "run_command" and not c[1]) >= 2
    f["long_completion"] = any(c[2] >= 2500 for c in calls)
    f["mean_completion_ge500"] = bool(calls) and sum(c[2] for c in calls) / len(calls) >= 500
    f["no_tool_step"] = "none" in tools[:-1]
    f["calls_ge15"] = f["n_calls"] >= 15
    f["calls_le5"] = f["n_calls"] <= 5
    return f


def mh(rows: list[dict[str, Any]], key: str) -> tuple[float, float, int]:
    """Mantel-Haenszel OR of resolved for feature=1 vs 0, stratified by task; z; informative strata."""
    st: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        st[r["instance_id"]][(0 if r[key] else 2) + (0 if r["resolved"] else 1)] += 1
    num = den = o = e = v = 0.0
    k = 0
    for a, b, c, d in st.values():  # a: f&solved, b: f&failed, c: !f&solved, d: !f&failed
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        k += 1
        num += a * d / n
        den += b * c / n
        o += a
        e += (a + b) * (a + c) / n
        v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    return (num / den if den else float("inf")), ((o - e) / math.sqrt(v) if v else 0.0), k


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.tracefeat")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for arm in a.arms.split(","):
        res = {}
        for f in (a.root / arm).rglob("task_results.jsonl"):
            for line in f.open():
                r = json.loads(line)
                res[r["instance_id"]] = r
        for f in (a.root / arm).rglob("trace_*.json"):
            iid = f.stem[len("trace_"):]
            if iid not in res:
                continue
            r = res[iid]
            row = {"arm": arm, "instance_id": iid, "resolved": bool(r.get("resolved")),
                   "error": (r.get("error") or "")[:60], **episode(json.loads(f.read_text()))}
            row["empty_patch"] = not r.get("agent_patch_size")
            rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "episodes.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    print(f"episodes {n} arms {len(a.arms.split(','))} solved {sum(r['resolved'] for r in rows) / n:.3f}")
    print(f"{'feature':26s} {'freq':>5s} {'solve|1':>7s} {'solve|0':>7s} {'MH_OR':>6s} {'z':>6s} {'strata':>6s}")
    for key in [k for k, v in rows[0].items() if isinstance(v, bool) and k != "resolved"]:
        on = [r for r in rows if r[key]]
        off = [r for r in rows if not r[key]]
        orr, z, k = mh(rows, key)
        s1 = sum(r["resolved"] for r in on) / len(on) if on else float("nan")
        s0 = sum(r["resolved"] for r in off) / len(off) if off else float("nan")
        print(f"{key:26s} {len(on) / n:5.2f} {s1:7.3f} {s0:7.3f} {orr:6.2f} {z:+6.1f} {k:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
