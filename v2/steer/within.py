"""Within-task behaviour contrast: which episode features differ between solved and unsolved runs of the same task? (CPU)

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Diagnosis step of the weight-programming loop. For every task with both solved and unsolved episodes in a
  mine directory, compute per-episode features from the request log (tool mix, failed edits, tests run, thinking,
  where the first edit falls ...) and report, per feature, the mean over tasks of (mean solved - mean unsolved),
  its t statistic over tasks, and the share of tasks where solved > unsolved. Task difficulty cancels.
  Reads <mine>/rollouts.jsonl + request logs. Usage: python -m v2.steer.within --mine DIR [--srcs a,b]
SRP/DRY check: Pass - v2.steer.endfeat classifies the end state only; this ranks many features within task.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

TEST = re.compile(r"pytest|unittest|tox|nosetests|manage\.py test|go test|npm test|cargo test|make test")


def feats(path: str) -> dict[str, float] | None:
    try:
        rows = [json.loads(l) for l in open(path)]
    except OSError:
        return None
    rows = [x for x in rows if x.get("is_agent_call", True)]
    if not rows:
        return None
    m = rows[-1]["messages"]
    ev = []  # (name, status, args, result-length)
    for i, x in enumerate(m):
        if x.get("role") == "tool" and m[i - 1].get("tool_calls"):
            fn = m[i - 1]["tool_calls"][-1]["function"]
            try:
                st = json.loads(x["content"]).get("status") or ""
            except (ValueError, AttributeError):
                st = ""
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"raw": args}
            ev.append((fn["name"], st, args or {}, len(str(x.get("content")))))
    n = len(ev)
    if n == 0:
        return None
    names = [e[0] for e in ev]
    edits = [i for i, e in enumerate(ev) if e[0] == "edit_file"]
    ok_edits = [i for i in edits if ev[i][1] == "ok"]
    runs = [i for i, e in enumerate(ev) if e[0] == "run_command"]
    cmd = lambda i: str(ev[i][2].get("command") or ev[i][2])
    tests = [i for i in runs if TEST.search(cmd(i))]
    last_ok = ok_edits[-1] if ok_edits else -1
    files_read = {str(ev[i][2].get("path") or ev[i][2].get("file_path")) for i, e in enumerate(ev) if e[0] == "read_file"}
    files_edit = {str(ev[i][2].get("path") or ev[i][2].get("file_path")) for i in ok_edits}
    first = edits[0] if edits else n
    f = {
        "n_calls": n, "n_read": names.count("read_file"), "n_run": len(runs), "n_edit": len(edits),
        "n_ok_edit": len(ok_edits), "n_fail_edit": len(edits) - len(ok_edits),
        "any_fail_edit": float(len(edits) > len(ok_edits)), "first_edit_idx": first,
        "first_edit_frac": first / n, "n_test_runs": len(tests), "any_test": float(bool(tests)),
        "test_before_edit": float(any(i < first for i in tests)),
        "test_after_last_edit": float(any(i > last_ok for i in tests)) if ok_edits else 0.0,
        "test_ok_after_last_edit": float(any(i > last_ok and ev[i][1] == "ok" for i in tests)) if ok_edits else 0.0,
        "runs_after_last_edit": sum(i > last_ok for i in runs) if ok_edits else 0,
        "calls_after_last_edit": n - 1 - last_ok if ok_edits else 0,
        "files_read": len(files_read), "files_edited": len(files_edit),
        "grep_runs": sum(bool(re.search(r"\bgrep\b|\brg\b", cmd(i))) for i in runs),
        "ls_find_runs": sum(bool(re.search(r"\bls\b|\bfind\b", cmd(i))) for i in runs),
        "git_diff_runs": sum("git diff" in cmd(i) or "git status" in cmd(i) for i in runs),
        "py_inline_runs": sum(bool(re.search(r"python3? -c|<<", cmd(i))) for i in runs),
        "pycompile_runs": sum("py_compile" in cmd(i) for i in runs),
        "err_results": sum(e[1] == "error" for e in ev), "err_frac": sum(e[1] == "error" for e in ev) / n,
        "read_frac": names.count("read_file") / n, "run_frac": len(runs) / n, "edit_frac": len(edits) / n,
        "mean_result_len": sum(e[3] for e in ev) / n,
        "edit_new_len": (sum(len(str(ev[i][2].get("new_string", ""))) for i in ok_edits) / len(ok_edits)) if ok_edits else 0,
        "edit_old_len": (sum(len(str(ev[i][2].get("old_string", ""))) for i in edits) / len(edits)) if edits else 0,
        "new_file_edits": sum(not str(ev[i][2].get("old_string", "")) for i in edits),
        "test_file_edits": sum("test" in str(ev[i][2].get("path") or ev[i][2].get("file_path")).lower() for i in ok_edits),
    }
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.within")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--srcs", default="")
    a = p.parse_args(argv)
    srcs = set(a.srcs.split(",")) if a.srcs else None
    T: dict[str, dict[bool, list[dict[str, float]]]] = collections.defaultdict(lambda: {True: [], False: []})
    for r in map(json.loads, (a.mine / "rollouts.jsonl").open()):
        if r.get("infra") or (srcs and r["src"] not in srcs) or not r.get("request_log_path"):
            continue
        f = feats(r["request_log_path"])
        if f is None:
            continue
        for k in ("think_tokens", "comp_tokens", "n_forced", "patch_chars", "duration"):
            f[k] = float(r.get(k) or 0)
        f["think_per_call"] = f["think_tokens"] / max(1, f["n_calls"])
        T[r["instance_id"]][str(r["resolved"]) == "True"].append(f)
    mixed = {t: v for t, v in T.items() if v[True] and v[False]}
    print(f"tasks {len(T)}  mixed {len(mixed)}  episodes {sum(len(v[True]) + len(v[False]) for v in T.values())}")
    keys = sorted(next(iter(mixed.values()))[True][0])
    out = []
    for k in keys:
        d = []
        for v in mixed.values():
            d.append(sum(x[k] for x in v[True]) / len(v[True]) - sum(x[k] for x in v[False]) / len(v[False]))
        mu = sum(d) / len(d)
        sd = math.sqrt(sum((x - mu) ** 2 for x in d) / max(1, len(d) - 1))
        ms = sum(sum(x[k] for x in v[True]) / len(v[True]) for v in mixed.values()) / len(mixed)
        mf = sum(sum(x[k] for x in v[False]) / len(v[False]) for v in mixed.values()) / len(mixed)
        out.append((mu / (sd / math.sqrt(len(d)) + 1e-12), k, ms, mf, sum(x > 0 for x in d), sum(x < 0 for x in d)))
    print(f"{'feature':26s}{'solved':>10s}{'unsolved':>10s}{'t':>8s}{'tasks +/-':>12s}")
    for t, k, ms, mf, up, dn in sorted(out, key=lambda x: -abs(x[0])):
        print(f"{k:26s}{ms:>10.2f}{mf:>10.2f}{t:>8.1f}{f'{up}/{dn}':>12s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
