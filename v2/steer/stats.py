"""Pathology statistics over the miner output (v2.steer.mine).

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: Reads <mine>/calls.jsonl.gz + rollouts.jsonl and prints / writes <mine>/stats.json: per source the rate of
  each pathology (budget-forced thinking, verbatim loops, Wait bursts, unclosed thinking, timeouts, no-submit, turns
  cap) per call and per rollout, and how each correlates with solve rate (conditional resolve rate with/without, and
  the share of timeouts / no-submits that co-occur with each call pathology). Loop thresholds are reported at several
  levels because no single verbatim-repetition cut is canonical.
  Usage: python -m v2.steer.stats --mine DIR [--exclude-infra]
SRP/DRY check: Pass - consumer of v2.steer.mine only.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Callable

# Call-level loop definitions: verbatim 16-gram repeated >= k times in the thinking channel.
LOOP_K = (3, 5, 10)


def is_loop(c: dict[str, Any], k: int = 5) -> bool:
    return c["max_ng"] >= k


def is_wait_burst(c: dict[str, Any]) -> bool:
    return c["wait512"] >= 6  # >= 6 'Wait' within 512 thinking tokens


def rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def cond(rows: list[dict[str, Any]], flag: Callable[[dict[str, Any]], bool], y: str) -> dict[str, Any]:
    a = [r for r in rows if flag(r)]
    b = [r for r in rows if not flag(r)]
    return {"n_with": len(a), "share": rate([flag(r) for r in rows]), f"{y}_with": rate([bool(r[y]) for r in a]),
            f"{y}_without": rate([bool(r[y]) for r in b])}


def quantiles(xs: list[float], qs=(0.5, 0.9, 0.97, 0.99)) -> dict[str, float]:
    s = sorted(xs)
    return {f"p{int(q * 100)}": s[min(len(s) - 1, int(q * len(s)))] for q in qs} if s else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.stats")
    p.add_argument("--mine", type=Path, required=True)
    a = p.parse_args(argv)
    ros = [json.loads(l) for l in (a.mine / "rollouts.jsonl").open()]
    ros = [r for r in ros if r["log_found"] and not r["infra"]]
    calls_by: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    with gzip.open(a.mine / "calls.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            calls_by[(c["src"], c["run_key"])].append(c)
    out: dict[str, Any] = {}
    for src in sorted({r["src"] for r in ros}):
        R = [r for r in ros if r["src"] == src]
        C = [c for r in R for c in calls_by[(src, r["run_key"])] if c["is_agent"] is not False]
        for r in R:  # rollout-level call pathology flags
            cs = calls_by[(src, r["run_key"])]
            r["any_forced"] = r["n_forced"] > 0
            for k in LOOP_K:
                r[f"any_loop{k}"] = any(is_loop(c, k) for c in cs)
            r["any_wait_burst"] = any(is_wait_burst(c) for c in cs)
            r["any_unclosed"] = r["n_unclosed"] > 0
        s: dict[str, Any] = {
            "rollouts": len(R), "calls": len(C), "resolve_rate": rate([r["resolved"] for r in R]),
            "rollout_rates": {k: rate([bool(r[k]) for r in R]) for k in
                              ("timeout", "no_submit", "turns_cap", "any_forced", "any_unclosed", "any_wait_burst",
                               *[f"any_loop{k}" for k in LOOP_K])},
            "patch_empty_rate": rate([not r["patch_chars"] for r in R]),
            "call_rates": {
                "forced": rate([c["forced"] for c in C]),
                "unclosed": rate([c["unclosed"] for c in C]),
                "finish_length": rate([c["finish"] == "length" for c in C]),
                "wait_burst": rate([is_wait_burst(c) for c in C]),
                **{f"loop{k}": rate([is_loop(c, k) for c in C]) for k in LOOP_K},
                "loop5_given_forced": rate([is_loop(c) for c in C if c["forced"]]),
                "loop5_given_not_forced": rate([is_loop(c) for c in C if not c["forced"]]),
                "wait_burst_given_forced": rate([is_wait_burst(c) for c in C if c["forced"]]),
            },
            "think_tokens_per_call": quantiles([c["n_think"] for c in C]),
            "mean_think_per_call": round(sum(c["n_think"] for c in C) / max(1, len(C)), 1),
            "wait_per_1k_think": round(1000 * sum(c["wait"] for c in C) / max(1, sum(c["n_think"] for c in C)), 2),
            "resolve_conditional": {k: cond(R, lambda r, k=k: bool(r[k]), "resolved") for k in
                                    ("any_forced", "any_loop5", "any_wait_burst", "any_unclosed", "timeout",
                                     "no_submit")},
            "timeout_conditional": {k: cond(R, lambda r, k=k: bool(r[k]), "timeout") for k in
                                    ("any_forced", "any_loop5", "any_wait_burst")},
            "no_submit_conditional": {k: cond(R, lambda r, k=k: bool(r[k]), "no_submit") for k in
                                      ("any_forced", "any_loop5", "any_wait_burst")},
            "forced_calls_share_of_think_tokens": round(
                sum(c["n_think"] for c in C if c["forced"]) / max(1, sum(c["n_think"] for c in C)), 4),
            "timeouts_mean_forced_calls": round(
                sum(r["n_forced"] for r in R if r["timeout"]) / max(1, sum(r["timeout"] for r in R)), 2),
            "non_timeouts_mean_forced_calls": round(
                sum(r["n_forced"] for r in R if not r["timeout"]) / max(1, sum(not r["timeout"] for r in R)), 3),
        }
        out[src] = s
    (a.mine / "stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
