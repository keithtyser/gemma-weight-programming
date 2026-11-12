"""Multi-run paired decision statistic for golden evals (promote rule v2).

Greedy runs on standard vLLM are not repeatable, so a run is a sample, not a fact. Per arm and task, the solve rate is
averaged over the runs; anchor and candidate are paired by task:

  d_t = mean_runs(candidate solved t) - mean_runs(anchor solved t)

  - mean delta (pooled and per part) = mean of d_t over tasks
  - CI90 by a repo-clustered bootstrap: resample repos with replacement, then tasks with replacement within each
    sampled repo, recompute the mean of d over the resampled tasks; 10,000 replicates, fixed seed (reproducible)
  - sign test (exact, two-sided) over the tasks whose means differ
  - noise floor: per-run solve rate of each arm (pooled and per part) and its SD across runs

Promote rule v2: pooled CI90 lower bound > 0 AND comp-part mean delta >= 0 AND >= 5 runs in each arm.
A task missing from a run (crash without a record) counts as unsolved in that run and is reported.
"""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np

RULE_V2 = {"min_runs": 5, "replicates": 10_000, "ci": 0.90, "seed": 20261106}


def task_means(runs: list[list[dict[str, Any]]]) -> tuple[dict[str, float], dict[str, str], int]:
    """{task: mean solved over runs}, {task: repo}, number of (run, task) cells missing (counted as unsolved)."""
    tasks: dict[str, str] = {}
    for run in runs:
        for r in run:
            tasks.setdefault(str(r["instance_id"]), str(r.get("repo") or ""))
    solved = {t: 0.0 for t in tasks}
    missing = 0
    for run in runs:
        seen = {}
        for r in run:
            seen[str(r["instance_id"])] = bool(r.get("resolved"))  # last record wins (re-runs append)
        for t in tasks:
            if t in seen:
                solved[t] += float(seen[t])
            else:
                missing += 1
    n = max(1, len(runs))
    return {t: v / n for t, v in solved.items()}, tasks, missing


def run_rates(runs: list[list[dict[str, Any]]], tasks: list[str]) -> list[float]:
    out = []
    for run in runs:
        seen = {str(r["instance_id"]): bool(r.get("resolved")) for r in run}
        out.append(sum(seen.get(t, False) for t in tasks) / max(1, len(tasks)))
    return out


def sign_test(n_pos: int, n_neg: int) -> float:
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, p)


def cluster_bootstrap(d: dict[str, float], repo: dict[str, str], *, replicates: int, ci: float,
                      seed: int) -> tuple[float, float]:
    """Two-stage (repo, then task within repo) bootstrap of the mean paired difference."""
    by_repo: dict[str, list[float]] = {}
    for t, v in d.items():
        by_repo.setdefault(repo.get(t, ""), []).append(v)
    groups = [np.asarray(v, dtype=float) for _, v in sorted(by_repo.items())]
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    g = len(groups)
    stats = np.empty(replicates)
    for b in range(replicates):
        total = 0.0
        count = 0
        for gi in rng.integers(0, g, size=g):
            arr = groups[gi]
            pick = arr[rng.integers(0, len(arr), size=len(arr))]
            total += float(pick.sum())
            count += len(arr)
        stats[b] = total / count
    lo, hi = np.quantile(stats, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_part(runs_a: list[list[dict[str, Any]]], runs_c: list[list[dict[str, Any]]], *,
                replicates: int = RULE_V2["replicates"], ci: float = RULE_V2["ci"],
                seed: int = RULE_V2["seed"]) -> dict[str, Any]:
    ma, repo_a, miss_a = task_means(runs_a)
    mc, repo_c, miss_c = task_means(runs_c)
    common = sorted(set(ma) & set(mc))
    repo = {t: repo_a.get(t) or repo_c.get(t, "") for t in common}
    d = {t: mc[t] - ma[t] for t in common}
    lo, hi = cluster_bootstrap(d, repo, replicates=replicates, ci=ci, seed=seed)
    n_pos = sum(1 for v in d.values() if v > 0)
    n_neg = sum(1 for v in d.values() if v < 0)
    ra, rc = run_rates(runs_a, common), run_rates(runs_c, common)
    return {
        "tasks": len(common), "repos": len(set(repo.values())),
        "runs_anchor": len(runs_a), "runs_candidate": len(runs_c),
        "anchor_rate": float(np.mean([ma[t] for t in common])) if common else float("nan"),
        "candidate_rate": float(np.mean([mc[t] for t in common])) if common else float("nan"),
        "mean_delta": float(np.mean(list(d.values()))) if d else float("nan"),
        f"ci{int(ci * 100)}_low": lo, f"ci{int(ci * 100)}_high": hi,
        "sign_test": {"candidate_better": n_pos, "anchor_better": n_neg, "tied": len(common) - n_pos - n_neg,
                      "two_sided_p": sign_test(n_pos, n_neg)},
        "noise_floor": {"anchor_run_rates": ra, "candidate_run_rates": rc,
                        "anchor_run_sd": float(np.std(ra, ddof=1)) if len(ra) > 1 else None,
                        "candidate_run_sd": float(np.std(rc, ddof=1)) if len(rc) > 1 else None},
        "missing_cells": {"anchor": miss_a, "candidate": miss_c},
        "only_in_anchor": sorted(set(ma) - set(mc)), "only_in_candidate": sorted(set(mc) - set(ma)),
        "per_task": {t: {"repo": repo[t], "anchor": ma[t], "candidate": mc[t], "delta": d[t]} for t in common},
    }


def decide_v2(comp_a: list[list[dict[str, Any]]], comp_c: list[list[dict[str, Any]]],
              reb_a: list[list[dict[str, Any]]], reb_c: list[list[dict[str, Any]]],
              *, min_runs: int = RULE_V2["min_runs"], replicates: int = RULE_V2["replicates"]) -> dict[str, Any]:
    """Promote rule v2 over both parts. Runs are paired by part, not by index (runs are exchangeable samples)."""
    rep: dict[str, Any] = {"rule": "promote iff pooled CI90 lower > 0 AND comp mean delta >= 0 AND >= "
                                   f"{min_runs} runs per arm", "replicates": replicates}
    rep["comp"] = paired_part(comp_a, comp_c, replicates=replicates)
    rep["rebench"] = paired_part(reb_a, reb_c, replicates=replicates)
    pooled_a = [ca + ra for ca, ra in zip(comp_a, reb_a)]
    pooled_c = [cc + rc for cc, rc in zip(comp_c, reb_c)]
    rep["pooled"] = paired_part(pooled_a, pooled_c, replicates=replicates)
    runs = min(len(comp_a), len(comp_c), len(reb_a), len(reb_c))
    reasons = []
    if runs < min_runs:
        reasons.append(f"too_few_runs:{runs}<{min_runs}")
    if not rep["pooled"]["ci90_low"] > 0:
        reasons.append(f"pooled_ci90_low_not_positive:{rep['pooled']['ci90_low']:.4f}")
    if not rep["comp"]["mean_delta"] >= 0:
        reasons.append(f"comp_delta_negative:{rep['comp']['mean_delta']:.4f}")
    rep["runs_per_arm"] = runs
    rep["promote"] = not reasons
    rep["reasons"] = reasons
    return rep
