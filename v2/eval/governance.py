"""Two-Tier Seal-and-Sprint Governance Engine.

Replaces heavy manual pre-registration paperwork and single-GPU lease lockouts with
fast, deterministic programmatic guardrails:
  - Tier 1 (Sprint): Unrestricted parallel iteration on `train_pool` across both RTX PRO 6000s.
  - Tier 2 (Seal & Promote): Cryptographic repo-disjointness assertion + paired multi-seed
    repository-cluster bootstrap & exact McNemar promotion gate before spending 1/day Kaggle LB slots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Any

from v2.constants import (
    FLAKY_GRADER_TASK_IDS,
    KAGGLE_SAFE_WALL_CLOCK_HOURS,
    ORIGINAL_SPLITV2_REPOS,
    SEALED_RICH_HTTPX_REPOS,
    canonicalize_repo,
)


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    mean_anchor_rate: float
    mean_candidate_rate: float
    mean_delta: float
    repo_cluster_ci90_low: float
    repo_cluster_ci90_high: float
    discordant_candidate_wins: int
    discordant_anchor_wins: int
    exact_mcnemar_one_sided_p: float
    seeds_evaluated: int
    paired_tasks_per_seed: int
    wall_clock_safe: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_split_seal_integrity(
    split_manifest: dict[str, Any],
    *,
    sealed_harbor_repos: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Cryptographically verify zero repository overlap across Train, Unseen-CV, and Sealed Holdouts."""
    train_repos = {canonicalize_repo(r) for r in split_manifest.get("train_repos", [])}
    cv_repos = {canonicalize_repo(r) for r in split_manifest.get("unseen_cv_repos", [])}
    orig_repos = set(ORIGINAL_SPLITV2_REPOS)
    rich_httpx = set(SEALED_RICH_HTTPX_REPOS)
    harbor_canon = {canonicalize_repo(r) for r in sealed_harbor_repos}

    train_ids = set(split_manifest.get("train_instance_ids", []))
    cv_ids = set(split_manifest.get("unseen_cv_instance_ids", []))

    violations: list[str] = []
    if train_repos & cv_repos:
        violations.append(f"train_vs_cv_repo_overlap:{sorted(train_repos & cv_repos)}")
    if train_repos & orig_repos:
        violations.append(f"train_vs_splitv2_repo_overlap:{sorted(train_repos & orig_repos)}")
    if train_repos & rich_httpx:
        violations.append(f"train_vs_rich_httpx_overlap:{sorted(train_repos & rich_httpx)}")
    if train_repos & harbor_canon:
        violations.append(f"train_vs_harbor_sealed_overlap:{sorted(train_repos & harbor_canon)}")
    if cv_repos & (orig_repos | rich_httpx | harbor_canon):
        violations.append(
            f"cv_vs_reserved_holdout_overlap:{sorted(cv_repos & (orig_repos | rich_httpx | harbor_canon))}"
        )
    if train_ids & cv_ids:
        violations.append(f"train_vs_cv_instance_overlap:{len(train_ids & cv_ids)}")

    expected_payload = {
        "split_seed": split_manifest.get("split_seed"),
        "train_repos": sorted(split_manifest.get("train_repos", [])),
        "unseen_cv_repos": sorted(split_manifest.get("unseen_cv_repos", [])),
        "train_instance_ids": sorted(split_manifest.get("train_instance_ids", [])),
        "unseen_cv_instance_ids": sorted(split_manifest.get("unseen_cv_instance_ids", [])),
    }
    computed_seal = hashlib.sha256(repr(expected_payload).encode("utf-8")).hexdigest()
    recorded_seal = split_manifest.get("seal_sha256")
    if recorded_seal and recorded_seal != computed_seal:
        violations.append(f"seal_sha256_mismatch:{recorded_seal}!={computed_seal}")

    return {
        "seal_valid": len(violations) == 0,
        "computed_seal_sha256": computed_seal,
        "train_repo_count": len(train_repos),
        "cv_repo_count": len(cv_repos),
        "violations": violations,
    }


def _exact_binomial_tail(wins: int, total_discordant: int) -> float:
    """Exact one-sided McNemar p-value P(Binomial(n, 0.5) >= wins)."""
    if total_discordant <= 0 or wins <= 0:
        return 1.0
    log_denom = total_discordant * math.log(2.0)
    prob = 0.0
    for k in range(wins, total_discordant + 1):
        log_comb = (
            math.lgamma(total_discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(total_discordant - k + 1)
        )
        prob += math.exp(log_comb - log_denom)
    return min(1.0, prob)


def summarize_failure_modes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Break unresolved rollouts down by failure mode (the empty-patch / timeout mix is what training must move)."""
    n = len(rows)
    if n == 0:
        return {"total": 0}

    def _err(r: dict[str, Any]) -> str:
        return str(r.get("error") or r.get("agent_error") or "")

    empty = sum(1 for r in rows if not r.get("resolved") and "Empty agent patch" in _err(r))
    timeout = sum(1 for r in rows if "session timeout" in _err(r) or "hard timeout" in _err(r))
    budget = sum(1 for r in rows if "tool call budget" in _err(r) or "turns budget" in _err(r))
    test_fail = sum(1 for r in rows if not r.get("resolved") and "JUnit" in _err(r))
    return {
        "total": n,
        "resolved_rate": round(sum(1 for r in rows if r.get("resolved")) / n, 4),
        "empty_patch_rate": round(empty / n, 4),
        "timeout_rate": round(timeout / n, 4),
        "budget_exhausted_rate": round(budget / n, 4),
        "tests_failed_rate": round(test_fail / n, 4),
        "mean_tool_calls": round(sum(int(r.get("tool_calls", 0)) for r in rows) / n, 2),
    }


def evaluate_paired_promotion(
    anchor_runs: list[list[dict[str, Any]]],
    candidate_runs: list[list[dict[str, Any]]],
    *,
    projected_120task_hours: float,
    min_seeds: int = 2,
    min_mean_delta: float = 0.02,
    max_mcnemar_p: float = 0.10,
    exclude_task_ids: frozenset[str] | set[str] = FLAKY_GRADER_TASK_IDS,
    bootstrap_samples: int = 2000,
    rng_seed: int = 20261101,
) -> PromotionDecision:
    """Compare Candidate vs Anchor on paired multi-seed unseen-repo CV runs.

    Each element of `anchor_runs` / `candidate_runs` is one seed's list of task dicts:
      `{"instance_id": str, "repo": str, "resolved": bool}`

    Promotion requires ALL of: mean delta >= `min_mean_delta`, repo-cluster bootstrap CI90 lower bound >= 0, and an
    exact one-sided McNemar p <= `max_mcnemar_p` over the pooled discordant pairs. (Before, the p-value was computed but
    never enforced, so a single discordant win with zero losses promoted at p = 0.5.) Tasks in `exclude_task_ids`
    (grader-flaky controls) are dropped first.
    """
    reasons: list[str] = []
    excluded = set(exclude_task_ids or ())
    anchor_runs = [[r for r in run if str(r["instance_id"]) not in excluded] for run in anchor_runs]
    candidate_runs = [[r for r in run if str(r["instance_id"]) not in excluded] for run in candidate_runs]
    if len(anchor_runs) != len(candidate_runs) or len(anchor_runs) == 0:
        raise ValueError("anchor_runs and candidate_runs must have equal non-zero seed count")

    n_seeds = len(anchor_runs)
    if n_seeds < min_seeds:
        reasons.append(f"insufficient_seeds:{n_seeds}<{min_seeds}")

    per_task_anchor: dict[str, list[float]] = {}
    per_task_cand: dict[str, list[float]] = {}
    task_to_repo: dict[str, str] = {}

    discordant_cand_wins = 0
    discordant_anchor_wins = 0

    for seed_idx, (a_list, c_list) in enumerate(zip(anchor_runs, candidate_runs, strict=True)):
        a_map = {str(r["instance_id"]): r for r in a_list}
        c_map = {str(r["instance_id"]): r for r in c_list}
        if set(a_map.keys()) != set(c_map.keys()):
            raise ValueError(f"Seed {seed_idx} task IDs do not match between anchor and candidate")
        for iid in sorted(a_map.keys()):
            a_ok = bool(a_map[iid]["resolved"])
            c_ok = bool(c_map[iid]["resolved"])
            repo = canonicalize_repo(str(a_map[iid].get("repo") or iid.split("__")[0].replace("_", "/")))
            task_to_repo[iid] = repo
            per_task_anchor.setdefault(iid, []).append(1.0 if a_ok else 0.0)
            per_task_cand.setdefault(iid, []).append(1.0 if c_ok else 0.0)
            if c_ok and not a_ok:
                discordant_cand_wins += 1
            elif a_ok and not c_ok:
                discordant_anchor_wins += 1

    all_iids = sorted(per_task_anchor.keys())
    n_tasks = len(all_iids)
    if n_tasks == 0:
        raise ValueError("Cannot evaluate promotion on 0 tasks")

    mean_anchor = sum(sum(v) / len(v) for v in per_task_anchor.values()) / n_tasks
    mean_cand = sum(sum(v) / len(v) for v in per_task_cand.values()) / n_tasks
    mean_delta = mean_cand - mean_anchor

    # Repository-cluster bootstrap for 90% CI on delta
    repo_to_iids: dict[str, list[str]] = {}
    for iid, repo in task_to_repo.items():
        repo_to_iids.setdefault(repo, []).append(iid)
    repos = sorted(repo_to_iids.keys())

    rng = random.Random(rng_seed)
    boot_deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_repos = [repos[rng.randrange(len(repos))] for _ in range(len(repos))]
        num = 0.0
        cnt = 0
        for r in sampled_repos:
            for iid in repo_to_iids[r]:
                t_a = sum(per_task_anchor[iid]) / len(per_task_anchor[iid])
                t_c = sum(per_task_cand[iid]) / len(per_task_cand[iid])
                num += t_c - t_a
                cnt += 1
        boot_deltas.append(num / max(1, cnt))

    boot_deltas.sort()
    ci90_low = boot_deltas[int(0.05 * len(boot_deltas))]
    ci90_high = boot_deltas[min(len(boot_deltas) - 1, int(0.95 * len(boot_deltas)))]

    mcnemar_p = _exact_binomial_tail(
        discordant_cand_wins, discordant_cand_wins + discordant_anchor_wins
    )

    wall_clock_safe = projected_120task_hours <= KAGGLE_SAFE_WALL_CLOCK_HOURS
    if not wall_clock_safe:
        reasons.append(
            f"projected_120task_hours_unsafe:{projected_120task_hours:.2f}h>{KAGGLE_SAFE_WALL_CLOCK_HOURS:.2f}h"
        )
    if mean_delta < min_mean_delta:
        reasons.append(f"mean_delta_below_threshold:{mean_delta:.4f}<{min_mean_delta:.4f}")
    if ci90_low < 0.0:
        reasons.append(f"repo_cluster_ci90_crosses_zero:{ci90_low:.4f}")
    if mcnemar_p > max_mcnemar_p:
        reasons.append(f"mcnemar_not_significant:p={mcnemar_p:.4f}>{max_mcnemar_p:.4f}")

    return PromotionDecision(
        promote=len(reasons) == 0,
        mean_anchor_rate=round(mean_anchor, 4),
        mean_candidate_rate=round(mean_cand, 4),
        mean_delta=round(mean_delta, 4),
        repo_cluster_ci90_low=round(ci90_low, 4),
        repo_cluster_ci90_high=round(ci90_high, 4),
        discordant_candidate_wins=discordant_cand_wins,
        discordant_anchor_wins=discordant_anchor_wins,
        exact_mcnemar_one_sided_p=round(mcnemar_p, 6),
        seeds_evaluated=n_seeds,
        paired_tasks_per_seed=n_tasks,
        wall_clock_safe=wall_clock_safe,
        reasons=tuple(reasons),
    )
