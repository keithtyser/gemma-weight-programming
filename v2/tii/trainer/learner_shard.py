"""Token-cost-balanced sharding of learner work units (rollout tries) across GPUs, from the subtree planner.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: Equal GPU utilization in the 2-GPU learner. A rollout's learner time is not its unique-token count: tries
  above the tree budget are split (plan_subtrees) and each group re-processes its ancestor path, and the 7 full-
  attention layers cost O(path position) per row. work_features() runs the planner on CPU and returns, per forward
  group, the processed tokens and the full-attention keys (sum over rows of path position + 1); estimate_seconds()
  turns those into predicted learner seconds with a linear model fitted on measured per-rollout learner times
  (fit_coefficients(), bench_tree_mfu per_rollout records); shard_work() assigns units to ranks largest-first to the
  least-loaded rank (deterministic LPT), then move/swap local improvement. Callers: v2/train/async_rl.shard_units (uses unit["work"] when present),
  v2/tii/tests/test_learner_shard.py.
SRP/DRY check: Pass - reuses tree.plan_subtrees / build_tree; the LPT rule is the one async_rl.shard_units used, now on
  predicted seconds instead of unique tokens.
"""

from __future__ import annotations

from typing import Sequence

from v2.tii.trainer.tree import build_tree, plan_subtrees

# Fitted on GPU0 (TII-Marlin, fast backward, adaptive 20k, no recompute, commit eecd42e) on 40 real rollouts
# (results/tii/mfu_adaptive_20000_s40.json -> results/tii/learner_cost_fit.json): median rel. error 1.9 % (max 5.9 %),
# validated on 12 other rollouts: 2.3 % (max 5.5 %).
# seconds = c_tok * processed_tokens + c_key * full_attention_keys + c_fwd * forwards
DEFAULT_COEF = {"c_tok": 1.40471e-4, "c_key": 4.38613e-9, "c_fwd": 0.0}


def work_features(sequences: Sequence[Sequence[int]], budget: int) -> dict:
    """Planner-exact learner work of one rollout (CPU only): per forward group, processed tokens and full-attention
    keys. The effective budget is max(budget, longest call), as AlignedGemma4.score_calls_subtrees uses."""
    if not sequences:
        return {"groups": 0, "processed": 0, "full_keys": 0, "per_group": []}
    eff = max(budget, max(len(s) for s in sequences))
    groups = plan_subtrees(sequences, eff)
    per = []
    for g in groups:
        tb = build_tree([sequences[i] for i in g])
        per.append({"tokens": tb.unique_tokens, "full_keys": int(sum(p + 1 for p in tb.row_pos))})
    return {"groups": len(per), "processed": sum(x["tokens"] for x in per),
            "full_keys": sum(x["full_keys"] for x in per), "per_group": per}


def estimate_seconds(feat: dict, coef: dict | None = None) -> float:
    c = coef or DEFAULT_COEF
    return c["c_tok"] * feat["processed"] + c["c_key"] * feat["full_keys"] + c["c_fwd"] * feat["groups"]


def shard_work(work: Sequence[float], world: int) -> list[int]:
    """Owner rank of every unit: largest predicted work first, to the least-loaded rank (ties -> lower rank, lower
    index). Deterministic, so every rank computes the same assignment."""
    order = sorted(range(len(work)), key=lambda i: (-float(work[i]), i))
    loads = [0.0] * world
    owner = [0] * len(work)
    for i in order:
        r = min(range(world), key=lambda k: (loads[k], k))
        owner[i] = r
        loads[r] += float(work[i])
    # local improvement (deterministic): move or swap units between the most and the least loaded rank while that
    # lowers the maximum load
    for _ in range(4 * len(work)):
        hi = max(range(world), key=lambda k: (loads[k], -k))
        lo = min(range(world), key=lambda k: (loads[k], k))
        gap = loads[hi] - loads[lo]
        best, best_gain = None, 1e-12
        for i in range(len(work)):
            if owner[i] != hi:
                continue
            wi = float(work[i])
            if 0 < wi < gap and min(wi, gap - wi) > best_gain:  # move i: new gap = |gap - 2 wi|
                best, best_gain = (i, None), min(wi, gap - wi)
            for j in range(len(work)):
                if owner[j] != lo:
                    continue
                d = wi - float(work[j])
                if 0 < d < gap and min(d, gap - d) > best_gain:
                    best, best_gain = (i, j), min(d, gap - d)
        if best is None:
            break
        i, j = best
        owner[i] = lo
        loads[hi] -= float(work[i])
        loads[lo] += float(work[i])
        if j is not None:
            owner[j] = hi
            loads[lo] -= float(work[j])
            loads[hi] += float(work[j])
    return owner


def imbalance(work: Sequence[float], owner: Sequence[int], world: int) -> float:
    """max rank load / mean rank load (1.0 = perfectly balanced)."""
    loads = [0.0] * world
    for w, o in zip(work, owner):
        loads[o] += float(w)
    mean = sum(loads) / world
    return max(loads) / mean if mean > 0 else 1.0


def fit_coefficients(rows: Sequence[tuple[dict, float]]) -> dict:
    """Non-negative least squares (3 coefficients, by projected coordinate descent) of measured seconds on
    (processed, full_keys, groups). rows = [(work_features(...), seconds), ...]."""
    import numpy as np

    X = np.array([[f["processed"], f["full_keys"], f["groups"]] for f, _ in rows], dtype=np.float64)
    y = np.array([s for _, s in rows], dtype=np.float64)
    scale = X.max(0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    beta = np.zeros(3)
    for _ in range(20000):
        for j in range(3):
            r = y - Xs @ beta + Xs[:, j] * beta[j]
            den = float(Xs[:, j] @ Xs[:, j])
            beta[j] = max(0.0, float(Xs[:, j] @ r) / den) if den > 0 else 0.0
    b = beta / scale
    pred = X @ b
    rel = np.abs(pred - y) / np.maximum(y, 1e-9)
    return {"c_tok": float(b[0]), "c_key": float(b[1]), "c_fwd": float(b[2]), "n": len(rows),
            "median_rel_err": float(np.median(rel)), "max_rel_err": float(rel.max())}
