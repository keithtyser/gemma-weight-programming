"""Dual-GPU evalhost evaluation & Two-Tier Seal-and-Sprint governance engine."""

from __future__ import annotations

from typing import Any

from v2.eval.governance import (
    PromotionDecision,
    evaluate_paired_promotion,
    verify_split_seal_integrity,
)

__all__ = [
    "DualGpuTopologyPlan",
    "PromotionDecision",
    "build_dual_gpu_topology_plan",
    "build_vllm_0213_command",
    "evaluate_paired_promotion",
    "verify_split_seal_integrity",
]


def __getattr__(name: str) -> Any:
    if name in {"DualGpuTopologyPlan", "build_dual_gpu_topology_plan", "build_vllm_0213_command"}:
        from v2.eval import gpu_runner

        return getattr(gpu_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
