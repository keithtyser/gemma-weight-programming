"""Quantization-exact Gemma 4 post-training stack (W4A16 LoRA, Trajectory Audit, OPD, RLOO+CISPO/DAPO)."""

from v2.train.distill_and_rl import (
    HARDWARE_PROFILES,
    compute_cispo_loss,
    compute_dapo_loss,
    compute_hybrid_opd_rloo_advantages,
    compute_rloo_advantages,
    compute_topk_reverse_kl_opd_loss,
)
from v2.train.int4_qat_lora import (
    GEMMA4_LANGUAGE_LORA_REGEX,
    build_loraplus_param_groups,
    is_valid_gemma4_lora_target,
)
from v2.train.trajectory_audit import audit_agent_trajectory

__all__ = [
    "GEMMA4_LANGUAGE_LORA_REGEX",
    "HARDWARE_PROFILES",
    "audit_agent_trajectory",
    "build_loraplus_param_groups",
    "compute_cispo_loss",
    "compute_dapo_loss",
    "compute_hybrid_opd_rloo_advantages",
    "compute_rloo_advantages",
    "compute_topk_reverse_kl_opd_loss",
    "is_valid_gemma4_lora_target",
]
