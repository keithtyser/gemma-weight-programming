"""Nov 2026 SOTA Multi-Turn SWE Post-Training Objectives & Hardware Scaling Profiles.

Implements the exact mathematical objectives from `sota_post_training_nov2026.md`:
  1. `compute_rloo_advantages`: Unbiased, std-free Leave-One-Out episode advantages
     + DAPO Dynamic Sampling filter (`0 < sum(R) < G`) + DAPO Soft Overlong Buffer penalty.
  2. `compute_cispo_loss` & `compute_dapo_loss`:
     - CISPO (`-sg(clip(r, 1-eps_low, 1+eps_high)) * A * log_pi`) + Truncated IS (`TIS`)
       for async `vllm==0.19.1` Marlin W4A16 <-> PyTorch training resilience.
     - DAPO Clip-Higher (`eps_low=0.20, eps_high=0.28`) with global active-token normalization.
  3. `compute_topk_reverse_kl_opd_loss`: Exact Top-K (`K=64`) + Tail-Bucket Reverse-KL
     On-Policy Distillation (`31B-RL Teacher -> E4B Student`, shared 262,144-token Gemma 4 vocab).
  4. `compute_hybrid_opd_rloo_advantages`: Stage-3 Hybrid `pytest` RLOO + Annealed Teacher Reverse-KL.
  5. `HARDWARE_PROFILES`: Ready-to-run configs for `evalhost` (`2x RTX PRO 6000`, 192 GiB)
     and burst scale-up (`8x NVIDIA B300`, 2.3 TB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareTrainingProfile:
    profile_id: str
    gpus: int
    vram_per_gpu_gib: int
    total_vram_gib: int
    recommended_lora_rank: int
    recommended_lora_alpha: int
    concurrent_docker_sandboxes: int
    rl_group_size: int
    max_seq_len: int
    overlong_buffer_start: int
    notes: str


HARDWARE_PROFILES: dict[str, HardwareTrainingProfile] = {
    "evalhost_2x_rtx_pro_6000": HardwareTrainingProfile(
        profile_id="evalhost_2x_rtx_pro_6000",
        gpus=2,
        vram_per_gpu_gib=96,
        total_vram_gib=192,
        recommended_lora_rank=64,
        recommended_lora_alpha=128,
        concurrent_docker_sandboxes=16,
        rl_group_size=6,
        max_seq_len=32768,
        overlong_buffer_start=26624,
        notes=(
            "GPU0 co-hosts E4B rollout + 31B-RL teacher prefill in W4A16 (or runs 31B async rollouts); "
            "GPU1 runs W_dequant BF16 + LoRA (r=64, LoRA+) updates."
        ),
    ),
    "burst_8x_b300": HardwareTrainingProfile(
        profile_id="burst_8x_b300",
        gpus=8,
        vram_per_gpu_gib=288,
        total_vram_gib=2304,
        recommended_lora_rank=64,
        recommended_lora_alpha=128,
        concurrent_docker_sandboxes=256,
        rl_group_size=8,
        max_seq_len=32768,
        overlong_buffer_start=26624,
        notes=(
            "GPUs 0..4 run 5x async vLLM rollout/teacher engines across 256 sandboxes; "
            "GPUs 5..7 run FSDP2 W_dequant BF16 + LoRA (r=64) RLOO+CISPO & Multi-LoRA Specialist Trio."
        ),
    ),
}


def apply_dapo_overlong_penalty(
    raw_reward: float,
    trajectory_tokens: int,
    *,
    overlong_buffer_start: int = 26624,
    max_seq_len: int = 32768,
    max_penalty: float = 0.5,
) -> float:
    """DAPO soft overlong buffer reward shaping preventing 32k context inflation."""
    if trajectory_tokens <= overlong_buffer_start:
        return float(raw_reward)
    if trajectory_tokens >= max_seq_len:
        return float(raw_reward) - float(max_penalty)
    span = max(1, max_seq_len - overlong_buffer_start)
    frac = (trajectory_tokens - overlong_buffer_start) / float(span)
    return float(raw_reward) - float(max_penalty) * frac


def compute_rloo_advantages(
    group_rewards: list[float],
    *,
    require_dynamic_contrast: bool = True,
) -> dict[str, Any]:
    """Compute exact Leave-One-Out (`RLOO`) advantages without std-division bias.

    Formula:
      `A_i = R_i - (1 / (G - 1)) * sum_{j != i} R_j`
    Also evaluates DAPO Dynamic Sampling contrast (`max(R) > min(R)`).
    """
    g = len(group_rewards)
    if g < 2:
        raise ValueError(f"RLOO requires group size G >= 2, got {g}")

    has_contrast = (max(group_rewards) - min(group_rewards)) > 1e-8
    if require_dynamic_contrast and not has_contrast:
        return {
            "keep_group": False,
            "advantages": [0.0] * g,
            "group_mean_reward": sum(group_rewards) / g,
        }

    total = sum(group_rewards)
    advantages = [float(r) - (total - float(r)) / float(g - 1) for r in group_rewards]
    return {
        "keep_group": True,
        "advantages": advantages,
        "group_mean_reward": total / g,
    }


def compute_cispo_loss(
    current_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    action_mask: Any,
    *,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
    tis_cap: float | None = 3.0,
) -> Any:
    """CISPO (MiniMax-M1) + Asymmetric Clip-Higher + Global Active-Token Normalization.

    `L_CISPO = - (1 / sum(M)) * sum_{M=1} sg(clip(r_{i,t}, 1 - eps_low, 1 + eps_high)) * A_{i,t} * log pi_theta`

    Preserves clean gradient flow even when async `vllm==0.19.1` W4A16 Marlin logprobs have slight
    numerical offset from PyTorch `W_dequant` BF16 forward passes.
    """
    import torch

    mask_f = action_mask.to(torch.float32)
    denom = torch.clamp(mask_f.sum(), min=1.0)

    log_ratio = (current_logprobs - old_logprobs).to(torch.float32)
    # Clamp exponent for numerical safety before exp
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    if tis_cap is not None:
        ratio = torch.clamp(ratio, max=float(tis_cap))

    clipped_weight = torch.clamp(ratio, min=1.0 - float(eps_low), max=1.0 + float(eps_high)).detach()
    adv_f = advantages.to(torch.float32)
    per_token = -clipped_weight * adv_f * current_logprobs.to(torch.float32) * mask_f
    return per_token.sum() / denom


def compute_dapo_loss(
    current_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    action_mask: Any,
    *,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
) -> Any:
    """DAPO (Yu et al., 2025) Asymmetric Clip-Higher Policy Optimization with Global Token Norm."""
    import torch

    mask_f = action_mask.to(torch.float32)
    denom = torch.clamp(mask_f.sum(), min=1.0)

    ratio = torch.exp(torch.clamp((current_logprobs - old_logprobs).to(torch.float32), -20.0, 20.0))
    clipped_ratio = torch.clamp(ratio, min=1.0 - float(eps_low), max=1.0 + float(eps_high))
    adv_f = advantages.to(torch.float32)

    surr1 = ratio * adv_f
    surr2 = clipped_ratio * adv_f
    obj = torch.minimum(surr1, surr2) * mask_f
    return -obj.sum() / denom


def compute_topk_reverse_kl_opd_loss(
    student_full_logits: Any,
    teacher_topk_indices: Any,
    teacher_topk_logprobs: Any,
    action_mask: Any,
    *,
    softcap: float = 30.0,
    tail_eps: float = 1e-7,
) -> Any:
    """Exact Top-K (`K=64`) + Tail-Bucket Reverse-KL On-Policy Distillation (`31B -> E4B`).

    Computes `KL(pi_student || pi_teacher)` on active student rollout tokens without transferring
    full 262,144-vocab teacher tensors across GPUs!
      `student_full_logits`: `[B, T, V]` (pre-softcap or softcapped if `softcap <= 0`)
      `teacher_topk_indices`: `[B, T, K]` int64 token IDs
      `teacher_topk_logprobs`: `[B, T, K]` normalized teacher log-probabilities
      `action_mask`: `[B, T]` binary mask (`1` on student generated tokens)
    """
    import torch
    import torch.nn.functional as F

    logits = student_full_logits.to(torch.float32)
    if softcap > 0.0:
        logits = float(softcap) * torch.tanh(logits / float(softcap))

    student_logprobs_all = F.log_softmax(logits, dim=-1)
    student_topk_logprobs = torch.gather(student_logprobs_all, dim=-1, index=teacher_topk_indices)
    student_topk_probs = torch.exp(student_topk_logprobs)

    t_topk_lp = teacher_topk_logprobs.to(torch.float32)
    t_topk_probs = torch.exp(t_topk_lp)

    # Top-K Reverse KL term: sum_{v in TopK} p_S(v) * (log p_S(v) - log p_T(v))
    topk_rkl = (student_topk_probs * (student_topk_logprobs - t_topk_lp)).sum(dim=-1)

    # Compressed tail bucket ensuring exact probability conservation on [0, 1]
    s_tail_prob = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=0.0, max=1.0)
    t_tail_prob = torch.clamp(1.0 - t_topk_probs.sum(dim=-1), min=float(tail_eps), max=1.0)
    tail_rkl = torch.where(
        s_tail_prob > float(tail_eps),
        s_tail_prob * (torch.log(torch.clamp(s_tail_prob, min=float(tail_eps))) - torch.log(t_tail_prob)),
        torch.zeros_like(s_tail_prob),
    )

    mask_f = action_mask.to(torch.float32)
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return ((topk_rkl + tail_rkl) * mask_f).sum() / denom


def compute_hybrid_opd_rloo_advantages(
    rloo_episode_advantages: Any,
    teacher_sampled_token_logprobs: Any,
    student_old_sampled_token_logprobs: Any,
    *,
    lambda_opd: float = 0.10,
    kl_clip: float = 5.0,
) -> Any:
    """Stage-3 E4B Hybrid Advantage: `pytest` RLOO + Annealed 31B Teacher Reverse-KL Shaping."""
    import torch

    # Broadcast scalar episode advantage [B] -> [B, T] if needed
    ep_adv = rloo_episode_advantages.to(torch.float32)
    if ep_adv.ndim == 1 and teacher_sampled_token_logprobs.ndim == 2:
        ep_adv = ep_adv.unsqueeze(-1)

    token_rkl_signal = torch.clamp(
        (teacher_sampled_token_logprobs - student_old_sampled_token_logprobs).to(torch.float32),
        min=-float(kl_clip),
        max=float(kl_clip),
    )
    return ep_adv + float(lambda_opd) * token_rkl_signal
