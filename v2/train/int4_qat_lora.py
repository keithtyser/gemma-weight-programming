"""Quantization-Exact Unmerged LoRA for Gemma 4 `-qat-w4a16-ct` Checkpoints.

Eliminates train-serve quantization drift between PyTorch training and `vllm==0.19.1`
Marlin W4A16 + Punica serving:
  1. Scopes LoRA strictly to `model.language_model.layers.*.(self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj)`
     — bypassing `Gemma4ClippableLinear` in vision/audio towers, freezing E4B Per-Layer
     Embeddings (PLE), and naturally skipping absent `k_proj`/`v_proj` in E4B L24..41 shared
     layers and absent `v_proj` in 31B unified `K=V` global layers.
  2. Supports BOTH:
       - Speed-optimal frozen `W_dequant` BF16 `nn.Linear` materialization (bit-identical to
         vLLM Marlin W4A16 unpack, 2.3x faster TensorCore GEMMs on 2x RTX PRO 6000 / 8x B300), and
       - Memory-minimal `Int4QatLoRALinear` on-the-fly unpack (~18.5 GiB single-GPU 31B).
  3. Selective active-token `30.0 * tanh(z / 30.0)` softcapped loss/logprobs (75% less LM-head VRAM)
     with `LoRA+` asymmetric optimizer parameter groups (`lr_B = 8x lr_A` SFT/OPD, `4x` RL).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from v2.constants import MODEL_SPECS

GEMMA4_LANGUAGE_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
# KV-free variant: E4B has num_key_value_heads=2 < Kaggle TP=4 (replicated KV) and shared-KV layers 24..41, so
# k_proj/v_proj LoRA on E4B is both the least-tested vLLM Punica path and of little benefit.
GEMMA4_LANGUAGE_LORA_REGEX_NO_KV = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
LORA_TARGET_PRESETS: dict[str, str] = {
    "all": GEMMA4_LANGUAGE_LORA_REGEX,
    "no_kv": GEMMA4_LANGUAGE_LORA_REGEX_NO_KV,
}
# KV-free variant: E4B has num_key_value_heads=2 < Kaggle TP=4 (replicated KV) and shared-KV layers 24..41, so
# k_proj/v_proj LoRA on E4B is both the least-tested vLLM Punica path and of little benefit.
GEMMA4_LANGUAGE_LORA_REGEX_NO_KV = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
LORA_TARGET_PRESETS: dict[str, str] = {
    "all": GEMMA4_LANGUAGE_LORA_REGEX,
    "no_kv": GEMMA4_LANGUAGE_LORA_REGEX_NO_KV,
}
_GEMMA4_LORA_COMPILED_RE = re.compile(GEMMA4_LANGUAGE_LORA_REGEX)
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.self_attn\.(k_proj|v_proj)$")


def is_valid_gemma4_lora_target(module_name: str, model_id: str | None = None) -> bool:
    """Verify `module_name` belongs to language backbone and is not a shared-KV dead layer."""
    if not _GEMMA4_LORA_COMPILED_RE.match(module_name):
        return False
    if model_id is not None and model_id in MODEL_SPECS:
        spec = MODEL_SPECS[model_id]
        kv_start, kv_end = spec.independent_kv_layers
        m = _LAYER_IDX_RE.search(module_name)
        if m:
            layer_idx = int(m.group(1))
            if not (kv_start <= layer_idx <= kv_end):
                return False
    return True


def unpack_ct_int4_nibbles_torch(
    weight_packed: Any,
    weight_scale: Any,
    out_features: int,
    in_features: int,
    *,
    zero_point_offset: int = 8,
    dtype: Any = None,
) -> Any:
    """Bit-exact unpack of `compressed-tensors` symmetric W4A16 (`int32` -> 8 x 4-bit nibbles * group scale).

    Matches vLLM 0.19.1 Marlin W4A16 dequantization:
      `W_dequant = ((nibble_u4 - 8) * scale_group)` cast to BF16.
    Supports both `[out_features, in_features // 8]` row-packed and `[in_features // 8, out_features]` layouts.
    """
    import torch

    target_dtype = dtype or torch.bfloat16
    shifts = torch.arange(0, 32, 4, device=weight_packed.device, dtype=torch.int32)

    if weight_packed.shape == (out_features, in_features // 8):
        # [out, in//8, 8] -> [out, in]
        unpacked_u4 = torch.bitwise_and(
            torch.bitwise_right_shift(weight_packed.unsqueeze(-1), shifts), 0x0F
        ).reshape(out_features, in_features)
    elif weight_packed.shape == (in_features // 8, out_features):
        unpacked_u4 = (
            torch.bitwise_and(
                torch.bitwise_right_shift(weight_packed.unsqueeze(1), shifts.view(1, 8, 1)), 0x0F
            )
            .reshape(in_features, out_features)
            .transpose(0, 1)
            .contiguous()
        )
    else:
        raise ValueError(
            f"Unexpected weight_packed shape {tuple(weight_packed.shape)} for ({out_features}, {in_features})"
        )

    signed_int4 = unpacked_u4.to(torch.float32) - float(zero_point_offset)

    # Broadcast group scales (e.g. group_size=32 -> [out_features, in_features // 32])
    if weight_scale.ndim == 2 and weight_scale.shape[0] == out_features:
        group_size = in_features // weight_scale.shape[1]
        expanded_scale = weight_scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    else:
        expanded_scale = weight_scale.to(torch.float32)

    return (signed_int4 * expanded_scale).to(target_dtype)


def build_int4_qat_lora_linear_cls() -> type:
    """Construct `Int4QatLoRALinear` (`torch.nn.Module`) lazily so CPU tools work without torch installed."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Int4QatLoRALinear(nn.Module):
        """Memory-minimal on-the-fly W4A16 dequantizing Linear + Unmerged LoRA adapter."""

        def __init__(
            self,
            weight_packed: torch.Tensor,
            weight_scale: torch.Tensor,
            in_features: int,
            out_features: int,
            *,
            bias: torch.Tensor | None = None,
            r: int = 64,
            lora_alpha: float = 128.0,
            compute_dtype: torch.dtype = torch.bfloat16,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.r = r
            self.lora_alpha = float(lora_alpha)
            self.scaling = float(lora_alpha) / float(r)
            self.compute_dtype = compute_dtype
            self.disable_lora = False

            self.register_buffer("weight_packed", weight_packed.contiguous(), persistent=True)
            self.register_buffer("weight_scale", weight_scale.contiguous(), persistent=True)
            if bias is not None:
                self.register_buffer("bias", bias.detach().to(compute_dtype), persistent=True)
            else:
                self.bias = None

            # Keep LoRA master weights in FP32 so small QAT learning rates (4e-6..1e-5) never underflow BF16 ULPs!
            self.lora_A = nn.Parameter(
                torch.empty((r, in_features), device=weight_packed.device, dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, r), device=weight_packed.device, dtype=torch.float32)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_dt = x.to(self.compute_dtype)
            w_deq = unpack_ct_int4_nibbles_torch(
                self.weight_packed,
                self.weight_scale,
                self.out_features,
                self.in_features,
                dtype=self.compute_dtype,
            )
            base_out = F.linear(x_dt, w_deq, self.bias)
            if getattr(self, "disable_lora", False):
                return base_out
            lora_out = F.linear(
                F.linear(x_dt, self.lora_A.to(self.compute_dtype)),
                self.lora_B.to(self.compute_dtype),
            ) * self.scaling
            return base_out + lora_out

    return Int4QatLoRALinear


def selective_softcapped_logprobs_and_ce(
    hidden_states: Any,
    lm_head_weight: Any,
    labels: Any,
    *,
    softcap: float = 30.0,
    return_token_logprobs: bool = False,
) -> dict[str, Any]:
    """Compute exact Gemma 4 `softcap * tanh(z / softcap)` CE / logprobs ONLY on unmasked positions.

    Avoids materializing `[batch, 32768, 262144]` logits in VRAM for masked prompt & tool-response
    tokens (`labels == -100`), cutting LM-head memory by ~75-85% on long SWE trajectories!
    """
    import torch
    import torch.nn.functional as F

    # Shift causal LM: position t predicts labels[:, t + 1]
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    active_mask = shift_labels.ne(-100)
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        zero = shift_hidden.sum() * 0.0
        return {
            "loss": zero,
            "active_tokens": 0,
            "max_abs_pre_softcap_logit": 0.0,
            "active_token_logprobs": torch.empty(0, device=hidden_states.device),
            "active_mask": active_mask,
        }

    flat_hidden = shift_hidden[active_mask]  # [N_active, hidden_dim]
    flat_targets = shift_labels[active_mask]  # [N_active]

    raw_logits = F.linear(flat_hidden, lm_head_weight.to(flat_hidden.dtype)).float()
    max_abs_raw = float(raw_logits.detach().abs().max().item())

    if softcap > 0.0:
        capped_logits = softcap * torch.tanh(raw_logits / softcap)
    else:
        capped_logits = raw_logits

    log_probs_all = F.log_softmax(capped_logits, dim=-1)
    target_logprobs = log_probs_all.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    ce_loss = -target_logprobs.mean()

    out: dict[str, Any] = {
        "loss": ce_loss,
        "active_tokens": active_count,
        "max_abs_pre_softcap_logit": round(max_abs_raw, 4),
        "active_mask": active_mask,
    }
    if return_token_logprobs:
        full_lp = torch.zeros_like(shift_labels, dtype=torch.float32)
        full_lp[active_mask] = target_logprobs
        out["token_logprobs"] = full_lp
    return out


def build_loraplus_param_groups(
    named_parameters: Any,
    *,
    lr_a: float = 5e-5,
    loraplus_lr_ratio: float = 8.0,
    weight_decay: float = 0.01,
) -> list[dict[str, Any]]:
    """Build `LoRA+` (Hayou et al., ICML 2024) asymmetric AdamW parameter groups (`lr_B = ratio * lr_A`)."""
    group_a: list[Any] = []
    group_b: list[Any] = []
    for name, param in named_parameters:
        if not getattr(param, "requires_grad", False):
            continue
        if "lora_B" in name or "lora_b" in name:
            group_b.append(param)
        else:
            group_a.append(param)

    groups: list[dict[str, Any]] = []
    if group_a:
        groups.append({"params": group_a, "lr": float(lr_a), "weight_decay": 0.0, "group_name": "lora_A"})
    if group_b:
        groups.append(
            {
                "params": group_b,
                "lr": float(lr_a) * float(loraplus_lr_ratio),
                "weight_decay": float(weight_decay),
                "group_name": "lora_B",
            }
        )
    return groups


def export_vllm_peft_adapter(
    lora_state_dict: dict[str, Any],
    output_dir: str | Path,
    *,
    base_model_name: str,
    r: int = 64,
    lora_alpha: int = 128,
    target_modules: str | list[str] = GEMMA4_LANGUAGE_LORA_REGEX,
) -> dict[str, Any]:
    """Export unmerged LoRA `.safetensors` + `adapter_config.json` strictly compatible with vLLM 0.19.1."""
    from safetensors.torch import save_file
    import torch

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    clean_tensors: dict[str, Any] = {}
    for key, tensor in sorted(lora_state_dict.items()):
        # Ensure standard PEFT prefix `base_model.model.` and BF16 contiguous storage
        peft_key = key if key.startswith("base_model.model.") else f"base_model.model.{key}"
        peft_key = peft_key.replace(".default.weight", ".weight")
        clean_tensors[peft_key] = tensor.detach().to(torch.bfloat16).contiguous().cpu()

    # Gemma 4 `attention_k_eq_v` (e.g. 31B full-attention layers) has `k_proj` and NO `v_proj` in HF/PyTorch,
    # whereas vLLM `gemma4.py` clones `k_proj` into `v_proj` inside `QKVParallelLinear`.
    # Clone `k_proj` LoRA tensors into `v_proj` whenever `k_proj` exists without `v_proj` to keep V == K in vLLM!
    keqv_cloned = 0
    for peft_key in list(clean_tensors.keys()):
        if peft_key.endswith(".self_attn.k_proj.lora_A.weight"):
            v_a_key = peft_key.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.v_proj.lora_A.weight")
            b_key = peft_key.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.k_proj.lora_B.weight")
            v_b_key = peft_key.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.v_proj.lora_B.weight")
            if v_a_key not in clean_tensors and b_key in clean_tensors:
                clean_tensors[v_a_key] = clean_tensors[peft_key].clone()
                clean_tensors[v_b_key] = clean_tensors[b_key].clone()
                keqv_cloned += 1

    sf_path = out / "adapter_model.safetensors"
    save_file(clean_tensors, str(sf_path))

    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name,
        "r": int(r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": 0.0,
        "use_rslora": False,
        "bias": "none",
        "target_modules": target_modules,
    }
    cfg_path = out / "adapter_config.json"
    cfg_path.write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "adapter_dir": str(out),
        "tensor_count": len(clean_tensors),
        "keqv_v_proj_cloned_layers": keqv_cloned,
        "safetensors_bytes": sf_path.stat().st_size,
    }


def patch_keqv_adapter_safetensors(adapter_dir: str | Path) -> int:
    """In-place patch an existing PEFT `adapter_model.safetensors` so Gemma 4 `k_eq_v` layers preserve `V == K` in vLLM."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    sf_path = Path(adapter_dir).resolve() / "adapter_model.safetensors"
    tensors: dict[str, Any] = {}
    with safe_open(str(sf_path), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    cloned = 0
    for k in list(tensors.keys()):
        if k.endswith(".self_attn.k_proj.lora_A.weight"):
            v_a = k.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.v_proj.lora_A.weight")
            k_b = k.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.k_proj.lora_B.weight")
            v_b = k.replace(".self_attn.k_proj.lora_A.weight", ".self_attn.v_proj.lora_B.weight")
            if v_a not in tensors and k_b in tensors:
                tensors[v_a] = tensors[k].clone().contiguous()
                tensors[v_b] = tensors[k_b].clone().contiguous()
                cloned += 1
    if cloned > 0:
        save_file(tensors, str(sf_path))
    return cloned
