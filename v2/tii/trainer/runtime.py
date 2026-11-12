"""Process setup and engine-numerics config for the aligned (trainer-inference invariant) training forward.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: The aligned trainer forward must run the same kernel *branches* as the rollout engine. vLLM chooses
  branches from process-wide state that is read at import time or on the first cuBLAS handle:
  `VLLM_BATCH_INVARIANT` (read by `triton_unified_attention` at import, by `layernorm.rms_norm`, by the LoRA op
  configs and by `UnquantizedLinearMethod`), the aten overrides installed by `enable_batch_invariant_mode()` (log_softmax,
  softmax, mean.dim, bmm; mm/linear only on sm80/89/100) and the cuBLAS workspace env vars. `setup_process()` reproduces
  the engine worker's `init_batch_invariance(TRITON_ATTN)` call in the trainer process before any vLLM kernel module is
  imported, then applies the engine track's kernel patches (if `v2/tii/engine/aligned_engine.py` exposes them) so that
  both sides run identical code. `EngineNumerics` records every engine choice the trainer forward depends on (linear
  backend, lm_head GEMM, custom-op vs native dispatch, LoRA geometry, KV block size, logprobs mode); it is filled from
  the engine track's settings, never guessed silently.
  Reads: environment, optional module `v2.tii.engine.aligned_engine`. Writes: process env vars, torch backend flags.
  Depends on: vLLM 0.19.1 inside /path/to/gemma-v2/tii-venv.
SRP/DRY check: Pass - no other module in the repo sets up vLLM batch-invariant state in a training process; we call
  vLLM's own `init_batch_invariance` instead of re-implementing it.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from typing import Any

_SETUP_DONE = False
_CFG_CTX: Any = None


@dataclasses.dataclass(frozen=True)
class EngineNumerics:
    """Every engine-side choice the trainer forward must mirror to be bitwise equal.

    linear_backend: "marlin_engine"  - avenue A: engine-track batch-invariant Marlin (torch.ops.tii_marlin.gemm);
                    "dense_bf16_bi"  - vLLM UnquantizedLinearMethod under VLLM_BATCH_INVARIANT=1
                                       (`linear_batch_invariant` -> Triton `matmul_persistent`) on dequantized bf16 weights;
                    "w4a16_engine"   - the engine track's batch-invariant W4A16 Triton GEMM on the packed INT4 bytes
                                       (looked up in v2.tii.engine.aligned_engine).
    lm_head_backend: "bi_persistent" - `matmul_persistent` (needs the engine to route lm_head there as well);
                     "torch_linear"  - `F.linear` (cuBLASLt on sm120; NOT M-invariant unless proven).
    custom_ops: True -> RMSNorm / GeluAndMul / RotaryEmbedding run their `forward_cuda` kernels (engine
                compile mode NONE with custom_ops all); False -> `forward_native` torch code in eager.
    lora_max_rank / lora_max_loras: the engine's `max_lora_rank` / `max_loras` (stacked LoRA buffers are padded
                to them, and the shrink/expand kernels reduce over the padded rank).
    lora_wrapped: modules the engine wraps with a LoRA layer; each gets a shrink+expand pass even when the
                adapter has no weight for it (zero slices), exactly as Punica does.
    kv_block_size: engine KV-cache block size (TRITON_ATTN requires a multiple of 16).
    logprobs_mode: vLLM `--logprobs-mode` ("processed_logprobs" in production).
    lora_shrink: the engine's Punica shrink (env TII_LORA_SHRINK): "pin" = vLLM shrink pinned to block_k 32 /
                split_k 1 (plugin P3); "split" = TII fixed split-K 16 shrink with ordered reduce
                (v2/tii/engine/lora_shrink_tii.py, plugin P3b). Defines the bytes of the LoRA buffer h.
    """

    linear_backend: str = "dense_bf16_bi"
    lm_head_backend: str = "bi_persistent"
    custom_ops: bool = True
    lora_max_rank: int = 16
    lora_max_loras: int = 1
    lora_wrapped: tuple[str, ...] = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    kv_block_size: int = 16
    logprobs_mode: str = "processed_logprobs"
    lora_shrink: str = "pin"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


AVENUE_TO_LINEAR = {"w4a16": "w4a16_engine", "marlin": "marlin_engine", "dense": "dense_bf16_bi"}


def _engine_module() -> Any | None:
    """The engine track's settings module, or None when it does not exist yet (we then run the defaults above)."""
    try:
        return importlib.import_module("v2.tii.engine.aligned_engine")
    except ModuleNotFoundError:
        return None


def engine_numerics(overrides: dict[str, Any] | None = None) -> EngineNumerics:
    """EngineNumerics from the engine module's `TRAINER_NUMERICS` dict (if present) plus explicit overrides."""
    base: dict[str, Any] = {}
    mod = _engine_module()
    if mod is not None and hasattr(mod, "TRAINER_NUMERICS"):
        base.update(dict(mod.TRAINER_NUMERICS))
    if overrides:
        base.update(overrides)
    if "lora_wrapped" in base:
        base["lora_wrapped"] = tuple(base["lora_wrapped"])
    base.setdefault("lora_shrink", os.environ.get("TII_LORA_SHRINK", "pin"))  # the engine plugin's switch
    if base["lora_shrink"] not in ("pin", "split"):
        raise ValueError(f"lora_shrink must be 'pin' or 'split', got {base['lora_shrink']!r}")
    return EngineNumerics(**base)


def setup_process(*, aligned_plugin: bool = True, linear: str | None = None, lora_shrink: str | None = None) -> dict[str, Any]:
    """Put this process into the engine's kernel-selection state. Call before importing vLLM kernel modules.

    Order mirrors an engine worker: (1) env (VLLM_BATCH_INVARIANT=1; TII_ALIGNED=1 + TII_LINEAR for the engine
    track's general plugin), (2) `load_general_plugins()` - the same entry point every vLLM process runs, which
    applies the engine track's kernel patches (W4A16 kernel registration, sm120 matmul overrides, LoRA shrink pin,
    attention masked-tile fix), (3) `init_batch_invariance(TRITON_ATTN)` as in gpu_worker, (4) a default VllmConfig
    context so vLLM CustomOps (RotaryEmbedding, RMSNorm) can be constructed outside an engine.
    Returns a receipt (env values, capability, applied patches) for the evidence record.
    """
    global _SETUP_DONE
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if aligned_plugin:
        os.environ["TII_ALIGNED"] = "1"
        if linear is not None:
            os.environ["TII_LINEAR"] = linear
        if lora_shrink is not None:  # same switch the engine plugin reads (P3 pin / P3b split)
            os.environ["TII_LORA_SHRINK"] = lora_shrink
        os.environ.setdefault("TII_DEV_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[3]))
    import torch

    from vllm.model_executor.layers import batch_invariant as bi
    from vllm.plugins import load_general_plugins
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not _SETUP_DONE:
        load_general_plugins()
        # Same call the engine's gpu_worker makes (init_worker_distributed_environment -> init_batch_invariance):
        # sets CUBLAS/NCCL env, installs the aten overrides, forces fp32 IEEE matmul.
        bi.init_batch_invariance(AttentionBackendEnum.TRITON_ATTN)
        from vllm.config import VllmConfig, set_current_vllm_config

        global _CFG_CTX
        _CFG_CTX = set_current_vllm_config(VllmConfig())
        _CFG_CTX.__enter__()  # held for the process lifetime (trainer constructs CustomOps outside an engine)
        _SETUP_DONE = True

    import vllm.envs as envs
    from vllm.v1.attention.ops import triton_unified_attention as tua

    if not envs.VLLM_BATCH_INVARIANT or not tua.is_batch_invariant:
        raise RuntimeError("VLLM_BATCH_INVARIANT is not active in this process; import order is wrong")
    applied: dict[str, Any] = {}
    try:
        from v2.tii.engine import vllm_plugin

        applied = {k: (v if isinstance(v, (bool, int, str, list, dict)) else str(v)) for k, v in vllm_plugin._APPLIED.items()}
    except Exception as exc:  # the engine plugin is optional for single-operator work; record why it is absent
        applied = {"error": repr(exc)}
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    return {
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "TII_ALIGNED": os.environ.get("TII_ALIGNED"),
        "TII_LINEAR": os.environ.get("TII_LINEAR"),
        "TII_LARGEM": os.environ.get("TII_LARGEM", ""),
        "TII_ATTN_FIX": os.environ.get("TII_ATTN_FIX"),
        "TII_LORA_SHRINK": os.environ.get("TII_LORA_SHRINK", "pin"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUBLASLT_WORKSPACE_SIZE": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
        "fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "capability": cap,
        "engine_plugin_applied": applied,
        "attention_2d_kernel": tua.kernel_unified_attention_2d.fn.__name__ if hasattr(tua.kernel_unified_attention_2d, "fn") else str(tua.kernel_unified_attention_2d),
        "torch": torch.__version__,
    }
