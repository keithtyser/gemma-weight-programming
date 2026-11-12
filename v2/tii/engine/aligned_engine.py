"""TII aligned rollout engine: the exact vLLM 0.19.1 settings (env, engine args, patched modules) on sm120.

Author: Claude Opus 5.5 (engine track), 2026-11-05.

This is the single source of truth for the engine side of the train-inference invariance contract. The trainer
track reads ENGINE_FACTS / KERNELS to call the same kernels with the same constants; the C3/C4 tests build their
engines only through `aligned_env()` + `aligned_engine_kwargs()`.

Two avenues (closure field `avenue`):
  "w4a16" (A, preferred): serve the INT4 compressed-tensors checkpoint; every quantized Linear runs
          torch.ops.tii.w4a16_gemm (v2/tii/engine/w4a16_triton.py), Marlin & co. disabled.
  "dense" (B): serve the bit-exact bf16 dequantized twin (v2/tii/engine/build_proxy.py / dequantize_checkpoint);
          every Linear runs vLLM's linear_batch_invariant -> matmul_persistent.
Shared by both: VLLM_BATCH_INVARIANT=1 (Triton RMSNorm, Triton log_softmax, TRITON_ATTN 2D kernel, LoRA
split_k=1), the vllm_plugin patches P2 (sm120 persistent-matmul aten overrides -> lm_head), P3 (LoRA shrink
block_k=32 pin), P4 (attention leading-masked-tile fix), CompilationMode.NONE with every custom op enabled (no
Inductor fusion), FULL CUDA graphs, processed logprobs.

Usage (offline):
    import os
    from v2.tii.engine.aligned_engine import aligned_env, aligned_engine_kwargs
    os.environ.update(aligned_env("w4a16"))          # before importing vllm
    from vllm import LLM
    llm = LLM(**aligned_engine_kwargs(model_path, enable_lora=True))
Server: `aligned_server_cmd(...)` returns the argv list for `vllm.entrypoints.openai.api_server`.
"""

from __future__ import annotations

import json
import os
from typing import Any

AVENUES = ("w4a16", "dense", "marlin")  # marlin = TII batch-invariant Marlin (v2/tii/engine/tii_marlin_op.py)
# the checkout this module is imported from (engine subprocesses must load the plugin from the same tree)
TII_DEV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TII_VENV_PYTHON = "/path/to/gemma-v2/tii-venv/bin/python"

# Kernels vLLM 0.19.1 may otherwise pick for compressed-tensors WNA16 on CUDA. Disabled as a second guard; P1
# also inserts the TII kernel at the head of the list.
_DISABLED_MP_KERNELS = "MarlinLinearKernel,MacheteLinearKernel,AllSparkLinearKernel,CutlassW4A8LinearKernel,ConchLinearKernel,ExllamaLinearKernel"

COMPILATION_CONFIG: dict[str, Any] = {
    "mode": 0,  # CompilationMode.NONE: no Dynamo/Inductor; every op runs its named kernel
    "custom_ops": ["all"],  # RMSNorm/GeluAndMul/rotary use forward_cuda (RMSNorm -> BI Triton kernel under BI)
    # whole-forward CUDA graphs for batches <= max capture size; eager above. TII_CUDAGRAPH_MODE=FULL_DECODE_ONLY: graphs
    # only for uniform decode batches (seg7 then takes its decode-only path; mixed steps run eager)
    "cudagraph_mode": os.environ.get("TII_CUDAGRAPH_MODE", "FULL"),
}

# Facts the trainer must mirror (all verified by executed tests listed in reports/2026-11-05-tii-engine-track.md).
ENGINE_FACTS: dict[str, Any] = {
    "vllm_version": "0.19.1",
    "gpu": "RTX PRO 6000 Blackwell (sm120)",
    "dtype": "bfloat16",
    "kv_cache_dtype": "auto (bf16)",
    "attention_backend": "TRITON_ATTN, 2D kernel only (VLLM_BATCH_INVARIANT=1), TILE_SIZE 32, BLOCK_M 16, "
                         "BLOCK_Q 4, kernel = v2.tii.engine.unified_attention_tii.kernel_unified_attention_2d_tii",
    "linear_w4a16": "torch.ops.tii.w4a16_gemm (v2.tii.engine.w4a16_triton), BLOCK_K=32 fixed, no split-K, "
                    "weights repacked by repack_ct_weight/repack_ct_scale; merged qkv and gate_up as in vLLM",
    "linear_marlin": "torch.ops.tii_marlin.gemm (v2.tii.engine.tii_marlin_op, KERNEL_VERSION tii-marlin-k128n64L128t3): "
                     "vLLM 0.19.1 Marlin, thread_k 128 / 4 k-warps, (128x64, 128 thr) for M-chunks < 128 rows and "
                     "(128x128, 256 thr) above, thread_m_blocks <= 3, no K-striping, no 8-row MMA, no atomics, fp32 acc; "
                     "weights = vLLM gptq_marlin_repack + marlin_permute_scales (tii_marlin_op.prepare_weights)",
    "preferred_avenue": "marlin (exact INT4, ~Marlin speed); w4a16 Triton and dense kept as alternatives",
    "linear_dense": "vllm.model_executor.layers.batch_invariant.linear_batch_invariant -> matmul_persistent "
                    "(bf16 config BLOCK 128x128x64, GROUP_SIZE_M 8, 3 stages, 8 warps)",
    "lm_head": "tied embed_tokens [262144,2560] bf16 via F.linear -> aten override (P2) -> matmul_persistent",
    "final_softcap": "bf16: logits/30 -> tanh -> *30, each rounded to bf16, then .float()",
    "temperature": "fp32 logits.div_(T) (T=1.0)",
    "logprobs": "processed_logprobs: fp32 logits -> batch_invariant log_softmax Triton kernel (BLOCK 1024)",
    "rms_norm": "batch_invariant.rms_norm Triton (fp32 sum of squares in 1024 chunks, x*inv_rms*w in fp32, "
                "one rounding); v_norm uses an all-ones weight",
    "gelu_and_mul": "torch.ops._C.gelu_tanh_and_mul (custom op, fp32 math one rounding)",
    "ple_gelu": "torch.nn.functional.gelu(approximate='tanh') eager on bf16",
    "rotary": "torch.ops._C.rotary_embedding (neox), cos/sin cache from vLLM get_rope",
    "lora_shrink": {"block_m": 32, "block_n": 16, "block_k": 32, "split_k": 1, "num_warps": 4,
                    "num_stages": 2, "group_size_m": 8, "out": "fp32 buffer"},
    "lora_expand": "vLLM default expand config (block_m 64, block_n 64|128, block_k 32), fp32 acc -> bf16, "
                   "bf16 add into base output; B pre-scaled by alpha/r in bf16 at load",
    "max_lora_rank": 16,
    "compilation_config": COMPILATION_CONFIG,
    # engine-speed options (engine track 2026-11-06, reports/2026-11-05-tii-engine-track.md "Update 2026-11-06")
    "attention_seg7": "TII_ATTN=seg7: v2.tii.engine.segmented_attention_v7.unified_attention_tii (fixed absolute "
                      "512-key segments, 32-key tiles, L as a bf16(P)@1 MMA chain, ordered _merge, IEEE div); the "
                      "trainer gets it through plugin P4c (ops.triton_unified_attention.unified_attention)",
    "bit_identical_options": {"TII_SEG7_CASCADE": "1", "TII_LORA_TARGETS": "packed+adapter q/k/v/o/gate/up/down",
                              "TII_LORA_SHRINK_MEMSET": "0", "TII_FUSE_NORMS": "1", "TII_FUSE_GELU": "1",
                              "TII_LOGSOFTMAX": "par"},
}


W4A16_VERSION = "tii-w4a16-v1"  # numerics of torch.ops.tii.w4a16_gemm; v2 = fixed split-K table SPLIT_K_V2


def aligned_env(avenue: str = "w4a16", *, attn_fix: bool = True, receipt_dir: str | None = None,
                w4a16_version: str = W4A16_VERSION, attention: str | None = None) -> dict[str, str]:
    """Environment for every process of the aligned engine (set before importing vllm). The trainer must import
    v2.tii.engine.w4a16_triton with the same TII_W4A16_VERSION in its environment."""
    assert avenue in AVENUES, avenue
    env = {
        "TII_W4A16_VERSION": w4a16_version,
        "TII_ATTN": attention or os.environ.get("TII_ATTN", "2d"),
        # "pin" = vLLM shrink with block_k=32/split_k=1 (P3); "split" = TII fixed split-K shrink (P3b)
        "TII_LORA_SHRINK": os.environ.get("TII_LORA_SHRINK", "pin"),  # "2d" = vLLM 2D kernel + fix (P4); "seg" = segmented attention (P4b)
        "TII_ALIGNED": "1",
        # "w4" = TII-W4 small-M GEMM for M <= 64 (bitwise identical to TII-Marlin; v2/tii/kernel/tii_w4.py)
        "TII_SMALLM": os.environ.get("TII_SMALLM", ""),
        # "w4l" = TII-W4L large-M GEMM for M > 64 (bitwise identical to TII-Marlin; v2/tii/kernel/tii_w4l.py)
        "TII_LARGEM": os.environ.get("TII_LARGEM", ""),
        "TII_LMHEAD": os.environ.get("TII_LMHEAD", "table"),  # lm_head tile table; bytes == vLLM matmul_persistent
        "TII_LORA_EXPAND": os.environ.get("TII_LORA_EXPAND", "small"),  # LoRA expand tiles; bytes == vLLM default config
        "TII_LINEAR": avenue,
        "TII_ATTN_FIX": "1" if attn_fix else "0",
        "TII_DEV_ROOT": TII_DEV_ROOT,
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_DISABLED_KERNELS": _DISABLED_MP_KERNELS,
    }
    # engine-speed options (bit-identical or recorded numerics): pass through / record the resolved values
    env["TII_LORA_TARGETS"] = os.environ.get("TII_LORA_TARGETS", ",".join(LORA_TARGET_MODULES))
    for k in ("TII_LORA_SHRINK_MEMSET", "TII_FUSE_NORMS", "TII_FUSE_QKV_NORM", "TII_FUSE_GELU", "TII_SEG7_CASCADE", "TII_LORA_SPECIALIZE", "TII_CUDAGRAPH_MODE",
              "TII_LOGSOFTMAX"):
        if k in os.environ:
            env[k] = os.environ[k]
    for k in os.environ:
        if k.startswith("TII_SEG7_"):
            env[k] = os.environ[k]
    if receipt_dir:
        env["TII_RECEIPT_DIR"] = receipt_dir
    return env


# vLLM wraps EVERY supported linear for LoRA (incl. the PLE per_layer_input_gate / per_layer_projection), and runs
# shrink + expand with all-zero weights where the adapter has none (4 of 12 LoRA launches per decoder layer). The RL
# adapters target q/o/gate/up/down, i.e. the packed vLLM modules below; the trainer already skips modules without
# adapter weights (aligned_forward._build_lora). TII_LORA_TARGETS=all restores vLLM's default (wrap everything).
# vLLM matches the suffix of the WRAPPED (packed) module names when wrapping and the suffix of the ADAPTER module names
# when loading (worker_manager warns "will be ignored" for q_proj / gate_proj / up_proj if only packed names are
# listed), so both spellings are listed; per_layer_input_gate / per_layer_projection stay unwrapped.
LORA_TARGET_MODULES = ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "gate_proj", "up_proj",
                       "down_proj")


def lora_target_modules() -> list[str] | None:
    v = os.environ.get("TII_LORA_TARGETS", ",".join(LORA_TARGET_MODULES))
    return None if v == "all" else [x for x in v.split(",") if x]


def lora_kwargs(max_loras: int, max_lora_rank: int = 16) -> dict[str, Any]:
    kw: dict[str, Any] = {"enable_lora": True, "max_loras": max_loras, "max_lora_rank": max_lora_rank}
    tm = lora_target_modules()
    if tm is not None:
        kw["lora_target_modules"] = tm
    if lora_specialize():
        kw["specialize_active_lora"] = True
    return kw


def lora_specialize() -> bool:
    """TII_LORA_SPECIALIZE=1: vLLM specialize_active_lora (LoRA kernel grids sized by the number of active adapters,
    one CUDA graph per power-of-2 count) instead of max_loras + 1 slots; the grid z only adds empty CTAs, numerics
    unchanged."""
    return os.environ.get("TII_LORA_SPECIALIZE", "0") == "1"


def aligned_engine_kwargs(
    model: str,
    *,
    enable_lora: bool = False,
    max_loras: int = 2,
    max_lora_rank: int = 16,
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.12,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "model": model,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "logprobs_mode": "processed_logprobs",
        "enable_prefix_caching": enable_prefix_caching,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "scheduler_reserve_full_isl": False,
        "attention_backend": "TRITON_ATTN",
        "compilation_config": dict(COMPILATION_CONFIG),
        "enforce_eager": enforce_eager,
        "seed": seed,
    }
    if enable_lora:
        kw.update(lora_kwargs(max_loras, max_lora_rank))
    return kw


def aligned_server_cmd(model: str, port: int, *, lora_modules: dict[str, str] | None = None,
                       max_model_len: int = 32768, gpu_memory_utilization: float = 0.8,
                       max_lora_rank: int = 16) -> list[str]:
    """argv for an OpenAI server with the aligned settings (env from aligned_env must be exported too).

    Mirrors repo v2/eval/gpu_runner.py production flags except the alignment deltas
    (-O0 + custom ops all + FULL graphs, TRITON_ATTN explicit)."""
    cmd = [
        TII_VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--host", "127.0.0.1",
        "--port", str(port), "--max-model-len", str(max_model_len), "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization), "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4", "--reasoning-config",
        json.dumps({"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}),
        "--no-scheduler-reserve-full-isl", "--logprobs-mode", "processed_logprobs", "--enable-prefix-caching",
        "--attention-backend", "TRITON_ATTN", "--compilation-config", json.dumps(COMPILATION_CONFIG),
    ]
    if lora_modules:
        cmd += ["--enable-lora", "--max-loras", str(len(lora_modules)), "--max-lora-rank", str(max_lora_rank),
                "--lora-modules", *[f"{k}={v}" for k, v in sorted(lora_modules.items())]]
        tm = lora_target_modules()
        if tm is not None:
            cmd += ["--lora-target-modules", *tm]
        if lora_specialize():
            cmd += ["--specialize-active-lora"]
    return cmd


def rollout_sampling_kwargs(seed: int | None = None, max_tokens: int = 256) -> dict[str, Any]:
    """Rollout sampling as in production RL: T=1, top_p=1, top_k=262144 (= vocab, i.e. off), logprobs=1."""
    kw: dict[str, Any] = {"temperature": 1.0, "top_p": 1.0, "top_k": 262144, "logprobs": 1, "max_tokens": max_tokens}
    if seed is not None:
        kw["seed"] = seed
    return kw
