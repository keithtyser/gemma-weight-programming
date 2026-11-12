"""Aligned operators: vLLM's own kernels in the forward, native PyTorch math in the backward.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: One `torch.autograd.Function` per value-producing operator of the Gemma 4 E4B logprob path. Each forward
  calls the exact vLLM 0.19.1 function the rollout engine executes (same Python entry, same kernel branch under
  VLLM_BATCH_INVARIANT=1), so the forward bytes are the engine's bytes. Each backward is the native derivative of
  the same math (fp32 where noted) under the gradient contract written next to it and collected in
  `GRADIENT_CONTRACT`; backward is not bitwise and is validated against an independent oracle (gate C7).
  Operators: RMSNorm (Triton BI kernel), GeluAndMul (`_C.gelu_tanh_and_mul`), RoPE (`_C.rotary_embedding`),
  frozen linear (pluggable: BI Triton persistent matmul on bf16 / engine W4A16 kernel), LoRA (Punica Triton
  shrink/expand with the engine's stacked-buffer geometry), attention (Triton unified attention over a paged view
  with an explicit chunk schedule = fixed absolute key tiles), scorer (lm_head GEMM, bf16 final softcap, fp32
  temperature divide, BI log_softmax, gather - the sampler's processed-logprob path).
  Reads/writes: tensors only. Callers: v2/tii/trainer/aligned_forward.py, tests v2/tii/tests/test_trainer_*.
  Depends on: vLLM 0.19.1 (tii-venv); runtime.setup_process() must run first.
SRP/DRY check: Pass - existing trainer math (v2/train/run_qat_lora_train.py FrozenBf16LoRALinear, grpo_train.token_logprobs)
  re-implements vLLM numerics in torch; this module instead calls vLLM's kernels and keeps torch only for backward.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import collections

import torch
import torch.nn.functional as F

# Executed-path counters (evidence that the aligned kernels actually ran; a test asserts they moved).
CALLS: collections.Counter = collections.Counter()

# Test hooks (never set in production): ORACLE routes every public op through an independent fp32 surrogate bridge
# (v2/tii/tests/_oracle_bridge.py, the C7 reference arm); PLANT scales one backward to prove a check can fail.
ORACLE: Any = None
PLANT: dict[tuple[str, str], float] = {}

# Opt-in per-operator GPU timing (diagnostic; synchronizes, never on in production): TIMING_ON = True fills TIMES.
import contextlib
import os as _os

TIMING_ON = _os.environ.get("TII_OP_TIMING") == "1"
# Fast backward (bwd_kernels.py fused Triton + flex attention). TII_FAST_BWD=0 selects the fp32 torch reference path.
FAST_BWD = _os.environ.get("TII_FAST_BWD", "1") == "1"
TIMES: collections.Counter = collections.Counter()


@contextlib.contextmanager
def timed(name: str):
    if not TIMING_ON:
        yield
        return
    torch.cuda.synchronize()
    import time as _t

    t0 = _t.perf_counter()
    yield
    torch.cuda.synchronize()
    TIMES[name] += _t.perf_counter() - t0

# --------------------------------------------------------------------------------------------------------------
# Gradient contract, one row per operator (rendered into the C6 report). "visible" = the forward the engine runs.
# --------------------------------------------------------------------------------------------------------------
GRADIENT_CONTRACT: dict[str, dict[str, str]] = {
    "rms_norm": {
        "visible": "vllm layernorm.rms_norm -> BI Triton _rms_norm_kernel: fp32 sum of squares, x*inv_rms*w in fp32, one bf16 rounding",
        "backward": "reuse (case A): fp32 analytic VJP of y=x*rsqrt(mean(x^2)+eps)*w recomputed from saved bf16 x; dx rounded to bf16 once",
        "trainable": "none (norm weights frozen in LoRA training; grad None)",
        "saved": "x (bf16), weight",
    },
    "gelu_tanh_and_mul": {
        "visible": "torch.ops._C.gelu_tanh_and_mul(out, cat(gate,up)): fp32 inside, one bf16 rounding",
        "backward": "reuse (case A): fp32 analytic derivative of 0.5g(1+tanh(k(g+0.044715g^3)))*u; dgate/dup rounded to bf16 once",
        "trainable": "none",
        "saved": "x=cat(gate,up) (bf16)",
    },
    "rotary_embedding": {
        "visible": "torch.ops._C.rotary_embedding (neox, in place on a clone) with the bf16 cos/sin cache vLLM builds on GPU",
        "backward": "reuse (case A): exact transpose of the rotation (cos, -sin) in fp32 from the same bf16 cache; rounded to bf16",
        "trainable": "none",
        "saved": "positions",
    },
    "frozen_linear": {
        "visible": "pluggable: dense_bf16_bi = vllm linear_batch_invariant (Triton matmul_persistent, fixed tile, fp32 acc); "
        "w4a16_engine = engine-track Triton INT4 kernel; marlin_engine = torch.ops.tii_marlin.gemm (batch-invariant Marlin, "
        "avenue A) on vLLM-repacked bytes of the same checkpoint",
        "backward": "reuse (case A dense; case B INT4 with STE): dX = dY @ W_ref with W_ref = the bf16 weight the kernel multiplies "
        "(dense) / dequant (q-8)*s in bf16 (INT4, = the deployed value; STE: the frozen INT4 grid gives no dW); bf16 GEMM with fp32 accumulation (torch.matmul: in an aligned sm120 process the engine plugin routes it to the BI Triton persistent matmul)",
        "trainable": "none (frozen base; no dW computed)",
        "saved": "nothing beyond the frozen weight",
    },
    "lora": {
        "visible": "Punica lora_shrink (x@A_dep^T into fp32 buffer, split_k=1 under BI) + lora_expand (bf16(buffer)@B_dep^T rounded to bf16, "
        "then bf16 add into the base output); A_dep=bf16(A), B_dep=bf16(bf16(B)*alpha/r) = bytes vLLM holds after load+optimize()",
        "backward": "new glue (case B: bf16 casts are STE): h=bf16(buffer); dB_dep=g^T h, dh=g B_dep, dA=dh^T x, dx+=dh A_dep, all fp32 "
        "accumulate from bf16 values; dA_master=dA, dB_master=dB_dep*alpha/r; dy_base=g (identity); dx rounded to bf16",
        "trainable": "lora_A, lora_B fp32 masters",
        "saved": "x (bf16, shared with the base linear), h (bf16 [slices,T,R])",
    },
    "attention": {
        "visible": "vllm triton unified_attention 2D kernel (BI), paged view of contiguous K/V, block table per chunk, "
        "chunk schedule reproducing the engine's prefill chunks and 1-query decode blocks (absolute 32-key tiles)",
        "backward": "glue (case A semantics): fp32 recompute of softmax(qk^T*scale + causal/window mask) per 256-query chunk; O, P from that "
        "one recompute (D=rowsum(dO*O_ref)); GQA dK/dV summed over the 4-head group in fixed order; dq/dk/dv rounded to bf16",
        "trainable": "none (inputs only)",
        "saved": "q, k, v (bf16), sequence boundaries",
    },
    "scorer": {
        "visible": "lm_head GEMM (pluggable, tied embedding) -> bf16 /30, tanh, *30 (LogitsProcessor) -> .float() -> div_ by fp32 "
        "temperature (Sampler) -> log_softmax(fp32) (BI Triton) -> gather",
        "backward": "reuse (case A): d/dlogits of log_softmax-gather in fp32 (onehot - p), / temperature, * (1 - tanh^2(z/30)) in fp32 from "
        "the bf16 raw logits, dH = dZ @ W in fp32 (vocab chunks of W upcast), rounded to bf16 once; W frozen",
        "trainable": "none (tied embedding frozen)",
        "saved": "hidden (bf16), targets, temperatures; logits recomputed per chunk",
    },
    "embedding_ple": {
        "visible": "F.embedding gathers (exact) * bf16 scalars; PLE projection through the frozen linear + RMSNorm",
        "backward": "none needed: no trainable parameter upstream of the embeddings",
        "trainable": "none",
        "saved": "-",
    },
    "elementwise": {
        "visible": "residual adds, *layer_scalar, F.gelu(tanh) of the PLE gate, gate*ple: the same eager torch bf16 ops as vLLM",
        "backward": "native autograd of the same ops (bf16)",
        "trainable": "none",
        "saved": "autograd default",
    },
}


# --------------------------------------------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------------------------------------------
def visible_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, custom_ops: bool) -> torch.Tensor:
    """vLLM RMSNorm.forward_cuda (custom op on) or forward_native (custom op off), no residual."""
    from vllm.model_executor.layers.layernorm import RMSNorm, rms_norm

    CALLS["rms_norm"] += 1
    if custom_ops:
        return rms_norm(x, weight, eps)  # VLLM_BATCH_INVARIANT -> rms_norm_batch_invariant (Triton)
    return RMSNorm.forward_static(x, eps, x.shape[-1], x.dtype, weight, None, None)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float, custom_ops: bool) -> torch.Tensor:
        y = visible_rms_norm(x, weight, eps, custom_ops)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_rmsnorm"):
            return RMSNormFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gy: torch.Tensor) -> tuple[Any, ...]:
        x, w = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise RuntimeError("RMSNorm weight is frozen under the gradient contract; trainable norm weights are not supported")
        if FAST_BWD:
            from v2.tii.trainer.bwd_kernels import rmsnorm_bwd

            return rmsnorm_bwd(x, w, gy, ctx.eps), None, None, None
        xf = x.float()
        n = x.shape[-1]
        rstd = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / n + ctx.eps)
        xhat = xf * rstd
        gw = gy.float() * w.float()
        dx = rstd * (gw - xhat * ((gw * xhat).sum(-1, keepdim=True) / n))
        return dx.to(x.dtype), None, None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, custom_ops: bool) -> torch.Tensor:
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.rms_norm(x, weight, eps, custom_ops)
    if torch.is_grad_enabled() and x.requires_grad:
        return RMSNormFn.apply(x, weight, eps, custom_ops)
    return visible_rms_norm(x, weight, eps, custom_ops)


# --------------------------------------------------------------------------------------------------------------
# GeluAndMul (tanh)
# --------------------------------------------------------------------------------------------------------------
def visible_gelu_and_mul(x: torch.Tensor, custom_ops: bool) -> torch.Tensor:
    d = x.shape[-1] // 2
    CALLS["gelu_tanh_and_mul"] += 1
    if custom_ops:
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        torch.ops._C.gelu_tanh_and_mul(out, x)  # GeluAndMul.forward_cuda
        return out
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]  # GeluAndMul.forward_native


_GELU_K = math.sqrt(2.0 / math.pi)


class GeluAndMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, custom_ops: bool) -> torch.Tensor:
        ctx.save_for_backward(x)
        return visible_gelu_and_mul(x, custom_ops)

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_gelu"):
            return GeluAndMulFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gy: torch.Tensor) -> tuple[Any, ...]:
        (x,) = ctx.saved_tensors
        if FAST_BWD:
            from v2.tii.trainer.bwd_kernels import gelu_and_mul_bwd

            return gelu_and_mul_bwd(x, gy), None
        d = x.shape[-1] // 2
        g = x[..., :d].float()
        u = x[..., d:].float()
        t = torch.tanh(_GELU_K * (g + 0.044715 * g * g * g))
        gelu = 0.5 * g * (1.0 + t)
        dgelu = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * _GELU_K * (1.0 + 3.0 * 0.044715 * g * g)
        gyf = gy.float()
        return torch.cat([gyf * u * dgelu, gyf * gelu], dim=-1).to(x.dtype), None


def gelu_and_mul(x: torch.Tensor, custom_ops: bool) -> torch.Tensor:
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.gelu_and_mul(x, custom_ops)
    if torch.is_grad_enabled() and x.requires_grad:
        return GeluAndMulFn.apply(x, custom_ops)
    return visible_gelu_and_mul(x, custom_ops)


# --------------------------------------------------------------------------------------------------------------
# Rotary embedding (neox; Gemma 4 default and proportional variants come from vLLM get_rope)
# --------------------------------------------------------------------------------------------------------------
def visible_rope(rope: Any, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor | None, custom_ops: bool):
    """Out-of-place wrapper around vLLM's in-place rotary: returns rotated copies of q (and k)."""
    q2 = q.clone()
    k2 = k.clone() if k is not None else None
    CALLS["rotary_embedding"] += 1
    if custom_ops:
        rope.forward_cuda(positions, q2, k2)
        return q2, k2
    return rope.forward_native(positions, q2, k2)


def _rope_transpose(rope: Any, positions: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Exact VJP of the neox rotation: per pair (x1, x2): dx1 = g1 cos + g2 sin, dx2 = g2 cos - g1 sin (fp32)."""
    cs = rope.cos_sin_cache.index_select(0, positions.flatten()).float()
    cos, sin = cs.chunk(2, dim=-1)  # [T, rot/2]
    hs, rot = rope.head_size, rope.rotary_dim
    T = positions.numel()
    gg = g.float().view(T, -1, hs)
    g1, g2, gp = gg[..., : rot // 2], gg[..., rot // 2 : rot], gg[..., rot:]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.cat([g1 * c + g2 * s, g2 * c - g1 * s, gp], dim=-1)
    return out.view(g.shape).to(g.dtype)


class RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor | None, positions: torch.Tensor, rope: Any, custom_ops: bool):
        ctx.rope = rope
        ctx.save_for_backward(positions)
        ctx.has_k = k is not None
        q2, k2 = visible_rope(rope, positions, q, k, custom_ops)
        return (q2, k2) if k is not None else (q2, q2.new_empty(0))

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_rope"):
            return RopeFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gq: torch.Tensor, gk: torch.Tensor) -> tuple[Any, ...]:
        (positions,) = ctx.saved_tensors
        if FAST_BWD and _os.environ.get("TII_FUSED_ROPE_BWD", "1") == "1":  # fused one-pass VJP, same fp32 math and roundings as _rope_transpose
            from v2.tii.trainer.bwd_kernels import rope_bwd

            r = ctx.rope
            dq = rope_bwd(gq, r.cos_sin_cache, positions, r.head_size, r.rotary_dim)
            dk = rope_bwd(gk, r.cos_sin_cache, positions, r.head_size, r.rotary_dim) if ctx.has_k else None
            return dq, dk, None, None, None
        dq = _rope_transpose(ctx.rope, positions, gq)
        dk = _rope_transpose(ctx.rope, positions, gk) if ctx.has_k else None
        return dq, dk, None, None, None


def rope_apply(rope: Any, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor | None, custom_ops: bool):
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.rope_apply(rope, positions, q, k, custom_ops)
    if torch.is_grad_enabled() and (q.requires_grad or (k is not None and k.requires_grad)):
        q2, k2 = RopeFn.apply(q, k, positions, rope, custom_ops)
        return q2, (k2 if k is not None else None)
    return visible_rope(rope, positions, q, k, custom_ops)


# --------------------------------------------------------------------------------------------------------------
# Frozen linear (base weights) with a pluggable visible kernel
# --------------------------------------------------------------------------------------------------------------
def dequant_int4_bf16(qweight: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """compressed-tensors pack-quantized uint4b8 -> bf16: ((nibble - 8) * scale) rounded once (= Marlin / trainer dequant)."""
    n, kp = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    u4 = torch.bitwise_and(torch.bitwise_right_shift(qweight.unsqueeze(-1), shifts), 0xF).reshape(n, kp * 8)
    s = scale.float().repeat_interleave(group, dim=1)
    return ((u4.float() - 8.0) * s).to(torch.bfloat16)


class FrozenLinear:
    """A frozen projection (no autograd parameters). Holds the engine-format weight and computes the visible GEMM.

    backend "dense_bf16_bi": `weight` [N, K] bf16; visible = vllm linear_batch_invariant (what UnquantizedLinearMethod
      runs under VLLM_BATCH_INVARIANT=1; with the engine plugin's sm120 overrides F.linear lands there too).
    backend "w4a16_engine": `qweight` int32 [K, N/8] (N packed) + `scale` bf16 [K/32, N] = the engine track's kernel
      layout (v2.tii.engine.w4a16_triton.repack_ct_*); visible = torch.ops.tii.w4a16_gemm, the op the engine's
      vLLM linear kernel calls.
    """

    def __init__(self, name: str, backend: str, *, weight: torch.Tensor | None = None,
                 qweight: torch.Tensor | None = None, scale: torch.Tensor | None = None) -> None:
        self.name = name
        self.backend = backend
        self.weight = weight
        self.qweight = qweight
        self.scale = scale
        if backend == "dense_bf16_bi":
            if weight is None:
                raise ValueError(f"{name}: dense_bf16_bi needs a bf16 weight")
            self.out_features, self.in_features = weight.shape
        elif backend == "w4a16_engine":
            if qweight is None or scale is None:
                raise ValueError(f"{name}: w4a16_engine needs packed int4 weight + scales (kernel layout)")
            self.in_features, self.out_features = qweight.shape[0], qweight.shape[1] * 8
        elif backend == "marlin_engine":
            # qweight/scale = compressed-tensors [N, K/8] int32 / [N, K/32] bf16 (kept for W_ref); the Marlin-layout
            # copy is made with the engine's own prepare_weights (vLLM gptq_marlin_repack + marlin_permute_scales).
            if qweight is None or scale is None:
                raise ValueError(f"{name}: marlin_engine needs compressed-tensors packed weight + scales")
            from v2.tii.engine import tii_marlin_op

            self.marlin = tii_marlin_op.prepare_weights(qweight, scale)
            self.out_features, self.in_features = qweight.shape[0], qweight.shape[1] * 8
        else:
            raise ValueError(f"unknown linear backend {backend!r}")

    def visible(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        CALLS[f"linear_{self.backend}"] += 1
        if self.backend == "dense_bf16_bi":
            from vllm.model_executor.layers.batch_invariant import linear_batch_invariant

            y = linear_batch_invariant(x2, self.weight)
        elif self.backend == "w4a16_engine":
            from v2.tii.engine.w4a16_triton import w4a16_gemm

            y = w4a16_gemm(x2, self.qweight, self.scale)
        else:
            from v2.tii.engine import tii_marlin_op

            q, sc, n, k = self.marlin
            y = tii_marlin_op.gemm(x2, q, sc, n, k)  # torch.ops.tii_marlin.gemm, what TiiMarlinLinearKernel.apply_weights calls
        return y.view(*x.shape[:-1], self.out_features)

    def weight_ref(self) -> torch.Tensor:
        """W_ref [N, K] of the gradient contract: the bf16 value the visible kernel multiplies."""
        if self.backend == "dense_bf16_bi":
            return self.weight
        if self.backend == "marlin_engine":
            return dequant_int4_bf16(self.qweight, self.scale)
        from v2.tii.engine.w4a16_triton import dequant_kn

        return dequant_kn(self.qweight, self.scale).t()


class FrozenLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lin: FrozenLinear) -> torch.Tensor:
        ctx.lin = lin
        ctx.x_shape = x.shape
        return lin.visible(x)

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_linear"):
            return FrozenLinearFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gy: torch.Tensor) -> tuple[Any, ...]:
        if FAST_BWD and ctx.lin.backend == "marlin_engine":
            from v2.tii.trainer.bwd_kernels import int4_dx

            gx = int4_dx(gy, ctx.lin.qweight, ctx.lin.scale)  # W_ref tiles dequantized on the fly (no bf16 copy)
        else:
            w = ctx.lin.weight_ref()
            gx = torch.matmul(gy.reshape(-1, gy.shape[-1]), w)  # dX = dY W_ref (bf16 in, fp32 accumulate, bf16 out)
        if PLANT:
            gx = gx * PLANT.get(("linear_dx", ctx.lin.name), 1.0)
        return gx.view(ctx.x_shape), None


def frozen_linear(x: torch.Tensor, lin: FrozenLinear) -> torch.Tensor:
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.frozen_linear(x, lin)
    if torch.is_grad_enabled() and x.requires_grad:
        return FrozenLinearFn.apply(x, lin)
    return lin.visible(x)


# --------------------------------------------------------------------------------------------------------------
# LoRA through Punica's Triton kernels
# --------------------------------------------------------------------------------------------------------------
class LoRASite:
    """One vLLM LoRA-wrapped linear (merged qkv: 3 slices, gate_up: 2, o/down: 1) with persistent stacked buffers.

    Buffers are allocated once and updated in place (`refresh`) because Punica caches kernel pointer tables by
    data_ptr (`_LORA_A_PTR_DICT`/`_LORA_B_PTR_DICT`): re-allocating could alias a stale entry.
    masters[s] = (A [r, in] fp32 Parameter, B [n_s, r] fp32 Parameter) or None for a slice without adapter.
    """

    def __init__(self, name: str, in_features: int, slices: Sequence[int], masters: Sequence[tuple[torch.nn.Parameter, torch.nn.Parameter] | None],
                 *, max_rank: int, max_loras: int, scaling: float, device: torch.device) -> None:
        self.name = name
        self.in_features = int(in_features)
        self.slices = tuple(int(s) for s in slices)
        self.masters = list(masters)
        self.max_rank = int(max_rank)
        self.scaling = float(scaling)
        self.a_stacked = tuple(torch.zeros(max_loras, 1, max_rank, in_features, dtype=torch.bfloat16, device=device) for _ in self.slices)
        self.b_stacked = tuple(torch.zeros(max_loras, 1, n, max_rank, dtype=torch.bfloat16, device=device) for n in self.slices)
        for m in self.masters:
            if m is not None and m[0].shape[0] > max_rank:
                raise ValueError(f"{name}: adapter rank {m[0].shape[0]} > engine max_lora_rank {max_rank}")

    def params(self) -> list[torch.nn.Parameter]:
        out: list[torch.nn.Parameter] = []
        for m in self.masters:
            if m is not None:
                out.extend(m)
        return out

    @torch.no_grad()
    def refresh(self) -> None:
        """Write deployed bytes into slot 0: A_dep = bf16(A); B_dep = bf16(B) *= scaling in bf16 (export + vLLM optimize())."""
        for s, m in enumerate(self.masters):
            self.a_stacked[s].zero_()
            self.b_stacked[s].zero_()
            if m is None:
                continue
            a, b = m
            r = a.shape[0]
            a_dep = a.detach().to(torch.bfloat16)
            b_dep = b.detach().to(torch.bfloat16)
            if self.scaling != 1:
                b_dep = b_dep * self.scaling  # LoRALayerWeights.optimize(): lora_b *= scaling on the bf16 tensor
            self.a_stacked[s][0, 0, :r, :].copy_(a_dep)
            self.b_stacked[s][0, 0, :, :r].copy_(b_dep)


def make_lora_meta(num_tokens: int, max_loras: int, device: torch.device) -> tuple[Any, ...]:
    """Punica kernel metadata for a batch where every token uses the adapter in slot 0."""
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    meta = LoRAKernelMeta.make(max_loras, num_tokens, device)
    meta.prepare_tensors(torch.zeros(num_tokens, dtype=torch.int32, device=device))
    return meta.meta_args(num_tokens, False)


# Punica shrink variant of the engine (EngineNumerics.lora_shrink; set by AlignedGemma4 from its numerics)
LORA_SHRINK = _os.environ.get("TII_LORA_SHRINK", "pin")


def visible_lora(x2: torch.Tensor, y_base: torch.Tensor, site: LoRASite, meta_args: tuple[Any, ...],
                 inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """PunicaWrapperGPU.add_lora_linear(y, x, a_stacked, b_stacked, 1.0, slices) on a copy of y_base (in place on y_base
    when the caller owns it: same kernels, same bytes). Returns (y, buffer)."""
    from vllm.lora.ops.triton_ops import lora_expand, lora_shrink

    if LORA_SHRINK == "split":  # engine plugin P3b: TII fixed split-K shrink, ordered reduce (same signature)
        from v2.tii.engine.lora_shrink_tii import tii_lora_shrink as lora_shrink
    elif LORA_SHRINK != "pin":
        raise ValueError(f"unknown LORA_SHRINK {LORA_SHRINK!r}")
    CALLS["lora_shrink_expand"] += 1
    CALLS[f"lora_shrink_{LORA_SHRINK}"] += 1
    y = y_base if inplace else y_base.clone()
    buffer = torch.empty((len(site.slices), x2.size(0), site.max_rank), dtype=torch.float32, device=x2.device)
    lora_shrink(x2, site.a_stacked, buffer, *meta_args, 1.0)
    lora_expand(buffer, site.b_stacked, y.view(-1, y.shape[-1]), *meta_args, offset_start=0, add_inputs=True)
    return y, buffer


class LoRAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y_base: torch.Tensor, site: LoRASite, meta_args: tuple[Any, ...], *params: torch.Tensor):
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y, buffer = visible_lora(x2, y_base, site, meta_args)
        ctx.site = site
        ctx.x_shape = x.shape
        ctx.save_for_backward(x2, buffer.to(torch.bfloat16))  # h = what lora_expand multiplies (CAST_TYPE)
        return y

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_lora"):
            return LoRAFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gy: torch.Tensor) -> tuple[Any, ...]:
        x2, h = ctx.saved_tensors
        site: LoRASite = ctx.site
        g2 = gy.reshape(-1, gy.shape[-1])
        if FAST_BWD:
            return LoRAFn._bw_fast(ctx, x2, h, site, g2, gy)
        xf = x2.float()
        dx = torch.zeros(x2.shape, dtype=torch.float32, device=x2.device)
        grads: list[torch.Tensor | None] = []
        off = 0
        for s, n in enumerate(site.slices):
            m = site.masters[s]
            if m is not None:
                a, b = m
                r = a.shape[0]
                if FAST_BWD:  # bf16 values, fp32 accumulation, no fp32 copies of the [T, n] slices
                    from v2.tii.trainer.bwd_kernels import mm_bf16_f32, mm_f32acc

                    gs16 = g2[:, off : off + n]
                    hs16 = h[s, :, :r]
                    db_dep = mm_bf16_f32(gs16.t(), hs16)  # [n, r]
                    dh = mm_bf16_f32(gs16, site.b_stacked[s][0, 0, :, :r])  # [T, r] fp32
                    da = mm_f32acc(dh.t().contiguous(), x2)  # [r, in], ~fp32 accurate
                    dx += mm_f32acc(dh, site.a_stacked[s][0, 0, :r, :])
                else:
                    gs = g2[:, off : off + n].float()
                    hs = h[s, :, :r].float()
                    a_dep = site.a_stacked[s][0, 0, :r, :].float()
                    b_dep = site.b_stacked[s][0, 0, :, :r].float()
                    db_dep = gs.t() @ hs  # [n, r]
                    dh = gs @ b_dep  # [T, r]
                    da = dh.t() @ xf  # [r, in]
                    dx += dh @ a_dep
                grads.append(da if ctx.needs_input_grad[4 + len(grads)] else None)
                grads.append(db_dep * site.scaling if ctx.needs_input_grad[4 + len(grads)] else None)
            off += n
        if PLANT and ("lora", site.name) in PLANT:
            grads = [g * PLANT[("lora", site.name)] if g is not None else None for g in grads]
        return (dx.to(x2.dtype).view(ctx.x_shape), gy, None, None, *grads)


def _lora_bw_fast(ctx: Any, x2: torch.Tensor, h: torch.Tensor, site: "LoRASite", g2: torch.Tensor, gy: torch.Tensor):
    """Consolidated LoRA backward (same contract: bf16 values, fp32 accumulation, dx rounded once):
    dh_s = g_s B_s (fp32), stacked over slices as DH [T, S*r] and split into bf16 hi + lo (hi + lo ~ fp32 DH);
    dx = [DH_hi | DH_lo] @ [A; A]  - ONE GEMM with K = 2*S*r, fp32 accumulation, bf16 output;
    dA = [DH_hi | DH_lo]^T @ x      - ONE GEMM, fp32 output, hi and lo halves summed;
    dB_s = g_s^T h_s (fp32 output) per slice."""
    from v2.tii.trainer.bwd_kernels import mm_bf16_f32, mm_tallk

    act = [(s_, n, site.masters[s_]) for s_, n in enumerate(site.slices)]
    offs = [0]
    for n in site.slices:
        offs.append(offs[-1] + n)
    live = [(s_, n, m) for s_, n, m in act if m is not None]
    dhs, a_deps, dbs = [], [], []
    for s_, n, m in live:
        r = m[0].shape[0]
        gs16 = g2[:, offs[s_] : offs[s_] + n]
        dbs.append(mm_tallk(gs16.t(), h[s_, :, :r]))  # [n, r]
        dhs.append(mm_bf16_f32(gs16, site.b_stacked[s_][0, 0, :, :r]))  # [T, r] fp32
        a_deps.append(site.a_stacked[s_][0, 0, :r, :])
    DH = torch.cat(dhs, 1)  # [T, S*r]
    hi = DH.to(torch.bfloat16)
    lo = (DH - hi.float()).to(torch.bfloat16)
    HL = torch.cat([hi, lo], 1)  # [T, 2*S*r]
    A2 = torch.cat(a_deps + a_deps, 0)  # [2*S*r, in]
    dx = torch.mm(HL, A2, out_dtype=torch.bfloat16)
    dA2 = mm_tallk(HL.t(), x2)  # [2*S*r, in]
    k = DH.shape[1]
    dA_all = dA2[:k] + dA2[k:]
    grads: list[torch.Tensor | None] = []
    col = 0
    for (s_, n, m), db in zip(live, dbs):
        r = m[0].shape[0]
        grads.append(dA_all[col : col + r] if ctx.needs_input_grad[4 + len(grads)] else None)
        grads.append(db * site.scaling if ctx.needs_input_grad[4 + len(grads)] else None)
        col += r
    if PLANT and ("lora", site.name) in PLANT:
        grads = [g * PLANT[("lora", site.name)] if g is not None else None for g in grads]
    return (dx.view(ctx.x_shape), gy, None, None, *grads)


LoRAFn._bw_fast = staticmethod(_lora_bw_fast)


def lora_apply(x: torch.Tensor, y_base: torch.Tensor, site: LoRASite, meta_args: tuple[Any, ...]) -> torch.Tensor:
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.lora_apply(x, y_base, site, meta_args)
    params = site.params()
    if torch.is_grad_enabled() and (x.requires_grad or y_base.requires_grad or any(p.requires_grad for p in params)):
        return LoRAFn.apply(x, y_base, site, meta_args, *params)
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    return visible_lora(x2, y_base, site, meta_args)[0]


# --------------------------------------------------------------------------------------------------------------
# Attention: Triton unified attention over a paged view, with an explicit chunk schedule
# --------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class AttnPlan:
    """Kernel launch metadata shared by every attention layer of one forward.

    seq_bounds: [(start, end)] token ranges of the logical sequences in the packed layout.
    chunks: per sequence, the engine's query chunks [(q_lo, q_hi)) relative to the sequence start (they partition
      [0, L)); a prompt prefilled in one step is one chunk, every decoded position is a 1-token chunk.
    The paged view pads every sequence to whole `block_size` pages; slot -> packed-token map is `page_src`.
    """

    seq_bounds: list[tuple[int, int]]
    block_size: int
    cu_q: torch.Tensor  # [num_chunks + 1] int32 query starts in packed order
    seqused_k: torch.Tensor  # [num_chunks] int32 = chunk end relative to its sequence
    block_table: torch.Tensor  # [num_chunks, max_pages] int32
    page_src: torch.Tensor  # [num_pages * block_size] int64 packed token index (or -1 for padding)
    num_pages: int
    max_q: int
    max_k: int
    # Backward geometry (independent of the forward chunk schedule). One segment per logical attention context:
    # (q_lo, q_hi, pos_lo, keys): query rows [q_lo, q_hi) sit at path positions [pos_lo, pos_lo + q_hi - q_lo) and
    # attend to keys at path positions 0..pos (causal), whose packed rows are `keys`: ("range", s) = rows s + arange,
    # or a LongTensor of packed rows (tree paths). row_node/row_pos/anc describe the same visibility per packed row
    # for mask-based kernels: key row visible from query row iff anc[node[q], node[k]] and pos[k] <= pos[q].
    segs: list = dataclasses.field(default_factory=list)
    row_node: torch.Tensor | None = None
    row_pos: torch.Tensor | None = None
    anc: torch.Tensor | None = None

    @staticmethod
    def build_tree(tb: Any, block_size: int, device: torch.device) -> "AttnPlan":
        """Tree plan from tree.TreeBatch: every trie node is one kernel sequence (query = its rows, keys = its path).
        A page that lies fully inside the ancestor positions is SHARED with the parent table (a prefix-cache hit); a
        page that overlaps the node gets its own copy of the ancestor tail rows plus the node rows (exact copies)."""
        import numpy as np

        nodes = tb.nodes
        tables: list[list[int]] = []
        paths: list = []
        src: list[int] = []
        cu, used = [0], []
        segs = []
        page = 0
        max_pages = 0
        for j, (par, lo, hi, pos_lo) in enumerate(nodes):
            own = np.arange(lo, hi)
            path = own if par < 0 else np.concatenate([paths[par], own])
            assert len(path) == pos_lo + (hi - lo)
            paths.append(path)
            end = len(path)
            npg = (end + block_size - 1) // block_size
            tab = []
            for b in range(npg):
                if par >= 0 and (b + 1) * block_size <= pos_lo:
                    tab.append(tables[par][b])  # shared prefix page
                else:
                    tab.append(page)
                    blk = path[b * block_size : (b + 1) * block_size].tolist()
                    src.extend(blk + [-1] * (block_size - len(blk)))
                    page += 1
            tables.append(tab)
            max_pages = max(max_pages, npg)
            cu.append(cu[-1] + hi - lo)
            used.append(end)
            contiguous = bool((np.diff(path) == 1).all()) if len(path) > 1 else True
            keys = ("range", int(path[0])) if contiguous else torch.tensor(path, dtype=torch.int64, device=device)
            segs.append((lo, hi, pos_lo, keys))
        bt = torch.zeros(len(tables), max_pages, dtype=torch.int32)
        for i, r in enumerate(tables):
            bt[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
        n = len(nodes)
        anc = torch.zeros(n, n, dtype=torch.bool)
        for j, node in enumerate(nodes):
            anc[j, j] = True
            if node[0] >= 0:
                anc[j] |= anc[node[0]]
        row_node = torch.empty(len(tb.ids), dtype=torch.int32)
        for j, node in enumerate(nodes):
            row_node[node[1]:node[2]] = j
        return AttnPlan(
            seq_bounds=[], block_size=block_size,
            cu_q=torch.tensor(cu, dtype=torch.int32, device=device),
            seqused_k=torch.tensor(used, dtype=torch.int32, device=device),
            block_table=bt.to(device), page_src=torch.tensor(src, dtype=torch.int64, device=device),
            num_pages=page, max_q=max(nd[2] - nd[1] for nd in nodes), max_k=max(used),
            segs=segs, row_node=row_node.to(device), row_pos=torch.tensor(tb.row_pos, dtype=torch.int32, device=device),
            anc=anc.to(device),
        )

    @staticmethod
    def build(seq_bounds: Sequence[tuple[int, int]], chunks: Sequence[Sequence[tuple[int, int]]], block_size: int,
              device: torch.device) -> "AttnPlan":
        cu = [0]
        used: list[int] = []
        rows: list[list[int]] = []
        src: list[int] = []
        page = 0
        max_pages = 0
        max_q = 1
        for (s, e), ch in zip(seq_bounds, chunks):
            L = e - s
            npg = (L + block_size - 1) // block_size
            pages = list(range(page, page + npg))
            page += npg
            src.extend(list(range(s, e)) + [-1] * (npg * block_size - L))
            max_pages = max(max_pages, npg)
            expect = 0
            for lo, hi in ch:
                if lo != expect or hi <= lo or hi > L:
                    raise ValueError(f"chunks must partition [0,{L}) in order; got {list(ch)}")
                expect = hi
                cu.append(cu[-1] + hi - lo)
                used.append(hi)
                rows.append(pages)
                max_q = max(max_q, hi - lo)
            if expect != L:
                raise ValueError(f"chunks cover [0,{expect}) but sequence length is {L}")
        bt = torch.zeros(len(rows), max_pages, dtype=torch.int32)
        for i, r in enumerate(rows):
            bt[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
        T = seq_bounds[-1][1] if seq_bounds else 0
        row_node = torch.empty(T, dtype=torch.int32)
        row_pos = torch.empty(T, dtype=torch.int32)
        for j, (s0, e0) in enumerate(seq_bounds):
            row_node[s0:e0] = j
            row_pos[s0:e0] = torch.arange(e0 - s0, dtype=torch.int32)
        return AttnPlan(
            seq_bounds=list(seq_bounds), block_size=block_size,
            cu_q=torch.tensor(cu, dtype=torch.int32, device=device),
            seqused_k=torch.tensor(used, dtype=torch.int32, device=device),
            block_table=bt.to(device), page_src=torch.tensor(src, dtype=torch.int64, device=device),
            num_pages=page, max_q=max_q, max_k=max(used) if used else 0,
            segs=[(s0, e0, 0, ("range", s0)) for s0, e0 in seq_bounds],
            row_node=row_node.to(device), row_pos=row_pos.to(device),
            anc=torch.eye(len(seq_bounds), dtype=torch.bool, device=device),
        )


def schedule_prompt_then_decode(length: int, prompt_len: int) -> list[tuple[int, int]]:
    """Engine schedule of a single-turn rollout: one prefill chunk for the prompt, then 1-token decode steps."""
    if not 0 < prompt_len <= length:
        raise ValueError(f"prompt_len {prompt_len} not in (0, {length}]")
    return [(0, prompt_len)] + [(p, p + 1) for p in range(prompt_len, length)]


def visible_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttnPlan, scale: float,
                      window: int | None) -> torch.Tensor:
    with timed("fwd_attention"):
        return _visible_attention(q, k, v, plan, scale, window)


def _learner_seg7_cfg_pre(d: int) -> tuple | None:
    """Learner-only seg7 prefill tile config for head size d (TII_ATTN=seg7). seg7's BLOCK_M / HEAD_SPLIT / warps /
    stages are free parameters (layout-independent per-row arithmetic: the bytes do not depend on them;
    tests/test_engine_segattn_v7.py, plus the full-model theta0 compare with this override against an engine dump made
    with the engine configs). The engine configs are tuned for decode and 2048-token prefill chunks; the learner
    whole-tree prefill (up to 20k rows) is 20 % faster at hd 512 with (32, 1, 8, 2) (tune_seg7_learner.py).
    Env TII_LEARNER_SEG7_PRE_<d>="bm,split,warps[,stages...]" overrides; "" = the engine config. Only the trainer
    process uses it (the rollout server never reads it)."""
    if _os.environ.get("TII_ATTN", "2d") != "seg7":
        return None
    env = _os.environ.get(f"TII_LEARNER_SEG7_PRE_{d}", LEARNER_SEG7_PRE.get(d, ""))
    return tuple(int(x) for x in env.split(",")) if env else None


LEARNER_SEG7_PRE = {512: "32,1,8,2"}


def _visible_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttnPlan, scale: float,
                       window: int | None) -> torch.Tensor:
    """q [T,Hq,D], k/v [T,Hkv,D] (bf16, packed) -> out [T,Hq,D] via unified_attention, as TritonAttentionImpl.forward calls it."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    T, hkv, d = k.shape
    bs = plan.block_size
    CALLS["unified_attention"] += 1
    # Paged copy of K/V (= reshape_and_cache_flash: an exact bf16 copy into pages); padding slots are zero.
    valid = plan.page_src >= 0
    kp = torch.zeros(plan.num_pages * bs, hkv, d, dtype=k.dtype, device=k.device)
    vp = torch.zeros_like(kp)
    kp[valid] = k[plan.page_src[valid]]
    vp[valid] = v[plan.page_src[valid]]
    out = torch.empty_like(q)
    nchunk = plan.seqused_k.numel()
    one = torch.ones(nchunk, hkv, dtype=torch.float32, device=q.device)  # layer._k_scale/_v_scale (1.0, unused for bf16)
    unified_attention(
        q=q, k=kp.view(plan.num_pages, bs, hkv, d), v=vp.view(plan.num_pages, bs, hkv, d), out=out,
        cu_seqlens_q=plan.cu_q, max_seqlen_q=plan.max_q, seqused_k=plan.seqused_k, max_seqlen_k=plan.max_k,
        softmax_scale=scale, causal=True, window_size=(window - 1, 0) if window else (-1, -1),
        block_table=plan.block_table, softcap=0, q_descale=None, k_descale=one, v_descale=one,
        **({"cfg_pre": cp} if (cp := _learner_seg7_cfg_pre(d)) is not None else {}),
    )
    return out


def plan_segs(plan_or_bounds: Any) -> list:
    """Backward segments of an AttnPlan, or of a plain [(start, end)] list (independent contiguous sequences)."""
    if isinstance(plan_or_bounds, AttnPlan):
        return plan_or_bounds.segs
    return [(s0, e0, 0, ("range", s0)) for s0, e0 in plan_or_bounds]


def seg_keys(keys: Any, lo: int, hi: int) -> Any:
    """Packed rows of path positions [lo, hi) of a segment: a slice for contiguous paths, else an index tensor."""
    if isinstance(keys, tuple):
        return slice(keys[1] + lo, keys[1] + hi)
    return keys[lo:hi]


def reference_attention_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gout: torch.Tensor,
                              plan_or_bounds: Any, scale: float, window: int | None,
                              chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 recompute VJP of causal (optionally sliding-window) GQA softmax attention, chunked over queries, for
    contiguous sequences or tree segments (the queries of a node attend to the rows of their root path).

    Uses explicit 3-D torch.bmm (no broadcasting matmul): in an aligned process aten::matmul may be overridden by
    the batch-invariant Triton kernels, whose 4-D path does not broadcast.
    """
    T, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    dq = torch.zeros(T, hq, d, dtype=torch.float32, device=q.device)
    dk = torch.zeros(T, hkv, d, dtype=torch.float32, device=q.device)
    dv = torch.zeros_like(dk)
    for q_lo, q_hi, pos_lo, keys in plan_segs(plan_or_bounds):
        for a in range(q_lo, q_hi, chunk):
            b = min(a + chunk, q_hi)
            pa, pb = pos_lo + a - q_lo, pos_lo + b - q_lo  # path positions of this query chunk
            klo = 0 if window is None else max(0, pa - window + 1)
            C, K = b - a, pb - klo
            kr = seg_keys(keys, klo, pb)
            qh = q[a:b].float().permute(1, 0, 2).reshape(hkv * g, C, d)
            go = gout[a:b].float().permute(1, 0, 2).reshape(hkv * g, C, d)
            kh = k[kr].float().permute(1, 0, 2).repeat_interleave(g, dim=0)  # [hkv*g, K, d]
            vh = v[kr].float().permute(1, 0, 2).repeat_interleave(g, dim=0)
            S = torch.bmm(qh, kh.transpose(1, 2).contiguous()) * scale  # [hkv*g, C, K]
            qpos = torch.arange(pa, pb, device=q.device)[:, None]
            kpos = torch.arange(klo, pb, device=q.device)[None, :]
            mask = kpos <= qpos
            if window is not None:
                mask &= (qpos - kpos) < window
            S = S.masked_fill(~mask, float("-inf"))
            P = torch.exp(S - S.amax(-1, keepdim=True))
            P = P / P.sum(-1, keepdim=True)
            O = torch.bmm(P, vh)
            dP = torch.bmm(go, vh.transpose(1, 2).contiguous())
            dS = P * (dP - (go * O).sum(-1, keepdim=True)) * scale
            dq[a:b] = torch.bmm(dS, kh).view(hq, C, d).permute(1, 0, 2)
            gk = torch.bmm(dS.transpose(1, 2).contiguous(), qh).view(hkv, g, K, d).sum(1).permute(1, 0, 2)
            gv = torch.bmm(P.transpose(1, 2).contiguous(), go).view(hkv, g, K, d).sum(1).permute(1, 0, 2)
            if isinstance(kr, slice):
                dk[kr] += gk
                dv[kr] += gv
            else:  # indices are unique within one path, so index_add_ is deterministic here
                dk.index_add_(0, kr, gk)
                dv.index_add_(0, kr, gv)
    return dq, dk, dv


class AttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttnPlan, scale: float, window: int | None):
        out = visible_attention(q, k, v, plan, scale, window)
        # fast backward: the forward output O is kept for D = rowsum(dO * O) (as FlashAttention-2 does), which
        # removes the O = P V recompute (one of six GEMMs on full layers, one of nine on sliding layers)
        ctx.has_o = FAST_BWD and _os.environ.get("TII_ATTN_BWD") != "flex" and _os.environ.get("TII_ATTN_RECOMPUTE_O") != "1"
        if ctx.has_o:
            ctx.save_for_backward(q, k, v, out)
        else:
            ctx.save_for_backward(q, k, v)
        ctx.plan, ctx.scale, ctx.window = plan, scale, window
        return out

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_attention"):
            return AttentionFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, gout: torch.Tensor) -> tuple[Any, ...]:
        if ctx.has_o:
            q, k, v, o = ctx.saved_tensors
        else:
            (q, k, v), o = ctx.saved_tensors, None
        if FAST_BWD and ctx.window is not None and _os.environ.get("TII_ATTN_BWD") == "flex":
            # optional alternative for sliding layers: FlexAttention recompute + backward
            from v2.tii.trainer.attn_bwd_flex import flex_attention_grads

            with timed("bwd_attn_sliding"):
                r = flex_attention_grads(q, k, v, gout.contiguous(), ctx.plan, ctx.scale, ctx.window)
            return (*r, None, None, None)
        if FAST_BWD:  # all layers: cuBLAS bf16 GEMMs + fused Triton softmax backward (windowed keys on sliding layers)
            from v2.tii.trainer.attn_bwd_chunked import chunked_full_attention_grads

            if ctx.window is None:
                with timed("bwd_attn_full"):
                    r = chunked_full_attention_grads(q, k, v, gout.contiguous(), ctx.plan, ctx.scale, o=o,
                                                      chunk=int(_os.environ.get("TII_ATTN_CHUNK", "1024")))
            else:
                from v2.tii.trainer.attn_bwd_chunked import banded_window_attention_grads_batched

                with timed("bwd_attn_sliding"):
                    r = banded_window_attention_grads_batched(q, k, v, gout.contiguous(), ctx.plan, ctx.scale, ctx.window,
                                                              o=o)
            return (*r, None, None, None)
        dq, dk, dv = reference_attention_grads(q, k, v, gout, ctx.plan, ctx.scale, ctx.window)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttnPlan, scale: float, window: int | None) -> torch.Tensor:
    if ORACLE is not None and torch.is_grad_enabled():
        return ORACLE.attention(q, k, v, plan, scale, window)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return AttentionFn.apply(q, k, v, plan, scale, window)
    return visible_attention(q, k, v, plan, scale, window)


# --------------------------------------------------------------------------------------------------------------
# Scorer: lm_head + final softcap + sampler processed logprobs
# --------------------------------------------------------------------------------------------------------------
def visible_lm_head(h: torch.Tensor, w: torch.Tensor, backend: str) -> torch.Tensor:
    CALLS[f"lm_head_{backend}"] += 1
    if backend == "bi_persistent":
        from vllm.model_executor.layers.batch_invariant import linear_batch_invariant

        return linear_batch_invariant(h, w)
    if backend == "torch_linear":
        return F.linear(h, w)  # UnquantizedEmbeddingMethod.apply -> default_unquantized_gemm
    raise ValueError(f"unknown lm_head backend {backend!r}")


def visible_logits(h: torch.Tensor, w: torch.Tensor, softcap: float | None, backend: str) -> torch.Tensor:
    """LogitsProcessor.forward: bf16 lm_head output, then soft cap as three bf16 ops."""
    logits = visible_lm_head(h, w, backend)
    if softcap is not None:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap
    return logits


def visible_token_logprobs(h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, temps: torch.Tensor,
                           softcap: float | None, backend: str, logprobs_mode: str,
                           forced: torch.Tensor | None = None, ent_out: list | None = None) -> torch.Tensor:
    """Sampler logprob of `targets` for rows of `h` (one row = one engine request's sampling position).
    forced: bool per row - the engine's ThinkingTokenBudgetLogitsProcessor set logits[row, target] = 1e9 there
    (logits_processors.py); applied after the fp32 cast and before temperature, as Sampler.forward does."""
    logits = visible_logits(h, w, softcap, backend)
    CALLS[f"sampler_{logprobs_mode}"] += 1
    if logprobs_mode == "raw_logprobs":
        lp = logits.log_softmax(dim=-1, dtype=torch.float32)  # Sampler.compute_logprobs on the bf16 logits
    elif logprobs_mode == "processed_logprobs":
        logits = logits.to(torch.float32)  # Sampler.forward
        if forced is not None and bool(forced.any()):  # Sampler.apply_logits_processors (thinking budget)
            rows = forced.nonzero().view(-1)
            logits[rows, targets[rows]] = 1e9
        logits.div_(temps.unsqueeze(dim=1))  # Sampler.apply_temperature (all_random: temp used as is)
        lp = logits.log_softmax(dim=-1, dtype=torch.float32)  # TopKTopPSampler.forward_native, no top-k/top-p
        if ent_out is not None:  # fused entropy diagnostic of the same processed distribution (no second lm_head)
            with torch.no_grad():
                ent_out.append(-(lp.exp() * lp).sum(-1))
    else:
        raise ValueError(f"unsupported logprobs_mode {logprobs_mode!r}")
    return lp.gather(-1, targets.view(-1, 1)).view(-1)


SCORER_BWD_CHUNK = 1024


class ScorerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, temps: torch.Tensor,
                softcap: float | None, backend: str, logprobs_mode: str, chunk: int,
                forced: torch.Tensor | None = None, ent_out: list | None = None) -> torch.Tensor:
        outs = [visible_token_logprobs(h[i : i + chunk], w, targets[i : i + chunk], temps[i : i + chunk], softcap, backend,
                                       logprobs_mode, None if forced is None else forced[i : i + chunk], ent_out)
                for i in range(0, h.shape[0], chunk)]
        # a forced row's processed logprob is 1e9/T - lse(..) = 0 exactly, independent of h: zero gradient
        ctx.forced = forced if (forced is not None and logprobs_mode == "processed_logprobs") else None
        ctx.save_for_backward(h, w, targets, temps)
        ctx.softcap, ctx.backend, ctx.mode, ctx.chunk = softcap, backend, logprobs_mode, chunk
        return torch.cat(outs) if outs else h.new_empty(0, dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, *g: torch.Tensor) -> tuple[Any, ...]:
        with timed("bwd_scorer"):
            return ScorerFn._bw(ctx, *g)

    @staticmethod
    def _bw(ctx: Any, glp: torch.Tensor) -> tuple[Any, ...]:
        h, w, targets, temps = ctx.saved_tensors
        if ctx.forced is not None:
            glp = glp.masked_fill(ctx.forced, 0.0)
        dh = torch.empty_like(h)
        if FAST_BWD:  # fused dz (one pass for the softmax stats, one for dz as bf16 hi/lo) + two bf16 GEMMs
            from v2.tii.trainer.bwd_kernels import scorer_dh

            ones = torch.ones_like(temps)
            # backward chunk >= SCORER_BWD_CHUNK rows: the bf16 head output is recomputed with the same batch-invariant
            # kernel (row values do not depend on the chunk), and the dz GEMM gets enough output tiles
            ch = max(ctx.chunk, SCORER_BWD_CHUNK) if ctx.backend == "bi_persistent" else ctx.chunk
            for i in range(0, h.shape[0], ch):
                raw = visible_lm_head(h[i : i + ch], w, ctx.backend)  # recompute the bf16 head output
                tt = temps[i : i + ch] if ctx.mode == "processed_logprobs" else ones[i : i + ch]
                dh[i : i + ch] = scorer_dh(raw, targets[i : i + ch], glp[i : i + ch], tt, ctx.softcap, w).to(h.dtype)
            if ctx.needs_input_grad[1]:
                raise RuntimeError("lm_head/embedding is frozen under the gradient contract")
            return dh, None, None, None, None, None, None, None, None, None
        for i in range(0, h.shape[0], ctx.chunk):
            hc = h[i : i + ctx.chunk]
            raw = visible_lm_head(hc, w, ctx.backend).float()  # recompute the bf16 head output
            if ctx.softcap is not None:
                tz = torch.tanh(raw / ctx.softcap)
                z = tz * ctx.softcap
            else:
                tz, z = None, raw
            t = temps[i : i + ctx.chunk].unsqueeze(1) if ctx.mode == "processed_logprobs" else 1.0
            p = torch.softmax(z / t, dim=-1)
            g = glp[i : i + ctx.chunk].float().unsqueeze(1)
            dz = -p * g
            dz.scatter_add_(1, targets[i : i + ctx.chunk].view(-1, 1), g)
            dz = dz / t
            if tz is not None:
                dz = dz * (1.0 - tz * tz)
            # dH = dZ W in fp32 (vocab chunks of W upcast on the fly): no bf16 rounding of dZ before the GEMM
            if FAST_BWD:
                from v2.tii.trainer.bwd_kernels import mm_f32acc

                acc = mm_f32acc(dz, w)  # hi/lo bf16 split: ~fp32-accurate dZ W on tensor cores
            else:
                acc = torch.zeros(hc.shape[0], w.shape[1], dtype=torch.float32, device=h.device)
                for v0 in range(0, w.shape[0], 32768):
                    acc += torch.mm(dz[:, v0 : v0 + 32768], w[v0 : v0 + 32768].float())
            dh[i : i + ctx.chunk] = acc.to(h.dtype)
        if ctx.needs_input_grad[1]:
            raise RuntimeError("lm_head/embedding is frozen under the gradient contract")
        return dh, None, None, None, None, None, None, None, None, None


def token_logprobs(h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, temps: torch.Tensor, softcap: float | None,
                   backend: str, logprobs_mode: str, chunk: int = 256, forced: torch.Tensor | None = None,
                   ent_out: list | None = None) -> torch.Tensor:
    """ent_out: if a list, the per-row entropy of the processed distribution is appended chunk by chunk (fused with
    the scorer forward; the oracle path does not fill it)."""
    if forced is not None and not bool(forced.any()):
        forced = None
    if ORACLE is not None and torch.is_grad_enabled():
        if forced is None:
            return ORACLE.token_logprobs(h, w, targets, temps, softcap, backend, logprobs_mode, chunk)
        return ORACLE.token_logprobs(h, w, targets, temps, softcap, backend, logprobs_mode, chunk, forced=forced)
    if torch.is_grad_enabled() and h.requires_grad:
        return ScorerFn.apply(h, w, targets, temps, softcap, backend, logprobs_mode, chunk, forced, ent_out)
    outs = [visible_token_logprobs(h[i : i + chunk], w, targets[i : i + chunk], temps[i : i + chunk], softcap, backend,
                                   logprobs_mode, None if forced is None else forced[i : i + chunk], ent_out)
            for i in range(0, h.shape[0], chunk)]
    return torch.cat(outs) if outs else h.new_empty(0, dtype=torch.float32)
