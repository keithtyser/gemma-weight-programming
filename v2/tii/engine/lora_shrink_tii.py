"""TII LoRA shrink with a FIXED split-K and an ordered reduction (batch-invariant and parallel).

Author: Claude Opus 5.5 (engine track), 2026-11-06. Derived from vLLM 0.19.1 lora/ops/triton_ops/lora_shrink_op.py
and kernel_utils.do_shrink_kernel (Apache-2.0; Punica, Chen et al. 2023).

Why: under VLLM_BATCH_INVARIANT vLLM runs the shrink with split_k=1, so x@A^T (K=2560..10240, N=rank 16) is computed
by ~2-6 CTAs per call at decode batch sizes: on the smoke-round server config this made LoRA cost ~42 % of the
decode throughput. Stock vLLM's split_k uses tl.atomic_add (order not fixed, not invariant).
What: SPLIT_K = 16 and BLOCK_K = 32 are fixed (numerics-defining): split s accumulates the 32-wide K tiles
s, s+16, s+32, ... (vLLM's own mm_k interleave) into its own fp32 partial, stored without atomics in
part[s, slice, token, r]; _tii_shrink_reduce then writes out = ((p0 + p1) + p2) + ... + p15 in fixed order. Every
output element therefore has one reduction order for every batch size, batch composition and BLOCK_M/BLOCK_N/warps
choice. Rows of tokens without a LoRA stay 0 (partials zeroed). Same signature as vLLM's _lora_shrink, installed by
vllm_plugin (P3b, TII_LORA_SHRINK=split) as punica_gpu.lora_shrink; the trainer calls tii_lora_shrink too.
"""

from __future__ import annotations

import os

import torch

from vllm.lora.ops.triton_ops.kernel_utils import mm_k
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr
from vllm.triton_utils import tl, triton

TII_SHRINK_SPLIT_K = 16
TII_SHRINK_BLOCK_K = 32
VERSION = f"tii-lora-shrink-s{TII_SHRINK_SPLIT_K}k{TII_SHRINK_BLOCK_K}"  # numerics; fused/two-pass are identical


@triton.jit
def _tii_shrink_kernel(input_ptr, lora_ptr, part_ptr, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                       lora_token_start_loc, lora_ids, scaling, input_d0_stride, input_d1_stride, lora_d0_stride,
                       lora_d1_stride, lora_d2_stride, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(axis=0)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    pid_m = pid_m_n // cta_n_num
    pid_n = pid_m_n % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset + offset_m)
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_id + rbn[None, :] * lora_d1_stride
             + offset_k[:, None] * lora_d2_stride)
    acc = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k, K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
               False, cur_lora_ptr.dtype.element_ty, False, base_k=pid_sk * BLOCK_K)
    acc *= scaling
    offset_cm = tl.arange(0, BLOCK_M)
    c_mask = (offset_cm[:, None] < cta_m_len) & (offset_n[None, :] < N)
    p = part_ptr + ((pid_sk * SLICE_NUM + slice_id) * M + ram[:, None]) * N + offset_n[None, :]
    tl.store(p, acc, mask=c_mask)


@triton.jit
def _tii_shrink_fused_kernel(input_ptr, lora_ptr, part_ptr, out_ptr, cnt_ptr, M, N, K, token_indices_sorted_by_lora_ids,
                             num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling, input_d0_stride,
                             input_d1_stride, lora_d0_stride, lora_d1_stride, lora_d2_stride, out_d0, out_d1, out_d2,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):
    """Same partial sums as _tii_shrink_kernel; the last of the SPLIT_K CTAs of a tile (atomic ticket) re-reads all
    SPLIT_K partials from memory and writes ((p0 + p1) + ...) + p15 -> bit-identical to kernel + _tii_shrink_reduce,
    without the memset and the separate reduce launch. Rows of no-LoRA tokens are never written (never read)."""
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(axis=0)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    pid_m = pid_m_n // cta_n_num
    pid_n = pid_m_n % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset + offset_m)
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_id + rbn[None, :] * lora_d1_stride
             + offset_k[:, None] * lora_d2_stride)
    acc = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k, K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
               False, cur_lora_ptr.dtype.element_ty, False, base_k=pid_sk * BLOCK_K)
    acc *= scaling
    offset_cm = tl.arange(0, BLOCK_M)
    c_mask = (offset_cm[:, None] < cta_m_len) & (offset_n[None, :] < N)
    plane = SLICE_NUM * M * N
    p = part_ptr + (slice_id * M + ram[:, None]) * N + offset_n[None, :]
    tl.store(p + pid_sk * plane, acc, mask=c_mask)
    tl.debug_barrier()
    tile = (lora_idx * SLICE_NUM + slice_id) * (cta_m_num * cta_n_num) + pid_m * cta_n_num + pid_n
    ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT_K - 1:
        r = tl.load(p, mask=c_mask, other=0.0, cache_modifier=".cg")
        for s in tl.static_range(1, SPLIT_K):
            r = r + tl.load(p + s * plane, mask=c_mask, other=0.0, cache_modifier=".cg")
        o = out_ptr + slice_id * out_d0 + ram[:, None] * out_d1 + offset_n[None, :] * out_d2
        tl.store(o, r, mask=c_mask)
        tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed", scope="gpu")


@triton.jit
def _tii_shrink_reduce(part_ptr, out_ptr, total, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    r = tl.load(part_ptr + offs, mask=m, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        r = r + tl.load(part_ptr + s * total + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, r, mask=m)


_CNT: dict = {}
_CFG_FUSED = tuple(int(x) for x in os.environ["TII_SHRINK_CFG"].split(",")) if os.environ.get("TII_SHRINK_CFG") else None


def _counters(n: int, device) -> torch.Tensor:
    """Self-resetting ticket counters (zeroed once; the last CTA of each tile resets its slot)."""
    buf = _CNT.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _CNT[device] = buf
    return buf


@torch.inference_mode()
def tii_lora_shrink_fused(inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted_by_lora_ids,
                          num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras,
                          scaling: float, zero_untouched: bool = True) -> None:
    """One launch per call; bit-identical to tii_lora_shrink (same partials, same ordered sum).
    zero_untouched=False skips zeroing the rows of tokens without an adapter (the vLLM engine path: the Punica
    expand only reads rows of tokens that have an adapter, so those rows are never read; saves one memset launch
    per LoRA call). Rows of adapter tokens are written identically either way."""
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return
    assert inputs.is_contiguous() and output_tensor.is_contiguous()
    M = inputs.size(0)
    lora_ptr_tensor, sd0, sd1, sd2 = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N, K = lora_a_weights[0].shape[-2:]
    S = len(lora_a_weights)
    part = torch.empty((TII_SHRINK_SPLIT_K, S, M, N), dtype=torch.float32, device=inputs.device)
    if zero_untouched:
        output_tensor.zero_()  # vLLM _lora_shrink semantics: rows of tokens without an adapter are 0
    BLOCK_M, BLOCK_N = 16, 16
    nw, ns = 4, 4  # 4 stages: K=10240 decode 8.8 -> 5.7 us (bench_lora_shrink_cfg; bytes identical to 2 stages)
    if _CFG_FUSED:  # TII_SHRINK_CFG=bm,warps,stages: tile/scheduling only (split-K order fixed -> same bytes)
        BLOCK_M, nw, ns = _CFG_FUSED
    nz = int(num_active_loras.item())
    cnt = _counters(nz * S * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), inputs.device)
    even_k = K % (TII_SHRINK_BLOCK_K * TII_SHRINK_SPLIT_K) == 0
    grid = (TII_SHRINK_SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), S, nz)
    _tii_shrink_fused_kernel[grid](inputs, lora_ptr_tensor, part, output_tensor, cnt, M, N, K,
                                   token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
                                   lora_ids, scaling, inputs.stride(0), inputs.stride(1), sd0, sd1, sd2,
                                   output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2),
                                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=TII_SHRINK_BLOCK_K, EVEN_K=even_k,
                                   SPLIT_K=TII_SHRINK_SPLIT_K, SLICE_NUM=S, num_warps=nw, num_stages=ns)


@torch.inference_mode()
def tii_lora_shrink_twopass(inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted_by_lora_ids,
                    num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras,
                    scaling: float) -> None:
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return
    assert inputs.dtype == lora_a_weights[0].dtype and inputs.dtype in (torch.float16, torch.bfloat16)
    assert inputs.is_contiguous() and output_tensor.is_contiguous()
    M = inputs.size(0)
    lora_ptr_tensor, sd0, sd1, sd2 = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N, K = lora_a_weights[0].shape[-2:]
    S = len(lora_a_weights)
    assert output_tensor.shape == (S, M, N), (tuple(output_tensor.shape), S, M, N)
    part = torch.zeros((TII_SHRINK_SPLIT_K, S, M, N), dtype=torch.float32, device=inputs.device)
    BLOCK_M, BLOCK_N = 16, 16
    even_k = K % (TII_SHRINK_BLOCK_K * TII_SHRINK_SPLIT_K) == 0
    grid = (TII_SHRINK_SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), S, num_active_loras.item())
    _tii_shrink_kernel[grid](inputs, lora_ptr_tensor, part, M, N, K, token_indices_sorted_by_lora_ids,
                             num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling, inputs.stride(0),
                             inputs.stride(1), sd0, sd1, sd2, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             BLOCK_K=TII_SHRINK_BLOCK_K, EVEN_K=even_k, SPLIT_K=TII_SHRINK_SPLIT_K, SLICE_NUM=S,
                             num_warps=4, num_stages=2)
    total = S * M * N
    _tii_shrink_reduce[(triton.cdiv(total, 1024),)](part, output_tensor, total, SPLIT_K=TII_SHRINK_SPLIT_K,
                                                    BLOCK=1024)


# Default entry point: the single-launch fused version (bit-identical to the two-pass reference above, verified on
# 216 cells incl. repeated calls; ~30 % faster at decode sizes).
tii_lora_shrink = tii_lora_shrink_fused


def tii_lora_shrink_engine(*args, **kwargs) -> None:
    """vLLM engine entry (P3b): tii_lora_shrink_fused without the no-adapter-row memset (see zero_untouched)."""
    tii_lora_shrink_fused(*args, zero_untouched=False, **kwargs)
