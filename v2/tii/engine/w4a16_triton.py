"""Batch-invariant W4A16 GEMM (Triton) shared by the aligned rollout engine and the trainer.

Author: Claude Opus 5.5 (engine track), 2026-11-05.

Origin: the kernel body is ported from vLLM main
`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` (Apache-2.0, vLLM contributors), which
vLLM 0.19.1 does not ship. Changes for the TII contract:
  * no split-K, ever; K is walked in fixed BLOCK_K = group_size = 32 steps from k=0 upward with one fp32
    accumulator per output element (`acc += dot(a, dequant(b))`), so the reduction order of every output element
    depends only on K -- never on M, on the batch, on the tile, or on the SM count;
  * the tile table (BLOCK_M, BLOCK_N, num_warps, num_stages) is an explicit, versioned function of M only
    (`tile_config`); it affects which CTA computes an element, not its reduction order. Its M-invariance is
    proven by executed tests (v2/tii/tests/test_engine_w4a16_invariance.py), not assumed;
  * dequant is `(q - 8).to(bf16) * scale_bf16` -> bf16, i.e. bf16(exact((q-8)*s)), identical bytes to
    v2.train.int4_qat_lora.unpack_ct_int4_nibbles_torch (checked in the same test);
  * registered as `torch.ops.tii.w4a16_gemm` so vLLM CUDA-graph capture and the trainer call one op.

Layout: activations a [M, K] bf16 contiguous; b_q [K, N//8] int32 with N packed (8 consecutive output columns
per int32, nibble j at bits 4j); scales [K//G, N] bf16. `repack_ct_weight` converts the compressed-tensors
checkpoint layout ([N, K//8] int32 with K packed, scales [N, K//G]) into this layout, byte-preserving per nibble.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

TII_W4A16_VERSION = "tii-w4a16-v1"
GROUP_SIZE = 32
ZP_BIAS = 8


@triton.jit
def _tii_w4a16_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    shifts_row = tl.arange(0, 8) * 4
    shifts_1d = tl.reshape(tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8)), (BLOCK_N,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        b_packed = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=mask_k[:, None] & (offs_bn[None, :] < N // 8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF
        group_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(scales_ptr + group_idx * N + offs_sn, mask=offs_sn < N, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))
        b_fp = (b - z).to(a.dtype) * scales
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def tile_config(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) as a function of the problem shape.

    BLOCK_K is not here: it is always GROUP_SIZE (32). Nothing in this table changes the per-element K order.
    """
    if M <= 16:
        return 16, 32, 4, 4
    if M <= 32:
        return 32, 32, 4, 4
    if M <= 64:
        return 64, 64, 4, 3
    return 128, 128, 8, 3


def tii_w4a16_gemm_impl(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    assert a.dtype == torch.bfloat16 and scales.dtype == torch.bfloat16 and b_q.dtype == torch.int32
    a = a.contiguous()
    M, K = a.shape
    N = b_q.shape[1] * 8
    assert b_q.shape == (K, N // 8) and b_q.is_contiguous(), (tuple(b_q.shape), K, N)
    assert scales.shape == (K // GROUP_SIZE, N) and scales.is_contiguous(), tuple(scales.shape)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0:
        return c
    bm, bn, nw, ns = tile_config(M, N, K)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _tii_w4a16_kernel[grid](
        a,
        b_q,
        scales,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE,
        ZP_BIAS=ZP_BIAS,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=GROUP_SIZE,
        num_warps=nw,
        num_stages=ns,
    )
    return c


# ---------------------------------------------------------------------------------------------------------------
# v2: fixed split-K. The split count S = split_k(N, K) depends only on the weight shape, never on M. Split s covers
# K rows [s*K/S, (s+1)*K/S); its partial p_s is the same fp32 acc chain as v1 restricted to that range; the result
# is bf16(((p_0 + p_1) + p_2) + ...). Two executions give identical bits by construction (verified by
# w4a16_split_equivalence_exp.py and tests/test_engine_w4a16_invariance.py --version v2):
#   inline : one CTA computes p_0..p_{S-1} in order and accumulates them (large M);
#   partial: one CTA per (tile, s) writes p_s to an fp32 workspace; _tii_reduce_kernel sums s = 0..S-1 in order
#            (small M, where v1 has too few CTAs to stream the weights).
# BLOCK_K (a multiple of 32 dividing K/S), BLOCK_M, BLOCK_N, warps and stages do not change the bits either.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _split_acc(a_ptr, b_ptr, s_ptr, offs_m, offs_n, offs_bn, shifts, sidx, M, N, k_per_split, sam, sak, sbk, sbn,
               ZP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(k_per_split // BK):
        offs_k = sidx * k_per_split + kk * BK + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak, mask=offs_m[:, None] < M, other=0.0)
        bp = tl.load(b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn, mask=offs_bn[None, :] < N // 8, other=0)
        b = tl.interleave(bp, bp)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF
        g = offs_k // G
        sc = tl.load(s_ptr + g[:, None] * N + offs_n[None, :], mask=offs_n[None, :] < N, other=1.0)
        bf = (b - ZP).to(a.dtype) * sc
        acc += tl.dot(a, bf, out_dtype=tl.float32)
    return acc


@triton.jit
def _tii_w4a16_v2_kernel(a_ptr, b_ptr, s_ptr, c_ptr, ws_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                         ZP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
                         SPLIT: tl.constexpr, PARTIAL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_bn = pid_n * (BN // 8) + tl.arange(0, BN // 8)
    shifts_1d = tl.reshape(tl.broadcast_to((tl.arange(0, 8) * 4)[None, :], (BN // 8, 8)), (BN,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BK, BN))
    kps = K // SPLIT
    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if PARTIAL:
        acc = _split_acc(a_ptr, b_ptr, s_ptr, offs_m, offs_n, offs_bn, shifts, pid_s, M, N, kps, sam, sak, sbk, sbn,
                         ZP, BM, BN, BK, G)
        tl.store(ws_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=omask)
    else:
        total = _split_acc(a_ptr, b_ptr, s_ptr, offs_m, offs_n, offs_bn, shifts, 0, M, N, kps, sam, sak, sbk, sbn,
                           ZP, BM, BN, BK, G)
        for s in tl.static_range(1, SPLIT):
            total = total + _split_acc(a_ptr, b_ptr, s_ptr, offs_m, offs_n, offs_bn, shifts, s, M, N, kps, sam, sak,
                                       sbk, sbn, ZP, BM, BN, BK, G)
        tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, total.to(c_ptr.type.element_ty), mask=omask)


@triton.jit
def _tii_reduce_kernel(ws_ptr, c_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < MN
    r = tl.load(ws_ptr + offs, mask=m, other=0.0)
    for s in tl.static_range(1, SPLIT):
        r = r + tl.load(ws_ptr + s * MN + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, r.to(c_ptr.type.element_ty), mask=m)


# Numerics-defining table for v2 (versioned with TII_W4A16_VERSION). Keys: (N, K) of every E4B linear as vLLM
# builds it (merged qkv / gate_up). Unlisted shapes use S = 1.
SPLIT_K_V2: dict[tuple[int, int], int] = {
    (3072, 2560): 4,    # qkv sliding
    (6144, 2560): 2,    # qkv full
    (2048, 2560): 4,    # q on KV-shared layers is still the merged qkv (3072/6144); kept for direct q calls
    (2560, 2048): 8,    # o sliding
    (2560, 4096): 8,    # o full
    (20480, 2560): 1,   # gate_up
    (2560, 10240): 8,   # down
    (256, 2560): 16,    # PLE per-layer gate
    (2560, 256): 2,     # PLE per-layer projection
    (10752, 2560): 2,   # per_layer_model_projection (real model)
    (1536, 2560): 4,    # per_layer_model_projection (C2 proxy)
}


def split_k(N: int, K: int) -> int:
    s = SPLIT_K_V2.get((N, K), 1)
    assert (K // GROUP_SIZE) % s == 0, (N, K, s)
    return s


def tile_config_v2(M: int, N: int, K: int) -> tuple[bool, int, int, int, int, int]:
    """(partial, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages); performance only, never numerics."""
    S = split_k(N, K)
    kps = K // S
    bk = 128 if kps % 128 == 0 else (64 if kps % 64 == 0 else 32)
    if M <= 16:
        return S > 1, 16, 32, bk, 4, 4
    if M <= 64:
        return S > 1, triton.next_power_of_2(M), 64, bk, 4, 3
    return False, 128, 128, 32, 8, 3


def tii_w4a16_gemm_v2_impl(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor,
                           cfg: tuple[bool, int, int, int, int, int] | None = None,
                           split: int | None = None) -> torch.Tensor:
    assert a.dtype == torch.bfloat16 and scales.dtype == torch.bfloat16 and b_q.dtype == torch.int32
    a = a.contiguous()
    M, K = a.shape
    N = b_q.shape[1] * 8
    assert b_q.shape == (K, N // 8) and b_q.is_contiguous() and scales.shape == (K // GROUP_SIZE, N)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0:
        return c
    S = split or split_k(N, K)  # explicit split is for tuning only; production calls use the table
    partial, bm, bn, bk, nw, ns = cfg or tile_config_v2(M, N, K)
    assert (K // S) % bk == 0 and bk % GROUP_SIZE == 0
    common = dict(ZP=ZP_BIAS, BM=bm, BN=bn, BK=bk, G=GROUP_SIZE, SPLIT=S, num_warps=nw, num_stages=ns)
    if partial and S > 1:
        ws = torch.empty((S, M, N), dtype=torch.float32, device=a.device)
        _tii_w4a16_v2_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn), S)](
            a, b_q, scales, c, ws, M, N, K, a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1), c.stride(0),
            c.stride(1), PARTIAL=True, **common)
        _tii_reduce_kernel[(triton.cdiv(M * N, 1024),)](ws, c, M * N, SPLIT=S, BLOCK=1024)
    else:
        _tii_w4a16_v2_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn), 1)](
            a, b_q, scales, c, c, M, N, K, a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1), c.stride(0),
            c.stride(1), PARTIAL=False, **common)
    return c


ACTIVE_VERSION = os.environ.get("TII_W4A16_VERSION", TII_W4A16_VERSION)
assert ACTIVE_VERSION in ("tii-w4a16-v1", "tii-w4a16-v2"), ACTIVE_VERSION


@torch.library.custom_op("tii::w4a16_gemm", mutates_args=())
def w4a16_gemm(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The op both sides call. Numerics are fixed by ACTIVE_VERSION (env TII_W4A16_VERSION at import)."""
    if ACTIVE_VERSION == "tii-w4a16-v2":
        return tii_w4a16_gemm_v2_impl(a, b_q, scales)
    return tii_w4a16_gemm_impl(a, b_q, scales)


@w4a16_gemm.register_fake
def _w4a16_gemm_fake(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], b_q.shape[1] * 8))


def repack_ct_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """compressed-tensors [N, K//8] int32 (K packed) -> kernel [K, N//8] int32 (N packed). Nibble-exact."""
    n, k8 = weight_packed.shape
    k = k8 * 8
    assert n % 8 == 0
    shifts = torch.arange(8, device=weight_packed.device, dtype=torch.int32) * 4
    w = ((weight_packed.unsqueeze(-1) >> shifts) & 0xF).reshape(n, k)
    w_kn = w.t().contiguous()
    return torch.sum((w_kn.view(k, n // 8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32).contiguous()


def repack_ct_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """compressed-tensors [N, K//G] -> kernel [K//G, N]."""
    return weight_scale.t().contiguous()


def dequant_kn(b_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference dequant of the kernel layout -> W^T [K, N] bf16 (bf16(exact((q-8)*s)))."""
    k, n8 = b_q.shape
    shifts = torch.arange(8, device=b_q.device, dtype=torch.int32) * 4
    q = ((b_q.unsqueeze(-1) >> shifts) & 0xF).reshape(k, n8 * 8)
    s = scales.float().repeat_interleave(GROUP_SIZE, dim=0)
    return ((q.float() - ZP_BIAS) * s).to(torch.bfloat16)
