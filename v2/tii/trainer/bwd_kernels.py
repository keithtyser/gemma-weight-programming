"""Fused Triton BACKWARD kernels for the aligned trainer (forward bytes untouched; backward under the gradient contract).

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: Replace the memory-bound multi-pass fp32 torch backward of the hottest operators by single-pass kernels with
  identical math (fp32 inside, one bf16 rounding of each output), and a GEMM that dequantizes the INT4 base on the fly:
    rmsnorm_bwd(x, w, gy, eps)   dx = rstd * (g*w - xhat * mean(g*w*xhat)), rstd from fp32 sum of squares
    gelu_and_mul_bwd(x, gy)      [dgate, dup] for y = gelu_tanh(gate) * up
    int4_dx(gy, q, s)            dX = dY @ W_ref with W_ref = bf16((q-8)*s) built per tile from the compressed-tensors
                                 bytes ([N, K/8] int32 K-packed, [N, K/32] bf16 scales) - no dequantized copy in memory
    mm_f32acc(a, b)              a @ b for fp32 a and bf16 b with ~fp32 accuracy: a = hi + lo (two bf16 parts), two bf16
                                 tensor-core GEMMs with fp32 output (torch.mm out_dtype) - error ~2^-17 relative
  Callers: ops.*Fn.backward when ops.FAST_BWD is True. Validated against the fp32 reference per operator
  (tests/test_trainer_ops.py backward cells) and whole-model by C7 v3.
SRP/DRY check: Pass - backward-only kernels; the fp32 torch versions stay in ops.py as the reference path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, w_ptr, gy_ptr, dx_ptr, n_cols, eps, x_stride, g_stride, d_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(gy_ptr + row * g_stride + cols, mask=m, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    xhat = x * rstd
    gw = g * w
    dot = tl.sum(gw * xhat, axis=0) / n_cols
    dx = rstd * (gw - xhat * dot)
    tl.store(dx_ptr + row * d_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=m)


def rmsnorm_bwd(x: torch.Tensor, w: torch.Tensor, gy: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    n = x.shape[-1]
    x2 = x.reshape(-1, n).contiguous()
    g2 = gy.reshape(-1, n).contiguous()
    dx = torch.empty_like(x2) if out is None else out  # out: [rows, n] view with unit column stride
    if out is not None:
        assert out.shape == x2.shape and out.stride(1) == 1
    _rmsnorm_bwd_kernel[(x2.shape[0],)](x2, w.contiguous(), g2, dx, n, eps, x2.stride(0), g2.stride(0), dx.stride(0),
                                       BLOCK=triton.next_power_of_2(n), num_warps=8 if n > 1024 else 4)
    return dx if out is not None else dx.view(x.shape)


@triton.jit
def _gelu_mul_bwd_kernel(x_ptr, gy_ptr, dx_ptr, n_rows, d, o_stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n_rows * d
    m = offs < total
    r = offs // d
    c = offs % d
    g = tl.load(x_ptr + r * (2 * d) + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + r * (2 * d) + d + c, mask=m, other=0.0).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    k0: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
    inner = k0 * (g + 0.044715 * g * g * g)
    e = tl.exp(-2.0 * tl.abs(inner))
    th = (1.0 - e) / (1.0 + e)
    th = tl.where(inner >= 0, th, -th)  # tanh(inner), overflow-safe
    gelu = 0.5 * g * (1.0 + th)
    dgelu = 0.5 * (1.0 + th) + 0.5 * g * (1.0 - th * th) * k0 * (1.0 + 3.0 * 0.044715 * g * g)
    tl.store(dx_ptr + r * o_stride + c, (gy * u * dgelu).to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dx_ptr + r * o_stride + d + c, (gy * gelu).to(dx_ptr.dtype.element_ty), mask=m)


def gelu_and_mul_bwd(x: torch.Tensor, gy: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d).contiguous()
    g2 = gy.reshape(-1, d).contiguous()
    dx = torch.empty_like(x2) if out is None else out
    if out is not None:
        assert out.shape == x2.shape and out.stride(1) == 1
    BLOCK = 1024
    n = x2.shape[0] * d
    _gelu_mul_bwd_kernel[(triton.cdiv(n, BLOCK),)](x2, g2, dx, x2.shape[0], d, dx.stride(0), BLOCK=BLOCK, num_warps=4)
    return dx if out is not None else dx.view(x.shape)


@triton.jit
def _dequant_ct_kernel(q_ptr, s_ptr, w_ptr, N, K8, BLOCK: tl.constexpr):
    """CT int4 [N, K/8] (nibble j of int32 c -> column 8c+j) + scales [N, K/32] -> bf16((q-8)*s) [N, K]."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # index over packed int32 words
    m = offs < N * K8
    n = offs // K8
    c = offs % K8
    packed = tl.load(q_ptr + offs, mask=m, other=0)
    sc = tl.load(s_ptr + n * (K8 // 4) + c // 4, mask=m, other=0.0).to(tl.float32)  # 32-col group = 4 words
    for j in tl.static_range(8):
        nib = (packed >> (4 * j)) & 0xF
        w = ((nib.to(tl.float32) - 8.0) * sc).to(tl.bfloat16)
        tl.store(w_ptr + n * (K8 * 8) + c * 8 + j, w, mask=m)


_SCRATCH: dict = {}


def dequant_ct(q: torch.Tensor, s: torch.Tensor, extra_rows: int = 0) -> torch.Tensor:
    """bf16 W_ref [N, K] in a reused scratch buffer (valid until the next call; never kept as a weight copy). With
    extra_rows E the view is [N + E, K]; rows N.. are left for the caller (LoRA A rows of a folded dX GEMM)."""
    N, K8 = q.shape
    need = (N + extra_rows) * K8 * 8
    buf = _SCRATCH.get("w")
    if buf is None or buf.numel() < need or buf.device != q.device:
        buf = torch.empty(max(need, 20480 * 2560 + 128 * 2560), dtype=torch.bfloat16, device=q.device)
        _SCRATCH["w"] = buf
    w = buf[:need].view(N + extra_rows, K8 * 8)
    BLOCK = 1024
    _dequant_ct_kernel[(triton.cdiv(N * K8, BLOCK),)](q, s, w, N, K8, BLOCK=BLOCK, num_warps=4)
    return w


def int4_dx(gy: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """dY [M, N] bf16 @ W_ref [N, K] (W_ref dequantized from CT int4 bytes into scratch) -> [M, K] bf16.

    torch.mm(..., out_dtype=bf16) dispatches aten::mm.dtype (cuBLAS, fp32 accumulation), which the batch-invariant
    aten::mm override does not intercept; backward GEMMs need no batch invariance.
    """
    g2 = gy.reshape(-1, gy.shape[-1])
    return torch.mm(g2, dequant_ct(q, s), out_dtype=torch.bfloat16)


def int4_dx_folded(G: torch.Tensor, q: torch.Tensor, s: torch.Tensor, a_rows: torch.Tensor) -> torch.Tensor:
    """Base dX and LoRA dx in ONE GEMM: G = [dY | DH_hi | DH_lo] [M, N + 2k] bf16 @ [W_ref; A; A] [N + 2k, K] -> [M, K]
    bf16 (fp32 accumulation over both, one bf16 rounding instead of three)."""
    N = q.shape[0]
    w = dequant_ct(q, s, extra_rows=a_rows.shape[0])
    w[N:].copy_(a_rows)
    return torch.mm(G, w, out_dtype=torch.bfloat16)


def mm_f32acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 result of a @ b where a is fp32 and b is bf16 (exact bf16 values), via a = hi + lo split."""
    hi = a.to(torch.bfloat16)
    lo = (a - hi.float()).to(torch.bfloat16)
    return torch.mm(hi, b, out_dtype=torch.float32) + torch.mm(lo, b, out_dtype=torch.float32)


def mm_bf16_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 @ bf16 with fp32 accumulation and fp32 output (tensor cores)."""
    return torch.mm(a, b, out_dtype=torch.float32)


def mm_tallk(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = torch.float32, chunk: int = 2048) -> torch.Tensor:
    """a [M, K] @ b [K, N] for a long reduction K (= tokens) and a small output (LoRA dA / dB): cuBLAS launches too
    few output tiles for M*N this small, so K is cut into fixed chunks, the chunks run as one batched GEMM (fp32
    accumulation, fp32 partials) and the partials are summed in a fixed order (deterministic). bf16 inputs."""
    M, K = a.shape
    N = b.shape[1]
    nc = K // chunk
    if nc < 2 or M * N >= (1 << 23) or min(M, N) < 16:
        return torch.mm(a, b, out_dtype=out_dtype)
    Kc = nc * chunk
    a3 = a[:, :Kc].unflatten(1, (nc, chunk)).permute(1, 0, 2)  # [nc, M, chunk] (strided view)
    b3 = b[:Kc].unflatten(0, (nc, chunk))  # [nc, chunk, N]
    out = torch.ops.aten.bmm.dtype(a3, b3, torch.float32).sum(0)  # not intercepted by the BI overrides
    if Kc < K:
        out = out + torch.mm(a[:, Kc:], b[Kc:], out_dtype=torch.float32)
    return out.to(out_dtype)


@triton.jit
def _scorer_dz_kernel(z_ptr, tgt_ptr, g_ptr, t_ptr, hi_ptr, lo_ptr, V, z_stride, o_stride, softcap,
                      HAS_CAP: tl.constexpr, BLOCK: tl.constexpr):
    """Per row: z = bf16 raw lm_head output. zc = cap*tanh(z/cap) (fp32), u = zc/T, p = softmax(u) (fp32);
    dz = (onehot(t) - p) * g / T * (1 - tanh^2) ; written as bf16 hi + bf16 lo (hi + lo ~ fp32 dz)."""
    row = tl.program_id(0).to(tl.int64)
    tgt = tl.load(tgt_ptr + row)
    g = tl.load(g_ptr + row).to(tl.float32)
    T = tl.load(t_ptr + row).to(tl.float32)
    m = -float("inf")
    ssum = 0.0
    for c0 in range(0, V, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        msk = cols < V
        z = tl.load(z_ptr + row * z_stride + cols, mask=msk, other=0.0).to(tl.float32)
        if HAS_CAP:
            a = z / softcap
            e = tl.exp(-2.0 * tl.abs(a))
            th = (1.0 - e) / (1.0 + e)
            th = tl.where(a >= 0, th, -th)
            u = th * softcap / T
        else:
            u = z / T
        u = tl.where(msk, u, -float("inf"))
        mn = tl.maximum(m, tl.max(u, axis=0))
        ssum = ssum * tl.exp(m - mn) + tl.sum(tl.exp(u - mn), axis=0)
        m = mn
    for c0 in range(0, V, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        msk = cols < V
        z = tl.load(z_ptr + row * z_stride + cols, mask=msk, other=0.0).to(tl.float32)
        if HAS_CAP:
            a = z / softcap
            e = tl.exp(-2.0 * tl.abs(a))
            th = (1.0 - e) / (1.0 + e)
            th = tl.where(a >= 0, th, -th)
            u = th * softcap / T
            dcap = 1.0 - th * th
        else:
            u = z / T
            dcap = 1.0
        p = tl.exp(u - m) / ssum
        dz = (tl.where(cols == tgt, 1.0, 0.0) - p) * g / T * dcap
        hi = dz.to(tl.bfloat16)
        lo = (dz - hi.to(tl.float32)).to(tl.bfloat16)
        tl.store(hi_ptr + row * o_stride + cols, hi, mask=msk)
        tl.store(lo_ptr + row * o_stride + cols, lo, mask=msk)


def scorer_dh(z: torch.Tensor, targets: torch.Tensor, g: torch.Tensor, temps: torch.Tensor, softcap: float | None,
              w: torch.Tensor) -> torch.Tensor:
    """dH [C, H] fp32 for rows of bf16 raw logits z [C, V] (one fused pass for dz, two bf16 GEMMs for hi/lo)."""
    C, V = z.shape
    HL = torch.empty((2 * C, V), dtype=z.dtype, device=z.device)  # rows [0, C) = dz hi, rows [C, 2C) = dz lo
    hi, lo = HL[:C], HL[C:]
    _scorer_dz_kernel[(C,)](z, targets.contiguous(), g.float().contiguous(), temps.float().contiguous(), hi, lo, V,
                            z.stride(0), hi.stride(0), float(softcap or 1.0), HAS_CAP=softcap is not None, BLOCK=4096,
                            num_warps=8)
    out = torch.mm(HL, w, out_dtype=torch.float32)  # one GEMM with M = 2C (C alone is too few output tiles)
    return out[:C] + out[C:]


@triton.jit
def _rope_bwd_kernel(g_ptr, cs_ptr, pos_ptr, out_ptr, H, hs, half, rot, cs_stride, BLOCK: tl.constexpr):
    """One program per (token, head): dx1 = g1 cos + g2 sin, dx2 = g2 cos - g1 sin (fp32, separate products as the
    torch reference computes them), pass-through of the non-rotary tail; bf16 out."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    j = tl.arange(0, BLOCK)
    m = j < half
    cos = tl.load(cs_ptr + pos * cs_stride + j, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(cs_ptr + pos * cs_stride + half + j, mask=m, other=0.0).to(tl.float32)
    base = (t * H + h) * hs
    g1 = tl.load(g_ptr + base + j, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(g_ptr + base + half + j, mask=m, other=0.0).to(tl.float32)
    a = g1 * cos
    b = g2 * sin
    d1 = a + b
    e = g2 * cos
    f = g1 * sin
    d2 = e - f
    tl.store(out_ptr + base + j, d1.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(out_ptr + base + half + j, d2.to(out_ptr.dtype.element_ty), mask=m)
    for o in range(rot, hs, BLOCK):  # non-rotary tail (partial rotary): copied through
        jj = o + j
        mt = jj < hs
        tl.store(out_ptr + base + jj, tl.load(g_ptr + base + jj, mask=mt, other=0.0), mask=mt)


def rope_bwd(g: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor, head_size: int, rotary_dim: int) -> torch.Tensor:
    """Fused VJP of the neox rotation (ops._rope_transpose in one pass)."""
    T = positions.numel()
    g2 = g.contiguous().view(T, -1, head_size)
    H = g2.shape[1]
    out = torch.empty_like(g2)
    half = rotary_dim // 2
    cs = cos_sin_cache.contiguous()
    _rope_bwd_kernel[(T, H)](g2, cs, positions.flatten().contiguous(), out, H, head_size, half, rotary_dim, cs.stride(0),
                             BLOCK=triton.next_power_of_2(half), num_warps=2,
                             enable_fp_fusion=False)  # no FMA contraction: same roundings as the torch reference
    return out.view(g.shape)
