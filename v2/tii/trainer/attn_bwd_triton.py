"""Fused FA2-style attention backward in Triton (no materialized scores), for full, sliding-window and tree attention.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: Backward of the visible vLLM attention with the same gradient contract as before (recompute inside the
  backward, O / LSE / P from ONE recompute, bf16 operands, fp32 accumulation, P and dS cast to bf16 for the dV / dK /
  dQ products), but without writing S, P, dP or dS to memory and without cuBLAS's slow bf16->fp32 batched path on
  sm120. Three kernels per batch of (rows = 4 query heads x C queries of one kv head, keys = gathered path window):
    stats : online softmax over key tiles -> lse (fp32) and D = rowsum(dO * O_ref) with O_ref from the same recompute;
    dkdv  : per key tile, loop over row tiles: P = exp(scale*QK^T - lse), dV += P^T dO, dP = dO V^T,
            dS = P (dP - D) scale, dK += dS^T Q   (fixed loop order: deterministic);
    dq    : per row tile, loop over key tiles: dQ += dS K.
  Masks: key column c is visible from row r iff kbase[b] + c <= qpos[r] (and qpos - kpos < W when W > 0) and
  c < Kl; per-row query positions (shared by the kv-head batches of one chunk/block) and per-batch key base
  positions cover contiguous sequences, tree paths and sliding windows. head_dim 512 accumulators are split into two
  256-wide halves (DSPLIT programs) so tiles fit sm120's 101 KB shared memory.
  Callers: attn_bwd_chunked (full: query chunks against gathered causal keys; sliding: banded blocks).
SRP/DRY check: Pass - replaces the GEMM + elementwise path inside attn_bwd_chunked; the fp32 reference stays as oracle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _visible(qpos, kbase, cols, Kl, W):
    kp = kbase + cols
    ok = (cols[None, :] < Kl) & (kp[None, :] >= 0) & (kp[None, :] <= qpos[:, None])
    ok = ok & ((W == 0) | (qpos[:, None] - kp[None, :] < W))
    return ok


@triton.jit
def _stats_kernel(Q, K, V, DO, O, LSE, DD, QPOS, KBASE, M, Kl, W, scale, QDIV,
                  sqb, sqm, skb, skn, svb, svn, sdb, sdm,
                  HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, HAS_O: tl.constexpr):
    pm = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < M
    dcols = tl.arange(0, HD)
    qpos = tl.load(QPOS + (b // QDIV) * M + rows, mask=rmask, other=-1)
    kbase = tl.load(KBASE + b)
    q = tl.load(Q + b * sqb + rows[:, None] * sqm + dcols[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    for n0 in range(0, Kl, BN):
        cols = n0 + tl.arange(0, BN)
        k = tl.load(K + b * skb + cols[:, None] * skn + dcols[None, :], mask=(cols < Kl)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        ok = _visible(qpos, kbase, cols, Kl, W)
        s = tl.where(ok, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new > -float("inf"), m_new, 0.0)
        p = tl.where(ok, tl.exp(s - m_safe[:, None]), 0.0)
        alpha = tl.where(m_new > -float("inf"), tl.exp(m_i - m_safe), 1.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        if not HAS_O:
            v = tl.load(V + b * svb + cols[:, None] * svn + dcols[None, :], mask=(cols < Kl)[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), 0.0)
    if HAS_O:  # saved forward output (same [B, M, D] layout and strides as DO)
        o = tl.load(O + b * sdb + rows[:, None] * sdm + dcols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    else:
        o = acc / l_safe[:, None]
    do = tl.load(DO + b * sdb + rows[:, None] * sdm + dcols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    dd = tl.sum(do * o, 1)
    tl.store(LSE + b * M + rows, lse, mask=rmask)
    tl.store(DD + b * M + rows, dd, mask=rmask)


@triton.jit
def _dkdv_kernel(Q, K, V, DO, LSE, DD, QPOS, KBASE, DK, DV, M, Kl, W, scale, QDIV,
                 sqb, sqm, skb, skn, svb, svn, sdb, sdm, sob, son,
                 HD: tl.constexpr, HDH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pn = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    half = tl.program_id(2)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < Kl
    dall = tl.arange(0, HD)
    dh = half * HDH + tl.arange(0, HDH)
    kbase = tl.load(KBASE + b)
    k = tl.load(K + b * skb + cols[:, None] * skn + dall[None, :], mask=cmask[:, None], other=0.0)
    v = tl.load(V + b * svb + cols[:, None] * svn + dall[None, :], mask=cmask[:, None], other=0.0)
    dk = tl.zeros([BN, HDH], tl.float32)
    dv = tl.zeros([BN, HDH], tl.float32)
    for m0 in range(0, M, BM):
        rows = m0 + tl.arange(0, BM)
        rmask = rows < M
        qpos = tl.load(QPOS + (b // QDIV) * M + rows, mask=rmask, other=-1)
        q = tl.load(Q + b * sqb + rows[:, None] * sqm + dall[None, :], mask=rmask[:, None], other=0.0)
        do = tl.load(DO + b * sdb + rows[:, None] * sdm + dall[None, :], mask=rmask[:, None], other=0.0)
        lse = tl.load(LSE + b * M + rows, mask=rmask, other=0.0)
        dd = tl.load(DD + b * M + rows, mask=rmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        ok = _visible(qpos, kbase, cols, Kl, W) & rmask[:, None]
        p = tl.where(ok, tl.exp(s - lse[:, None]), 0.0)
        do_h = tl.load(DO + b * sdb + rows[:, None] * sdm + dh[None, :], mask=rmask[:, None], other=0.0)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do_h)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dd[:, None]) * scale
        q_h = tl.load(Q + b * sqb + rows[:, None] * sqm + dh[None, :], mask=rmask[:, None], other=0.0)
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q_h)
    tl.store(DK + b * sob + cols[:, None] * son + dh[None, :], dk, mask=cmask[:, None])
    tl.store(DV + b * sob + cols[:, None] * son + dh[None, :], dv, mask=cmask[:, None])


@triton.jit
def _dq_kernel(Q, K, V, DO, LSE, DD, QPOS, KBASE, DQ, M, Kl, W, scale, QDIV,
               sqb, sqm, skb, skn, svb, svn, sdb, sdm,
               HD: tl.constexpr, HDH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    half = tl.program_id(2)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < M
    dall = tl.arange(0, HD)
    dh = half * HDH + tl.arange(0, HDH)
    qpos = tl.load(QPOS + (b // QDIV) * M + rows, mask=rmask, other=-1)
    kbase = tl.load(KBASE + b)
    q = tl.load(Q + b * sqb + rows[:, None] * sqm + dall[None, :], mask=rmask[:, None], other=0.0)
    do = tl.load(DO + b * sdb + rows[:, None] * sdm + dall[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + b * M + rows, mask=rmask, other=0.0)
    dd = tl.load(DD + b * M + rows, mask=rmask, other=0.0)
    dq = tl.zeros([BM, HDH], tl.float32)
    for n0 in range(0, Kl, BN):
        cols = n0 + tl.arange(0, BN)
        cmask = cols < Kl
        k = tl.load(K + b * skb + cols[:, None] * skn + dall[None, :], mask=cmask[:, None], other=0.0)
        v = tl.load(V + b * svb + cols[:, None] * svn + dall[None, :], mask=cmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        ok = _visible(qpos, kbase, cols, Kl, W) & rmask[:, None]
        p = tl.where(ok, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dd[:, None]) * scale
        k_h = tl.load(K + b * skb + cols[:, None] * skn + dh[None, :], mask=cmask[:, None], other=0.0)
        dq += tl.dot(ds.to(tl.bfloat16), k_h)
    tl.store(DQ + b * sqb + rows[:, None] * sqm + dh[None, :], dq.to(tl.bfloat16), mask=rmask[:, None])


def _cfg(hd: int):
    """(BM_stats, BN_stats, BM, BN, DSPLIT, warps, stages): tiles sized for 101 KB shared memory on sm120."""
    if hd >= 512:
        return 32, 32, 16, 32, 2, 4, 1
    return 64, 32, 32, 32, 1, 4, 1


def fused_attention_bwd(qh: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, go: torch.Tensor, qpos: torch.Tensor,
                        kbase: torch.Tensor, qdiv: int, scale: float, window: int, o: torch.Tensor | None = None):
    """qh/go [B, M, D] bf16 (rows), kc/vc [B, Kl, D] bf16; qpos int64 [B // qdiv, M]; kbase int64 [B].
    Returns dq [B, M, D] bf16, dk/dv [B, Kl, D] fp32."""
    B, M, D = qh.shape
    Kl = kc.shape[1]
    qh, kc, vc, go = qh.contiguous(), kc.contiguous(), vc.contiguous(), go.contiguous()
    qpos, kbase = qpos.contiguous(), kbase.contiguous()
    if o is not None:
        o = o.contiguous()
        assert o.stride() == go.stride()
    bms, bns, bm, bn, dsplit, nw, ns = _cfg(D)
    lse = torch.empty(B, M, dtype=torch.float32, device=qh.device)
    dd = torch.empty_like(lse)
    W = int(window or 0)
    _stats_kernel[(triton.cdiv(M, bms), B)](qh, kc, vc, go, o if o is not None else go, lse, dd, qpos, kbase, M, Kl, W,
                                            scale, qdiv,
                                            qh.stride(0), qh.stride(1), kc.stride(0), kc.stride(1), vc.stride(0),
                                            vc.stride(1), go.stride(0), go.stride(1),
                                            HD=D, BM=bms, BN=bns, HAS_O=o is not None,
                                            num_warps=8 if D >= 512 else nw, num_stages=ns)
    dk = torch.empty(B, Kl, D, dtype=torch.float32, device=qh.device)
    dv = torch.empty_like(dk)
    dq = torch.empty_like(qh)
    _dkdv_kernel[(triton.cdiv(Kl, bn), B, dsplit)](qh, kc, vc, go, lse, dd, qpos, kbase, dk, dv, M, Kl, W, scale, qdiv,
                                                  qh.stride(0), qh.stride(1), kc.stride(0), kc.stride(1), vc.stride(0),
                                                  vc.stride(1), go.stride(0), go.stride(1), dk.stride(0), dk.stride(1),
                                                  HD=D, HDH=D // dsplit, BM=bm, BN=bn, num_warps=nw, num_stages=ns)
    _dq_kernel[(triton.cdiv(M, bm), B, dsplit)](qh, kc, vc, go, lse, dd, qpos, kbase, dq, M, Kl, W, scale, qdiv,
                                               qh.stride(0), qh.stride(1), kc.stride(0), kc.stride(1), vc.stride(0),
                                               vc.stride(1), go.stride(0), go.stride(1),
                                               HD=D, HDH=D // dsplit, BM=bm, BN=bn, num_warps=nw, num_stages=ns)
    return dq, dk, dv
