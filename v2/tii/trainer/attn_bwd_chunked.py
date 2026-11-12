"""Chunked attention backward for full (causal, no window) layers: cuBLAS bf16 GEMMs + fused Triton softmax kernels.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: Backward of the visible vLLM attention for the head_dim-512 full-attention layers, where FlexAttention is
  limited by sm120's 101 KB shared memory. Same gradient contract as attn_bwd_flex (FA2-style): recompute the forward
  inside the backward so O, LSE and P come from ONE recompute; bf16 operands, fp32 accumulation; P and dS cast to bf16
  for the dV / dQ / dK GEMMs. Per sequence and query chunk (C rows):
    S  = Q K^T            (aten.bmm.dtype -> fp32, cuBLAS tensor cores; not intercepted by the BI aten overrides)
    lse = logsumexp(scale*S) over causal keys          (Triton row kernel, fp32)
    P  = exp(scale*S - lse)  -> bf16                    (Triton)
    O  = P V (fp32), D = rowsum(dO * O)
    dP = dO V^T (fp32);  dS = P*(dP - D)*scale -> bf16 (Triton, P recomputed in fp32 from S)
    dV += P^T dO; dQ = dS K; dK += dS^T Q  (fp32 accumulation over chunks in fixed order)
  GQA: the 4 query heads of a kv head are stacked as rows of one batch entry, so dK/dV sum over them in the GEMM.
  Callers: ops.AttentionFn (FAST_BWD, window is None). Validated per op (test_trainer_ops) and by C7 v3.
SRP/DRY check: Pass - full-attention specialisation of the backward; sliding layers keep attn_bwd_flex.
"""

from __future__ import annotations

from typing import Sequence

import os

import torch
import triton
import triton.language as tl

# Full-attention layers: "bmm" (default, measured faster on real trees at head_dim 512: large cuBLAS tiles) or
# "triton" (fused FA2-style kernels, attn_bwd_triton.py). Sliding layers always use the batched fused kernel.
ATTN_KERNEL = os.environ.get("TII_ATTN_KERNEL", "bmm")

_bmm_dtype = torch.ops.aten.bmm.dtype


def _bmm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """bf16 x bf16 -> fp32 batched GEMM (cuBLASLt aten.bmm.dtype, not intercepted by the BI overrides)."""
    if min(a.shape[1], a.shape[2], b.shape[2]) < 16:
        # tiny tree segments: cuBLASLt has no bf16->fp32 algorithm for N/M of 1..15 here; use an fp32 GEMM of the
        # same bf16 values (exact products, fp32 accumulation - the same contract)
        return torch.matmul(a.float(), b.float())
    if a.shape[0] <= 16:
        # few batches (the KV heads): one cuBLAS mm.dtype per batch is 1.4-2.8x faster than aten.bmm.dtype at these
        # shapes on sm120 (bmm.dtype picks a WMMA kernel for the fp32-output S = Q K^T: 73 vs 201 TFLOPS), same
        # values (fp32 accumulation, bitwise equal in the microbenchmark)
        out = torch.empty(a.shape[0], a.shape[1], b.shape[2], dtype=out_dtype, device=a.device)
        for i in range(a.shape[0]):
            torch.mm(a[i], b[i], out_dtype=out_dtype, out=out[i])
        return out
    return _bmm_dtype(a, b, out_dtype)


@triton.jit
def _lse_kernel(s_ptr, lse_ptr, Kl, C, q0, kbase, W, scale, s_stride, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)  # row = (batch b, group j, query qi) flattened; qi = r % C
    qpos = q0 + (r % C)  # query position on its path; key column c sits at path position kbase + c
    m = -float("inf")
    acc = 0.0
    for c0 in range(0, Kl, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        kp = kbase + cols
        ok = (cols < Kl) & (kp <= qpos) & ((W == 0) | (qpos - kp < W))
        x = tl.load(s_ptr + r * s_stride + cols, mask=ok, other=0.0) * scale
        x = tl.where(ok, x, -float("inf"))
        mn = tl.maximum(m, tl.max(x, axis=0))
        acc = acc * tl.exp(m - mn) + tl.sum(tl.where(ok, tl.exp(x - mn), 0.0), axis=0)
        m = mn
    tl.store(lse_ptr + r, m + tl.log(acc))


@triton.jit
def _p_kernel(s_ptr, lse_ptr, p_ptr, Kl, C, q0, kbase, W, scale, s_stride, p_stride, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    qpos = q0 + (r % C)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    inb = cols < Kl
    kp = kbase + cols
    ok = inb & (kp <= qpos) & ((W == 0) | (qpos - kp < W))
    lse = tl.load(lse_ptr + r)
    x = tl.load(s_ptr + r * s_stride + cols, mask=ok, other=0.0) * scale
    p = tl.where(ok, tl.exp(x - lse), 0.0)
    tl.store(p_ptr + r * p_stride + cols, p.to(tl.bfloat16), mask=inb)


@triton.jit
def _ds_kernel(s_ptr, dp_ptr, lse_ptr, d_ptr, ds_ptr, Kl, C, q0, kbase, W, scale, s_stride, ds_stride,
               BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    qpos = q0 + (r % C)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    inb = cols < Kl
    kp = kbase + cols
    ok = inb & (kp <= qpos) & ((W == 0) | (qpos - kp < W))
    lse = tl.load(lse_ptr + r)
    dd = tl.load(d_ptr + r)
    x = tl.load(s_ptr + r * s_stride + cols, mask=ok, other=0.0) * scale
    p = tl.where(ok, tl.exp(x - lse), 0.0)
    dp = tl.load(dp_ptr + r * s_stride + cols, mask=ok, other=0.0)
    ds = p * (dp - dd) * scale
    tl.store(ds_ptr + r * ds_stride + cols, ds.to(tl.bfloat16), mask=inb)


def chunked_full_attention_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gout: torch.Tensor,
                                 plan_or_bounds, scale: float, chunk: int = 1024, window: int | None = None,
                                 o: torch.Tensor | None = None):
    """Causal (optionally sliding-window) GQA attention backward over contiguous sequences or tree segments (query
    rows of a node attend to the packed rows of their root path). q/gout [T,Hq,D], k/v [T,Hkv,D] bf16 -> dq, dk, dv
    bf16. With a window W, a query chunk at path positions [q0, q0+C) only touches keys [max(0, q0-W+1), q0+C), so the
    work is proportional to (C + W) instead of the path length. dK/dV of shared rows accumulate every branch's
    contribution (fp32, fixed order, unique indices per path)."""
    from v2.tii.trainer.ops import plan_segs, seg_keys

    T, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    W = int(window or 0)
    dq = torch.empty_like(q)
    dk_acc = torch.zeros(T, hkv, d, dtype=torch.float32, device=q.device)
    dv_acc = torch.zeros_like(dk_acc)
    BLOCK = 1024
    for q_lo, q_hi, pos_lo, keys in plan_segs(plan_or_bounds):
        for a in range(q_lo, q_hi, chunk):
            b = min(a + chunk, q_hi)
            C = b - a
            q0 = pos_lo + a - q_lo  # path position of the first query of this chunk
            kb = max(0, q0 - W + 1) if W else 0  # first visible key position
            ke = q0 + C
            Kl = ke - kb
            kr = seg_keys(keys, kb, ke)
            kc = k[kr].permute(1, 0, 2).contiguous()  # [hkv, Kl, d]
            vc = v[kr].permute(1, 0, 2).contiguous()
            qh = q[a:b].view(C, hkv, g, d).permute(1, 2, 0, 3).reshape(hkv, g * C, d).contiguous()
            go = gout[a:b].view(C, hkv, g, d).permute(1, 2, 0, 3).reshape(hkv, g * C, d).contiguous()
            if ATTN_KERNEL == "triton":
                from v2.tii.trainer.attn_bwd_triton import fused_attention_bwd

                qpos = (q0 + torch.arange(g * C, device=q.device) % C).view(1, g * C).to(torch.int64)
                kbase = torch.full((hkv,), kb, dtype=torch.int64, device=q.device)
                dqh, dkc, dvc = fused_attention_bwd(qh, kc, vc, go, qpos, kbase, hkv, scale, W)
                dq[a:b] = dqh.view(hkv, g, C, d).permute(2, 0, 1, 3).reshape(C, hq, d)
                dkc, dvc = dkc.permute(1, 0, 2), dvc.permute(1, 0, 2)  # [Kl, hkv, d]
                if isinstance(kr, slice):
                    dk_acc[kr] += dkc
                    dv_acc[kr] += dvc
                else:
                    dk_acc.index_add_(0, kr, dkc)
                    dv_acc.index_add_(0, kr, dvc)
                continue
            Kl0 = Kl
            if Kl % 8:
                Kl = (Kl + 7) // 8 * 8
                kc = torch.cat([kc, kc.new_zeros(hkv, Kl - Kl0, d)], 1)
                vc = torch.cat([vc, vc.new_zeros(hkv, Kl - Kl0, d)], 1)
            S = _bmm(qh, kc.transpose(1, 2), torch.float32)  # [hkv, gC, Kl]
            R = hkv * g * C
            lse = torch.empty(R, dtype=torch.float32, device=q.device)
            S2 = S.view(R, Kl)
            _lse_kernel[(R,)](S2, lse, Kl, C, q0, kb, W, scale, S2.stride(0), BLOCK=BLOCK, num_warps=4)
            P = torch.empty(hkv, g * C, Kl, dtype=torch.bfloat16, device=q.device)
            P2 = P.view(R, Kl)
            grid = (R, triton.cdiv(Kl, BLOCK))
            _p_kernel[grid](S2, lse, P2, Kl, C, q0, kb, W, scale, S2.stride(0), P2.stride(0), BLOCK=BLOCK, num_warps=4)
            if o is not None:  # D = rowsum(dO * O) from the saved forward output (no P V recompute)
                oh = o[a:b].view(C, hkv, g, d).permute(1, 2, 0, 3).reshape(hkv, g * C, d)
                Dr = (go.float() * oh.float()).sum(-1).reshape(R)
            else:
                O = _bmm(P, vc, torch.float32)  # [hkv, gC, d]
                Dr = (go.float() * O).sum(-1).reshape(R)
                del O
            dP = _bmm(go, vc.transpose(1, 2), torch.float32)  # [hkv, gC, Kl]
            dS = torch.empty_like(P)
            _ds_kernel[grid](S2, dP.view(R, Kl), lse, Dr, dS.view(R, Kl), Kl, C, q0, kb, W, scale, S2.stride(0), Kl,
                             BLOCK=BLOCK, num_warps=4)
            del S, dP
            dvc = _bmm(P.transpose(1, 2), go, torch.float32).permute(1, 0, 2)[:Kl0]  # [Kl, hkv, d]
            dkc = _bmm(dS.transpose(1, 2), qh, torch.float32).permute(1, 0, 2)[:Kl0]
            dqh = _bmm(dS, kc, torch.float32)  # [hkv, gC, d]
            dq[a:b] = dqh.view(hkv, g, C, d).permute(2, 0, 1, 3).reshape(C, hq, d).to(q.dtype)
            if isinstance(kr, slice):
                dk_acc[kr] += dkc
                dv_acc[kr] += dvc
            else:  # unique indices within one path chunk: deterministic
                dk_acc.index_add_(0, kr, dkc)
                dv_acc.index_add_(0, kr, dvc)
            del P, dS, dqh, dkc, dvc
    return dq, dk_acc.to(k.dtype), dv_acc.to(v.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Banded (sliding-window) backward: all 256-query blocks of a segment in ONE batched GEMM per product, each block
# against its own fixed-width key window [p0 - W + 1, p0 + C) gathered by index. Masks use per-row query positions and
# per-batch key base positions. dK/dV scatter in 3 colour classes of non-overlapping windows (no atomics).
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def _band_lse_kernel(s_ptr, lse_ptr, qpos_ptr, kbase_ptr, rows_per_b, Kw, W, scale, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    qpos = tl.load(qpos_ptr + r)
    kb = tl.load(kbase_ptr + r // rows_per_b)
    m = -float("inf")
    acc = 0.0
    for c0 in range(0, Kw, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        kp = kb + cols
        ok = (cols < Kw) & (kp >= 0) & (kp <= qpos) & (qpos - kp < W)
        x = tl.load(s_ptr + r * Kw + cols, mask=ok, other=0.0) * scale
        x = tl.where(ok, x, -float("inf"))
        mn = tl.maximum(m, tl.max(x, axis=0))
        acc = acc * tl.exp(m - mn) + tl.sum(tl.where(ok, tl.exp(x - mn), 0.0), axis=0)
        m = mn
    tl.store(lse_ptr + r, m + tl.log(acc))


@triton.jit
def _band_p_kernel(s_ptr, lse_ptr, qpos_ptr, kbase_ptr, p_ptr, rows_per_b, Kw, W, scale, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    qpos = tl.load(qpos_ptr + r)
    kb = tl.load(kbase_ptr + r // rows_per_b)
    lse = tl.load(lse_ptr + r)
    for c0 in range(0, Kw, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        inb = cols < Kw
        kp = kb + cols
        ok = inb & (kp >= 0) & (kp <= qpos) & (qpos - kp < W)
        x = tl.load(s_ptr + r * Kw + cols, mask=ok, other=0.0) * scale
        tl.store(p_ptr + r * Kw + cols, tl.where(ok, tl.exp(x - lse), 0.0).to(tl.bfloat16), mask=inb)


@triton.jit
def _band_ds_kernel(s_ptr, dp_ptr, lse_ptr, d_ptr, qpos_ptr, kbase_ptr, ds_ptr, rows_per_b, Kw, W, scale,
                    BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    qpos = tl.load(qpos_ptr + r)
    kb = tl.load(kbase_ptr + r // rows_per_b)
    lse = tl.load(lse_ptr + r)
    dd = tl.load(d_ptr + r)
    for c0 in range(0, Kw, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        inb = cols < Kw
        kp = kb + cols
        ok = inb & (kp >= 0) & (kp <= qpos) & (qpos - kp < W)
        x = tl.load(s_ptr + r * Kw + cols, mask=ok, other=0.0) * scale
        p = tl.where(ok, tl.exp(x - lse), 0.0)
        dp = tl.load(dp_ptr + r * Kw + cols, mask=ok, other=0.0)
        tl.store(ds_ptr + r * Kw + cols, (p * (dp - dd) * scale).to(tl.bfloat16), mask=inb)


def banded_window_attention_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gout: torch.Tensor,
                                  plan_or_bounds, scale: float, window: int, block: int = 256):
    """Sliding-window GQA attention backward (contiguous sequences or tree segments), same contract as
    chunked_full_attention_grads, batched over all query blocks of a segment."""
    from v2.tii.trainer.ops import plan_segs, seg_keys

    T, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    W = int(window)
    Kw = block + W - 1
    dq = torch.empty_like(q)
    dk_acc = torch.zeros(T, hkv, d, dtype=torch.float32, device=q.device)
    dv_acc = torch.zeros_like(dk_acc)
    dev = q.device
    for q_lo, q_hi, pos_lo, keys in plan_segs(plan_or_bounds):
        L = q_hi - q_lo
        nb = (L + block - 1) // block
        Lp = pos_lo + L
        path = (torch.arange(Lp, device=dev) + keys[1]) if isinstance(keys, tuple) else keys[:Lp]
        p0 = pos_lo + torch.arange(nb, device=dev) * block  # first query position of each block
        kbase = p0 - (W - 1)  # per-block key window base position (may be negative -> masked)
        kpos = kbase[:, None] + torch.arange(Kw, device=dev)[None, :]  # [nb, Kw]
        kidx = path[kpos.clamp(0, Lp - 1)]  # packed key rows (clamped positions are masked out)
        # queries: pad the last block with copies of row q_lo (masked through qpos > valid below)
        qrow = q_lo + torch.arange(nb * block, device=dev)
        qvalid = qrow < q_hi
        qrow = torch.where(qvalid, qrow, torch.full_like(qrow, q_lo))
        qpos_q = pos_lo + torch.arange(nb * block, device=dev)  # [nb*block]
        kc = k[kidx.reshape(-1)].view(nb, Kw, hkv, d).permute(0, 2, 1, 3).reshape(nb * hkv, Kw, d)
        vc = v[kidx.reshape(-1)].view(nb, Kw, hkv, d).permute(0, 2, 1, 3).reshape(nb * hkv, Kw, d)
        qh = q[qrow].view(nb, block, hkv, g, d).permute(0, 2, 3, 1, 4).reshape(nb * hkv, g * block, d)
        gsel = gout[qrow] * qvalid[:, None, None].to(gout.dtype)  # padded query rows carry zero gradient
        go = gsel.view(nb, block, hkv, g, d).permute(0, 2, 3, 1, 4).reshape(nb * hkv, g * block, d)
        rows_per_b = g * block
        R = nb * hkv * rows_per_b
        # per-row query position: rows ordered (block, kv head, group member, query in block)
        qpos_r = qpos_q.view(nb, 1, 1, block).expand(nb, hkv, g, block).reshape(R).to(torch.int64)
        qpos_r = torch.where(qvalid.view(nb, 1, 1, block).expand(nb, hkv, g, block).reshape(R), qpos_r,
                             torch.full_like(qpos_r, -1))  # padded rows see no keys
        # .contiguous(): with nb == 1 the expand+reshape is a stride-0 VIEW; Triton indexes these as dense arrays
        kbase_b = kbase.view(nb, 1).expand(nb, hkv).reshape(nb * hkv).to(torch.int64).contiguous()
        qpos_r = qpos_r.contiguous()
        if ATTN_KERNEL == "triton":
            from v2.tii.trainer.attn_bwd_triton import fused_attention_bwd

            # per-block row positions (rows ordered group member j, query in block), -1 for padded queries
            qp = torch.where(qvalid, qpos_q, torch.full_like(qpos_q, -1)).view(nb, 1, block).expand(nb, g, block)
            dqh, dkw, dvw = fused_attention_bwd(qh, kc, vc, go, qp.reshape(nb, g * block).to(torch.int64).contiguous(),
                                                kbase_b, hkv, scale, W)
            dqh = dqh.view(nb, hkv, g, block, d).permute(0, 3, 1, 2, 4).reshape(nb * block, hq, d)
            dq[q_lo:q_hi] = dqh[:L]
            dkw = dkw.view(nb, hkv, Kw, d).permute(0, 2, 1, 3)
            dvw = dvw.view(nb, hkv, Kw, d).permute(0, 2, 1, 3)
            valid_k = (kpos >= 0) & (kpos < Lp)
            ncol = (Kw + block - 1) // block
            for c in range(min(ncol, nb)):
                sel = torch.arange(c, nb, ncol, device=dev)
                idx = kidx[sel][valid_k[sel]]
                dk_acc.index_add_(0, idx, dkw[sel][valid_k[sel]])
                dv_acc.index_add_(0, idx, dvw[sel][valid_k[sel]])
            continue
        S = _bmm(qh, kc.transpose(1, 2), torch.float32)  # [nb*hkv, g*block, Kw]
        lse = torch.empty(R, dtype=torch.float32, device=dev)
        _band_lse_kernel[(R,)](S, lse, qpos_r, kbase_b, rows_per_b, Kw, W, scale, BLOCK=1024, num_warps=4)
        lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))  # rows without keys (padding)
        P = torch.empty(nb * hkv, rows_per_b, Kw, dtype=torch.bfloat16, device=dev)
        _band_p_kernel[(R,)](S, lse, qpos_r, kbase_b, P, rows_per_b, Kw, W, scale, BLOCK=1024, num_warps=4)
        O = _bmm(P, vc, torch.float32)
        Dr = (go.float() * O).sum(-1).reshape(R)
        del O
        dP = _bmm(go, vc.transpose(1, 2), torch.float32)
        dS = torch.empty_like(P)
        _band_ds_kernel[(R,)](S, dP, lse, Dr, qpos_r, kbase_b, dS, rows_per_b, Kw, W, scale, BLOCK=1024, num_warps=4)
        del S, dP
        dvw = _bmm(P.transpose(1, 2), go, torch.float32).view(nb, hkv, Kw, d).permute(0, 2, 1, 3)  # [nb, Kw, hkv, d]
        dkw = _bmm(dS.transpose(1, 2), qh, torch.float32).view(nb, hkv, Kw, d).permute(0, 2, 1, 3)
        dqh = _bmm(dS, kc, torch.float32).view(nb, hkv, g, block, d).permute(0, 3, 1, 2, 4).reshape(nb * block, hq, d)
        dq[q_lo:q_hi] = dqh[:L].to(q.dtype)
        valid_k = (kpos >= 0) & (kpos < Lp)  # [nb, Kw] real key positions only
        ncol = (Kw + block - 1) // block  # windows of blocks j and j+ncol never overlap
        for c in range(min(ncol, nb)):
            sel = torch.arange(c, nb, ncol, device=dev)
            if sel.numel() == 0:
                continue
            idx = kidx[sel][valid_k[sel]]
            dk_acc.index_add_(0, idx, dkw[sel][valid_k[sel]])
            dv_acc.index_add_(0, idx, dvw[sel][valid_k[sel]])
        del P, dS, dqh, dkw, dvw
    return dq, dk_acc.to(k.dtype), dv_acc.to(v.dtype)


def _band_plan(plan_or_bounds, T: int, W: int, block: int, g: int, hkv: int, device):
    """Gather/scatter geometry of ALL sliding-window query blocks of a forward (every segment / tree node), cached on
    the AttnPlan: it is identical for every sliding layer of the forward."""
    from v2.tii.trainer.ops import AttnPlan, plan_segs

    cache = getattr(plan_or_bounds, "_band_cache", None) if isinstance(plan_or_bounds, AttnPlan) else None
    key = (W, block, T)
    if cache is not None and key in cache:
        return cache[key]
    Kw = block + W - 1
    qrows, qpos, kidx, kpos_all = [], [], [], []
    for q_lo, q_hi, pos_lo, keys in plan_segs(plan_or_bounds):
        L = q_hi - q_lo
        nb = (L + block - 1) // block
        Lp = pos_lo + L
        path = (torch.arange(Lp, device=device) + keys[1]) if isinstance(keys, tuple) else keys[:Lp]
        p0 = pos_lo + torch.arange(nb, device=device) * block
        kpos = (p0 - (W - 1))[:, None] + torch.arange(Kw, device=device)[None, :]
        kidx.append(path[kpos.clamp(0, Lp - 1)])
        kpos_all.append(torch.where((kpos >= 0) & (kpos < Lp), kpos, torch.full_like(kpos, -1)))
        r = q_lo + torch.arange(nb * block, device=device)
        valid = r < q_hi
        qrows.append(torch.where(valid, r, torch.full_like(r, q_lo)))
        qpos.append(torch.where(valid, pos_lo + torch.arange(nb * block, device=device), torch.full_like(r, -1)))
    qrows, qpos = torch.cat(qrows), torch.cat(qpos)
    kidx, kpos = torch.cat(kidx), torch.cat(kpos_all)
    NB = kidx.shape[0]
    # kbase per block = path position of key column 0 (may be negative: masked in the kernel)
    kbase = torch.cat([((pos_lo + torch.arange((q_hi - q_lo + block - 1) // block, device=device) * block) - (W - 1))
                       for q_lo, q_hi, pos_lo, _k in plan_segs(plan_or_bounds)])
    valid_k = kpos >= 0
    geo = {
        "NB": NB, "Kw": Kw,
        "qrows": qrows, "qvalid": qpos >= 0,
        "qpos_rows": qpos.view(NB, 1, block).expand(NB, g, block).reshape(NB, g * block).contiguous().to(torch.int64),
        "kidx": kidx.reshape(-1).contiguous(),
        "kbase_b": kbase.view(NB, 1).expand(NB, hkv).reshape(NB * hkv).contiguous().to(torch.int64),
        "scatter_idx": kidx[valid_k].contiguous(), "valid_k": valid_k,
        "out_rows": torch.nonzero(qpos >= 0).view(-1), "out_dst": qrows[qpos >= 0],
    }
    if isinstance(plan_or_bounds, AttnPlan):
        if cache is None:
            cache = {}
            plan_or_bounds._band_cache = cache
        cache[key] = geo
    return geo


def banded_window_attention_grads_batched(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gout: torch.Tensor,
                                          plan_or_bounds, scale: float, window: int, block: int = 256,
                                          o: torch.Tensor | None = None):
    """Sliding-window GQA attention backward for ALL segments of a forward in one fused kernel call: every 256-query
    block (of any segment or tree node) attends its own gathered key window. dK/dV are scattered with a sort-based
    deterministic accumulate (shared-prefix rows may receive contributions from several segments)."""
    from v2.tii.trainer.attn_bwd_triton import fused_attention_bwd

    T, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    W = int(window)
    geo = _band_plan(plan_or_bounds, T, W, block, g, hkv, q.device)
    NB, Kw = geo["NB"], geo["Kw"]
    kc = k[geo["kidx"]].view(NB, Kw, hkv, d).permute(0, 2, 1, 3).reshape(NB * hkv, Kw, d)
    vc = v[geo["kidx"]].view(NB, Kw, hkv, d).permute(0, 2, 1, 3).reshape(NB * hkv, Kw, d)
    qh = q[geo["qrows"]].view(NB, block, hkv, g, d).permute(0, 2, 3, 1, 4).reshape(NB * hkv, g * block, d)
    gsel = gout[geo["qrows"]] * geo["qvalid"][:, None, None].to(gout.dtype)
    go = gsel.view(NB, block, hkv, g, d).permute(0, 2, 3, 1, 4).reshape(NB * hkv, g * block, d)
    oh = None
    if o is not None:  # saved forward output for D = rowsum(dO * O); padded rows have dO = 0
        oh = o[geo["qrows"]].view(NB, block, hkv, g, d).permute(0, 2, 3, 1, 4).reshape(NB * hkv, g * block, d)
    dqh, dkw, dvw = fused_attention_bwd(qh, kc, vc, go, geo["qpos_rows"], geo["kbase_b"], hkv, scale, W, o=oh)
    dqh = dqh.view(NB, hkv, g, block, d).permute(0, 3, 1, 2, 4).reshape(NB * block, hq, d)
    dq = torch.empty_like(q)
    dq[geo["out_dst"]] = dqh[geo["out_rows"]]
    vk = geo["valid_k"]
    dkw = dkw.view(NB, hkv, Kw, d).permute(0, 2, 1, 3)[vk]  # [n, hkv, d]
    dvw = dvw.view(NB, hkv, Kw, d).permute(0, 2, 1, 3)[vk]
    dk = torch.zeros(T, hkv, d, dtype=torch.float32, device=q.device)
    dv = torch.zeros_like(dk)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:  # index_put_(accumulate=True) has a sort-based deterministic CUDA implementation in deterministic mode
        dk.index_put_((geo["scatter_idx"],), dkw, accumulate=True)
        dv.index_put_((geo["scatter_idx"],), dvw, accumulate=True)
    finally:
        torch.use_deterministic_algorithms(prev)
    return dq, dk.to(k.dtype), dv.to(v.dtype)
