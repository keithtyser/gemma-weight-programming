"""TII segmented attention: fixed absolute key segments shared by prefill, chunked prefill and decode.

Author: Claude Opus 5.5 (engine track), 2026-11-05. Derived from vLLM 0.19.1 triton_unified_attention (Apache-2.0).

Keys of every sequence are cut into absolute segments of SEG_TILES * TILE keys ([s*SEG, (s+1)*SEG)). For each query
row and segment s the kernel computes the per-segment online-softmax state (m_s, l_s, acc_s) from a fresh state
(M=-inf, L=0, ACC=0) over the segment's 32-key tiles in increasing order; segments are then merged in increasing s
with one fixed merge rule (`_merge`). Two executions produce identical bytes by construction:
  * `_seg_attn` mode 0: one CTA per (q-block, kv-head) walks its segments sequentially and merges in registers
                      (prefill, chunked prefill, mixed steps);
  * `_seg_attn` mode 1 + `_seg_reduce`: one CTA per (q-block, kv-head, segment) writes (m_s, l_s, acc_s) to fp32
                      buffers, a reduce kernel merges them per row in increasing s (decode-only steps: this is the
                      split-KV parallelism the 2D kernel lacks at decode).
  The mode is a runtime argument of ONE compiled kernel: two separately compiled kernels gave different QK^T /
  row-sum bits (Triton picks layouts per kernel), the shared binary removes that.
Fully masked tiles and segments are exact no-ops (M stays -inf, merges guarded by `where`), so which masked tiles a
q-block visits does not matter. All three kernels run with enable_fp_fusion=False so no mul+add is contracted
differently in different kernels. Features: causal, optional sliding window, optional softcap, GQA; no sinks,
alibi, qq-bias, fp8 or multimodal prefix ranges (asserted).
"""

from __future__ import annotations

import os

import torch

from triton.language.extra.cuda import libdevice

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import apply_softcap, cdiv_fn, find_seq_idx

SEG_TILES = 16  # 16 tiles x 32 keys = 512-key absolute segments
TILE = 32
VERSION = f"tii-segattn-v6-seg{SEG_TILES * TILE}"


@triton.jit
def _merge(M, L, ACC, m, l, acc):
    # Segment merge with explicitly rounded libdevice ops (__nv_fmul_rn/__nv_fadd_rn/__nv_fsub_rn/__nv_expf): the
    # compiler may not contract or re-associate them, so the in-register merge (mode 0) and _seg_reduce (mode 1)
    # produce the same bits. (Plain `L * a + l * b` differed by 1 ulp between the two kernels.)
    valid = m > float("-inf")
    Mn = tl.where(valid, tl.maximum(M, m), M)
    a = tl.where(valid, libdevice.exp(libdevice.sub_rn(M, Mn)), 1.0)
    b = tl.where(valid, libdevice.exp(libdevice.sub_rn(tl.where(valid, m, 0.0), Mn)), 0.0)
    L2 = tl.where(valid, libdevice.add_rn(libdevice.mul_rn(L, a), libdevice.mul_rn(l, b)), L)
    ACC2 = tl.where(valid[:, None], libdevice.add_rn(libdevice.mul_rn(ACC, a[:, None]),
                                                     libdevice.mul_rn(acc, b[:, None])), ACC)
    return Mn, L2, ACC2


@triton.jit
def _segment(Q, key_cache_ptr, value_cache_ptr, block_tables_ptr, block_table_offset, kv_head_idx, offs_d, offs_t,
             dim_mask, query_mask_0, query_mask_1, query_pos, context_len, max_seq_prefix_len, qpos_lo, t0, t1, scale,
             softcap, stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3, stride_v_cache_0,
             stride_v_cache_1, stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr,
             BLOCK_M: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
             USE_SOFTCAP: tl.constexpr):
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    for j in range(t0, t1):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)
        v_offset = (physical_block_idx[:, None] * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
                    + offs_d[None, :] * stride_v_cache_3 + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
                    + offs_d[:, None] * stride_k_cache_3 + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
        K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j_safe = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j_safe[:, None])
        Pb = P.to(V.dtype)
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j_safe)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if SLIDING_WINDOW:
            V = tl.where((context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0)
        acc += tl.dot(Pb, V)
    return M, L, acc


@triton.jit
def _prologue(query_ptr, query_start_len_ptr, seq_lens_ptr, num_seqs, query_stride_0, query_stride_1,
              num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr, BLOCK_Q: tl.constexpr,
              BLOCK_M: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
              TILE_SIZE: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    start = tl.load(query_start_len_ptr + seq_idx)
    stop = tl.load(query_start_len_ptr + seq_idx + 1)
    qlen = stop - start
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < qlen, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    Q = tl.load(query_ptr + query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1
                + offs_d[None, :], mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - qlen
    max_seq_prefix_len = tl.minimum(context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv
                                    + 1, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles
    qpos_lo = q_block_local_idx * BLOCK_Q
    if SLIDING_WINDOW > 0:
        qpos_hi = tl.minimum(qpos_lo + (BLOCK_M - 1) // num_queries_per_kv, qlen - 1)
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
    active = q_block_local_idx * BLOCK_Q < qlen
    return (seq_idx, kv_head_idx, offs_d, query_pos, query_offset_0, query_offset_1, dim_mask, query_mask_0,
            query_mask_1, Q, context_len, max_seq_prefix_len, tile_start, tile_end, qpos_lo, active)


# do_not_specialize: Triton otherwise compiles a separate binary when an int argument equals 1 (or is divisible by
# 16); per_segment/num_seg/num_seqs/block_table_stride must not create mode- or batch-dependent binaries.
@triton.jit(do_not_specialize=["per_segment", "num_seg", "num_seqs", "block_table_stride"])
def _seg_attn(output_ptr, m_buf, l_buf, acc_buf, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
              seq_lens_ptr, scale, softcap, per_segment, num_seg, num_query_heads: tl.constexpr,
              num_queries_per_kv: tl.constexpr, block_table_stride: tl.int64, query_stride_0: tl.int64,
              query_stride_1: tl.int64, output_stride_0: tl.int64, output_stride_1: tl.int64,
              BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,
              HEAD_SIZE_PADDED: tl.constexpr, USE_SOFTCAP: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
              stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64, stride_k_cache_2: tl.int64,
              stride_k_cache_3: tl.constexpr, stride_v_cache_0: tl.int64, stride_v_cache_1: tl.int64,
              stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr, query_start_len_ptr, BLOCK_Q: tl.constexpr,
              num_seqs: tl.int32, BLOCK_M: tl.constexpr, SEG_TILES: tl.constexpr):
    """One binary for both execution modes (per_segment is a runtime argument, so the QK/PV dots and the row
    reductions are the same machine code in both): per_segment == 0 -> CTA walks all segments of its q-block,
    merges in registers, writes the normalized output; per_segment == 1 -> CTA (q-block, head, s=program_id(2))
    writes the raw (m, l, acc) of segment s for _seg_reduce."""
    (seq_idx, kv_head_idx, offs_d, query_pos, query_offset_0, query_offset_1, dim_mask, query_mask_0, query_mask_1, Q,
     context_len, max_seq_prefix_len, tile_start, tile_end, qpos_lo, active) = _prologue(
        query_ptr, query_start_len_ptr, seq_lens_ptr, num_seqs, query_stride_0, query_stride_1, num_query_heads,
        num_queries_per_kv, BLOCK_Q, BLOCK_M, HEAD_SIZE, HEAD_SIZE_PADDED, TILE_SIZE, SLIDING_WINDOW)
    if not active:
        return
    offs_t = tl.arange(0, TILE_SIZE)
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    ACC = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    s_lo = tile_start // SEG_TILES
    s_hi = cdiv_fn(tile_end, SEG_TILES)
    if per_segment:
        sp = tl.program_id(2)
        if (sp < s_lo) | (sp >= s_hi):  # segment outside this q-block's key range: nothing to write or read
            return
        s_lo = sp
        s_hi = sp + 1
    for s in range(s_lo, s_hi):
        t0 = tl.maximum(s * SEG_TILES, tile_start)
        t1 = tl.minimum((s + 1) * SEG_TILES, tile_end)
        m, l, acc = _segment(Q, key_cache_ptr, value_cache_ptr, block_tables_ptr, block_table_offset, kv_head_idx,
                             offs_d, offs_t, dim_mask, query_mask_0, query_mask_1, query_pos, context_len,
                             max_seq_prefix_len, qpos_lo, t0, t1, scale, softcap, stride_k_cache_0, stride_k_cache_1,
                             stride_k_cache_2, stride_k_cache_3, stride_v_cache_0, stride_v_cache_1, stride_v_cache_2,
                             stride_v_cache_3, BLOCK_SIZE, TILE_SIZE, BLOCK_M, HEAD_SIZE_PADDED, SLIDING_WINDOW,
                             USE_SOFTCAP)
        if per_segment:
            M, L, ACC = m, l, acc
        else:
            M, L, ACC = _merge(M, L, ACC, m, l, acc)
    rmask = query_mask_0 & query_mask_1
    if per_segment:
        row = (query_offset_0 * num_query_heads + query_offset_1) * num_seg + tl.program_id(2)
        tl.store(m_buf + row, M, mask=rmask)
        tl.store(l_buf + row, L, mask=rmask)
        tl.store(acc_buf + row[:, None] * HEAD_SIZE_PADDED + offs_d[None, :], ACC, mask=rmask[:, None])
    else:
        out = tl.div_rn(ACC, tl.broadcast_to(L[:, None], ACC.shape))  # IEEE division in every path
        tl.store(output_ptr + query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1
                 + offs_d[None, :], out, mask=dim_mask[None, :] & rmask[:, None])


@triton.jit(do_not_specialize=["output_stride_0", "output_stride_1"])
def _seg_reduce(output_ptr, m_buf, l_buf, acc_buf, seq_lens_ptr, output_stride_0: tl.int64, output_stride_1: tl.int64,
                num_query_heads: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
                NUM_SEG: tl.constexpr, TILE_SIZE: tl.constexpr, SEG_TILES: tl.constexpr,
                SLIDING_WINDOW: tl.constexpr):
    """Decode-only: token i is the single query of sequence i. Merges exactly the segments the parallel kernel wrote
    for that row (same tile_start/tile_end formulas as _prologue for a 1-token q-block), in increasing order."""
    tok = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    base = (tok * num_query_heads + h) * NUM_SEG
    seq_len = tl.load(seq_lens_ptr + tok)
    tile_end = cdiv_fn(seq_len, TILE_SIZE)
    tile_start = 0
    if SLIDING_WINDOW > 0:
        tile_start = tl.maximum(0, (seq_len - SLIDING_WINDOW) // TILE_SIZE)
    M = tl.full([1], float("-inf"), dtype=tl.float32)
    L = tl.zeros([1], dtype=tl.float32)
    ACC = tl.zeros([1, HEAD_SIZE_PADDED], dtype=tl.float32)
    for s in range(tile_start // SEG_TILES, cdiv_fn(tile_end, SEG_TILES)):
        m = tl.load(m_buf + base + s + tl.zeros([1], dtype=tl.int32))
        l = tl.load(l_buf + base + s + tl.zeros([1], dtype=tl.int32))
        acc = tl.load(acc_buf + (base + s) * HEAD_SIZE_PADDED + offs_d)[None, :]
        M, L, ACC = _merge(M, L, ACC, m, l, acc)
    out = tl.div_rn(ACC, tl.broadcast_to(L[:, None], ACC.shape))  # IEEE division in every path
    tl.store(output_ptr + tok * output_stride_0 + h * output_stride_1 + offs_d[None, :], out,
             mask=(offs_d < HEAD_SIZE)[None, :])


def unified_attention_tii(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                          window_size, block_table, softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None,
                          num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None,
                          softmax_segm_expsum=None, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
                          mm_prefix_range=None, use_alibi_sqrt=False, force_path: str | None = None):
    """Drop-in for vllm unified_attention (same signature) for the TII aligned engine."""
    assert causal and q_descale is None and alibi_slopes is None and qq_bias is None and sinks is None
    assert mm_prefix_range is None and output_scale is None and q.dtype in (torch.bfloat16, torch.float16)
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    nqpk = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    BLOCK_M = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    BLOCK_Q = BLOCK_M // nqpk
    total_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    hp = triton.next_power_of_2(head_size)
    sw = 1 + window_size[0] if window_size[0] >= 0 else 0
    common = dict(scale=softmax_scale, softcap=softcap, num_query_heads=num_query_heads, num_queries_per_kv=nqpk,
                  block_table_stride=block_table.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
                  BLOCK_SIZE=block_size, TILE_SIZE=TILE, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=hp,
                  USE_SOFTCAP=(softcap > 0), SLIDING_WINDOW=sw, stride_k_cache_0=k.stride(0),
                  stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
                  stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2),
                  stride_v_cache_3=v.stride(3), query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q,
                  num_seqs=num_seqs, BLOCK_M=BLOCK_M, SEG_TILES=SEG_TILES)
    # num_warps depends only on the head size (never on the mode or batch), so both modes share one binary.
    common["num_warps"] = int(os.environ.get("TII_SEG_WARPS_HD512", "4")) if hp >= 512 else 4
    path = force_path or ("par" if max_seqlen_q == 1 else "2d")
    if path == "par":
        assert q.shape[0] == num_seqs, "segment-parallel path is decode-only (one query per sequence)"
    T = q.shape[0]
    if path == "2d":
        dummy = torch.empty((16,), dtype=torch.float32, device=q.device)  # same pointer dtypes as mode 1
        _seg_attn[(total_q_blocks, num_kv_heads, 1)](
            out, dummy, dummy, dummy, q, k, v, block_table, seqused_k, per_segment=0, num_seg=1,
            output_stride_0=out.stride(0), output_stride_1=out.stride(1), enable_fp_fusion=False, **common)
        return out
    # Static segment count from the block-table width (= max context the table can address), not from
    # max_seqlen_k: a FULL CUDA graph captured at a short context must still cover every segment on replay.
    # Segments past a sequence's end are empty (m=-inf) and skipped by the merge.
    num_seg = triton.cdiv(triton.cdiv(block_table.shape[1] * block_size, TILE), SEG_TILES)
    m_buf = torch.empty((T, num_query_heads, num_seg), dtype=torch.float32, device=q.device)
    l_buf = torch.empty_like(m_buf)
    acc_buf = torch.empty((T, num_query_heads, num_seg, hp), dtype=torch.float32, device=q.device)
    _seg_attn[(total_q_blocks, num_kv_heads, num_seg)](
        out, m_buf, l_buf, acc_buf, q, k, v, block_table, seqused_k, per_segment=1, num_seg=num_seg,
        output_stride_0=out.stride(0), output_stride_1=out.stride(1), enable_fp_fusion=False, **common)
    _seg_reduce[(T, num_query_heads)](out, m_buf, l_buf, acc_buf, seqused_k, out.stride(0), out.stride(1),
                                      num_query_heads, head_size, hp, num_seg, TILE, SEG_TILES, sw,
                                      enable_fp_fusion=False)
    return out
