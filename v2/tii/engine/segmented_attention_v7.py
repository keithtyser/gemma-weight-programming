"""TII segmented attention v7: fixed absolute key segments, layout-independent per-row arithmetic.

Author: Claude Opus 5.5 (engine track), 2026-11-06. Derived from vLLM 0.19.1 triton_unified_attention (Apache-2.0)
and TII segmented attention v6 (segmented_attention_tii.py).

Numerics (the contract; identical to what any q-block size / head split / mode / batch produces):
  keys of a sequence are cut into absolute 512-key segments [s*512, (s+1)*512); 32-key tiles inside.
  per row r and segment s, from (M=-inf, L=0, ACC=0), for each tile j of s in increasing order:
      S    = scale * (q_r . K_j)                       (bf16 MMA, fp32 accumulate, K-chain over head_dim)
      S    = -inf where masked (causal / sliding window)
      m    = max(M, rowmax(S));  m_safe = m if m > -inf else 0
      P    = exp(S - m_safe)
      l    = bf16(P) @ ones (bf16 MMA, fp32 accumulate; ROWSUM_SEQ=2: L carried as a 16-column MMA accumulator
             chain L = L * a + bf16(P) @ 1, the same instruction chain as ACC, so layout-independent like ACC)
      a    = exp(M - m_safe)
      ACC  = ACC * a + bf16(P) @ V_j                   (bf16 MMA, fp32 accumulate onto ACC*a)
      L    = L * a + l;  M = m
  segments merged in increasing s with _merge (libdevice rn ops), out = ACC / L (IEEE div).
What v7 changes against v6 (so separately compiled binaries can be used for different shapes):
  * the tile row-sum is a sequential fp32 chain in key order (an fp32 "ieee" dot with a ones matrix, which Triton
    lowers to an FMA k-loop), so it does not depend on how Triton distributes the 32 key columns over threads and
    warps (tl.sum's tree does, and that tree changes with BLOCK_M / num_warps);
  * BLOCK_M (rows per CTA) and HEAD_SPLIT (value dims per CTA, a grid dimension) are free parameters: each CTA
    recomputes S over the full head_dim and accumulates only its value-dim slice, so the per-row bytes are the same;
  * per-segment partials are indexed by sequence (decode rows of mixed steps can take the split-KV path too).
Whether the bytes are really independent of BLOCK_M / HEAD_SPLIT / num_warps is an empirical property of the
generated code: tests/test_engine_segattn_v7.py checks it exhaustively on the E4B shapes before any use.
"""

from __future__ import annotations

import os

import torch

from triton.language.extra.cuda import libdevice

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import apply_softcap, cdiv_fn, find_seq_idx

SEG_TILES = 16
TILE = 32
VERSION = f"tii-segattn-v7-seg{SEG_TILES * TILE}"


@triton.jit
def _merge(M, L, ACC, m, l, acc):
    valid = m > float("-inf")
    Mn = tl.where(valid, tl.maximum(M, m), M)
    a = tl.where(valid, libdevice.exp(libdevice.sub_rn(M, Mn)), 1.0)
    b = tl.where(valid, libdevice.exp(libdevice.sub_rn(tl.where(valid, m, 0.0), Mn)), 0.0)
    L2 = tl.where(valid, libdevice.add_rn(libdevice.mul_rn(L, a), libdevice.mul_rn(l, b)), L)
    ACC2 = tl.where(valid[:, None], libdevice.add_rn(libdevice.mul_rn(ACC, a[:, None]),
                                                     libdevice.mul_rn(acc, b[:, None])), ACC)
    return Mn, L2, ACC2


@triton.jit
def _rowsum(P, ONES, ROWSUM_SEQ: tl.constexpr):
    if ROWSUM_SEQ == 1:
        # [BM, 32] @ [32, 16] in fp32 "ieee": FMA k-loop in key order; every column equals the row sum, max picks it
        return tl.max(tl.dot(P, ONES, input_precision="ieee"), axis=1)
    else:
        return tl.sum(P, axis=1)


@triton.jit
def _segment(Q, key_cache_ptr, value_cache_ptr, block_tables_ptr, block_table_offset, kv_head_idx, offs_d, offs_dv,
             offs_t, dim_mask, dimv_mask, row_mask, query_pos, ctx_rows, context_len, max_seq_prefix_len, qpos_lo,
             t0, t1,
             scale, softcap, ONES, stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
             stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE: tl.constexpr,
             TILE_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, DV: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
             USE_SOFTCAP: tl.constexpr, ROWSUM_SEQ: tl.constexpr):
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    LM = tl.zeros([BLOCK_M, 16], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
    for j in range(t0, t1):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)
        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
                    + offs_d[:, None] * stride_k_cache_3 + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
        v_offset = (physical_block_idx[:, None] * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
                    + offs_dv[None, :] * stride_v_cache_3 + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
        K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offset, mask=dimv_mask[None, :] & tile_mask[:, None], other=0.0)
        query_abs_pos = ctx_rows[:, None] + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
        S = scale * tl.dot(Q, K)
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(row_mask[:, None] & seq_mask, S, float("-inf"))
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j_safe = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j_safe[:, None])
        alpha = tl.exp(M - m_j_safe)
        Pb = P.to(V.dtype)
        if ROWSUM_SEQ == 2:
            # L as a 16-column MMA accumulator chain, exactly like acc: LM = LM * alpha + bf16(P) @ 1
            LM = LM * alpha[:, None]
            LM += tl.dot(Pb, ONES)
        else:
            l_j = _rowsum(P, ONES, ROWSUM_SEQ)
            L = L * alpha + l_j
        acc = acc * alpha[:, None]
        M = m_j
        if SLIDING_WINDOW:
            V = tl.where((context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0)
        acc += tl.dot(Pb, V)
    if ROWSUM_SEQ == 2:
        L = tl.max(LM, axis=1)  # all 16 columns are the same chain
    return M, L, acc


@triton.jit
def _tile(j, M, LM, acc, Q, key_cache_ptr, value_cache_ptr, block_tables_ptr, block_table_offset, kv_head_idx, offs_d,
          offs_dv, offs_t, dim_mask, dimv_mask, row_mask, query_pos, ctx_rows, context_len, max_seq_prefix_len,
          qpos_lo, scale, softcap, ONES, stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
          stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE: tl.constexpr,
          TILE_SIZE: tl.constexpr, SLIDING_WINDOW: tl.constexpr, USE_SOFTCAP: tl.constexpr):
    """One 32-key tile of the contract (ROWSUM_SEQ=2), the same expressions as the body of _segment."""
    seq_offset = j * TILE_SIZE + offs_t
    tile_mask = seq_offset < max_seq_prefix_len
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)
    k_offset = (physical_block_idx[None, :] * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3 + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
    v_offset = (physical_block_idx[:, None] * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
                + offs_dv[None, :] * stride_v_cache_3 + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
    K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
    V = tl.load(value_cache_ptr + v_offset, mask=dimv_mask[None, :] & tile_mask[:, None], other=0.0)
    query_abs_pos = ctx_rows[:, None] + query_pos[:, None]
    seq_mask = seq_offset[None, :] <= query_abs_pos
    if SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
    S = scale * tl.dot(Q, K)
    if USE_SOFTCAP:
        S = apply_softcap(S, softcap)
    S = tl.where(row_mask[:, None] & seq_mask, S, float("-inf"))
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j_safe = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j_safe[:, None])
    alpha = tl.exp(M - m_j_safe)
    Pb = P.to(V.dtype)
    LM = LM * alpha[:, None]
    LM += tl.dot(Pb, ONES)
    acc = acc * alpha[:, None]
    if SLIDING_WINDOW:
        V = tl.where((context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0)
    acc += tl.dot(Pb, V)
    return m_j, LM, acc


@triton.jit(do_not_specialize=["num_seg", "num_seqs", "block_table_stride"])
def _seg_attn_v7(output_ptr, m_buf, l_buf, acc_buf, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
                 seq_lens_ptr, scale, softcap, mode: tl.constexpr, num_seg, num_query_heads: tl.constexpr,
                 num_queries_per_kv: tl.constexpr, block_table_stride: tl.int64, query_stride_0: tl.int64,
                 query_stride_1: tl.int64, output_stride_0: tl.int64, output_stride_1: tl.int64,
                 BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,
                 HEAD_SIZE_PADDED: tl.constexpr, DV: tl.constexpr, USE_SOFTCAP: tl.constexpr,
                 SLIDING_WINDOW: tl.constexpr, stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64,
                 stride_k_cache_2: tl.int64, stride_k_cache_3: tl.constexpr, stride_v_cache_0: tl.int64,
                 stride_v_cache_1: tl.int64, stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr,
                 query_start_len_ptr, BLOCK_Q: tl.constexpr, num_seqs: tl.int32, BLOCK_M: tl.constexpr,
                 SEG_TILES: tl.constexpr, ROWSUM_SEQ: tl.constexpr, memb_ptr, cnt_ptr, CASCADE: tl.constexpr,
                 ACC_GMEM: tl.constexpr, FLAT: tl.constexpr):
    """grid (q-blocks [mode 1: sequences], kv-heads * HEAD_SPLIT, Z).
    mode 0: all q-blocks, CTA walks its segments and merges in registers, writes normalized output (Z = 1).
    mode 1: decode q-blocks only (qlen == 1): CTA (q-block, head, s = program_id(2)) writes segment s's raw state
            into the per-sequence buffers for _seg_reduce_v7.
    mode 2: like mode 0 but skips decode q-blocks (mixed steps: their decode rows run modes 1 + reduce).
    CASCADE (mode 1): the CTA of (sequence i, segment s) serves the member list memb[i, s, :cnt] built by
    _seg_groups (cnt = 1 and memb = [i] for a private segment; cnt = 0 when another member leads): rows are
    member x query head, keys/values are read once for all members.
    ACC_GMEM (modes 0/2): the merged accumulator lives in a per-CTA fp32 scratch slice (acc_buf) instead of
    registers, so only one [BLOCK_M, DV] accumulator is live in the tile loop (head_dim 512 prefill otherwise
    spills). Same values, same _merge: identical bits."""
    q_block_global_idx = tl.program_id(0)
    HEAD_SPLIT: tl.constexpr = HEAD_SIZE_PADDED // DV
    kv_head_idx = tl.program_id(1) // HEAD_SPLIT
    dv_idx = tl.program_id(1) % HEAD_SPLIT
    if mode == 1:  # grid axis 0 = sequence; a decode sequence has exactly one q-block
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
    else:
        seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx
    start = tl.load(query_start_len_ptr + seq_idx)
    stop = tl.load(query_start_len_ptr + seq_idx + 1)
    qlen = stop - start
    if q_block_local_idx * BLOCK_Q >= qlen:
        return
    if mode == 1:
        if qlen != 1:
            return
    if mode == 2:
        if qlen == 1:
            return
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_dv = dv_idx * DV + tl.arange(0, DV)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    dim_mask = offs_d < HEAD_SIZE
    dimv_mask = offs_dv < HEAD_SIZE
    row_mask = (query_pos < qlen) & (query_offset_1 < num_query_heads)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - qlen
    ctx_rows = context_len + 0 * offs_m
    member = seq_idx + 0 * offs_m
    if CASCADE:
        if mode == 1:
            gs = seq_idx * num_seg + tl.program_id(2)
            cnt = tl.load(cnt_ptr + gs)
            if cnt == 0:
                return
            mslot = offs_m // num_queries_per_kv
            member = tl.load(memb_ptr + gs * BLOCK_Q + mslot, mask=mslot < cnt, other=seq_idx)
            query_pos = 0 * offs_m
            query_offset_0 = tl.load(query_start_len_ptr + member)
            ctx_rows = tl.load(seq_lens_ptr + member) - 1
            row_mask = (mslot < cnt) & (query_offset_1 < num_query_heads)
    Q = tl.load(query_ptr + query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1
                + offs_d[None, :], mask=dim_mask[None, :] & row_mask[:, None], other=0.0)
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
    if ROWSUM_SEQ == 2:
        ONES = tl.full([TILE_SIZE, 16], 1.0, dtype=tl.bfloat16)
    else:
        ONES = tl.full([TILE_SIZE, 16], 1.0, dtype=tl.float32)
    offs_t = tl.arange(0, TILE_SIZE)
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    ACC = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
    s_lo = tile_start // SEG_TILES
    s_hi = cdiv_fn(tile_end, SEG_TILES)
    if mode == 1:
        sp = tl.program_id(2)
        if (sp < s_lo) | (sp >= s_hi):
            return
        s_lo = sp
        s_hi = sp + 1
    if ACC_GMEM:
        pid_lin = (tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)).to(tl.int64)
        scr_ptrs = acc_buf + (pid_lin * BLOCK_M + offs_m)[:, None] * DV + tl.arange(0, DV)[None, :]
    if FLAT:
        # modes 0/2: one software-pipelined loop over all tiles; a segment's state is merged at its last tile and
        # reset for the next one -- the same per-segment states and the same merge order as the segment loop
        # below, without restarting the pipeline every 16 tiles.
        m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        LM = tl.zeros([BLOCK_M, 16], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
        for j in range(tile_start, tile_end):
            m, LM, acc = _tile(j, m, LM, acc, Q, key_cache_ptr, value_cache_ptr, block_tables_ptr,
                               block_table_offset, kv_head_idx, offs_d, offs_dv, offs_t, dim_mask, dimv_mask,
                               row_mask, query_pos, ctx_rows, context_len, max_seq_prefix_len, qpos_lo, scale,
                               softcap, ONES, stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                               stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE,
                               TILE_SIZE, SLIDING_WINDOW, USE_SOFTCAP)
            if ((j + 1) % SEG_TILES == 0) | (j == tile_end - 1):
                M, L, ACC = _merge(M, L, ACC, m, tl.max(LM, axis=1), acc)
                m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                LM = tl.zeros([BLOCK_M, 16], dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
        s_hi = s_lo  # segment loop below is skipped
    for s in range(s_lo, s_hi):
        t0 = tl.maximum(s * SEG_TILES, tile_start)
        t1 = tl.minimum((s + 1) * SEG_TILES, tile_end)
        m, l, acc = _segment(Q, key_cache_ptr, value_cache_ptr, block_tables_ptr, block_table_offset, kv_head_idx,
                             offs_d, offs_dv, offs_t, dim_mask, dimv_mask, row_mask, query_pos, ctx_rows,
                             context_len,
                             max_seq_prefix_len, qpos_lo, t0, t1, scale, softcap, ONES, stride_k_cache_0,
                             stride_k_cache_1, stride_k_cache_2, stride_k_cache_3, stride_v_cache_0,
                             stride_v_cache_1, stride_v_cache_2, stride_v_cache_3, BLOCK_SIZE, TILE_SIZE, BLOCK_M,
                             DV, SLIDING_WINDOW, USE_SOFTCAP, ROWSUM_SEQ)
        if mode == 1:
            M, L, ACC = m, l, acc
        else:
            if ACC_GMEM:
                ACCp = tl.load(scr_ptrs, mask=tl.full([BLOCK_M, DV], 1, tl.int1) & (s > s_lo), other=0.0)
                M, L, ACCp = _merge(M, L, ACCp, m, l, acc)
                tl.store(scr_ptrs, ACCp)
                tl.debug_barrier()  # scratch is re-read next segment, possibly by other threads of the CTA
            else:
                M, L, ACC = _merge(M, L, ACC, m, l, acc)
    if ACC_GMEM:
        if mode != 1:
            ACC = tl.load(scr_ptrs, mask=tl.full([BLOCK_M, DV], 1, tl.int1) & (s_hi > s_lo), other=0.0)
    if mode == 1:
        # decode q-block: rows are the heads of this kv head (query_pos == 0); index by sequence
        row = (member * num_query_heads + query_offset_1) * num_seg + tl.program_id(2)
        tl.store(m_buf + row, M, mask=row_mask & (dv_idx == 0))
        tl.store(l_buf + row, L, mask=row_mask & (dv_idx == 0))
        tl.store(acc_buf + row[:, None] * HEAD_SIZE_PADDED + offs_dv[None, :], ACC, mask=row_mask[:, None])
    else:
        out = tl.div_rn(ACC, tl.broadcast_to(L[:, None], ACC.shape))
        tl.store(output_ptr + query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1
                 + offs_dv[None, :], out, mask=dimv_mask[None, :] & row_mask[:, None])


@triton.jit(do_not_specialize=["num_seqs", "num_seg", "block_table_stride"])
def _seg_groups(memb_ptr, cnt_ptr, block_tables_ptr, seq_lens_ptr, query_start_len_ptr, num_seqs, num_seg,
                block_table_stride, MAXS: tl.constexpr, G: tl.constexpr, SEG_KEYS: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    """Cascade work lists (decode rows only). Segment s of decode sequence i is SHARED with decode sequence j when
    both contain the whole segment ((s+1)*SEG_KEYS <= seq_len - 1, so every key is causal for the decode row)
    and the physical block holding the segment's last key is the same block: with prefix caching that block's
    hash covers every earlier token, so the two sequences have identical tokens up to it and, the engine being
    batch-invariant, identical K/V bytes in every block of the segment. Members of a shared segment are ranked by
    sequence index and cut into chunks of G; chunk leader = first member of the chunk: cnt[leader, s] = chunk
    size, memb[leader, s, :] = members; other members get cnt = 0. Private segments: cnt = 1, memb = [i]."""
    i = tl.program_id(0)
    oj = tl.arange(0, MAXS)
    jv = oj < num_seqs
    st = tl.load(query_start_len_ptr + oj, mask=jv, other=0)
    en = tl.load(query_start_len_ptr + oj + 1, mask=jv, other=0)
    sl = tl.load(seq_lens_ptr + oj, mask=jv, other=0)
    dec = jv & ((en - st) == 1)
    s = tl.program_id(1)  # grid (num_seqs, num_seg): one program per (sequence, segment)
    lastb = ((s + 1) * SEG_KEYS - 1) // BLOCK_SIZE
    full = dec & ((s + 1) * SEG_KEYS <= sl - 1)
    key = tl.load(block_tables_ptr + oj.to(tl.int64) * block_table_stride + lastb, mask=full, other=-1)
    key_i = tl.sum(tl.where(oj == i, key, 0), axis=0)
    full_i = tl.sum(tl.where((oj == i) & full, 1, 0), axis=0) > 0
    eq = full & (key == key_i) & full_i
    rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    rank_i = tl.sum(tl.where(oj == i, rank, 0), axis=0)
    chunk = rank_i // G
    inchunk = eq & ((rank // G) == chunk)
    leader = tl.min(tl.where(inchunk, oj, MAXS), axis=0)
    base = (i * num_seg + s) * G
    if full_i:
        if leader == i:
            tl.store(cnt_ptr + i * num_seg + s, tl.sum(inchunk.to(tl.int32), axis=0))
            tl.store(memb_ptr + base + (rank - chunk * G), oj, mask=inchunk)
        else:
            tl.store(cnt_ptr + i * num_seg + s, 0)
    else:
        tl.store(cnt_ptr + i * num_seg + s, 1)
        tl.store(memb_ptr + base, i)

@triton.jit(do_not_specialize=["output_stride_0", "output_stride_1", "num_seqs"])
def _seg_reduce_v7(output_ptr, m_buf, l_buf, acc_buf, seq_lens_ptr, query_start_len_ptr, num_seqs,
                   output_stride_0: tl.int64, output_stride_1: tl.int64, num_query_heads: tl.constexpr,
                   HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, NUM_SEG: tl.constexpr,
                   TILE_SIZE: tl.constexpr, SEG_TILES: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    """grid (num_seqs, num_query_heads): merges the segments of the decode row of sequence i (skips qlen != 1)."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    start = tl.load(query_start_len_ptr + i)
    stop = tl.load(query_start_len_ptr + i + 1)
    if stop - start != 1:
        return
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    base = (i * num_query_heads + h) * NUM_SEG
    seq_len = tl.load(seq_lens_ptr + i)
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
    out = tl.div_rn(ACC, tl.broadcast_to(L[:, None], ACC.shape))
    tl.store(output_ptr + start * output_stride_0 + h * output_stride_1 + offs_d[None, :], out,
             mask=(offs_d < HEAD_SIZE)[None, :])


def _cfg(hp: int, kind: str):
    """(BLOCK_M, HEAD_SPLIT, num_warps[, num_stages[, acc_gmem[, flat]]]) per head size and launch kind;
    env TII_SEG7_<KIND>_<HP>=bm,split,warps[,stages[,gmem[,flat]]]."""
    env = os.environ.get(f"TII_SEG7_{kind.upper()}_{hp}")
    if env:
        return tuple(int(x) for x in env.split(","))
    # fastest measured on the rental RTX PRO 6000 (bench_segattn_v7, E4B shapes, 48 seqs ~9.6k ctx / 2048-token
    # chunk at 9k): dec 512 (16,1,4,2) 1148 us vs 2D 1548; dec 256 (16,1,4) 37.5 vs 39.2; pre 512 flat (16,1,4,2,0,1)
    # 3963 vs 2D 4058; pre 256 (32,1,4) 115 vs 225. Register use: tests/segattn_v7_regprobe.py (CPU, ptxas -v).
    if kind == "decc":  # cascade decode: 32 rows = 8 group members x 4 query heads per CTA (G = 8)
        return (32, 1, 8, 2)
    if kind == "dec":
        return (16, 1, 4, 2) if hp >= 512 else (16, 1, 4)
    return (16, 1, 4, 2, 0, 1) if hp >= 512 else (32, 1, 4)


def unified_attention_tii(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                          window_size, block_table, softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None,
                          num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None,
                          softmax_segm_expsum=None, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
                          mm_prefix_range=None, use_alibi_sqrt=False, force_path: str | None = None,
                          cfg_pre: tuple | None = None, cfg_dec: tuple | None = None, rowsum_seq: int = 2,
                          cascade: bool | None = None):
    """Drop-in for vllm unified_attention. force_path: None (auto: decode-only -> 'par', else 'mixed'),
    '2d' (mode 0 for everything), 'par' (decode-only split-KV), 'mixed' (mode 2 + decode rows split-KV)."""
    assert causal and q_descale is None and alibi_slopes is None and qq_bias is None and sinks is None
    assert mm_prefix_range is None and output_scale is None and q.dtype in (torch.bfloat16, torch.float16)
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    nqpk = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    hp = triton.next_power_of_2(head_size)
    sw = 1 + window_size[0] if window_size[0] >= 0 else 0

    if cascade is None:
        cascade = os.environ.get("TII_SEG7_CASCADE", "0") == "1"
    cascade = bool(cascade) and sw == 0  # sliding layers: their keys are recent, never a shared prefix
    lists = {"memb": torch.empty((1,), dtype=torch.int32, device=q.device)}
    lists["cnt"] = lists["memb"]

    def launch(mode, cfg, z, m_buf, l_buf, acc_buf, num_seg):
        bm, hs, nw = cfg[:3]
        ns = cfg[3] if len(cfg) > 3 else 1  # software pipelining depth: changes scheduling, not arithmetic
        gmem = bool(cfg[4]) if len(cfg) > 4 and mode != 1 else False
        flat = bool(cfg[5]) if len(cfg) > 5 and mode != 1 else False
        assert not (flat and (gmem or rowsum_seq != 2)), "FLAT needs the MMA row-sum chain and register ACC"
        bm = max(bm, triton.next_power_of_2(nqpk))
        bq = bm // nqpk
        total_q_blocks = num_seqs if mode == 1 else q.shape[0] // bq + num_seqs
        if gmem:
            acc_buf = torch.empty((total_q_blocks * num_kv_heads * hs * bm * (hp // hs),), dtype=torch.float32,
                                  device=q.device)
        _seg_attn_v7[(total_q_blocks, num_kv_heads * hs, z)](
            out, m_buf, l_buf, acc_buf, q, k, v, block_table, seqused_k, softmax_scale, softcap, mode, num_seg,
            num_query_heads=num_query_heads, num_queries_per_kv=nqpk, block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0),
            output_stride_1=out.stride(1), BLOCK_SIZE=block_size, TILE_SIZE=TILE, HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=hp, DV=hp // hs, USE_SOFTCAP=(softcap > 0), SLIDING_WINDOW=sw,
            stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=bq, num_seqs=num_seqs, BLOCK_M=bm, SEG_TILES=SEG_TILES, ROWSUM_SEQ=int(rowsum_seq),
            memb_ptr=lists["memb"], cnt_ptr=lists["cnt"], CASCADE=(cascade and mode == 1), ACC_GMEM=gmem, FLAT=flat,
            num_warps=nw,
            num_stages=ns, enable_fp_fusion=False)

    cpre = cfg_pre or _cfg(hp, "pre")
    cdec = cfg_dec or _cfg(hp, "decc" if cascade else "dec")
    path = force_path or ("par" if max_seqlen_q == 1 else "mixed")
    dummy = torch.empty((16,), dtype=torch.float32, device=q.device)
    if path == "2d":
        launch(0, cpre, 1, dummy, dummy, dummy, 1)
        return out
    if path == "mixed":
        launch(2, cpre, 1, dummy, dummy, dummy, 1)
    num_seg = triton.cdiv(triton.cdiv(block_table.shape[1] * block_size, TILE), SEG_TILES)
    m_buf = torch.empty((num_seqs, num_query_heads, num_seg), dtype=torch.float32, device=q.device)
    l_buf = torch.empty_like(m_buf)
    acc_buf = torch.empty((num_seqs, num_query_heads, num_seg, hp), dtype=torch.float32, device=q.device)
    if cascade:
        g = max(cdec[0], triton.next_power_of_2(nqpk)) // nqpk
        lists["memb"] = torch.empty((num_seqs, num_seg, g), dtype=torch.int32, device=q.device)
        lists["cnt"] = torch.empty((num_seqs, num_seg), dtype=torch.int32, device=q.device)
        _seg_groups[(num_seqs, num_seg)](lists["memb"], lists["cnt"], block_table, seqused_k, cu_seqlens_q, num_seqs,
                                 num_seg, block_table.stride(0), MAXS=max(16, triton.next_power_of_2(num_seqs)),
                                 G=g, SEG_KEYS=SEG_TILES * TILE, BLOCK_SIZE=block_size)
    launch(1, cdec, num_seg, m_buf, l_buf, acc_buf, num_seg)
    _seg_reduce_v7[(num_seqs, num_query_heads)](out, m_buf, l_buf, acc_buf, seqused_k, cu_seqlens_q, num_seqs,
                                                out.stride(0), out.stride(1), num_query_heads, head_size, hp,
                                                num_seg, TILE, SEG_TILES, sw, enable_fp_fusion=False)
    return out
