"""Row-parallel log_softmax with the SAME bytes as vLLM's batch-invariant _log_softmax_kernel.

Author: Claude Opus 5.5 (engine track), 2026-11-06.

vLLM 0.19.1 batch_invariant._log_softmax_kernel runs one program per row: max over 1024-wide chunks (exact, order
free), sum_exp = ((0 + s_0) + s_1) + ... + s_{C-1} with s_k = tl.sum(where(mask, exp(x - max), 0)) of chunk k, then
out = (x - max) - log(sum_exp). At decode (48 rows x 262,144 vocab) that is 48 CTAs walking 256 chunks each, ~280 us.
Here the same quantities are computed by (rows x chunks) programs:
  K1 chunk max m_k;  K2 row max = max_k m_k (exact) and s_k with the identical tl.sum (BLOCK_SIZE 1024, 4 warps);
  K3 every program re-adds s_0..s_{C-1} sequentially from 0.0 in the same order, takes the same tl.log, and writes
  its chunk. Bit-identical by construction (tests/test_engine_log_softmax.py checks it against the vLLM kernel).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

BLOCK = 1024  # = vLLM batch_invariant.log_softmax BLOCK_SIZE (defines the chunk tl.sum tree)
VERSION = "tii-logsoftmax-par-v1"


@triton.jit
def _ls_chunk_max(x_ptr, cmax_ptr, x_stride, n_cols, n_chunks, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(x_ptr + row * x_stride + col_idx, mask=mask, other=-float("inf"))
    tl.store(cmax_ptr + row * n_chunks + k, tl.max(vals))


@triton.jit
def _ls_chunk_sum(x_ptr, cmax_ptr, csum_ptr, x_stride, n_cols, n_chunks, BLOCK_SIZE: tl.constexpr,
                  CPAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    ck = tl.arange(0, CPAD)
    max_val = tl.max(tl.load(cmax_ptr + row * n_chunks + ck, mask=ck < n_chunks, other=-float("inf")))
    col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(x_ptr + row * x_stride + col_idx, mask=mask, other=0.0)
    exp_vals = tl.exp(vals - max_val)
    tl.store(csum_ptr + row * n_chunks + k, tl.sum(tl.where(mask, exp_vals, 0.0)))


@triton.jit
def _ls_write(x_ptr, out_ptr, cmax_ptr, csum_ptr, x_stride, out_stride, n_cols, n_chunks, BLOCK_SIZE: tl.constexpr,
              CPAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    ck = tl.arange(0, CPAD)
    max_val = tl.max(tl.load(cmax_ptr + row * n_chunks + ck, mask=ck < n_chunks, other=-float("inf")))
    sum_exp = 0.0
    for j in range(0, n_chunks):
        sum_exp += tl.load(csum_ptr + row * n_chunks + j)
    log_sum_exp = tl.log(sum_exp)
    col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(x_ptr + row * x_stride + col_idx, mask=mask)
    output = vals - max_val - log_sum_exp
    tl.store(out_ptr + row * out_stride + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in for vllm batch_invariant.log_softmax (same semantics, same bytes)."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    original_shape = input.shape
    x = input.reshape(-1, input.shape[-1]).contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0:
        return out.reshape(original_shape)
    n_chunks = triton.cdiv(n_cols, BLOCK)
    cpad = max(16, triton.next_power_of_2(n_chunks))
    cmax = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=x.device)
    csum = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=x.device)
    grid = (n_rows, n_chunks)
    _ls_chunk_max[grid](x, cmax, x.stride(0), n_cols, n_chunks, BLOCK_SIZE=BLOCK)
    _ls_chunk_sum[grid](x, cmax, csum, x.stride(0), n_cols, n_chunks, BLOCK_SIZE=BLOCK, CPAD=cpad)
    _ls_write[grid](x, out, cmax, csum, x.stride(0), out.stride(0), n_cols, n_chunks, BLOCK_SIZE=BLOCK, CPAD=cpad)
    return out.reshape(original_shape)
