"""lm_head GEMM for the TII aligned engine: vLLM's batch-invariant persistent matmul kernel with an M-dependent
tile table (engine track, 2026-11-06).

vLLM's matmul_persistent uses one bf16 config (128x128x64, 8 warps, 3 stages) for every M; at decode sizes (M<=64)
that streams the 1.34 GB tied lm_head at ~0.5-0.9 TB/s. The output bits of matmul_kernel_persistent do not depend on
the tile config (per-element K chain of fused MMAs in fixed BLOCK_K-independent order: verified bitwise across a config
sweep at M=8/48/64/128 and in tests/test_engine_lm_head.py for 21 M values), so the table below changes speed only: trainer calls of vLLM's
matmul_persistent produce the same bytes. Installed by vllm_plugin P2c (TII_LMHEAD=table, the default) by rebinding batch_invariant.matmul_persistent.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers import batch_invariant as bi
from vllm.triton_utils import triton


def _cfg(M: int):
    if M <= 16:
        return 16, 64, 32, 4, 4
    if M <= 64:
        return 64, 128, 64, 8, 3
    return 128, 128, 64, 8, 3  # vLLM default


_ORIG = bi.matmul_persistent
LM_HEAD_N = 262144  # Gemma 4 vocab; the table is applied to the lm_head only, every other matmul keeps vLLM's call


def tii_matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return _ORIG(a, b, bias)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0:
        return c
    bm, bn, bk, nw, ns = _cfg(M)
    sms = bi.num_compute_units(a.device.index)
    grid = (min(sms, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)
    bi.matmul_kernel_persistent[grid](
        a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        NUM_SMS=sms, A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8,
        num_warps=nw, num_stages=ns)
    return c


def routed_matmul_persistent(a, b, bias=None):
    if b.ndim == 2 and b.shape[1] == LM_HEAD_N:
        return tii_matmul_persistent(a, b, bias)
    return _ORIG(a, b, bias)


def install() -> None:
    """Route vLLM's batch_invariant.matmul_persistent (called by its aten mm/matmul/linear overrides) through the
    table for N == 262144. Output bytes are unchanged (tests/test_engine_lm_head.py)."""
    bi.matmul_persistent = routed_matmul_persistent
