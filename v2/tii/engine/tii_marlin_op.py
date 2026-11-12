"""TII batch-invariant Marlin W4A16 op (avenue A, kernel version tii-marlin-k128n64L128t3). Engine track, 2026-11-05.

The extension is vLLM v0.19.1's Marlin with the host scheduler patched (v2/tii/engine/tii_marlin/prepare_tii_marlin.py):
one k-family thread config (thread_k 128, 4 k-warps): (128,64,128 threads) for M chunks < 128 rows and
(128,128,256 threads) for >= 128, thread_m_blocks <= 3, no 8-row MMA path, no K-striping (one CTA per output tile,
full K), no atomics. Every output element therefore has the same reduction order for every M; verified bitwise
on all E4B shapes (tests/test_engine_tii_marlin.py). Weights use vLLM's own Marlin repack, so the engine and the
trainer can share bytes: prepare_weights(weight_packed, weight_scale) on the compressed-tensors tensors.
"""

from __future__ import annotations

import os

import torch

KERNEL_VERSION = "tii-marlin-k128n64L128t3"
SO_PATH = os.environ.get(
    "TII_MARLIN_SO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tii_marlin", "build_k128n64L128t3",
                 "tii_marlin_k128n64L128t3.so"))
_loaded = False
_ws: dict[int, torch.Tensor] = {}


def load() -> None:
    global _loaded
    if not _loaded:
        torch.ops.load_library(SO_PATH)
        _loaded = True


def workspace(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _ws:
        sms = torch.cuda.get_device_properties(idx).multi_processor_count
        _ws[idx] = torch.zeros(sms * 4, dtype=torch.int32, device=device)  # never touched (slice_count == 1)
    return _ws[idx]


def prepare_weights(weight_packed: torch.Tensor, weight_scale: torch.Tensor, group_size: int = 32):
    """compressed-tensors [N, K//8] int32 + [N, K//G] bf16 -> Marlin layout (vLLM's repack, byte-preserving)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (marlin_make_empty_g_idx,
                                                                            marlin_permute_scales)

    n, k8 = weight_packed.shape
    k = k8 * 8
    dev = weight_packed.device
    q = ops.gptq_marlin_repack(weight_packed.t().contiguous(), perm=marlin_make_empty_g_idx(dev), size_k=k, size_n=n,
                               num_bits=4, is_a_8bit=False)
    s = marlin_permute_scales(weight_scale.t().contiguous(), size_k=k, size_n=n, group_size=group_size,
                              is_a_8bit=False)
    return q, s, n, k


# TII_LARGEM=w4l (kernel track, 2026-11-06): rows >= tii_w4l.MIN_M go to the TII-W4L large-M kernel
# (v2/tii/kernel/tii_w4l.py), bitwise identical to this op for every M (tests/test_kernel_tii_w4l.py). Both the
# learner (trainer/ops.py FrozenLinear "marlin_engine") and the engine M > 64 path call this function.
LARGEM = os.environ.get("TII_LARGEM", "")
_w4l = None


def gemm(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    global _w4l
    load()
    x2 = x.reshape(-1, x.shape[-1])
    if LARGEM == "w4l":
        if _w4l is None:
            from v2.tii.kernel import tii_w4l as _m
            _w4l = _m
        if _w4l.use(n, k, x2.shape[0]):
            return _w4l.gemm(x2, q, s, n, k).reshape(*x.shape[:-1], n)
    out = torch.ops.tii_marlin.gemm(x2, q, s, workspace(x.device), n, k)
    return out.reshape(*x.shape[:-1], n)
