"""TII-W4: batch-invariant W4A16 GEMM = new small-M kernel (M <= 64) + TII-Marlin (M > 64). Kernel track, 2026-11-05.

Bitwise identical to tii_marlin_op.gemm for every M by construction (see tii_w4.cu: the four TII-Marlin k-warp
chains run in a 4-CTA cluster and are combined as c0 + (c1 + (c3 + c2))); verified by tests/test_kernel_tii_w4.py.
Same weights as TII-Marlin (vLLM Marlin repack via tii_marlin_op.prepare_weights), same call signature.
"""

from __future__ import annotations

import os

import torch

from v2.tii.engine import tii_marlin_op

KERNEL_VERSION = "tii-w4-v1 (small-M: 4 chain CTAs + L2 last-arriver combine; M>64 or table -1: tii-marlin-k128n64L128t4s4)"
_HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.environ.get("TII_W4_SO", os.path.join(_HERE, "build", "tii_w4_v1", "tii_w4_v1.so"))
# TII-Marlin rebuilt with thread_m_blocks 4 for 48 < M <= 64 (one weight pass; TII_SMALL_TM4=1 in prepare_tii_marlin.py).
# Bitwise identical to tii-marlin-k128n64L128t3 (same k-warp order; checked in the op test).
MARLIN_T4S4_SO = os.path.join(_HERE, "..", "engine", "tii_marlin", "build_k128n64L128t4s4", "tii_marlin_k128n64L128t4s4.so")
SMALL_M_MAX = 64
_loaded = False
_warm: set[int] = set()


def load() -> None:
    global _loaded
    if not _loaded:
        if "TII_MARLIN_SO" not in os.environ:
            tii_marlin_op.SO_PATH = os.path.abspath(MARLIN_T4S4_SO)
        tii_marlin_op.load()
        torch.ops.load_library(SO_PATH)
        _loaded = True


def warmup(device: torch.device) -> None:
    """Run every tabled small-M config once (before CUDA-graph capture): sets kernel attributes, allocates the
    L2 exchange workspace and its zeroed counters outside any capture."""
    load()
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _warm:
        return
    with torch.cuda.device(idx):
        for (n, k, m), c in sorted(CFG.items()):
            if c < 0:
                continue
            q = torch.zeros(k // 16, n * 2, dtype=torch.int32, device=device)
            s = torch.zeros(k // 32, n, dtype=torch.bfloat16, device=device)
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
            torch.ops.tii_w4.gemm(x, q, s, n, k, c, 0)
        torch.cuda.synchronize(device)
    _warm.add(idx)


def enc(nt: int = 1, kt: int = 1, wh: int = 1, sm: int = 0, cl: int = 0, stages: int = 4) -> int:
    return nt | kt << 4 | wh << 8 | sm << 12 | cl << 16 | stages << 20


# (N, K, ceil16(M)) -> cfg code or -1 (= TII-Marlin); schedule only (the arithmetic of every output element is the same for all codes,
# tested); filled from the tuning sweep (results/tune_*.jsonl) by tii_w4_table.py.
CFG: dict[tuple[int, int, int], int] = {}
try:
    from v2.tii.kernel.tii_w4_table import TABLE as _T
    CFG.update(_T)
except ImportError:
    pass


def cfg_for(n: int, k: int, m: int) -> int:
    env = os.environ.get("TII_W4_CFG")
    if env:
        return int(env)
    c = CFG.get((n, k, max(16, -(-m // 16) * 16)))
    if c is None:
        c = enc(1, 2 if (k // 128) % 2 == 0 else 1, 1, 0, 0)
    return c


def gemm(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    load()
    x2 = x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    c = cfg_for(n, k, m) if 1 <= m <= SMALL_M_MAX else -1
    if c >= 0:
        out = torch.ops.tii_w4.gemm(x2, q, s, n, k, c, 0)
    else:
        out = tii_marlin_op.gemm(x2, q, s, n, k)  # TII-Marlin, or TII-W4L under TII_LARGEM=w4l (same bits)
    return out.reshape(*x.shape[:-1], n)


def gemm_small(x2: torch.Tensor, q, s, n: int, k: int, cfg: int, plant: int = 0) -> torch.Tensor:
    load()
    return torch.ops.tii_w4.gemm(x2, q, s, n, k, cfg, plant)
