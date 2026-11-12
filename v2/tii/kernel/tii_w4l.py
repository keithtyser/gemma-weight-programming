"""TII-W4L: large-M batch-invariant W4A16 GEMM, bitwise TII-Marlin (tii-marlin-k128n64L128t3) for every M.
Kernel track, 2026-11-06.

Production configs use the chain-sequential kernel (tii_w4l.cu, TII-W4C): the four TII-Marlin k-warp chains of an
output tile run one after another (3, 2, 1, 0) with a running fp32 sum, run = c3; run = c2 + run; run = c1 + run;
out = bf16(c0 + run), which is bitwise c0 + (c1 + (c3 + c2)) (IEEE addition is commutative). Same Marlin weight
and scale bytes (tii_marlin_op.prepare_weights), same dequant / scale / mma code. Verified bitwise by
tests/test_kernel_tii_w4l.py. Schedule configs (tile shape, stages, tile order, grid) never change an output bit.
"""

from __future__ import annotations

import os

import torch

KERNEL_VERSION = "tii-w4l-v1 (chain-sequential, persistent, 64x256 CTA / 64x32 warp tiles, overlapped stage boundary; bitwise tii-marlin-k128n64L128t3)"
_HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.environ.get("TII_W4L_SO", os.path.join(_HERE, "build", "tii_w4l_v1", "tii_w4l_v1.so"))
_loaded = False


def load() -> None:
    global _loaded
    if not _loaded:
        torch.ops.load_library(SO_PATH)
        _loaded = True


def enc(shape: int, t: int, stages: int, order: int = 1, grid: int = 0) -> int:
    """shape: CTA tile / warp tile (tii_w4l.cu tii_w4l_gemm; 22 = 64x256 CTA of 1x8 warps of 64x32); t = 128-k tiles
    per pipeline stage (1, 2, 4; (K/128) % t == 0); order 0 = m-fastest, 1 = n-fastest tile walk; grid 0 = #SMs."""
    return shape | t << 8 | stages << 12 | order << 16 | grid << 20


DEFAULT = enc(22, 4, 2, 1)
# (N, K) -> [(max_M, cfg), ...] (first entry with M <= max_M); from the rental A/B (b7.json, M 2096 / 8192, 06-Nov).
# Performance only: every cfg gives the same bits (tests/test_kernel_tii_w4l.py checks all of ALL_CFGS).
_BIG = 1 << 30
CFG: dict[tuple[int, int], list[tuple[int, int]]] = {
    (3072, 2560): [(_BIG, enc(22, 4, 2, 1))],                            # qkv sliding
    (6144, 2560): [(_BIG, enc(22, 4, 2, 0))],                            # qkv global
    (2560, 2048): [(_BIG, enc(22, 4, 2, 1))],                            # o sliding
    (2560, 4096): [(_BIG, enc(22, 4, 2, 1))],                            # o global
    (20480, 2560): [(_BIG, enc(22, 4, 2, 0))],                           # gate_up
    (2560, 10240): [(_BIG, enc(22, 4, 2, 1))],                           # down
    (256, 2560): [(4096, enc(24, 4, 3, 1)), (_BIG, enc(22, 4, 2, 1))],   # per_layer_input_gate
    (2560, 256): [(_BIG, enc(22, 2, 4, 1))],                             # per_layer_projection (K/128 = 2)
    (10752, 2560): [(_BIG, enc(22, 4, 2, 0))],                           # per_layer_model_projection
}
# rows below MIN_M stay on TII-Marlin / TII-W4 small-M (same bits; chosen for speed)
MIN_M = int(os.environ.get("TII_W4L_MIN_M", "65"))
ALL_CFGS = [enc(22, 4, 2, 1), enc(22, 2, 2, 1), enc(22, 2, 4, 1), enc(22, 1, 4, 1), enc(22, 4, 2, 0),
            enc(22, 4, 2, 1, 100), enc(16, 4, 2, 1), enc(16, 2, 4, 1), enc(17, 4, 2, 1), enc(18, 4, 3, 1),
            enc(21, 2, 4, 1), enc(23, 4, 2, 1), enc(23, 2, 3, 1), enc(24, 4, 3, 1), enc(24, 2, 4, 1)]


# Per-shape smallest M routed to TII-W4L (below it TII-Marlin is faster: W4L 64x256 tiles leave too few CTAs at
# small M; qkv sliding at M ~2096 loses to wave quantization). From the rental crossover run (job5, 06-Nov,
# M = 65 / 96 / 128 / 300 / 828 / 2096 / 8192 / 16384). Speed only: both kernels give the same bits for every M.
MIN_M_SHAPE: dict[tuple[int, int], int] = {
    (3072, 2560): 4096,   # qkv sliding
    (6144, 2560): 512,    # qkv global
    (2560, 2048): 1024,   # o sliding
    (2560, 4096): 1024,   # o global
    (20480, 2560): 600,   # gate_up
    (2560, 10240): 1024,  # down
    (256, 2560): 2048,    # per_layer_input_gate
    (2560, 256): 600,     # per_layer_projection
    (10752, 2560): 1024,  # per_layer_model_projection
}


def use(n: int, k: int, m: int) -> bool:
    if "TII_W4L_MIN_M" in os.environ:  # override (tests, A/B)
        return m >= MIN_M
    return m >= max(MIN_M, MIN_M_SHAPE.get((n, k), 1024))


_warm: set[tuple[int, int, int]] = set()


def warmup(n: int, k: int, device: torch.device) -> None:
    """Run the production cfg once for (N, K) outside any CUDA-graph capture (sets the kernel's shared-memory
    attribute and occupancy query on first use)."""
    load()
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if (n, k, idx) in _warm:
        return
    with torch.cuda.device(idx):
        q = torch.zeros(k // 16, n * 2, dtype=torch.int32, device=device)
        s = torch.zeros(k // 32, n, dtype=torch.bfloat16, device=device)
        x = torch.zeros(MIN_M, k, dtype=torch.bfloat16, device=device)
        for c in {cfg_for(n, k, MIN_M)} | {c for _, c in CFG.get((n, k), ())}:
            torch.ops.tii_w4l.gemm(x, q, s, n, k, c, 0)
        torch.cuda.synchronize(device)
    _warm.add((n, k, idx))


def cfg_for(n: int, k: int, m: int) -> int:
    env = os.environ.get("TII_W4L_CFG")
    if env:
        return int(env, 0)
    for max_m, c in CFG.get((n, k), ()):
        if m <= max_m:
            return c
    return DEFAULT if (k // 128) % 4 == 0 else enc(22, 2, 4, 1)


def gemm_cfg(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int, k: int, cfg: int,
             plant: int = 0) -> torch.Tensor:
    load()
    x2 = x.reshape(-1, x.shape[-1])
    return torch.ops.tii_w4l.gemm(x2, q, s, n, k, cfg, plant).reshape(*x.shape[:-1], n)


# planted control for engine-level gates ONLY (TII_W4L_PLANT=1: chains combined ((c0+c1)+c2)+c3, must FAIL C3)
PLANT = int(os.environ.get("TII_W4L_PLANT", "0"))


def gemm(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    load()
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.ops.tii_w4l.gemm(x2, q, s, n, k, cfg_for(n, k, x2.shape[0]), PLANT)
    return out.reshape(*x.shape[:-1], n)
