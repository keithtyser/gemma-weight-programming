"""Fast attention backward for the aligned trainer: FlexAttention recompute (bf16 inputs, fp32 softmax/accumulate).

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: Backward of the visible vLLM attention (forward bytes are untouched). Gradient contract (case A, same as the
  fp32 reference it replaces): recompute the forward with the training kernel inside the backward, so O, LSE and P
  come from ONE forward (FlexAttention, fp32 online softmax, bf16 Q/K/V tiles), then FlexAttention's backward
  (FA2-style: P and dS cast to bf16 for the dV/dQ/dK GEMMs, fp32 accumulation, deterministic per-block loops).
  Masks: per-row visibility from AttnPlan.row_node/row_pos/anc (packed sequences or a call tree) plus the
  sliding window (pos_q - pos_k < W).
  GQA via enable_gqa (dK/dV summed over the 4 query heads of each kv head inside the kernel). Tile sizes are pinned
  small enough for sm120's 101 KB shared memory at head_dim 512.
  Callers: ops.AttentionFn when ops.ATTN_BWD == "flex". Depends on torch 2.10 flex_attention (compiled once per shape
  family; dynamic=True to avoid recompiles across lengths).
SRP/DRY check: Pass - replaces ops.reference_attention_grads only on the fast path; the fp32 reference stays as oracle.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_COMPILED = {}

# Every padded length (multiple of PAD) and every ancestry-matrix size is a separate compiled specialization; the
# default dynamo limit (8) makes flex silently fall back to its eager path (slow, and it crashed at T~20k). Allow
# enough specializations for lengths up to 128k and power-of-two ancestry sizes.
import torch._dynamo.config as _dcfg

for _name in ("cache_size_limit", "recompile_limit"):
    if hasattr(_dcfg, _name):
        setattr(_dcfg, _name, max(getattr(_dcfg, _name), 512))
if hasattr(_dcfg, "accumulated_cache_size_limit"):
    _dcfg.accumulated_cache_size_limit = max(_dcfg.accumulated_cache_size_limit, 4096)
if hasattr(_dcfg, "accumulated_recompile_limit"):
    _dcfg.accumulated_recompile_limit = max(_dcfg.accumulated_recompile_limit, 4096)


def _kernel_options(d: int) -> dict:
    if d >= 512:
        return {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16,
                "num_stages": 1, "num_warps": 4}
    return {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32,
            "num_stages": 1, "num_warps": 4}


def _flex():
    if "f" not in _COMPILED:
        _COMPILED["f"] = torch.compile(flex_attention, dynamic=False)
    return _COMPILED["f"]


_MASKS: dict = {}
PAD = 2048  # pad T to a multiple of this (bounded set of compiled shapes); pad rows are their own masked document


def flex_attention_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gout: torch.Tensor,
                         plan, scale: float, window: int | None):
    """q [T,Hq,D], k/v [T,Hkv,D] bf16 packed; plan = ops.AttnPlan (contiguous sequences or a tree). Key row visible
    from query row iff anc[node[q], node[k]] and pos[k] <= pos[q] (and pos[q] - pos[k] < window). Returns bf16 grads."""
    T, hq, d = q.shape
    Tp = ((T + PAD - 1) // PAD) * PAD
    hit = _MASKS.get((id(plan), Tp, window))
    if hit is not None and hit[0] is plan:
        bm = hit[1]
    else:  # one block mask per (plan, window), shared by every layer of the forward/backward
        n = plan.anc.shape[0]
        node = torch.full((Tp,), n, dtype=torch.int32, device=q.device)  # pad rows: their own extra node
        node[:T] = plan.row_node
        pos = torch.zeros(Tp, dtype=torch.int32, device=q.device)
        pos[:T] = plan.row_pos
        na = 1 << max(3, n.bit_length())  # power-of-two size (> n): few distinct shapes to compile
        anc = torch.zeros(na, na, dtype=torch.bool, device=q.device)
        anc[:n, :n] = plan.anc
        anc[n, n] = True

        if window is None:
            def mask_mod(b, h, qi, ki):
                return anc[node[qi], node[ki]] & (pos[ki] <= pos[qi])
        else:
            def mask_mod(b, h, qi, ki):
                return anc[node[qi], node[ki]] & (pos[ki] <= pos[qi]) & (pos[qi] - pos[ki] < window)

        bm = create_block_mask(mask_mod, None, None, Tp, Tp, device=q.device, BLOCK_SIZE=128, _compile=True)
        if len(_MASKS) > 8:
            _MASKS.clear()
        _MASKS[(id(plan), Tp, window)] = (plan, bm)

    def pad(t):
        if Tp == T:
            return t
        return torch.cat([t, t.new_zeros(Tp - T, *t.shape[1:])], 0)

    with torch.enable_grad():
        qq = pad(q.detach()).permute(1, 0, 2).unsqueeze(0).requires_grad_(True)  # [1, Hq, Tp, D]
        kk = pad(k.detach()).permute(1, 0, 2).unsqueeze(0).requires_grad_(True)
        vv = pad(v.detach()).permute(1, 0, 2).unsqueeze(0).requires_grad_(True)
        out = _flex()(qq, kk, vv, block_mask=bm, scale=scale, enable_gqa=True, kernel_options=_kernel_options(d))
        dq, dk, dv = torch.autograd.grad(out, (qq, kk, vv), pad(gout).permute(1, 0, 2).unsqueeze(0))
    return (dq[0, :, :T].permute(1, 0, 2).contiguous(), dk[0, :, :T].permute(1, 0, 2).contiguous(),
            dv[0, :, :T].permute(1, 0, 2).contiguous())
