"""Memory-lean MLP block for the aligned trainer: one autograd Function, no GEMM recompute, gate_up offloaded to host.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: The MLP half of Gemma4DecoderLayer (pre_feedforward_layernorm -> gate_up (+LoRA) -> GeluAndMul -> down (+LoRA)
  -> post_feedforward_layernorm). FORWARD: the same visible vLLM calls, in the same order, as the unfused ops path
  (ops.visible_rms_norm, FrozenLinear.visible, ops.visible_lora, ops.visible_gelu_and_mul), so its bytes are the
  engine's bytes (C5). MEMORY: autograd keeps only the block input x (bf16 [T, 2560]), the gate_up output gu
  ([T, 20480], optionally offloaded to pinned host memory and prefetched back during backward), the down output dn
  ([T, 2560]) and the two small LoRA shrink buffers - instead of x, norm output, gu, GELU output and dn. BACKWARD
  (same gradient contract as the per-op Functions): post-norm VJP from dn; act = GeluAndMul(gu) and the norm output
  y = RMSNorm(x) are recomputed with the visible kernels (elementwise, cheap; no GEMM is recomputed); down LoRA and
  base dX (INT4 dequant-scratch GEMM); GELU VJP; gate_up LoRA and base dX; pre-norm VJP.
  Callers: AlignedGemma4._mlp_block when mlp_fused is on. Depends on ops.py, bwd_kernels.py.
SRP/DRY check: Pass - fuses existing ops' forward calls and backward cores; no new numerics.
"""

from __future__ import annotations

from typing import Any

import torch

from v2.tii.trainer import ops


class HostOffloader:
    """Async device->pinned-host copies of large saved tensors during forward, prefetched back in reverse order during
    backward (one copy stream). Pinned buffers are pooled by size and reused across steps."""

    def __init__(self) -> None:
        import os

        # Disabled by default: on 05-Nov-2026 an unbounded per-shape pinned pool exhausted host RAM (OOM kill) on the
        # shared host. Needs a hard-capped arena before any further use.
        if os.environ.get("TII_ALLOW_HOST_OFFLOAD") != "1":
            raise RuntimeError("host offload disabled (unbounded pinned memory incident); set TII_ALLOW_HOST_OFFLOAD=1 only with a capped arena")
        self.stream = torch.cuda.Stream()
        self.pool: dict[tuple, list[torch.Tensor]] = {}
        self.items: list[dict] = []
        self.bytes_moved = 0

    def reset(self) -> None:
        for it in self.items:
            if it.get("host") is not None:
                self.pool.setdefault((it["shape"], it["dtype"]), []).append(it["host"])
        self.items = []

    def pack(self, t: torch.Tensor) -> int:
        key = (tuple(t.shape), t.dtype)
        free = self.pool.get(key)
        host = free.pop() if free else torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            host.copy_(t, non_blocking=True)
        t.record_stream(self.stream)  # the allocator keeps t's memory until the copy is done
        ev = torch.cuda.Event()
        ev.record(self.stream)
        self.items.append({"host": host, "ev": ev, "shape": key[0], "dtype": t.dtype, "dev": None, "dev_ev": None,
                           "device": t.device})
        self.bytes_moved += t.numel() * t.element_size()
        return len(self.items) - 1

    def _prefetch(self, i: int) -> None:
        if i < 0 or i >= len(self.items):
            return
        it = self.items[i]
        if it["dev"] is not None or it["host"] is None:
            return
        dev = torch.empty(it["shape"], dtype=it["dtype"], device=it["device"])
        self.stream.wait_event(it["ev"])
        with torch.cuda.stream(self.stream):
            dev.copy_(it["host"], non_blocking=True)
        dev.record_stream(self.stream)
        ev = torch.cuda.Event()
        ev.record(self.stream)
        it["dev"], it["dev_ev"] = dev, ev

    def unpack(self, i: int) -> torch.Tensor:
        self._prefetch(i)
        it = self.items[i]
        torch.cuda.current_stream().wait_event(it["dev_ev"])
        t = it["dev"]
        it["dev"] = None
        self._prefetch(i - 1)  # backward runs the layers in reverse: start the next copy now
        return t


OFFLOADER: HostOffloader | None = None
FOLD_LORA_DX = __import__("os").environ.get("TII_FOLD_LORA_DX", "1") == "1"


def lora_k2(site: ops.LoRASite) -> int:
    """Columns of [DH_hi | DH_lo] of a site (2 x sum of live ranks)."""
    return 2 * sum(m[0].shape[0] for m in site.masters if m is not None)


def _lora_bwd_core(x2: torch.Tensor, h: torch.Tensor, site: ops.LoRASite, g2: torch.Tensor,
                   hl_out: torch.Tensor | None = None):
    """LoRA backward of one site (contract of ops._lora_bw_fast). Returns (dx bf16 [T, in], [dA, dB, ...]). With
    hl_out ([T, 2k] view) the hi|lo parts are written there and dx is left to the caller's folded GEMM: returns
    (None, grads, A2) with A2 = [A; A] the matching weight rows."""
    from v2.tii.trainer.bwd_kernels import mm_bf16_f32, mm_tallk

    offs = [0]
    for n in site.slices:
        offs.append(offs[-1] + n)
    live = [(s_, n, site.masters[s_]) for s_, n in enumerate(site.slices) if site.masters[s_] is not None]
    dhs, a_deps, dbs = [], [], []
    for s_, n, m in live:
        r = m[0].shape[0]
        gs16 = g2[:, offs[s_] : offs[s_] + n]
        dbs.append(mm_tallk(gs16.t(), h[s_, :, :r]))
        dhs.append(mm_bf16_f32(gs16, site.b_stacked[s_][0, 0, :, :r]))
        a_deps.append(site.a_stacked[s_][0, 0, :r, :])
    DH = torch.cat(dhs, 1)
    k = DH.shape[1]
    A2 = torch.cat(a_deps + a_deps, 0)
    if hl_out is not None:
        hl_out[:, :k].copy_(DH)
        hl_out[:, k:].copy_(DH - hl_out[:, :k].float())
        HL, dx = hl_out, None
    else:
        hi = DH.to(torch.bfloat16)
        lo = (DH - hi.float()).to(torch.bfloat16)
        HL = torch.cat([hi, lo], 1)
        dx = torch.mm(HL, A2, out_dtype=torch.bfloat16)
    dA2 = mm_tallk(HL.t(), x2)
    dA = dA2[:k] + dA2[k:]
    grads, col = [], 0
    for (s_, n, m), db in zip(live, dbs):
        r = m[0].shape[0]
        grads += [dA[col : col + r], db * site.scaling]
        col += r
    if ops.PLANT and ("lora", site.name) in ops.PLANT:
        grads = [g * ops.PLANT[("lora", site.name)] for g in grads]
    if hl_out is not None:
        return None, grads, A2
    return dx, grads


class MLPBlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, blk: dict, meta: Any, *params: torch.Tensor) -> torch.Tensor:
        cu, eps = blk["custom_ops"], blk["eps"]
        y = ops.visible_rms_norm(x, blk["w_pre"], eps, cu)
        gu = blk["gate_up"].visible(y)
        h_gu = None
        if blk["site_gu"] is not None:
            gu, buf = ops.visible_lora(y.reshape(-1, y.shape[-1]).contiguous(), gu, blk["site_gu"], meta, inplace=True)
            h_gu = buf.to(torch.bfloat16)
        act = ops.visible_gelu_and_mul(gu, cu)
        dn = blk["down"].visible(act)
        h_dn = None
        if blk["site_dn"] is not None:
            dn, buf = ops.visible_lora(act.reshape(-1, act.shape[-1]).contiguous(), dn, blk["site_dn"], meta, inplace=True)
            h_dn = buf.to(torch.bfloat16)
        out = ops.visible_rms_norm(dn, blk["w_post"], eps, cu)
        ctx.blk, ctx.meta = blk, meta
        if OFFLOADER is not None:
            ctx.gu_handle = OFFLOADER.pack(gu)
            ctx.save_for_backward(x, dn, h_gu, h_dn)
        else:
            ctx.gu_handle = None
            ctx.save_for_backward(x, dn, h_gu, h_dn, gu)
        return out

    @staticmethod
    def backward(ctx: Any, g_out: torch.Tensor) -> tuple[Any, ...]:
        with ops.timed("bwd_mlp_block"):
            return MLPBlockFn._bw(ctx, g_out)

    @staticmethod
    def _bw(ctx: Any, g_out: torch.Tensor) -> tuple[Any, ...]:
        from v2.tii.trainer.bwd_kernels import gelu_and_mul_bwd, int4_dx, int4_dx_folded, rmsnorm_bwd

        blk = ctx.blk
        if ctx.gu_handle is not None:
            x, dn, h_gu, h_dn = ctx.saved_tensors
            gu = OFFLOADER.unpack(ctx.gu_handle)
        else:
            x, dn, h_gu, h_dn, gu = ctx.saved_tensors
        cu, eps = blk["custom_ops"], blk["eps"]
        T = x.shape[0]

        def base_dx(lin, g):
            if lin.backend == "marlin_engine":
                return int4_dx(g, lin.qweight, lin.scale)
            return torch.matmul(g, lin.weight_ref())

        # folded dX: base dX and LoRA dx in one GEMM over [g | DH_hi | DH_lo] (planted linear_dx controls scale the
        # base dX alone, so they use the unfolded path)
        fold = FOLD_LORA_DX and not ops.PLANT and blk["down"].backend == "marlin_engine"
        act = ops.visible_gelu_and_mul(gu, cu)  # recomputed elementwise (bitwise the forward value)
        if fold and blk["site_dn"] is not None and blk["site_gu"] is not None:
            n_dn, k_dn = dn.shape[-1], lora_k2(blk["site_dn"])
            G1 = torch.empty(T, n_dn + k_dn, dtype=dn.dtype, device=dn.device)
            g_dn = rmsnorm_bwd(dn, blk["w_post"], g_out.contiguous(), eps, out=G1[:, :n_dn])
            _, grads_dn, A2 = _lora_bwd_core(act.reshape(T, -1), h_dn, blk["site_dn"], g_dn, hl_out=G1[:, n_dn:])
            g_act = int4_dx_folded(G1, blk["down"].qweight, blk["down"].scale, A2)
            del G1
            n_gu, k_gu = gu.shape[-1], lora_k2(blk["site_gu"])
            G2 = torch.empty(T, n_gu + k_gu, dtype=gu.dtype, device=gu.device)
            g_gu = gelu_and_mul_bwd(gu, g_act, out=G2[:, :n_gu])
            del g_act
            y = ops.visible_rms_norm(x, blk["w_pre"], eps, cu)  # recomputed (elementwise)
            _, grads_gu, A2 = _lora_bwd_core(y.reshape(T, -1), h_gu, blk["site_gu"], g_gu, hl_out=G2[:, n_gu:])
            g_y = int4_dx_folded(G2, blk["gate_up"].qweight, blk["gate_up"].scale, A2)
            del G2
            g_x = rmsnorm_bwd(x, blk["w_pre"], g_y.view(x.shape), eps)
            return (g_x, None, None, *grads_gu, *grads_dn)
        g_dn = rmsnorm_bwd(dn, blk["w_post"], g_out.contiguous(), eps)
        grads_dn: list = []
        g_act = base_dx(blk["down"], g_dn.reshape(T, -1))
        if ops.PLANT:  # planted error on the base linear dX only (as ops.FrozenLinearFn)
            g_act = g_act * ops.PLANT.get(("linear_dx", blk["down"].name), 1.0)
        if blk["site_dn"] is not None:
            dxl, grads_dn = _lora_bwd_core(act.reshape(T, -1), h_dn, blk["site_dn"], g_dn.reshape(T, -1))
            g_act = g_act + dxl
        g_gu = gelu_and_mul_bwd(gu, g_act)
        y = ops.visible_rms_norm(x, blk["w_pre"], eps, cu)  # recomputed (elementwise)
        grads_gu: list = []
        g_y = base_dx(blk["gate_up"], g_gu.reshape(T, -1))
        if ops.PLANT:
            g_y = g_y * ops.PLANT.get(("linear_dx", blk["gate_up"].name), 1.0)
        if blk["site_gu"] is not None:
            dxl, grads_gu = _lora_bwd_core(y.reshape(T, -1), h_gu, blk["site_gu"], g_gu.reshape(T, -1))
            g_y = g_y + dxl
        g_x = rmsnorm_bwd(x, blk["w_pre"], g_y.view(x.shape), eps)
        return (g_x, None, None, *grads_gu, *grads_dn)
