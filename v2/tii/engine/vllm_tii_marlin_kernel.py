"""vLLM MPLinearKernel: stock Marlin weight repack + the TII batch-invariant Marlin GEMM. Engine track, 2026-11-05.
Registered by vllm_plugin.py (patch P1, TII_LINEAR=marlin).

TII_SMALLM=w4 (kernel track, 2026-11-06): apply_weights calls v2.tii.kernel.tii_w4.gemm instead, i.e. the small-M
kernel for M <= 64 where it is faster and TII-Marlin otherwise; bitwise identical to tii_marlin_op.gemm for every M
(v2/tii/tests/test_kernel_tii_w4.py), so the trainer side is unchanged.

TII_LARGEM=w4l (kernel track, 2026-11-06): tii_marlin_op.gemm sends M >= tii_w4l.MIN_M to the TII-W4L large-M kernel
(bitwise identical, v2/tii/tests/test_kernel_tii_w4l.py); set the same env for the trainer."""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.scalar_type import scalar_types

from v2.tii.engine import tii_marlin_op, vllm_plugin

import os

SMALLM = os.environ.get("TII_SMALLM", "")
if SMALLM == "w4":
    from v2.tii.kernel import tii_w4


class TiiMarlinLinearKernel(MarlinLinearKernel):
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type != scalar_types.uint4b8 or c.act_type != torch.bfloat16 or c.group_size != 32:
            return False, "TII Marlin: uint4b8, bf16, group 32 only"
        if c.zero_points or c.has_g_idx:
            return False, "TII Marlin: no zero points / act-order"
        k, n = c.partition_weight_shape
        if k % 128 or n % 64:
            return False, f"TII Marlin: K % 128 and N % 64 required (K={k}, N={n})"
        return super().can_implement(c)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)  # stock Marlin repack/permute (byte-preserving)
        if SMALLM == "w4":
            tii_w4.warmup(layer.weight_packed.device if hasattr(layer, "weight_packed") else torch.device("cuda"))
            if "P1_smallm_w4" not in vllm_plugin._APPLIED:
                vllm_plugin._receipt("P1_smallm_w4", version=tii_w4.KERNEL_VERSION, so=tii_w4.SO_PATH,
                                     marlin_so=tii_marlin_op.SO_PATH, table_entries=len(tii_w4.CFG))
        else:
            tii_marlin_op.load()
        if tii_marlin_op.LARGEM == "w4l":  # TII-W4L for M > 64 (kernel track): set kernel attributes before capture
            from v2.tii.kernel import tii_w4l
            k, n = self.config.partition_weight_shape
            tii_w4l.warmup(n, k, layer.weight_packed.device if hasattr(layer, "weight_packed") else torch.device("cuda"))
            if "P1_largem_w4l" not in vllm_plugin._APPLIED:
                vllm_plugin._receipt("P1_largem_w4l", version=tii_w4l.KERNEL_VERSION, so=tii_w4l.SO_PATH,
                                     min_m=tii_w4l.MIN_M, table_entries=len(tii_w4l.CFG))

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        vllm_plugin.CALLS["w4a16_apply"] += 1
        w_q, w_s, _, _ = self._get_weight_params(layer)
        k, n = self.config.partition_weight_shape
        out = (tii_w4.gemm if SMALLM == "w4" else tii_marlin_op.gemm)(x, w_q, w_s, n, k)
        if bias is not None:
            out.add_(bias)
        return out
