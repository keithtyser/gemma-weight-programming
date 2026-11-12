"""vLLM 0.19.1 MPLinearKernel wrapper around the TII batch-invariant W4A16 Triton GEMM.

Author: Claude Opus 5.5 (engine track), 2026-11-05. Registered by v2/tii/engine/vllm_plugin.py (patch P1).
Accepts exactly what this project serves: symmetric uint4b8, group size 32, bf16 activations, no act-order,
no zero points. Anything else is refused so vLLM falls through to its stock list (and the run is not aligned).
"""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.scalar_type import scalar_types

from v2.tii.engine import vllm_plugin
from v2.tii.engine.w4a16_triton import GROUP_SIZE, repack_ct_scale, repack_ct_weight, w4a16_gemm


class TiiTritonW4A16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type != scalar_types.uint4b8:
            return False, f"weight type {c.weight_type} (only uint4b8)"
        if c.act_type != torch.bfloat16:
            return False, "only bf16 activations"
        if c.group_size != GROUP_SIZE:
            return False, f"group size {c.group_size} (only {GROUP_SIZE})"
        if c.zero_points or c.has_g_idx:
            return False, "zero points / act-order not supported"
        k, n = c.partition_weight_shape
        if n % 8 or k % GROUP_SIZE:
            return False, f"shape K={k} N={n}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def repack_w(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)  # [N, K//8]
            x.data = repack_ct_weight(x.data)
            return x

        def repack_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)  # [N, K//G]
            x.data = repack_ct_scale(x.data)
            return x

        self._transform_param(layer, self.w_q_name, repack_w)
        self._transform_param(layer, self.w_s_name, repack_s)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        vllm_plugin.CALLS["w4a16_apply"] += 1
        w_q, w_s, _, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        out = w4a16_gemm(x_2d, w_q, w_s)
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (self.config.partition_weight_shape[1],))
