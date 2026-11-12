"""Aligned Gemma 4 E4B (+LoRA) training forward whose per-token logprobs equal vLLM's rollout logprobs bitwise.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: A teacher-forced, packed (THD) re-implementation of vLLM 0.19.1 `gemma4.py` (Gemma4Model /
  Gemma4DecoderLayer / Gemma4Attention / Gemma4MLP + LogitsProcessor + Sampler) that calls, for every operator that
  produces a value, the same vLLM kernel the rollout engine executes (see ops.py), in the same order, on the same
  weight bytes, with LoRA applied through Punica's kernels on the engine's stacked-buffer geometry. Each line of
  `_layer` / `_attention` mirrors the corresponding line of vllm/model_executor/models/gemma4.py (cited inline).
  Inputs: token ids of one or more full sequences (prompt + response) packed contiguously, their boundaries and the
  engine's per-sequence query chunk schedule (prompt prefill chunks, 1-token decode steps). Output: final hidden
  states and, via `score`, per-token processed logprobs (fp32) of the given target tokens.
  Trainable parameters: LoRA A/B fp32 masters only (frozen INT4/bf16 base, frozen norms and embeddings).
  Reads: a compressed-tensors Gemma 4 checkpoint dir (full model or the C2 proxy; INT4 `weight_packed` or dense
  `weight`), optionally a PEFT adapter dir. Writes nothing (export via v2.train.int4_qat_lora.export_vllm_peft_adapter).
  Depends on: runtime.setup_process() having run (engine kernel state), vLLM 0.19.1, v2/tii/engine/w4a16_triton.py
  for the INT4 backend.
SRP/DRY check: Pass - v2/train/run_qat_lora_train.py builds the HF model with torch numerics; this module is the
  vLLM-numerics twin and reuses its PEFT key conventions and the shared exporter instead of duplicating them.
"""

from __future__ import annotations

import os

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from v2.tii.trainer import ops
from v2.tii.trainer.runtime import EngineNumerics

LM = "model.language_model."
LORA_TARGETS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _Layer:
    """Frozen state of one decoder layer (no autograd parameters)."""

    def __init__(self) -> None:
        self.lin: dict[str, ops.FrozenLinear] = {}
        self.norm: dict[str, torch.Tensor] = {}
        self.sites: dict[str, ops.LoRASite] = {}


class AlignedGemma4(torch.nn.Module):
    def __init__(self, ckpt_dir: str | Path, numerics: EngineNumerics, *, device: str | torch.device = "cuda",
                 lora_rank: int | None = None, lora_alpha: float | None = None,
                 lora_targets: Sequence[str] = LORA_TARGETS, checkpoint_layers: bool = False,
                 checkpoint_policy: str | None = None) -> None:
        super().__init__()
        from safetensors import safe_open
        from vllm.model_executor.layers.rotary_embedding import get_rope

        self.dev = torch.device(device)
        self.num = numerics
        self.checkpoint_layers = checkpoint_layers
        # "full": recompute whole decoder layers in backward; "mlp": recompute only the MLP/PLE block; "none"
        self.checkpoint_policy = checkpoint_policy or ("full" if checkpoint_layers else "none")
        # gate runs: TII_CKPT_POLICY=none|mlp|full overrides the caller's policy (the learner's adaptive policy uses
        # none and mlp per forward; the gates run once per policy). Recorded in the receipt.
        self.ckpt_override = os.environ.get("TII_CKPT_POLICY") or None
        if self.ckpt_override:
            if self.ckpt_override not in ("none", "mlp", "full"):
                raise ValueError(f"TII_CKPT_POLICY must be none|mlp|full, got {self.ckpt_override!r}")
            self.checkpoint_policy = self.ckpt_override
        # engine logits processors that change processed logprobs (logits_processors.ThinkingBudget or None); the
        # caller sets it to the rollout server's reasoning config + the requests' thinking_token_budget
        self.thinking_budget = None
        self.forced_by_completion: dict[int, list[int]] = {}  # id(completion list) -> resolved forced ids
        # fused MLP block (lean saved state) on the fast backward path; gate_up offload to pinned host memory optional
        self.mlp_fused = ops.FAST_BWD
        self.offload_gate_up = False
        # Memory-planned recompute for tree scoring (score_calls_subtrees): forwards of <= no_ckpt_tokens unique tokens
        # run with NO activation recomputation; a forward forced larger by a single long call (a call cannot be split)
        # uses long_ckpt_policy ("mlp": only the MLP block is recomputed). checkpoint_policy is the default otherwise.
        self.no_ckpt_tokens = 20000
        self.long_ckpt_policy = "mlp"
        self.adaptive_ckpt = False
        ckpt = Path(ckpt_dir)
        cfg = json.loads((ckpt / "config.json").read_text())
        tc = cfg.get("text_config", cfg)
        self.cfg = tc
        self.H = tc["hidden_size"]
        self.nL = tc["num_hidden_layers"]
        self.nh, self.nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
        self.eps = tc["rms_norm_eps"]
        self.ple_dim = tc.get("hidden_size_per_layer_input") or 0
        self.softcap = tc.get("final_logit_softcapping")
        if tc.get("attention_k_eq_v") or tc.get("enable_moe_block") or tc.get("use_double_wide_mlp"):
            raise NotImplementedError("k_eq_v / MoE / double-wide MLP variants are not in the E4B scope")
        if tc.get("attn_logit_softcapping"):
            raise NotImplementedError("attention softcap is not in the E4B scope")
        self.types = tc["layer_types"]
        first_shared = self.nL - tc.get("num_kv_shared_layers", 0)
        # gemma4.py Gemma4Attention.__init__: shared layer -> last non-shared layer of the same type
        self.kv_source: dict[int, int] = {}
        for i in range(first_shared, self.nL):
            prev = self.types[:first_shared]
            self.kv_source[i] = len(prev) - 1 - prev[::-1].index(self.types[i])
        self.kv_sources = set(self.kv_source.values())
        self.window = tc["sliding_window"]

        # vLLM builds the model under set_default_torch_dtype(bf16) and torch.device(cuda): python-float buffers are
        # bf16 tensors created on the GPU. Reproduce each one with the same expression.
        with torch.device(self.dev):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                self.normalizer = torch.tensor(self.H**0.5)  # Gemma4Model.normalizer
                self.embed_scale_per_layer = torch.tensor(self.ple_dim**0.5)
                self.per_layer_input_scale = torch.rsqrt(torch.tensor(2.0))
                self.per_layer_projection_scale = torch.tensor(self.H**-0.5)
                self.ones = {d: torch.ones(d) for d in (tc["head_dim"], tc.get("global_head_dim", tc["head_dim"]))}  # v_norm weight
                self.ropes = {}
                for lt in ("sliding_attention", "full_attention"):
                    hd = tc.get("global_head_dim", tc["head_dim"]) if lt == "full_attention" else tc["head_dim"]
                    self.ropes[lt] = get_rope(hd, max_position=tc["max_position_embeddings"], is_neox_style=True,
                                              rope_parameters=dict(tc["rope_parameters"][lt]))
            finally:
                torch.set_default_dtype(prev)

        f = safe_open(str(ckpt / "model.safetensors"), framework="pt", device="cpu")
        keys = set(f.keys())

        def t(name: str) -> torch.Tensor:
            return f.get_tensor(name).to(self.dev)

        def lin(name: str, parts: Sequence[str]) -> ops.FrozenLinear:
            """Merged frozen linear (vLLM QKVParallelLinear / MergedColumnParallelLinear concatenates along N)."""
            if all(f"{p}.weight_packed" in keys for p in parts):
                packed = [f.get_tensor(f"{p}.weight_packed") for p in parts]
                scales = [f.get_tensor(f"{p}.weight_scale") for p in parts]
                if numerics.linear_backend == "marlin_engine":  # merged along N in CT layout, then Marlin-prepared
                    q = torch.cat([p.to(self.dev) for p in packed], 0).contiguous()
                    sc = torch.cat([x.to(self.dev) for x in scales], 0).contiguous()
                    return ops.FrozenLinear(name, "marlin_engine", qweight=q, scale=sc)
                if numerics.linear_backend == "w4a16_engine":
                    from v2.tii.engine.w4a16_triton import repack_ct_scale, repack_ct_weight

                    q = torch.cat([repack_ct_weight(p.to(self.dev)) for p in packed], dim=1).contiguous()
                    s = torch.cat([repack_ct_scale(x.to(self.dev)) for x in scales], dim=1).contiguous()
                    return ops.FrozenLinear(name, "w4a16_engine", qweight=q, scale=s)
                w = torch.cat([ops.dequant_int4_bf16(p.to(self.dev), x.to(self.dev)) for p, x in zip(packed, scales)], 0)
                return ops.FrozenLinear(name, "dense_bf16_bi", weight=w.contiguous())
            if all(f"{p}.weight" in keys for p in parts):
                if numerics.linear_backend != "dense_bf16_bi":
                    raise ValueError(f"{name}: dense checkpoint needs linear_backend=dense_bf16_bi")
                w = torch.cat([t(f"{p}.weight") for p in parts], 0)
                return ops.FrozenLinear(name, "dense_bf16_bi", weight=w.contiguous())
            raise KeyError(f"weights for {parts} not found")

        self.embed = t(LM + "embed_tokens.weight")  # also the tied lm_head (ParallelLMHead tied to embed_tokens)
        self.receipt: dict[str, Any] = {}
        if "lm_head.weight" in keys:  # the checkpoint carries a copy; vLLM ties to embed_tokens
            self.receipt["lm_head_copy_equals_embed"] = bool(torch.equal(f.get_tensor("lm_head.weight").to(self.dev), self.embed))
        self.final_norm = t(LM + "norm.weight")
        if self.ple_dim:
            self.ple_table = t(LM + "embed_tokens_per_layer.weight")
            self.ple_proj = lin("per_layer_model_projection", [LM + "per_layer_model_projection"])
            self.ple_proj_norm = t(LM + "per_layer_projection_norm.weight")

        self.layers: list[_Layer] = []
        for i in range(self.nL):
            p = f"{LM}layers.{i}."
            L = _Layer()
            shared = i in self.kv_source
            attn = [p + "self_attn.q_proj"] + ([] if shared else [p + "self_attn.k_proj", p + "self_attn.v_proj"])
            L.lin["qkv"] = lin(f"{i}.qkv", attn)
            L.lin["o"] = lin(f"{i}.o", [p + "self_attn.o_proj"])
            L.lin["gate_up"] = lin(f"{i}.gate_up", [p + "mlp.gate_proj", p + "mlp.up_proj"])
            L.lin["down"] = lin(f"{i}.down", [p + "mlp.down_proj"])
            if self.ple_dim:
                L.lin["ple_gate"] = lin(f"{i}.ple_gate", [p + "per_layer_input_gate"])
                L.lin["ple_proj"] = lin(f"{i}.ple_proj", [p + "per_layer_projection"])
                L.norm["post_ple"] = t(p + "post_per_layer_input_norm.weight")
            for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                L.norm[n] = t(p + n + ".weight")
            L.norm["q"] = t(p + "self_attn.q_norm.weight")
            if not shared:
                L.norm["k"] = t(p + "self_attn.k_norm.weight")
            L.layer_scalar = t(p + "layer_scalar")
            self.layers.append(L)

        # LoRA: fp32 masters named by the HF/PEFT module path (export/load keep the PEFT convention).
        from vllm.v1.attention.ops import triton_unified_attention as _tua

        k2d = _tua.kernel_unified_attention_2d
        # engine attention fix (P4) present => prefill bytes == decode bytes, so a whole sequence may be one chunk
        self.attn_fix_active = getattr(getattr(k2d, "fn", k2d), "__name__", "") == "kernel_unified_attention_2d_tii"
        self.receipt["attn_fix_active"] = self.attn_fix_active
        self.receipt["checkpoint_policy"] = self.checkpoint_policy
        self.receipt["ckpt_override"] = self.ckpt_override
        # LoRA shrink variant (numerics-defining for the LoRA buffer h): must match the engine plugin's switch
        env_shrink = os.environ.get("TII_LORA_SHRINK", "pin")
        if self.num.lora_shrink != env_shrink:
            raise ValueError(f"EngineNumerics.lora_shrink={self.num.lora_shrink!r} but TII_LORA_SHRINK={env_shrink!r}: "
                             "the trainer would not use the engine's shrink")
        ops.LORA_SHRINK = self.num.lora_shrink
        self.receipt["lora_shrink"] = {"mode": self.num.lora_shrink}
        if self.num.lora_shrink == "split":
            from v2.tii.engine.lora_shrink_tii import VERSION as _SHRINK_VERSION

            self.receipt["lora_shrink"]["version"] = _SHRINK_VERSION
        self.lora_params = torch.nn.ParameterDict()
        self.lora_enabled = lora_rank is not None
        self.lora_scaling = (float(lora_alpha) / float(lora_rank)) if lora_rank else 1.0
        self.lora_rank = lora_rank
        if self.lora_enabled:
            self._build_lora(lora_rank, lora_targets)

    # ------------------------------------------------------------------------------------------------ LoRA
    @staticmethod
    def _pkey(module: str, ab: str) -> str:
        return module.replace(".", "__") + "__" + ab  # ParameterDict keys may not contain "."

    def _build_lora(self, r: int, targets: Sequence[str]) -> None:
        tc = self.cfg
        hd_s, hd_f = tc["head_dim"], tc.get("global_head_dim", tc["head_dim"])
        inter = tc["intermediate_size"]

        def master(module: str, n_in: int, n_out: int) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
            a = torch.nn.Parameter(torch.zeros(r, n_in, dtype=torch.float32, device=self.dev))
            b = torch.nn.Parameter(torch.zeros(n_out, r, dtype=torch.float32, device=self.dev))
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # PEFT default init (B = 0)
            self.lora_params[self._pkey(module, "A")] = a
            self.lora_params[self._pkey(module, "B")] = b
            return a, b

        for i, L in enumerate(self.layers):
            hd = hd_f if self.types[i] == "full_attention" else hd_s
            q_n, kv_n = self.nh * hd, self.nkv * hd
            p = f"{LM}layers.{i}."
            mk = lambda name, n_in, n_out: master(p + name, n_in, n_out) if name.split(".")[-1] in targets else None  # noqa: E731
            site = lambda nm, n_in, slices, ms: ops.LoRASite(  # noqa: E731
                f"{i}.{nm}", n_in, slices, ms, max_rank=self.num.lora_max_rank, max_loras=self.num.lora_max_loras,
                scaling=self.lora_scaling, device=self.dev)
            # vLLM wraps the merged modules; k/v slices have no adapter (zero stacked buffers), also on KV-shared layers
            L.sites["qkv"] = site("qkv", self.H, (q_n, kv_n, kv_n), [mk("self_attn.q_proj", self.H, q_n), None, None])
            L.sites["o"] = site("o", q_n, (self.H,), [mk("self_attn.o_proj", q_n, self.H)])
            L.sites["gate_up"] = site("gate_up", self.H, (inter, inter), [mk("mlp.gate_proj", self.H, inter), mk("mlp.up_proj", self.H, inter)])
            L.sites["down"] = site("down", inter, (self.H,), [mk("mlp.down_proj", inter, self.H)])
            for nm in list(L.sites):
                if all(m is None for m in L.sites[nm].masters):
                    del L.sites[nm]  # module without adapter weights: see ops.GRADIENT_CONTRACT / report (signed zero only)

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """PEFT-convention {module}.lora_A.weight / lora_B.weight (fp32 masters) for export_vllm_peft_adapter."""
        out = {}
        for k, v in self.lora_params.items():
            mod, ab = k.rsplit("__", 1)
            out[f"{mod.replace('__', '.')}.lora_{ab}.weight"] = v.detach()
        return out

    @torch.no_grad()
    def load_adapter(self, adapter_dir: str | Path) -> int:
        """Load a PEFT adapter (as exported for vLLM) into the fp32 masters. Returns the number of modules loaded."""
        from safetensors.torch import load_file

        ad = Path(adapter_dir)
        cfg = json.loads((ad / "adapter_config.json").read_text())
        if int(cfg["r"]) != self.lora_rank or abs(float(cfg["lora_alpha"]) / float(cfg["r"]) - self.lora_scaling) > 0:
            raise ValueError(f"adapter r/alpha {cfg['r']}/{cfg['lora_alpha']} != model {self.lora_rank}/{self.lora_scaling}")
        sd = load_file(str(ad / "adapter_model.safetensors"))
        n = 0
        seen = set()
        for k, v in sd.items():
            m = re.match(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
            if not m:
                raise KeyError(f"unexpected adapter key {k}")
            pk = self._pkey(m.group(1), m.group(2))
            if pk not in self.lora_params:
                raise KeyError(f"adapter module {m.group(1)} not a LoRA target of this model")
            self.lora_params[pk].copy_(v.to(torch.float32))
            seen.add(pk)
            n += 1
        missing = set(self.lora_params.keys()) - seen
        if missing:
            raise KeyError(f"adapter is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
        return n // 2

    def refresh_lora(self) -> None:
        for L in self.layers:
            for s in L.sites.values():
                s.refresh()

    # ------------------------------------------------------------------------------------------------ forward
    def _proj(self, L: _Layer, name: str, x: torch.Tensor, lora: bool) -> torch.Tensor:
        """vLLM linear (+ BaseLinearLayerWithLoRA.apply: base GEMM, then Punica add_lora_linear in place)."""
        y = ops.frozen_linear(x, L.lin[name])
        site = L.sites.get(name) if lora else None
        if site is None:
            return y
        if name == "qkv" and y.shape[-1] != sum(site.slices):
            # KV-shared layer: the engine's merged qkv also produces (discarded) k/v columns; Punica treats the
            # three slices independently, so the q columns are the same with zero placeholders for k/v.
            pad = y.new_zeros(y.shape[0], sum(site.slices) - y.shape[-1])
            return ops.lora_apply(x, torch.cat([y, pad], -1), site, self._meta)[:, : y.shape[-1]]
        return ops.lora_apply(x, y, site, self._meta)

    def _attention(self, i: int, L: _Layer, x: torch.Tensor, positions: torch.Tensor, plan: ops.AttnPlan,
                   kv_store: dict[int, tuple[torch.Tensor, torch.Tensor]], lora: bool) -> torch.Tensor:
        lt = self.types[i]
        hd = self.cfg.get("global_head_dim", self.cfg["head_dim"]) if lt == "full_attention" else self.cfg["head_dim"]
        T = x.shape[0]
        cu = self.num.custom_ops
        qkv = self._proj(L, "qkv", x, lora)  # g4: qkv, _ = self.qkv_proj(hidden_states)
        q_n, kv_n = self.nh * hd, self.nkv * hd
        shared = i in self.kv_source
        q = qkv[:, :q_n]
        q = ops.rms_norm(q.unflatten(-1, (self.nh, hd)), L.norm["q"], self.eps, cu).flatten(-2, -1)  # q_norm
        if not shared:
            k = qkv[:, q_n : q_n + kv_n]
            v = qkv[:, q_n + kv_n : q_n + 2 * kv_n]
            k = ops.rms_norm(k.unflatten(-1, (self.nkv, hd)), L.norm["k"], self.eps, cu).flatten(-2, -1)  # k_norm
            q, k = ops.rope_apply(self.ropes[lt], positions, q, k, cu)  # q, k = self.rotary_emb(positions, q, k)
            v = ops.rms_norm(v.unflatten(-1, (self.nkv, hd)), self.ones[hd], self.eps, cu).flatten(-2, -1)  # v_norm
            if i in self.kv_sources:  # only KV-source layers' k/v are read later (by KV-shared layers)
                kv_store[i] = (k, v)
        else:
            q, _ = ops.rope_apply(self.ropes[lt], positions, q, None, cu)  # q = self.rotary_emb(positions, q, k)[0]
            k, v = kv_store[self.kv_source[i]]  # reads the target layer's cache (kv_sharing_target_layer_name)
        window = self.window if lt == "sliding_attention" else None
        o = ops.attention(q.view(T, self.nh, hd), k.view(T, self.nkv, hd), v.view(T, self.nkv, hd), plan, 1.0, window)
        return self._proj(L, "o", o.view(T, q_n), lora)  # output, _ = self.o_proj(attn_output)

    def _layer(self, i: int, h: torch.Tensor, ple_i: torch.Tensor | None, positions: torch.Tensor, plan: ops.AttnPlan,
               kv_store: dict, lora: bool) -> torch.Tensor:
        """Gemma4DecoderLayer.forward, line by line (attention block, then MLP + PLE block)."""
        x = self._attn_block(i, h, positions, plan, kv_store, lora)
        if self.checkpoint_policy == "mlp" and torch.is_grad_enabled() and x.requires_grad:
            # selective checkpointing: only the MLP/PLE block (the largest activations, [T, 20480]) is recomputed;
            # the recompute is the same deterministic visible forward, so it reproduces the forward bytes exactly
            return checkpoint(self._mlp_block, i, x, ple_i, lora, use_reentrant=False)
        return self._mlp_block(i, x, ple_i, lora)

    def _attn_block(self, i: int, h: torch.Tensor, positions: torch.Tensor, plan: ops.AttnPlan, kv_store: dict,
                    lora: bool) -> torch.Tensor:
        L = self.layers[i]
        cu = self.num.custom_ops
        residual = h
        x = ops.rms_norm(residual, L.norm["input_layernorm"], self.eps, cu)
        x = self._attention(i, L, x, positions, plan, kv_store, lora)
        x = ops.rms_norm(x, L.norm["post_attention_layernorm"], self.eps, cu)
        return x + residual

    def _mlp_block(self, i: int, x: torch.Tensor, ple_i: torch.Tensor | None, lora: bool) -> torch.Tensor:
        L = self.layers[i]
        cu = self.num.custom_ops
        residual = x
        if self.mlp_fused and ops.ORACLE is None and torch.is_grad_enabled() and x.requires_grad:
            # one autograd Function for the MLP half: same visible calls, lean saved state (mlp_block.py)
            from v2.tii.trainer.mlp_block import MLPBlockFn

            site_gu = L.sites.get("gate_up") if lora else None
            site_dn = L.sites.get("down") if lora else None
            blk = {"w_pre": L.norm["pre_feedforward_layernorm"], "w_post": L.norm["post_feedforward_layernorm"],
                   "gate_up": L.lin["gate_up"], "down": L.lin["down"], "site_gu": site_gu, "site_dn": site_dn,
                   "eps": self.eps, "custom_ops": cu}
            params = (site_gu.params() if site_gu else []) + (site_dn.params() if site_dn else [])
            x = MLPBlockFn.apply(x, blk, self._meta if lora else None, *params)
        else:
            x = ops.rms_norm(x, L.norm["pre_feedforward_layernorm"], self.eps, cu)
            x = self._proj(L, "gate_up", x, lora)  # Gemma4MLP: gate_up_proj
            x = ops.gelu_and_mul(x, cu)  # GeluAndMul(approximate="tanh")
            x = self._proj(L, "down", x, lora)
            x = ops.rms_norm(x, L.norm["post_feedforward_layernorm"], self.eps, cu)
        x = x + residual
        if ple_i is not None:
            gate = self._proj(L, "ple_gate", x, False)  # per_layer_input_gate (ReplicatedLinear, no adapter)
            gate = F.gelu(gate, approximate="tanh")
            gated = gate * ple_i
            contrib = self._proj(L, "ple_proj", gated, False)
            contrib = ops.rms_norm(contrib, L.norm["post_ple"], self.eps, cu)
            x = x + contrib
        return x * L.layer_scalar

    def forward_hidden(self, input_ids: torch.Tensor, seq_bounds: Sequence[tuple[int, int]],
                       chunks: Sequence[Sequence[tuple[int, int]]], *, lora: bool | None = None,
                       h0_grad: bool = False) -> torch.Tensor:
        """Final-norm hidden states [T, H] (bf16) for packed sequences (Gemma4Model.forward, normal path)."""
        positions = torch.cat([torch.arange(e - s, device=self.dev) for s, e in seq_bounds])
        plan = ops.AttnPlan.build(seq_bounds, chunks, self.num.kv_block_size, self.dev)
        return self.forward_rows(input_ids, positions, plan, lora=lora, h0_grad=h0_grad)

    def forward_rows(self, input_ids: torch.Tensor, positions: torch.Tensor, plan: ops.AttnPlan, *,
                     lora: bool | None = None, h0_grad: bool = False) -> torch.Tensor:
        """Final-norm hidden states for packed rows with explicit positions and attention plan (sequences or a tree)."""
        lora = self.lora_enabled if lora is None else (lora and self.lora_enabled)
        T = input_ids.numel()
        ids = input_ids.to(self.dev).long()
        positions = positions.to(self.dev).long()
        from v2.tii.trainer import mlp_block

        if self.offload_gate_up and self.mlp_fused:
            if mlp_block.OFFLOADER is None:
                mlp_block.OFFLOADER = mlp_block.HostOffloader()
            mlp_block.OFFLOADER.reset()  # the previous step's backward has consumed its items
        else:
            mlp_block.OFFLOADER = None
        if lora:
            self.refresh_lora()
            self._meta = ops.make_lora_meta(T, self.num.lora_max_loras, self.dev)
        cu = self.num.custom_ops
        with torch.no_grad():  # no trainable parameter upstream of the embeddings
            h = F.embedding(ids, self.embed) * self.normalizer  # embed_input_ids
            ple = None
            if self.ple_dim:
                tok = torch.where((ids >= 0) & (ids < self.ple_table.shape[0]), ids, torch.zeros_like(ids))
                ple_emb = (F.embedding(tok, self.ple_table) * self.embed_scale_per_layer).reshape(T, self.nL, self.ple_dim)
                proj = ops.frozen_linear(h, self.ple_proj)  # per_layer_model_projection (no adapter)
                proj = proj * self.per_layer_projection_scale
                proj = proj.reshape(T, self.nL, self.ple_dim)
                proj = ops.rms_norm(proj, self.ple_proj_norm, self.eps, cu)
                ple = (proj + ple_emb) * self.per_layer_input_scale
        if h0_grad:  # C7: expose dLoss/dX at the decoder input (embedding output) for the oracle comparison
            h = h.detach().requires_grad_(True)
            self.h0 = h
        kv_store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i in range(self.nL):
            ple_i = ple[:, i, :] if ple is not None else None
            if (self.checkpoint_policy == "full" and torch.is_grad_enabled() and h.requires_grad
                    and i not in self.kv_sources):
                # recompute the (deterministic) layer in backward; KV-source layers stay un-checkpointed because
                # their k/v feed the shared layers through kv_store
                h = checkpoint(self._layer, i, h, ple_i, positions, plan, kv_store, lora, use_reentrant=False)
            else:
                h = self._layer(i, h, ple_i, positions, plan, kv_store, lora)
        return ops.rms_norm(h, self.final_norm, self.eps, cu)  # self.norm(hidden_states)

    def score(self, hidden: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor, temperature: float | torch.Tensor,
              chunk: int = 256, forced: Any = None, ent_out: list | None = None) -> torch.Tensor:
        """Processed logprob of targets[j] from the logits at packed row positions[j] (fp32). forced[j] (bool) marks
        tokens the engine's thinking-budget logits processor forced (logits_processors.forced_targets)."""
        h = hidden.index_select(0, positions.to(self.dev))
        temps = temperature if isinstance(temperature, torch.Tensor) else torch.full((h.shape[0],), float(temperature), device=self.dev)
        if forced is not None and not isinstance(forced, torch.Tensor):
            forced = torch.tensor([f >= 0 for f in forced], dtype=torch.bool)
        if forced is not None:
            forced = forced.to(self.dev)
        return ops.token_logprobs(h, self.embed, targets.to(self.dev).long(), temps.to(self.dev, torch.float32), self.softcap,
                                  self.num.lm_head_backend, self.num.logprobs_mode, chunk, forced=forced,
                                  ent_out=ent_out)

    def forced_of(self, calls: Sequence[tuple[Sequence[int], Sequence[int]]]) -> list[int] | None:
        """Engine-forced token per sampled token of calls (None when no processor is configured). Calls whose
        completion list is registered in self.forced_by_completion (id(completion) -> per-token forced ids, resolved
        with the engine's logprobs: logits_processors.resolve_forced) use that; others the per-token replay rule."""
        if self.thinking_budget is None:
            return None
        from v2.tii.trainer.logits_processors import forced_targets

        out: list[int] = []
        for p, g in calls:
            r = self.forced_by_completion.get(id(g))
            out += r if r is not None else forced_targets([(p, g)], self.thinking_budget)
        return out

    def register_engine_logprobs(self, calls: Sequence[tuple[Sequence[int], Sequence[int]]],
                                 engine_lps: Sequence[Sequence[float]]) -> int:
        """Resolve the engine-forced tokens of calls from their recorded engine logprobs (exact under the engine's
        async-scheduling blind spots; logits_processors.resolve_forced) and register them for forced_of. Returns the
        number of forced tokens."""
        from v2.tii.trainer.logits_processors import resolve_forced

        n = 0
        for (p, g), e in zip(calls, engine_lps):
            r = resolve_forced(p, g, e, self.thinking_budget)
            self.forced_by_completion[id(g)] = r
            n += sum(1 for x in r if x >= 0)
        return n

    def score_calls(self, calls: Sequence[tuple[Sequence[int], Sequence[int]]], temperature: float, *,
                    lora: bool | None = None, h0_grad: bool = False, tree: Any = None) -> list[torch.Tensor]:
        """EXACT shared-prefix scoring of an episode: calls = [(prompt_ids P_k, completion_ids G_k)].

        The sequences S_k = P_k + G_k are packed into a token trie (tree.build_tree): every trie token is computed
        once; each completion token is scored from the logits of its own predecessor on its own path, i.e. in exactly
        the context the engine sampled it in. Attention runs the engine kernel with one kernel sequence per trie node
        whose block table reuses the ancestor pages (prefix-cache semantics; exact only with the engine attention fix,
        which this method requires). Backward accumulates every branch's gradient into the shared prefix rows.
        Returns per call the fp32 logprobs of G_k.
        """
        from v2.tii.trainer.tree import build_tree

        if not self.attn_fix_active:
            raise ValueError("tree scoring needs the engine attention fix (P4): prefix-cache hit == miss bitwise")
        seqs = [list(p) + list(g) for p, g in calls]
        tb = tree if tree is not None else build_tree(seqs)
        plan = ops.AttnPlan.build_tree(tb, self.num.kv_block_size, self.dev)
        ids = torch.tensor(tb.ids, dtype=torch.long, device=self.dev)
        hidden = self.forward_rows(ids, torch.tensor(tb.row_pos, device=self.dev), plan, lora=lora, h0_grad=h0_grad)
        pos_idx, tgt, counts = [], [], []
        for (p, g), rows, seq in zip(calls, tb.seq_rows, seqs):
            P = len(p)
            pos_idx.extend(rows[P - 1 : len(seq) - 1].tolist())  # logits at the predecessor row on this call path
            tgt.extend(seq[P:])
            counts.append(len(seq) - P)
        lp = self.score(hidden, torch.tensor(pos_idx, device=self.dev), torch.tensor(tgt, device=self.dev), temperature,
                        forced=self.forced_of(calls))
        out, k = [], 0
        for n in counts:
            out.append(lp[k : k + n])
            k += n
        return out

    DEFAULT_TREE_BUDGET = 56000  # unique tokens per forward: ~57 GB peak (full checkpointing) on the 96 GB GPU

    def score_calls_subtrees(self, calls: Sequence[tuple[Sequence[int], Sequence[int]]], temperature: float, *,
                             budget: int | None = None, lora: bool | None = None, loss_fn: Any = None) -> list[torch.Tensor]:
        """Exact tree scoring of an episode whose trie may exceed one forward: the calls are split by
        tree.plan_subtrees into groups of <= `budget` unique tokens; each group is one exact tree forward (it recomputes
        only its own ancestor path), so every token keeps its exact context.

        loss_fn(group_call_indices, group_logprobs) -> scalar: if given, each group's loss is backpropagated right after
        its forward (bounded memory) and the LoRA .grad accumulates the groups' gradients - the same sum as the unsplit
        tree's backward. Returns per call fp32 logprobs (detached when loss_fn is given). self.last_split records the
        plan (groups, unique tokens per group, recomputed-prefix overhead).
        """
        from v2.tii.trainer.tree import build_tree, plan_subtrees

        seqs = [list(p) + list(g) for p, g in calls]
        # a single call cannot be split: the effective budget is at least the longest call of the episode
        budget = max(int(budget or self.DEFAULT_TREE_BUDGET), max(len(x) for x in seqs))
        groups = plan_subtrees(seqs, budget)
        out: list[Any] = [None] * len(calls)
        uniq: list[int] = []
        policies: list[str] = []
        for g in groups:
            sub = [calls[i] for i in g]
            tb = build_tree([seqs[i] for i in g])
            uniq.append(tb.unique_tokens)
            saved_policy = self.checkpoint_policy
            if self.adaptive_ckpt:
                self.checkpoint_policy = "none" if tb.unique_tokens <= self.no_ckpt_tokens else self.long_ckpt_policy
                policies.append(self.checkpoint_policy)
            if loss_fn is None:
                lps = self.score_calls(sub, temperature, lora=lora, tree=tb)
            else:
                with torch.enable_grad():
                    lps = self.score_calls(sub, temperature, lora=lora, tree=tb)
                    loss_fn(g, lps).backward()
                lps = [x.detach() for x in lps]
            self.checkpoint_policy = saved_policy
            for i, lp in zip(g, lps):
                out[i] = lp
        whole = build_tree(seqs).unique_tokens if len(groups) > 1 else (uniq[0] if uniq else 0)
        self.last_split = {"groups": len(groups), "budget": budget, "unique_per_group": uniq, "unique_unsplit": whole,
                           "processed": sum(uniq), "overhead_tokens": sum(uniq) - whole, "policies": policies}
        return out

    def rollout_logprobs(self, sequences: Sequence[Sequence[int]], prompt_lens: Sequence[int], temperature: float,
                         *, lora: bool | None = None, prompt_chunks: Sequence[Sequence[tuple[int, int]]] | None = None,
                         h0_grad: bool = False, one_shot: bool | None = None) -> list[torch.Tensor]:
        """Per-sequence logprobs of the response tokens (positions prompt_len .. L-1), engine schedule reproduced."""
        if one_shot is None:
            one_shot = self.attn_fix_active and prompt_chunks is None
        if one_shot and not self.attn_fix_active:
            raise ValueError("one-shot prefill scoring needs the engine attention fix (P4) to be bitwise")
        bounds, chunks, flat = [], [], []
        pos_idx, tgt = [], []
        off = 0
        for j, (seq, P) in enumerate(zip(sequences, prompt_lens)):
            L = len(seq)
            bounds.append((off, off + L))
            if one_shot:
                # Whole sequence as one prefill chunk. Bitwise equal to the engine's prefill+decode schedule ONLY with
                # the engine's attention fix (P4: leading fully-masked tile keeps M=-inf); proved by C5 cells.
                chunks.append([(0, L)])
            else:
                pre = list(prompt_chunks[j]) if prompt_chunks is not None else [(0, P)]
                chunks.append(pre + [(p, p + 1) for p in range(P, L)])
            flat.extend(seq)
            pos_idx.extend(range(off + P - 1, off + L - 1))  # logits at position p predict token p+1
            tgt.extend(seq[P:])
            off += L
        ids = torch.tensor(flat, dtype=torch.long, device=self.dev)
        hidden = self.forward_hidden(ids, bounds, chunks, lora=lora, h0_grad=h0_grad)
        lp = self.score(hidden, torch.tensor(pos_idx, device=self.dev), torch.tensor(tgt, device=self.dev), temperature,
                        forced=self.forced_of([(list(seq[:P]), list(seq[P:])) for seq, P in zip(sequences, prompt_lens)]))
        out, k = [], 0
        for seq, P in zip(sequences, prompt_lens):
            n = len(seq) - P
            out.append(lp[k : k + n])
            k += n
        return out
