"""TII fused RMSNorm epilogues for the Gemma 4 decoder layer: fewer launches, the SAME bytes.

Author: Claude Opus 5.5 (engine track), 2026-11-06.

Under VLLM_BATCH_INVARIANT=1 every RMSNorm of vLLM 0.19.1 runs batch_invariant._rms_norm_kernel (one row per program,
fp32 sum of squares over BLOCK_SIZE=1024 chunks with tl.sum, inv = 1 / sqrt(sum / n + eps), out = bf16(x * inv * w)),
and the residual adds / layer scalar run as separate torch bf16 elementwise kernels (fp32 op, one bf16 rounding).
A Gemma 4 decoder layer launches 5 norms + 3 adds + 1 scalar mul. _tii_norm_add computes, per row,

    t = bf16(x * inv(x) * w1)            exactly the BI norm expressions (same chunking, same tl.sum, same options)
    h = bf16(f32(t) + f32(r))            torch bf16 add (fp32 add, one rounding)            [HAS_RES]
    h = bf16(f32(h) * f32(s))            torch bf16 mul by the layer scalar                 [HAS_SCALE]
    y = bf16(h * inv(h) * w2)            the next BI norm of h                              [HAS_NORM2]

so one launch replaces norm (+ add) (+ mul) (+ norm) with bit-identical outputs (tests/test_engine_fused_norms.py
checks this against the unfused vLLM ops, and C3 compares a fused engine with an unfused one). Nothing changes for the
trainer: its forward produces the same bytes as the unfused ops.
Patch (vllm_plugin P5, TII_FUSE_NORMS=1): gemma4._run_decoder_layers -> run_decoder_layers_fused, which per layer does
    [input norm only for the first layer of a slice] -> attention -> K1: h = post_attn_norm(a) + res, m = pre_ff_norm(h)
    -> MLP -> K2: h2 = post_ff_norm(mlp) + h -> PLE gate/gelu/mul/projection (unchanged torch ops)
    -> K4: h3 = (post_ple_norm(p) + h2) * layer_scalar, y = next layer's input_norm(h3) (inside a slice).
"""

from __future__ import annotations

import os

import torch

from triton.language.extra.cuda import libdevice

from vllm.triton_utils import tl, triton

from v2.tii.engine import hang_watch as _HW  # noqa: E402

FUSE_GELU = os.environ.get("TII_FUSE_GELU", "0") == "1"  # enable only after the exhaustive bf16 test passes

BLOCK = 1024  # = vLLM batch_invariant.rms_norm BLOCK_SIZE (defines the tl.sum tree; do not change)
VERSION = "tii-fused-norms-v1"


@triton.jit
def _tii_norm_add(x_ptr, r_ptr, w1_ptr, s_ptr, w2_ptr, h_ptr, y_ptr, x_stride, r_stride, h_stride, y_stride, n_cols,
                  eps1, eps2, HAS_RES: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_NORM2: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = x_ptr + row * x_stride
    hr = h_ptr + row * h_stride
    # pass 1: BI sum of squares of x (same statements as batch_invariant._rms_norm_kernel)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(xr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps1)
    inv_rms = 1.0 / rms
    # pass 2: normalize, weight, round; residual add; layer scalar; second sum of squares
    sum_sq2 = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(xr + col_idx, mask=mask, other=0.0)
        weight = tl.load(w1_ptr + col_idx, mask=mask, other=1.0)
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        h = output_f32.to(vals.dtype)
        if HAS_RES:
            res = tl.load(r_ptr + row * r_stride + col_idx, mask=mask, other=0.0)
            h = (h.to(tl.float32) + res.to(tl.float32)).to(vals.dtype)
        if HAS_SCALE:
            sc = tl.load(s_ptr)
            h = (h.to(tl.float32) * sc.to(tl.float32)).to(vals.dtype)
        tl.store(hr + col_idx, h, mask=mask)
        if HAS_NORM2:
            h_f32 = h.to(tl.float32)
            sq2 = h_f32 * h_f32
            sum_sq2 += tl.sum(tl.where(mask, sq2, 0.0))
    if HAS_NORM2:
        mean_sq2 = sum_sq2 / n_cols
        rms2 = tl.sqrt(mean_sq2 + eps2)
        inv_rms2 = 1.0 / rms2
        yr = y_ptr + row * y_stride
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(hr + col_idx, mask=mask, other=0.0)
            weight = tl.load(w2_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms2 * weight_f32
            tl.store(yr + col_idx, output_f32.to(vals.dtype), mask=mask)


@triton.jit
def _tii_qkv_norm(qkv_ptr, wq_ptr, wk_ptr, q_ptr, k_ptr, v_ptr, qkv_stride, eps, HEAD: tl.constexpr,
                  NH: tl.constexpr, NKV: tl.constexpr, DO_KV: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """grid (tokens, NH [+ 2 NKV]): per-head BI RMSNorm of q (weight wq), k (weight wk), v (no weight = the all-ones
    weight of vLLM's has_weight=False RMSNorm: x * inv * 1.0 == x * inv), read straight from the fused qkv
    projection output (no .contiguous() copies), written to contiguous q [T, NH*HEAD], k/v [T, NKV*HEAD].
    Same statements as batch_invariant._rms_norm_kernel for a row of HEAD columns (one BLOCK_SIZE chunk)."""
    t = tl.program_id(0).to(tl.int64)
    hidx = tl.program_id(1)
    if hidx < NH:
        src = qkv_ptr + t * qkv_stride + hidx * HEAD
        dst = q_ptr + t * (NH * HEAD) + hidx * HEAD
        w_ptr = wq_ptr
    elif hidx < NH + NKV:
        src = qkv_ptr + t * qkv_stride + (NH + (hidx - NH)) * HEAD
        dst = k_ptr + t * (NKV * HEAD) + (hidx - NH) * HEAD
        w_ptr = wk_ptr
    else:
        src = qkv_ptr + t * qkv_stride + (NH + NKV + (hidx - NH - NKV)) * HEAD
        dst = v_ptr + t * (NKV * HEAD) + (hidx - NH - NKV) * HEAD
        w_ptr = wk_ptr  # unused (no weight)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, HEAD, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < HEAD
        vals = tl.load(src + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / HEAD
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for col_offset in range(0, HEAD, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < HEAD
        vals = tl.load(src + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        if hidx < NH + NKV:
            weight = tl.load(w_ptr + col_idx, mask=mask, other=1.0)
            output_f32 = vals_f32 * inv_rms * weight.to(tl.float32)
        else:
            output_f32 = vals_f32 * inv_rms * 1.0
        tl.store(dst + col_idx, output_f32.to(vals.dtype), mask=mask)


def qkv_norm(qkv: torch.Tensor, nh: int, nkv: int, head: int, wq: torch.Tensor, wk: torch.Tensor | None, eps: float,
             do_kv: bool):
    """(q, k, v) normalized from the fused qkv output [T, (nh + 2 nkv) * head]; k/v None unless do_kv."""
    assert qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[1] == (nh + 2 * nkv) * head
    t = qkv.shape[0]
    q = torch.empty((t, nh * head), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((t, nkv * head), dtype=qkv.dtype, device=qkv.device) if do_kv else q
    v = torch.empty((t, nkv * head), dtype=qkv.dtype, device=qkv.device) if do_kv else q
    if t > 0:
        _tii_qkv_norm[(t, nh + (2 * nkv if do_kv else 0))](qkv, wq, wk if wk is not None else wq, q, k, v,
                                                            qkv.stride(0), eps, HEAD=head, NH=nh, NKV=nkv,
                                                            DO_KV=do_kv, BLOCK_SIZE=BLOCK)
    return q, (k if do_kv else None), (v if do_kv else None)


def attention_forward_fused(self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
    """Gemma4Attention.forward (vLLM 0.19.1) with q/k/v norms in one launch and no copies; same bytes."""
    qkv, _ = self.qkv_proj(hidden_states)
    ok = (qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.dtype == torch.bfloat16
          and getattr(self.v_norm, "has_weight", True) is False
          and all(getattr(n, "variance_size_override", None) is None for n in (self.q_norm, self.k_norm, self.v_norm))
          and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon == self.v_norm.variance_epsilon)
    if not ok:
        _ORIG["attn_fallbacks"] = _ORIG.get("attn_fallbacks", 0) + 1
        return _ORIG["attn_forward"](self, positions, hidden_states, **kwargs)
    shared = self.is_kv_shared_layer
    q, k, v = qkv_norm(qkv, self.num_heads, self.num_kv_heads, self.head_dim, self.q_norm.weight.data,
                       self.k_norm.weight.data, self.q_norm.variance_epsilon, do_kv=not shared)
    if not shared:
        q, k = self.rotary_emb(positions, q, k)
    else:  # as vLLM: raw k/v slices (RoPE on k is discarded; the shared layer reads the donor layer's KV cache)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.rotary_emb(positions, q, k)[0]
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


# torch GeluCUDAKernelImpl (tanh): kBeta = M_SQRT2 * M_2_SQRTPI * 0.5 (double, then float), kKappa = 0.044715f
_GELU_BETA = 1.41421356237309504880 * 1.12837916709551257390 * 0.5


@triton.jit
def _tii_gelu_mul(g_ptr, p_ptr, o_ptr, n_cols, g_stride, p_stride, o_stride, BETA: tl.constexpr,
                  BLOCK: tl.constexpr):
    """o = bf16(f32(bf16(gelu_tanh(g))) * f32(p)): torch gelu(approximate='tanh') (opmath fp32, nvcc FMA contraction
    of x + kKappa * x^3) followed by the torch bf16 mul; p may be a strided view (per_layer_inputs[:, layer, :])."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(g_ptr + row * g_stride + cols, mask=mask, other=0.0)
    p = tl.load(p_ptr + row * p_stride + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    x_cube = xf * xf * xf
    inner = BETA * tl.fma(0.044715, x_cube, xf)
    out = 0.5 * xf * (1.0 + libdevice.tanh(inner))
    gb = out.to(x.dtype)
    tl.store(o_ptr + row * o_stride + cols, (gb.to(tl.float32) * p.to(tl.float32)).to(x.dtype), mask=mask)


def gelu_mul(gate: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
    assert gate.dim() == 2 and pli.dim() == 2 and gate.shape == pli.shape and gate.stride(1) == 1 and pli.stride(1) == 1
    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    m, n = gate.shape
    if m > 0:
        _tii_gelu_mul[(m, triton.cdiv(n, 256))](gate, pli, out, n, gate.stride(0), pli.stride(0), out.stride(0),
                                                 BETA=_GELU_BETA, BLOCK=256)
    return out


def norm_add(x: torch.Tensor, w1: torch.Tensor, eps1: float, residual: torch.Tensor | None = None,
             scale: torch.Tensor | None = None, w2: torch.Tensor | None = None, eps2: float = 0.0):
    """Returns (h, y): h = [norm1(x) (+ residual)] (* scale) and y = norm2(h) (None without w2). 2D inputs."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n).contiguous()  # the BI rms_norm makes its input contiguous too
    m = x2.shape[0]
    h = torch.empty_like(x2)
    y = torch.empty_like(x2) if w2 is not None else h
    r2 = residual.reshape(-1, n) if residual is not None else x2
    if residual is not None and r2.stride(-1) != 1:
        r2 = r2.contiguous()
    if m > 0:
        _tii_norm_add[(m,)](x2, r2, w1.contiguous(), scale if scale is not None else w1,
                            w2.contiguous() if w2 is not None else w1, h, y, x2.stride(0), r2.stride(0), h.stride(0),
                            y.stride(0), n, eps1, eps2, HAS_RES=residual is not None, HAS_SCALE=scale is not None,
                            HAS_NORM2=w2 is not None, BLOCK_SIZE=BLOCK)
    return h.reshape(x.shape), (y.reshape(x.shape) if w2 is not None else None)


def _layer_forward(layer, positions, hidden_states, normed, per_layer_input, next_norm, **kwargs):
    """Gemma4DecoderLayer.forward (vLLM 0.19.1, no MoE block) with the K1/K2/K4 fusions; same bytes."""
    residual = hidden_states
    if normed is None:
        normed = layer.input_layernorm(residual)
    a = layer.self_attn(positions=positions, hidden_states=normed, **kwargs)
    pa, pf = layer.post_attention_layernorm, layer.pre_feedforward_layernorm
    h, m_in = norm_add(a, pa.weight.data, pa.variance_epsilon, residual=residual, w2=pf.weight.data,
                       eps2=pf.variance_epsilon)
    mo = layer.mlp(m_in)
    po = layer.post_feedforward_layernorm
    h2, _ = norm_add(mo, po.weight.data, po.variance_epsilon, residual=h)
    nw = next_norm.weight.data if next_norm is not None else None
    ne = next_norm.variance_epsilon if next_norm is not None else 0.0
    if per_layer_input is not None and layer.per_layer_input_gate is not None:
        gate = layer.per_layer_input_gate(h2)
        if FUSE_GELU and gate.dim() == 2 and gate.stride(1) == 1 and per_layer_input.stride(-1) == 1:
            gated = gelu_mul(gate, per_layer_input)
        else:
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated = gate * per_layer_input
        plc = layer.per_layer_projection(gated)
        pn = layer.post_per_layer_input_norm
        h3, y = norm_add(plc, pn.weight.data, pn.variance_epsilon, residual=h2, scale=layer.layer_scalar, w2=nw,
                         eps2=ne)
        if _HW._STATE["installed"]:
            _HW.trace("layer_end", f"layer={getattr(layer, 'layer_idx', '?')} T={h3.shape[0]}")
        return h3, y
    h3 = h2 * layer.layer_scalar
    return h3, (next_norm(h3) if next_norm is not None else None)


def supported(layer) -> bool:
    ok = not getattr(layer, "enable_moe_block", False)
    for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                 "post_feedforward_layernorm"):
        nm = getattr(layer, name)
        ok = ok and getattr(nm, "has_weight", True) and getattr(nm, "variance_size_override", None) is None
    pn = getattr(layer, "post_per_layer_input_norm", None)
    if pn is not None:
        ok = ok and getattr(pn, "has_weight", True) and getattr(pn, "variance_size_override", None) is None
    ls = getattr(layer, "layer_scalar", None)
    return ok and ls is not None and ls.numel() == 1 and ls.dtype == torch.bfloat16


_ORIG = {}


def decoder_layer_forward_fused(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
    """Gemma4DecoderLayer.forward drop-in (the normal, non-fast-prefill Gemma4Model.forward loop). vLLM's layer
    returns (h, None) and ignores the incoming residual; here the residual slot carries the NEXT layer's input-norm
    output y computed by K4 (self._tii_next_norm, linked by model_forward_linked; None for the last layer, so the
    model's final norm sees residual None exactly as before). A layer that falls back simply recomputes its norm."""
    nxt = getattr(self, "_tii_next_norm", "unlinked")
    if isinstance(nxt, str) or hidden_states.dtype != torch.bfloat16 or not supported(self):
        _ORIG["layer_fallbacks"] = _ORIG.get("layer_fallbacks", 0) + 1
        h, _ = _ORIG["layer_forward"](self, positions, hidden_states, None, per_layer_input=per_layer_input, **kwargs)
        return h, None
    _ORIG["layer_fused_calls"] = _ORIG.get("layer_fused_calls", 0) + 1
    return _layer_forward(self, positions, hidden_states, residual, per_layer_input, nxt, **kwargs)


def model_forward_linked(self, *args, **kwargs):
    """Gemma4Model.forward wrapper: links each decoder layer to the next layer's input_layernorm once."""
    if not getattr(self, "_tii_linked", False):
        from itertools import islice

        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        for i, layer in enumerate(layers):
            layer._tii_next_norm = layers[i + 1].input_layernorm if i + 1 < len(layers) else None
        self._tii_linked = True
    return _ORIG["model_forward"](self, *args, **kwargs)


def run_decoder_layers_fused(decoder_layers, layer_idx_start, positions, hidden_states, per_layer_inputs=None,
                             **kwargs):
    """Drop-in for vllm gemma4._run_decoder_layers (same arguments, same returned hidden states). Falls back to the
    original function when a layer has a feature the fused path does not cover (MoE block, weightless norms, ...)."""
    if hidden_states.dtype != torch.bfloat16 or not all(supported(layer) for layer in decoder_layers):
        _ORIG["fallbacks"] = _ORIG.get("fallbacks", 0) + 1
        return _ORIG["fn"](decoder_layers, layer_idx_start, positions, hidden_states, per_layer_inputs, **kwargs)
    _ORIG["fused_calls"] = _ORIG.get("fused_calls", 0) + 1
    normed = None
    nl = len(decoder_layers)
    for idx, layer in enumerate(decoder_layers):
        layer_idx = idx + layer_idx_start
        lpi = per_layer_inputs[:, layer_idx, :] if per_layer_inputs is not None else None
        nxt = decoder_layers[idx + 1].input_layernorm if idx + 1 < nl else None
        hidden_states, normed = _layer_forward(layer, positions, hidden_states, normed, lpi, nxt, **kwargs)
    return hidden_states
