"""vLLM 0.19.1 general plugin for the TII aligned rollout engine (sm120, Gemma 4 E4B).

Author: Claude Opus 5.5 (engine track), 2026-11-05.

Loaded in every vLLM process through the `vllm.general_plugins` entry point of the tiny shim package in
v2/tii/engine/plugin_pkg (installed into tii-venv only). It does nothing unless TII_ALIGNED=1, so the BI=0
production path stays byte-identical. With TII_ALIGNED=1 it applies, before model construction:

  P1 w4a16   : registers TiiTritonW4A16LinearKernel first in the CUDA mixed-precision kernel list
               (and VLLM_DISABLED_KERNELS should contain MarlinLinearKernel as a second guard).
  P2 sm120mm : vLLM's enable_batch_invariant_mode installs the Triton persistent-matmul aten overrides
               (mm/addmm/matmul/linear) only on sm100/80/89; on sm120 it only shrinks the cuBLAS workspace, so
               the tied lm_head (F.linear) runs cuBLASLt. P2 installs the same overrides on capability 12.x.
  P3 lora    : pins the Punica shrink kernel to block_k=32, split_k=1 for every M (0.19.1 uses block_k 256 for
               M<128 and 32 otherwise, even under VLLM_BATCH_INVARIANT=1).
  P4 attn    : (TII_ATTN_FIX=1, default on) replaces Triton unified attention's 2D kernel with a copy whose online
               softmax keeps M=-inf for rows that have seen only fully-masked tiles (see unified_attention_tii.py).

Every patch writes a receipt line to $TII_RECEIPT_DIR/receipt-<pid>.jsonl when that variable is set, and the
patched callables count their own invocations so a test can assert that the aligned path actually executed.
"""

from __future__ import annotations

import json
import os
import time

_APPLIED: dict[str, object] = {}
CALLS: dict[str, int] = {"w4a16_apply": 0, "lora_shrink_cfg": 0, "lora_expand_small": 0}


def _receipt(event: str, **kw) -> None:
    d = os.environ.get("TII_RECEIPT_DIR")
    rec = {"t": time.time(), "pid": os.getpid(), "event": event, **kw}
    _APPLIED[event] = kw or True
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"receipt-{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def _p1_w4a16(which: str = "w4a16") -> None:
    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.platforms import PlatformEnum

    if which == "marlin":
        from v2.tii.engine.vllm_tii_marlin_kernel import TiiMarlinLinearKernel as K
    else:
        from v2.tii.engine.vllm_w4a16_kernel import TiiTritonW4A16LinearKernel as K

    lst = _POSSIBLE_KERNELS[PlatformEnum.CUDA]
    if K not in lst:
        lst.insert(0, K)
    _receipt("P1_" + which, kernels=[k.__name__ for k in lst])


def _p2_sm120_matmul() -> None:
    import vllm.model_executor.layers.batch_invariant as bi
    from vllm.platforms import current_platform

    orig = bi.enable_batch_invariant_mode
    if getattr(orig, "_tii_wrapped", False):
        return

    def enable_batch_invariant_mode_tii():
        already = bi._batch_invariant_MODE
        orig()
        cap = current_platform.get_device_capability()
        if already or cap is None or cap.major != 12:
            return
        lib = bi._batch_invariant_LIB
        lib.impl("aten::mm", bi.mm_batch_invariant, "CUDA")
        lib.impl("aten::addmm", bi.addmm_batch_invariant, "CUDA")
        lib.impl("aten::matmul", bi.matmul_batch_invariant, "CUDA")
        lib.impl("aten::linear", bi.linear_batch_invariant, "CUDA")
        _receipt("P2_sm120_matmul_overrides", capability=f"{cap.major}.{cap.minor}")
        _receipt("worker_state", **worker_state())

    enable_batch_invariant_mode_tii._tii_wrapped = True
    bi.enable_batch_invariant_mode = enable_batch_invariant_mode_tii
    _receipt("P2_wrapper_installed")


def _p2c_lm_head_table() -> None:
    """lm_head (N=262144) persistent matmul with an M-bucketed tile table; bytes identical to vLLM's
    matmul_persistent (tests/test_engine_lm_head.py), 1.4-1.9x faster at decode M."""
    from v2.tii.engine import persistent_matmul_tii as pm

    pm.install()
    _receipt("P2c_lm_head_table", table="M<=16:(16,64,32,4,4) M<=64:(64,128,64,8,3) else vLLM default")


SHRINK_PIN = {"block_m": 32, "block_n": 16, "block_k": 32, "split_k": 1, "num_warps": 4, "num_ctas": 1,
              "group_size_m": 8, "num_stages": 2, "max_nreg": None}
# LoRA expand (K = rank = 16 <= block_k: one fused dot per output element, so the tile config does not change bits;
# tests/test_engine_lora_expand.py). Smaller tiles: 1.3-1.7x faster on qkv/o at M=48, neutral on gate_up/down.
EXPAND_SMALL = {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 4, "num_ctas": 1, "num_stages": 2,
                "max_nreg": None}
EXPAND_SMALL_MAX_M = 256
# wide outputs (gate_up, N = 10240 per slice): 16x128 tiles, 6.7 -> 5.4 us at M=48 (tests/bench_lora_expand_cfg.py;
# same bytes for every tile config since rank 16 is one fused dot)
EXPAND_WIDE = {"block_m": 16, "block_n": 128, "block_k": 32, "num_warps": 4, "num_ctas": 1, "num_stages": 2,
               "max_nreg": None}
EXPAND_WIDE_MIN_N = 8192


def _p3_lora_pin() -> None:
    import vllm.lora.ops.triton_ops.utils as u

    orig = u.get_lora_op_configs
    if getattr(orig, "_tii_wrapped", False):
        return

    def get_lora_op_configs_tii(op_type, max_loras, batch, hidden_size, rank, num_slices, add_inputs=None,
                                moe_intermediate_size=None):
        cfg = dict(orig(op_type, max_loras, batch, hidden_size, rank, num_slices, add_inputs, moe_intermediate_size))
        if op_type == "shrink":
            cfg.update(SHRINK_PIN)
            CALLS["lora_shrink_cfg"] += 1
        elif op_type == "expand" and os.environ.get("TII_LORA_EXPAND", "small") in ("small", "small1") \
                and batch <= EXPAND_SMALL_MAX_M and rank <= EXPAND_SMALL["block_k"]:
            wide = hidden_size >= EXPAND_WIDE_MIN_N and os.environ.get("TII_LORA_EXPAND", "small") == "small"
            cfg.update(EXPAND_WIDE if wide else EXPAND_SMALL)  # small1 = the round-1 table (no wide tiles)
            CALLS["lora_expand_small"] += 1
        return cfg

    get_lora_op_configs_tii._tii_wrapped = True
    u.get_lora_op_configs = get_lora_op_configs_tii
    patched = []
    import importlib

    for mod in ("vllm.lora.ops.triton_ops.lora_shrink_op", "vllm.lora.ops.triton_ops.lora_expand_op"):
        m = importlib.import_module(mod)
        if hasattr(m, "get_lora_op_configs"):
            m.get_lora_op_configs = get_lora_op_configs_tii
            patched.append(mod)
    _receipt("P3_lora_shrink_pin", modules=patched, shrink=SHRINK_PIN,
             expand=EXPAND_SMALL if os.environ.get("TII_LORA_EXPAND", "small") == "small" else "vllm")


def _p3b_lora_shrink_split() -> None:
    """P3b: Punica shrink -> TII fixed split-K (16) shrink with ordered reduction (v2/tii/engine/lora_shrink_tii.py)."""
    import vllm.lora.punica_wrapper.punica_gpu as pg

    from v2.tii.engine.lora_shrink_tii import VERSION, tii_lora_shrink, tii_lora_shrink_engine

    # TII_LORA_SHRINK_MEMSET=0 (default): skip zeroing the rows of no-adapter tokens, which the expand never reads
    memset = os.environ.get("TII_LORA_SHRINK_MEMSET", "0") == "1"
    pg.lora_shrink = _traced(tii_lora_shrink if memset else tii_lora_shrink_engine, "lora_shrink", _shrink_info)
    _receipt("P3b_lora_shrink_split", version=VERSION, memset=memset)


def _p4c_segmented_attention_v7() -> None:
    """P4c (TII_ATTN=seg7): unified attention -> TII segmented attention v7 in BOTH the Triton backend (engine) and
    the ops module (what the trainer's visible_attention imports), so engine and learner run the same function.
    Configs from env TII_SEG7_{PRE,DEC}_{256,512} (recorded in the receipt)."""
    import vllm.v1.attention.backends.triton_attn as ta
    import vllm.v1.attention.ops.triton_unified_attention as tua

    from v2.tii.engine import segmented_attention_v7 as v7

    fn = _traced(v7.unified_attention_tii, "seg7", _attn_info)
    ta.unified_attention = fn
    tua.unified_attention = fn
    _receipt("P4c_segmented_attention_v7", version=v7.VERSION,
             cfg={f"{k}_{hp}": v7._cfg(hp, k) for k in ("pre", "dec") for hp in (256, 512)})


def _p4b_segmented_attention() -> None:
    """P4b: the Triton attention backend calls TII segmented attention (fixed absolute 512-key segments; prefill =
    sequential in-register merge, decode = segment-parallel + ordered reduce, one binary)."""
    import vllm.v1.attention.backends.triton_attn as ta

    from v2.tii.engine.segmented_attention_tii import VERSION, unified_attention_tii

    ta.unified_attention = unified_attention_tii
    _receipt("P4b_segmented_attention", version=VERSION)


def _p4_attention() -> None:
    import vllm.v1.attention.ops.triton_unified_attention as tua

    from v2.tii.engine.unified_attention_tii import kernel_unified_attention_2d_tii

    if tua.kernel_unified_attention_2d is not kernel_unified_attention_2d_tii:
        tua.kernel_unified_attention_2d = kernel_unified_attention_2d_tii
    _receipt("P4_attention_2d_masked_tile_fix")


def worker_state() -> dict:
    """What is actually installed in this process (read back from the live modules, not from flags)."""
    import vllm.envs as envs
    import vllm.lora.ops.triton_ops.lora_shrink_op as shrink
    import vllm.v1.attention.ops.triton_unified_attention as tua
    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.platforms import PlatformEnum

    return {
        "mp_kernel_head": _POSSIBLE_KERNELS[PlatformEnum.CUDA][0].__name__,
        "disabled_kernels": list(envs.VLLM_DISABLED_KERNELS),
        "attn_2d_kernel": getattr(tua.kernel_unified_attention_2d, "fn", tua.kernel_unified_attention_2d).__name__,
        "attn_backend_entry": __import__("vllm.v1.attention.backends.triton_attn", fromlist=["x"]).unified_attention.__name__,
        "attn_is_batch_invariant": bool(tua.is_batch_invariant),
        "lora_shrink_cfg_wrapped": bool(getattr(shrink.get_lora_op_configs, "_tii_wrapped", False)),
        "lora_shrink_entry": __import__("vllm.lora.punica_wrapper.punica_gpu", fromlist=["x"]).lora_shrink.__name__
        if hasattr(__import__("vllm.lora.punica_wrapper.punica_gpu", fromlist=["x"]).lora_shrink, "__name__")
        else str(__import__("vllm.lora.punica_wrapper.punica_gpu", fromlist=["x"]).lora_shrink),
        "vllm_batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
        "linear_smallm": os.environ.get("TII_SMALLM", ""),
        "linear_largem": os.environ.get("TII_LARGEM", ""),
        "lm_head_matmul": getattr(__import__("vllm.model_executor.layers.batch_invariant", fromlist=["x"]).matmul_persistent,
                                  "__qualname__", "?"),
    }


def _traced(fn, name: str, info_fn):
    """Wrap an op so the hang watch (P7) records its launch; a no-op unless the watch is installed."""
    from v2.tii.engine import hang_watch

    def w(*a, **k):
        out = fn(*a, **k)
        if hang_watch._STATE["installed"]:
            try:
                info = info_fn(a, k)
            except Exception:  # noqa: BLE001
                info = ""
            hang_watch.trace(name, info)
        return out

    w.__name__ = getattr(fn, "__name__", name)
    w.__wrapped__ = fn
    return w


def _attn_info(a, k) -> str:
    q = k.get("q", a[0] if a else None)
    sk = k.get("seqused_k")
    ws = k.get("window_size")
    return f"T={q.shape[0]} hd={q.shape[2]} seqs={sk.shape[0] if sk is not None else '?'} maxq={k.get('max_seqlen_q')} "            f"win={ws} bt={tuple(k['block_table'].shape) if 'block_table' in k else '?'}"


def _shrink_info(a, k) -> str:
    x = a[0] if a else k.get("inputs")
    w = a[1] if len(a) > 1 else k.get("lora_a_weights")
    return f"M={x.shape[0]} K={x.shape[1]} slices={len(w)}"


def _p7_hang_watch() -> None:
    """P7: hang dump (v2/tii/engine/hang_watch.py) in the vLLM EngineCore process; TII_HANG_WATCH=0 disables."""
    import sys

    if os.environ.get("TII_HANG_WATCH", "1") != "1":
        return
    if "v2.tii.trainer.runtime" in sys.modules and os.environ.get("TII_HANG_WATCH_FORCE") != "1":
        return  # the learner process (its forward never runs GPUModelRunner); every vLLM process installs it
    from v2.tii.engine import hang_watch

    _receipt("P7_hang_watch", **hang_watch.install())


def register() -> None:
    if os.environ.get("TII_ALIGNED", "0") != "1":
        return
    if "registered" in _APPLIED:
        _p7_hang_watch()  # a forked child (EngineCore) inherits the patches but not the watchdog thread
        return
    _receipt("registered", env={k: v for k, v in os.environ.items() if k.startswith(("TII_", "VLLM_BATCH",
                                                                                     "VLLM_DISABLED"))})
    if os.environ.get("TII_LINEAR", "w4a16") in ("w4a16", "marlin"):
        _p1_w4a16(os.environ.get("TII_LINEAR", "w4a16"))
    _p2_sm120_matmul()
    _p3_lora_pin()
    if os.environ.get("TII_ATTN_FIX", "1") == "1":
        _p4_attention()
    if os.environ.get("TII_LORA_SHRINK", "pin") == "split":
        _p3b_lora_shrink_split()
    if os.environ.get("TII_ATTN", "2d") == "seg":
        _p4b_segmented_attention()
    if os.environ.get("TII_ATTN", "2d") == "seg7":
        _p4c_segmented_attention_v7()
    if os.environ.get("TII_LMHEAD", "table") == "table":
        _p2c_lm_head_table()
    if os.environ.get("TII_FUSE_NORMS", "0") == "1":
        _p5_fused_norms()
    if os.environ.get("TII_LOGSOFTMAX", "vllm") == "par":
        _p6_log_softmax()
    _p7_hang_watch()


def _p6_log_softmax() -> None:
    """P6 (TII_LOGSOFTMAX=par): the aten::_log_softmax BI override calls a row-parallel kernel with the same bytes
    as vLLM's one-program-per-row _log_softmax_kernel (v2/tii/engine/log_softmax_tii.py)."""
    import vllm.model_executor.layers.batch_invariant as bi

    from v2.tii.engine import log_softmax_tii as ls

    bi.log_softmax = _traced(ls.log_softmax, "log_softmax", lambda a, k: f"shape={tuple(a[0].shape)}")
    _receipt("P6_log_softmax_par", version=ls.VERSION)


def _p5_fused_norms() -> None:
    """P5 (TII_FUSE_NORMS=1): Gemma 4 decoder layers with fused RMSNorm + residual add (+ layer scalar) (+ next
    norm) epilogues, bit-identical to the unfused BI ops (v2/tii/engine/fused_norms.py)."""
    import vllm.model_executor.models.gemma4 as g4

    from v2.tii.engine import fused_norms as fn

    if getattr(g4._run_decoder_layers, "__name__", "") != "run_decoder_layers_fused":  # fast-prefill (YOCO) path
        fn._ORIG["fn"] = g4._run_decoder_layers
        g4._run_decoder_layers = fn.run_decoder_layers_fused
    if getattr(g4.Gemma4DecoderLayer.forward, "__name__", "") != "decoder_layer_forward_fused":  # normal path
        fn._ORIG["layer_forward"] = g4.Gemma4DecoderLayer.forward
        g4.Gemma4DecoderLayer.forward = fn.decoder_layer_forward_fused
        fn._ORIG["model_forward"] = g4.Gemma4Model.forward
        g4.Gemma4Model.forward = fn.model_forward_linked
    qkv = os.environ.get("TII_FUSE_QKV_NORM", "1") == "1"
    if qkv and getattr(g4.Gemma4Attention.forward, "__name__", "") != "attention_forward_fused":
        fn._ORIG["attn_forward"] = g4.Gemma4Attention.forward
        g4.Gemma4Attention.forward = fn.attention_forward_fused
    _receipt("P5_fused_norms", version=fn.VERSION, qkv_norm=qkv)
