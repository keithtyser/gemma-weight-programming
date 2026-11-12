"""GPU passes of the weight-edit spike over the aligned E4B forward: residual activations and per-token logprobs.

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: Runs the trainer's aligned forward (v2/tii/trainer/aligned_forward.py, the vLLM-numerics twin the RL learner
  uses: INT4 base through the engine GEMM, LoRA through Punica) teacher-forced on logged rollout calls.
    acts  For each call in <states>/states.jsonl: forward prompt + completion up to the last labelled position (one
          prefill chunk), capture the residual stream after EVERY decoder layer (output of `_layer`, i.e. after the
          attention write, MLP write, PLE write and layer_scalar) at the labelled positions, and save
          <out>/acts_<adapter>.pt = {"h": [N, 42, 2560] bf16, "meta": [{call, abs, tag, label, split}]}. The
          run saves <out>/acts_<adapter>.part.pt every --save-every calls and resumes from it. The
          first --check calls are also scored over the whole completion and compared with the engine's logged
          logprobs (fidelity receipt: proves the forward is the logged policy).
    nll   For calls in a --calls jsonl (normal on-policy trajectories), score every completion token under each
          --adapter and write <out>/nll.jsonl rows {call, adapter, lp_sum, n, lp_think_sum, n_think}. With samples
          drawn from adapter p (screen_w2f = e4b_wave2_final at T=1), mean(lp_p - lp_q) per token is an unbiased
          estimate of KL(p || q): the "is general behaviour intact" check for an edited adapter q.
  Every adapter (r16 source or r17-19 folded) is zero-padded to rank 32 inside one model (lora_rank 32, scaling 2):
  the LoRA product, hence the function, is unchanged; numerics are those of an r32 engine, which is fine for
  directions/KL (no bitwise claim is made here, the receipt reports the measured difference).
  Reads request logs (prompt/completion token ids + engine logprobs), the base checkpoint, PEFT adapter dirs.
  Writes only under --out. GPU: one card, ~16-20 GB peak at 32k tokens.
  The TII-Marlin .so is a build artifact (not in git): in a fresh worktree set TII_MARLIN_SO to a built copy, e.g.
  /path/to/gemma-v2/tii-dev/v2/tii/engine/tii_marlin/build_k128n64L128t3/tii_marlin_k128n64L128t3.so.
  Usage: CUDA_VISIBLE_DEVICES=1 python -m v2.steer.collect acts --states DIR --adapter ref=ADAPTER --out DIR
         CUDA_VISIBLE_DEVICES=1 python -m v2.steer.collect nll --calls F.jsonl --adapter ref=A --adapter ed=B --out DIR
SRP/DRY check: Pass - model construction copies the AlignedGemma4 setup of v2/tii/trainer/measure_full.py; hidden
  capture is a wrapper around AlignedGemma4._layer (no model code duplicated).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v2.tii.trainer import runtime

PAD_RANK = 32


def load_padded(model: Any, adapter_dir: Path) -> None:
    """Copy a PEFT adapter of rank r <= 32 into the model's rank-32 masters, zero-padded, keeping B A * scaling equal:
    the model's scaling is 2 (alpha 64 / r 32); the adapter's is alpha/r; B is rescaled by (alpha/r)/2."""
    import re

    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, s = int(cfg["r"]), float(cfg["lora_alpha"]) / float(cfg["r"])
    if r > PAD_RANK:
        raise ValueError(f"adapter rank {r} > {PAD_RANK}")
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    seen = set()
    with torch.no_grad():
        for p in model.lora_params.values():
            p.zero_()
        for k, v in sd.items():
            m = re.match(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
            if not m:
                raise KeyError(k)
            pk = model._pkey(m.group(1), m.group(2))
            t = v.to(torch.float32).to(model.dev)
            if m.group(2) == "A":
                model.lora_params[pk][:r].copy_(t)
            else:
                model.lora_params[pk][:, :r].copy_(t * (s / model.lora_scaling))
            seen.add(pk)
    missing = set(model.lora_params) - seen
    if missing:
        raise KeyError(f"{adapter_dir}: missing {len(missing)} tensors e.g. {sorted(missing)[:2]}")


def build_model(model_path: str) -> Any:
    runtime.setup_process(linear="marlin")
    from v2.tii.trainer.aligned_forward import AlignedGemma4

    num = runtime.engine_numerics({"linear_backend": runtime.AVENUE_TO_LINEAR["marlin"], "lm_head_backend": "bi_persistent",
                                   "custom_ops": True, "lora_max_rank": PAD_RANK, "lora_max_loras": 1,
                                   "kv_block_size": 16, "logprobs_mode": "processed_logprobs"})
    return AlignedGemma4(model_path, num, lora_rank=PAD_RANK, lora_alpha=2 * PAD_RANK, checkpoint_policy="none")


def read_call(log_path: str, idx: int) -> dict[str, Any]:
    with open(log_path) as fh:
        for i, line in enumerate(fh):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"{log_path} has no call {idx}")


def install_capture(model: Any) -> dict[str, Any]:
    """Wrap model._layer so that, when state['rows'] is set, the output of every layer is gathered at those rows."""
    state: dict[str, Any] = {"rows": None, "out": {}}
    orig = model._layer

    def hooked(i, h, *args, **kw):  # noqa: ANN001 - mirrors AlignedGemma4._layer
        y = orig(i, h, *args, **kw)
        if state["rows"] is not None:
            state["out"][i] = y.index_select(0, state["rows"]).to("cpu")
        return y

    model._layer = hooked
    # MLP input (after pre_feedforward_layernorm: what gate_proj / up_proj read) and the RMS of the down_proj output
    # (before post_feedforward_layernorm), per layer, gathered at the same rows when state["mlp"] is set.
    state.update({"mlp": False, "xin": {}, "yrms": {}})
    lidx = {id(L): i for i, L in enumerate(model.layers)}
    orig_proj = model._proj

    def hooked_proj(L, name, x, lora):  # noqa: ANN001 - mirrors AlignedGemma4._proj
        y = orig_proj(L, name, x, lora)
        if state["mlp"] and state["rows"] is not None:
            if name == "gate_up":
                state["xin"][lidx[id(L)]] = x.index_select(0, state["rows"]).to("cpu")
            elif name == "down":
                state["yrms"][lidx[id(L)]] = y.index_select(0, state["rows"]).float().pow(2).mean(-1).sqrt().to("cpu")
        return y

    model._proj = hooked_proj
    return state


def cmd_acts(a: argparse.Namespace) -> int:
    import torch

    model = build_model(a.model)
    cap = install_capture(model)
    rows = [json.loads(l) for l in (a.states / "states.jsonl").open()]
    if a.limit:
        rows = rows[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.adapter:
        name, ad = spec.split("=", 1)
        load_padded(model, Path(ad))
        H: list[torch.Tensor] = []
        X: list[torch.Tensor] = []
        Y: list[torch.Tensor] = []
        meta: list[dict[str, Any]] = []
        receipt: list[dict[str, Any]] = []
        t0, ntok = time.time(), 0
        # Resume: a stopped run leaves <out>/acts_<name>.part.pt (written every --save-every calls); continue after it.
        part = a.out / f"acts_{name}.part.pt"
        start = 0
        if part.exists():
            pd = torch.load(part, map_location="cpu")
            H, meta, receipt, start = [pd["h"]], pd["meta"], pd["receipt"], pd["done_calls"]
            print(f"[{name}] resuming after {start} calls", flush=True)
        for j, row in enumerate(rows):
            if j < start:
                continue
            call = read_call(row["request_log_path"], row["call"])
            prompt, comp = call["prompt_token_ids"], call["completion_token_ids"]
            P = len(prompt)
            check = j < a.check
            end = len(comp) if check else max(p["abs"] for p in row["positions"]) - P + 1
            seq = prompt + comp[:end]
            L = len(seq)
            pos = torch.tensor([p["abs"] for p in row["positions"]], device=model.dev)
            cap["rows"], cap["out"], cap["xin"], cap["yrms"], cap["mlp"] = pos, {}, {}, {}, bool(a.mlp_in)
            with torch.no_grad():
                ids = torch.tensor(seq, dtype=torch.long, device=model.dev)
                hidden = model.forward_hidden(ids, [(0, L)], [[(0, L)]], lora=True)
                if check:  # fidelity receipt vs the engine's logged processed logprobs (T = 1)
                    lp = model.score(hidden, torch.arange(P - 1, L - 1, device=model.dev),
                                     torch.tensor(seq[P:], device=model.dev), 1.0,
                                     forced=model.forced_of([(prompt, comp)]))
                    ref = torch.tensor(call["completion_logprobs"][: L - P], device=lp.device)
                    d = (lp.float() - ref).abs()
                    receipt.append({"call": row["run_key"], "n": int(d.numel()), "max_abs": float(d.max()),
                                    "mean_abs": float(d.mean()), "bitwise_frac": float((d == 0).float().mean())})
            cap["rows"] = None
            H.append(torch.stack([cap["out"][i] for i in range(len(cap["out"]))], 1).to(torch.bfloat16))
            if a.mlp_in:
                X.append(torch.stack([cap["xin"][i] for i in range(len(cap["xin"]))], 1).to(torch.bfloat16))
                Y.append(torch.stack([cap["yrms"][i] for i in range(len(cap["yrms"]))], 1))
            for p in row["positions"]:
                meta.append({"src": row["src"], "run_key": row["run_key"], "call": row["call"], "abs": p["abs"],
                             "tag": p["tag"], "label": p["label"], "split": row["split"],
                             "instance_id": row["instance_id"],
                             **{k: v for k, v in p.items() if k not in ("abs", "tag", "label")}})
            ntok += L
            if a.save_every and (j + 1) % a.save_every == 0:
                torch.save({"h": torch.cat(H, 0), "meta": meta, "receipt": receipt, "done_calls": j + 1}, part)
            if j % 50 == 0:
                print(f"[{name}] {j}/{len(rows)} calls, {ntok / max(1e-9, time.time() - t0):.0f} tok/s", flush=True)
        extra = {"xin": torch.cat(X, 0), "yrms": torch.cat(Y, 0)} if a.mlp_in else {}
        torch.save({"h": torch.cat(H, 0), "meta": meta, "receipt": receipt, "adapter": ad, **extra,
                    "tok_per_s": ntok / max(1e-9, time.time() - t0)}, a.out / f"acts_{name}.pt")
        part.unlink(missing_ok=True)
        print(json.dumps({"adapter": name, "positions": len(meta), "receipt": receipt[:5]}), flush=True)
    return 0


def cmd_nll(a: argparse.Namespace) -> int:
    import torch

    model = build_model(a.model)
    calls = [json.loads(l) for l in a.calls.open()]
    if a.limit:
        calls = calls[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "nll.jsonl").open("a") as fo:
        for spec in a.adapter:
            name, ad = spec.split("=", 1)
            load_padded(model, Path(ad))
            for c in calls:
                call = read_call(c["request_log_path"], c["call"])
                prompt, comp = call["prompt_token_ids"], call["completion_token_ids"]
                with torch.no_grad():
                    lp = model.rollout_logprobs([prompt + comp], [len(prompt)], 1.0, lora=True, one_shot=True)[0]
                lp = lp.float().cpu()
                te = c.get("think_end") or len(comp)
                fo.write(json.dumps({"run_key": c["run_key"], "call": c["call"], "adapter": name,
                                     "lp_sum": float(lp.sum()), "n": int(lp.numel()),
                                     "lp_think_sum": float(lp[:te].sum()), "n_think": int(min(te, lp.numel())),
                                     "engine_lp_sum": float(sum(call["completion_logprobs"]))}) + "\n")
                fo.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.collect")
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm in ("acts", "nll"):
        s = sub.add_parser(nm)
        s.add_argument("--model", default="/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct")
        s.add_argument("--adapter", action="append", required=True, help="NAME=DIR (repeatable)")
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--limit", type=int, default=0)
    sub.choices["acts"].add_argument("--states", type=Path, required=True)
    sub.choices["acts"].add_argument("--check", type=int, default=8, help="calls scored vs engine logprobs")
    sub.choices["acts"].add_argument("--save-every", type=int, default=100, help="partial-save period (calls) for resume")
    sub.choices["acts"].add_argument("--mlp-in", action="store_true",
                                     help="also save MLP inputs (xin) and down-output RMS (yrms); no partial saves")
    sub.choices["nll"].add_argument("--calls", type=Path, required=True)
    a = p.parse_args(argv)
    if getattr(a, "mlp_in", False):
        a.save_every = 0
    return cmd_acts(a) if a.cmd == "acts" else cmd_nll(a)


if __name__ == "__main__":
    raise SystemExit(main())
