"""D1 part 2: greedy regeneration of the decision call, compiled adapter vs inference-time hooks (HF transformers).

Engine for every arm: transformers Gemma4ForConditionalGeneration, bf16, weights = the INT4 groups of the served
checkpoint dequantised to dense bf16 by compressed-tensors (checked here against d1_fidelity.dequant). This is NOT the
vLLM numerics path (dense bf16 GEMM instead of the INT4 marlin kernel); all arms share it. compressed-tensors wraps
every Linear to fake-quantise its weight on each forward; the weights are already on the INT4 grid, so that wrapper is
switched off (disable_quantization) and the switch is checked at start-up on 512 tokens (written to meta.json). Arms are batch rows of one
batched forward (same prompt), realised by per-row hooks:
  base  nothing
  A     adapter as vLLM applies it (unmerged LoRA: y += (alpha/r) * B(A x) on the gate/up/down projections), bf16
  A2    (only with --a2) the same adapter in a second batch row: checks that batch rows are deterministic
  N     neuron hook: MLP output += gelu_tanh(g.xh) * (u.xh) * v (fp32, cast back); old neuron not removed
  V     vector hook: layer output += m * d_unit (gated edit: scaled per position by a(xh) / a_ref)
  C, CV the control adapter and its vector hook
Decoding as v2/steer/replay.py: greedy, max 3072 tokens, stop at token 49 (or a generation_config EOS id), thinking
budget 2560 by the per-token rule of v2/tii/trainer/logits_processors.py (after <|channel>, token budget+1 is forced
to <channel|>); read-outs identical to replay.gen. Prompt is prefilled in chunks of 2048 tokens (exact, causal).

Usage: python d1_regen.py --points points.jsonl --out DIR --pair explore_more_0.25 [--limit N] [--minutes M]
"""
import argparse, json, random, re, sys, time, warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_fidelity import E, MODEL, PAIRS, PFX, WRITE, dequant, edit_spec  # noqa: E402

END_THINK, END_CALL = 101, 49
NAME = re.compile(r"<\|tool_call>call:([a-z_]+)")
OLD = re.compile(r'old_string:<\|"\|>(.*?)<\|"\|>', re.S)


class Arms:
    def __init__(self, model, specs, loras):
        self.rows, self.specs, self.loras, self.xh = [], specs, loras, {}
        layers = model.model.language_model.layers
        for li in WRITE:
            L = layers[li]
            L.mlp.register_forward_pre_hook(lambda mod, args, li=li: self.xh.__setitem__(li, args[0]))
            for nm in ("gate", "up", "down"):
                getattr(L.mlp, nm + "_proj").register_forward_hook(
                    lambda mod, args, out, li=li, nm=nm: self.proj(li, nm, args[0], out))
            L.mlp.register_forward_hook(lambda mod, args, out, li=li: self.neuron(li, out))
            L.register_forward_hook(lambda mod, args, out, li=li: self.vec(li, out))

    def act(self, li, b, s):
        xh = self.xh[li][b].float()
        return F.gelu(xh @ s["g"], approximate="tanh") * (xh @ s["u"])

    def proj(self, li, nm, x, out):
        for b, (_, ad, _, _) in enumerate(self.rows):
            if ad:
                A, Bm, s = self.loras[ad][(li, nm)]
                out[b] = out[b] + s * F.linear(F.linear(x[b], A), Bm)
        return out

    def neuron(self, li, out):
        for b, (_, _, ne, _) in enumerate(self.rows):
            if ne:
                s = self.specs[ne][li]
                out[b] = (out[b].float() + self.act(li, b, s)[:, None] * s["v"][None, :]).to(out.dtype)
        return out

    def vec(self, li, out):
        assert torch.is_tensor(out)
        for b, (_, _, _, ve) in enumerate(self.rows):
            if ve:
                s = self.specs[ve][li]
                add = (s["m"] * s["d_unit"])[None, :]
                if s["mode"] == "gated":
                    add = (self.act(li, b, s) / s["a_ref"])[:, None] * add
                out[b] = (out[b].float() + add).to(out.dtype)
        return out


def last(seq, t):
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == t:
            return i
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=2560)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--a2", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard", default="0/1", help="i/n: length tier i of n of the (limited) point set")
    a = p.parse_args()
    t0 = time.time()
    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    import transformers
    from transformers.utils.quantization_config import CompressedTensorsConfig

    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, quantization_config=CompressedTensorsConfig(run_compressed=False)).cuda().eval()
    dev = "cuda"
    start = tok.encode("<|channel>", add_special_tokens=False)
    assert len(start) == 1 and tok.encode("<channel|>", add_special_tokens=False) == [END_THINK]
    START = start[0]
    stops = {END_CALL} | set(json.loads((MODEL / "generation_config.json").read_text())["eos_token_id"])
    base = safe_open(str(MODEL / "model.safetensors"), "pt")
    W, layers = {}, model.model.language_model.layers
    for li in WRITE:
        bp = f"model.language_model.layers.{li}."
        W[li] = (dequant(base, bp + "mlp.gate_proj"), dequant(base, bp + "mlp.up_proj"), dequant(base, bp + "mlp.down_proj"),
                 base.get_tensor(bp + "post_feedforward_layernorm.weight").float(),
                 float(base.get_tensor(bp + "layer_scalar").float()))
        for k, nm in enumerate(("gate", "up", "down")):  # HF dense weights == our dequantised INT4 weights
            w = getattr(layers[li].mlp, nm + "_proj").weight
            assert w.dtype == torch.bfloat16 and torch.equal(w.float().cpu(), W[li][k]), (li, nm)
    zero_sd = load_file(str(E / "zero" / "adapter_model.safetensors"))
    real, ctrl = PAIRS[a.pair]
    specs, loras = {}, {}
    for n in (real, ctrl):
        sp = edit_spec(n, base, zero_sd, W)
        specs[n] = {li: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()} for li, s in sp.items()}
        cfg = json.loads((E / n / "adapter_config.json").read_text())
        sc = float(cfg["lora_alpha"]) / float(cfg["r"])
        sd = load_file(str(E / n / "adapter_model.safetensors"))
        loras[n] = {(li, nm): (sd[f"{PFX}{li}.mlp.{nm}_proj.lora_A.weight"].to(dev, torch.bfloat16),
                               sd[f"{PFX}{li}.mlp.{nm}_proj.lora_B.weight"].to(dev, torch.bfloat16), sc)
                    for li in WRITE for nm in ("gate", "up", "down")}
    del W
    arms = Arms(model, specs, loras)
    arms.rows = [("base", None, None, None), ("A", real, None, None), ("N", None, real, None),
                 ("V", None, None, real), ("C", ctrl, None, None), ("CV", None, None, ctrl)]
    if a.a2:
        arms.rows.append(("A2", real, None, None))
    names = [r[0] for r in arms.rows]
    B = len(names)
    pts = [json.loads(x) for x in a.points.open()]
    rng = random.Random(a.seed)
    by = {lab: [x for x in pts if x["label"] == lab] for lab in (1, 0)}
    for v in by.values():
        rng.shuffle(v)
    order = [x for pr in zip(by[1], by[0]) for x in pr]  # the same seeded, label-balanced order as d1_fidelity
    if a.limit:
        order = order[:a.limit]
    si, sn = map(int, a.shard.split("/"))
    if sn > 1:  # length tiers of the limited set (longest tier first, last tier shortest-first) so parallel shards fit one GPU
        srt = sorted(order, key=lambda x: -len(x["prompt"]))
        k = -(-len(srt) // sn)
        order = srt[si * k:(si + 1) * k][::-1 if si == sn - 1 else 1]
    a.out.mkdir(parents=True, exist_ok=True)
    from compressed_tensors.quantization import disable_quantization
    chk = torch.tensor(order[0]["prompt"][:512], device=dev)[None, :].repeat(B, 1)  # row 0 = base
    z_on = model(input_ids=chk).logits[0].float()
    model.apply(disable_quantization)
    z_off = model(input_ids=chk).logits[0].float()
    meta = {"pair": a.pair, "arms": names, "shard": a.shard, "fake_quant_off_check": {
        "max_abs_logit_diff": float((z_on - z_off).abs().max()), "argmax_equal": float((z_on.argmax(-1) == z_off.argmax(-1)).float().mean()),
        "kl_mean": float((torch.log_softmax(z_on, -1).exp() * (torch.log_softmax(z_on, -1) - torch.log_softmax(z_off, -1))).sum(-1).mean())}}
    (a.out / f"meta_{si}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    del z_on, z_off
    print("loaded", round(time.time() - t0), names, flush=True)
    done, rows_name = 0, f"rows_{si}.jsonl"
    have = {(r["run_key"], r["call"]) for r in map(json.loads, (a.out / rows_name).open())} if a.resume and (a.out / rows_name).exists() else set()
    with (a.out / rows_name).open("a" if a.resume else "w") as fh:
        for pt in order:
            if (time.time() - t0) / 60 > a.minutes:
                break
            if (pt["run_key"], pt["call"]) in have:
                continue
            prompt = pt["prompt"]
            assert not last(prompt, START) > last(prompt, END_THINK)  # prompt does not end inside a thinking section
            t1 = time.time()
            ids = torch.tensor(prompt, device=dev)[None, :].repeat(B, 1)
            past, out = None, None
            for c in range(0, ids.shape[1], 2048):
                out = model(input_ids=ids[:, c:c + 2048], past_key_values=past, use_cache=True)
                past = out.past_key_values
            lg = out.logits[:, -1].float()
            t2 = time.time()
            lp0 = torch.log_softmax(lg, -1)
            kl0 = {f"{x}|{y}": float((lp0[names.index(x)].exp() * (lp0[names.index(x)] - lp0[names.index(y)])).sum())
                   for x, y in (("base", "A"), ("N", "A"), ("V", "A"), ("CV", "C"), ("C", "A"))}
            toks = [[] for _ in range(B)]
            fin = [None] * B
            in_think, cnt, force = [False] * B, [0] * B, [False] * B
            for _ in range(a.max_tokens):
                nxt = lg.argmax(-1)
                for b in range(B):
                    if fin[b]:
                        nxt[b] = 0
                        continue
                    if force[b]:
                        nxt[b], force[b] = END_THINK, False
                    t = int(nxt[b])
                    if t in stops:
                        fin[b] = "stop"
                        continue
                    toks[b].append(t)
                    if t == START:
                        in_think[b], cnt[b] = True, 0
                    elif t == END_THINK:
                        in_think[b], cnt[b] = False, 0
                    elif in_think[b]:
                        cnt[b] += 1
                    if in_think[b] and cnt[b] >= a.budget:
                        in_think[b], force[b] = False, True
                if all(fin):
                    break
                out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                lg = out.logits[:, -1].float()
            row = {"run_key": pt["run_key"], "call": pt["call"], "label": pt["label"], "P": len(prompt), "kl_first": kl0,
                   "sec": round(time.time() - t1, 1), "prefill_sec": round(t2 - t1, 1),
                   "steps": max(len(t) for t in toks), "arms": {}}
            for b, nm in enumerate(names):
                text = tok.decode(toks[b], skip_special_tokens=False)
                m = NAME.search(text)
                o = OLD.search(text) if m and m.group(1) == "edit_file" else None
                row["arms"][nm] = {"tool": m.group(1) if m else "none", "n_tokens": len(toks[b]),
                                   "old_ok": None if o is None else any(o.group(1) in t for t in pt["seen"]),
                                   "think": toks[b].index(END_THINK) if END_THINK in toks[b] else len(toks[b]),
                                   "finish": fin[b] or "length", "text_head": text[:200], "text_tail": text[-200:],
                                   "same_as_A": toks[b] == toks[1], "same_as_base": toks[b] == toks[0],
                                   "common_prefix_A": next((i for i, (x, y) in enumerate(zip(toks[b], toks[1])) if x != y),
                                                           min(len(toks[b]), len(toks[1])))}
            del past, out
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            if done <= 3 or done % 10 == 0:
                print(done, round(time.time() - t0), row["sec"], row["prefill_sec"], row["steps"], {n: (r["tool"], r["n_tokens"]) for n, r in row["arms"].items()},
                      torch.cuda.max_memory_allocated() >> 20, flush=True)
    print("done", done, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
