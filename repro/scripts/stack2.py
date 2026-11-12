"""Round-3 edits: stacks of existing neuron edits from different authors, repacked into one LoRA.

    python stack2.py --out NAME --srcs editA[:scale] editB[:scale] ...   (run in $S/edits)
Every source is a rank-16 LoRA (alpha 32) whose live slots each overwrite one MLP neuron j of a layer (row j of
gate_proj and up_proj, column j of down_proj). The authors picked the same nearly silent neurons, so a plain sum of
the weight changes (stack.py) does not give two neurons: it gives one neuron with mixed weights. Here the first source
keeps its neurons; when a later source wants a neuron that is already taken, its neuron is moved to the next
quietest free neuron j2 of that layer: the new rows/column are the source's edited values (base[j] + change), written
over base[j2]. scale multiplies a source's change to down_proj (what its neurons write). Output: rank 32, alpha 64.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/path/to/gemma-v2/wsp_arena/claude/code")
import build  # noqa: E402  (dequant, BASE, ACTS, PFX)

R = 32
MODS = ["gate_proj", "up_proj", "down_proj"]


def neuron_changes(t, li, sc):
    """{j: {mod: change vector}} for one source and layer; change is the full weight change (alpha / r = 2 applied)."""
    out = defaultdict(dict)
    for mod in MODS:
        A = t[f"{build.PFX}{li}.mlp.{mod}.lora_A.weight"].double()
        B = t[f"{build.PFX}{li}.mlp.{mod}.lora_B.weight"].double()
        for s in range(A.shape[0]):
            if float(A[s].abs().max()) == 0 or float(B[:, s].abs().max()) == 0:
                continue
            side, vec = (A[s], B[:, s]) if mod == "down_proj" else (B[:, s], A[s])   # side indexes neurons
            nz = side.nonzero().flatten().tolist()
            assert len(nz) == 1, (li, mod, s, len(nz))        # one neuron per slot
            j = nz[0]
            ch = 2.0 * float(side[j]) * vec * (sc if mod == "down_proj" else 1.0)
            out[j][mod] = out[j].get(mod, 0) + ch
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--srcs", nargs="+", required=True)
    a = p.parse_args()
    srcs = [(s.split(":")[0], float(s.split(":")[1]) if ":" in s else 1.0) for s in a.srcs]
    ts = [load_file(f"{n}/adapter_model.safetensors") for n, _ in srcs]
    cfg = json.loads(Path(srcs[0][0], "adapter_config.json").read_text())
    for n, _ in srcs:
        c = json.loads(Path(n, "adapter_config.json").read_text())
        assert c["r"] == 16 and c["lora_alpha"] == 32 and not c.get("use_rslora"), n
        assert c["target_modules"] == cfg["target_modules"], n
    assert all(t.keys() == ts[0].keys() for t in ts)
    layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in ts[0]})
    base = safe_open(str(next(build.BASE.glob("*.safetensors"))), "pt")
    acts = None
    out = {k: torch.zeros(R if ".lora_A" in k else v.shape[0], v.shape[1] if ".lora_A" in k else R, dtype=v.dtype)
           for k, v in ts[0].items()}
    rep_layers, max_err = {}, 0.0
    for li in layers:
        per_src = [neuron_changes(t, li, sc) for t, (_, sc) in zip(ts, srcs)]
        if not any(per_src):
            continue
        bp = f"model.language_model.layers.{li}.mlp."
        W = {m: build.dequant(base, bp + m).double() for m in MODS}
        taken, plan, order = set(), [], None
        want = set().union(*[set(d) for d in per_src])
        for (name, _), d in zip(srcs, per_src):
            for j in sorted(d):
                j2 = j
                if j in taken:
                    if order is None:
                        if acts is None:
                            acts = torch.load(build.ACTS, map_location="cpu")
                        X = acts["xin"][:, li].float()
                        A0 = F.gelu(X @ W["gate_proj"].float().T, approximate="tanh") * (X @ W["up_proj"].float().T)
                        w_post = base.get_tensor(f"model.language_model.layers.{li}.post_feedforward_layernorm.weight").float()
                        quiet = A0.abs().mean(0) * (W["down_proj"].float() * w_post[:, None]).norm(dim=0)
                        order = torch.argsort(quiet).tolist()
                    j2 = next(x for x in order if x not in taken and x not in want)
                taken.add(j2)
                plan.append((name, j, j2, d[j]))
        assert len(plan) <= R, (li, len(plan))
        for s, (name, j, j2, ch) in enumerate(plan):
            for mod in MODS:
                c = ch.get(mod)
                if c is None:
                    continue
                old = W[mod][:, j] if mod == "down_proj" else W[mod][j]
                old2 = W[mod][:, j2] if mod == "down_proj" else W[mod][j2]
                vec = (old + c - old2) / 2.0                     # alpha / r = 2
                ka, kb = (f"{build.PFX}{li}.mlp.{mod}.lora_A.weight", f"{build.PFX}{li}.mlp.{mod}.lora_B.weight")
                if mod == "down_proj":
                    out[ka][s, j2], out[kb][:, s] = 1.0, vec.to(out[kb].dtype)
                else:
                    out[kb][j2, s], out[ka][s] = 1.0, vec.to(out[ka].dtype)
                # check: the stack's neuron j2 equals the source's edited neuron j
                got = old2 + 2.0 * (out[kb][:, s].double() if mod == "down_proj" else out[ka][s].double())
                max_err = max(max_err, float((got - (old + c)).abs().max()))
        rep_layers[li] = [{"src": n, "neuron": j, "placed_at": j2} for n, j, j2, _ in plan]
    cfg["r"], cfg["lora_alpha"] = R, 2 * R
    od = Path(a.out)
    od.mkdir(exist_ok=True)
    (od / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    save_file(out, str(od / "adapter_model.safetensors"))
    rep = {"edit": a.out, "stack_of": [{"edit": n, "scale": s} for n, s in srcs], "layers": rep_layers,
           "moved": sum(x["neuron"] != x["placed_at"] for v in rep_layers.values() for x in v),
           "max_abs_error_neuron_weights": max_err}
    (od / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}))
    for li, v in rep_layers.items():
        print(li, [(x["src"][:6], x["neuron"], x["placed_at"]) for x in v])


if __name__ == "__main__":
    main()
