"""D1: does the compiled weight edit equal exact inference-time activation addition? (activation fidelity, one engine)

Engine for every arm: v2.tii.trainer.aligned_forward.AlignedGemma4 (the vLLM-numerics twin used by v2.steer.collect:
INT4 base weights through the marlin GEMM, LoRA through the Punica path, bf16 activations), built by
v2.steer.collect.build_model. The base arm is the `zero` adapter (all lora_B = 0), exactly what vLLM serves as `base`.

Arms per edit pair (real edit E, its matched control K), same token sequence for every arm:
  base   zero adapter
  A      adapter E loaded as served (rank padded to 32 with zeros, scaling kept)
  N      zero adapter + "neuron hook": at each write layer l the MLP output y (before post_feedforward_layernorm) gets
         + a(xh) * v, with a(xh) = gelu_tanh(g . xh) * (u . xh) computed in fp32 from the MLP input xh;
         g, u, v are the new gate row, up row and down column of the edited neuron, reconstructed from the adapter file
         (dequantised INT4 base row/column + 2 B A). The overwritten neuron j is NOT removed. This is the exact intended
         write of the hand-written neuron as an inference-time hook.
  V      zero adapter + "vector hook" (classic activation addition): the layer output gets + m * d_unit at every
         position, d_unit = unit(w_post * v) (the residual direction the edit writes along),
         m = layer_scalar * sqrt(n) * beta / sqrt(1 + beta^2) * |w_post * unit(v)| (the builder's predicted shift).
         For a gated edit the vector is scaled per position by a(xh) / a_ref (a_ref = the builder's mean ON activation).
  C, CN, CV   the same three for the control adapter K.
Sequence = logged prompt + logged completion of the decision call (teacher forced), positions 0..L-1.
Residual metrics use the layer outputs (after layer_scalar) at CAP layers; "last" = last prompt position (the decision
point), "all" = every position. KL rows = the rows that predict the completion tokens (P-1 .. L-2); logits are
computed here in fp32 as softcap(h_final @ embed^T) (not the engine's lm-head kernel; the same for all arms).

Usage: python d1_fidelity.py --points points.jsonl --out DIR [--limit N] [--minutes M] [--resume]
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, "/path/to/gemma-v2/tii-steer")
MODEL = Path("/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct")
E = Path("/path/to/gemma-v2/results/steer/edits")
PFX = "base_model.model.model.language_model.layers."
WRITE = (18, 20, 22)
PAIRS = {"explore_more_0.25": ("bneg_t0.25", "bnm25_s1"), "before_first_edit_gate": ("ac_xp_pre", "ac_xp_const_ctrl")}
G4 = float(F.gelu(torch.tensor(4.0), approximate="tanh"))


def dequant(base, k):  # as wsp_arena/claude/code/build.py
    q, s = base.get_tensor(k + ".weight_packed"), base.get_tensor(k + ".weight_scale")
    n, kp = q.shape
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    u4 = torch.bitwise_and(torch.bitwise_right_shift(q.unsqueeze(-1), sh), 0xF).reshape(n, kp * 8)
    return ((u4.float() - 8.0) * s.float().repeat_interleave(32, dim=1)).to(torch.bfloat16).float()


def edit_spec(name, base, zero_sd, W):
    """New (gate row, up row, down column) of the edited neuron per write layer, from the adapter file; plus checks."""
    d = E / name
    cfg = json.loads((d / "adapter_config.json").read_text())
    r, s = int(cfg["r"]), float(cfg["lora_alpha"]) / float(cfg["r"])
    sd = load_file(str(d / "adapter_model.safetensors"))
    assert 1 <= r <= 32 and cfg["lora_alpha"] == 2 * r
    assert {k.replace(".lora_A", "").replace(".lora_B", "") for k in sd} == \
        {k.replace(".lora_A", "").replace(".lora_B", "") for k in zero_sd}
    assert all(torch.isfinite(t.float()).all() for t in sd.values())
    changed = sorted(k for k, t in sd.items() if ".lora_B" in k and float(t.float().abs().max()) > 0)
    want = sorted(f"{PFX}{li}.mlp.{nm}_proj.lora_B.weight" for li in WRITE for nm in ("gate", "up", "down"))
    assert changed == want, (name, changed[:3])  # nothing but the three MLP matrices of the write layers
    rep_p = d / "neuron_report.json"
    rep = json.loads((rep_p if rep_p.exists() else d / "report.json").read_text())
    out = {}
    for li in WRITE:
        k = f"{PFX}{li}.mlp."
        dW = {nm: s * (sd[k + nm + "_proj.lora_B.weight"].float() @ sd[k + nm + "_proj.lora_A.weight"].float())
              for nm in ("gate", "up", "down")}
        jg = (dW["gate"].abs().sum(1) > 0).nonzero().flatten().tolist()
        ju = (dW["up"].abs().sum(1) > 0).nonzero().flatten().tolist()
        jd = (dW["down"].abs().sum(0) > 0).nonzero().flatten().tolist()
        assert jg == ju == jd and len(jg) == 1, (name, li, jg, ju, jd)
        j = jg[0]
        Gt, Up, D, w_post, ls = W[li]
        g, u, v = Gt[j] + dW["gate"][j], Up[j] + dW["up"][j], D[:, j] + dW["down"][:, j]
        n = v.numel()
        R = rep["layers"][str(li)]
        if "neurons" in R:  # wsp_arena/claude build.py report
            nr = R["neurons"][0]
            beta, gap, mode = float(nr["beta_on"]), float(nr["gap"]), nr["mode"]
            a_ref = float(nr.get("act_heldout_on_mean", G4))
            assert nr["j"] == j
        else:               # v2/steer/neuron.py report
            beta, gap, mode = float(v.norm()) * G4 / (float(R["rho"]) * math.sqrt(n)), float(R["gap"]), rep["args"]["mode"]
            a_ref = G4
            assert R["neuron"] == j and mode == "const"
        uh = v / v.norm()
        wu = w_post * uh
        m = ls * math.sqrt(n) * beta / math.sqrt(1 + beta * beta) * float(wu.norm())
        out[li] = {"j": j, "g": g, "u": u, "v": v, "d_unit": wu / wu.norm(), "m": m, "beta": beta, "gap": gap,
                   "mode": mode, "a_ref": a_ref, "ls": ls}
    return out


def padded(model, name):
    """v2.steer.collect.load_padded, but returns the rank-32 padded tensors instead of writing them."""
    import re
    d = E / name
    cfg = json.loads((d / "adapter_config.json").read_text())
    r, s = int(cfg["r"]), float(cfg["lora_alpha"]) / float(cfg["r"])
    sd = load_file(str(d / "adapter_model.safetensors"))
    out = {pk: torch.zeros_like(p) for pk, p in model.lora_params.items()}
    seen = set()
    for k, v in sd.items():
        mm = re.match(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
        pk = model._pkey(mm.group(1), mm.group(2))
        t = v.to(torch.float32).to(model.dev)
        if mm.group(2) == "A":
            out[pk][:r].copy_(t)
        else:
            out[pk][:, :r].copy_(t * (s / model.lora_scaling))
        seen.add(pk)
    assert seen == set(model.lora_params)
    return out


def mv(x, w):  # [T, H] . [H]; torch.matmul is patched by the engine (batch-invariant kernels, 2-D only)
    return (x * w[None, :]).sum(-1)


class Hooks:
    def __init__(self, model, cap):
        self.neuron, self.vec, self.cap, self.xh, self.cap_layers = {}, {}, {}, {}, set(cap)
        lidx = {id(L): i for i, L in enumerate(model.layers)}
        orig_layer, orig_proj = model._layer, model._proj

        def layer(i, h, *a, **k):
            y = orig_layer(i, h, *a, **k)
            if i in self.vec:
                vec, gate = self.vec[i]
                if gate is None:
                    add = vec[None, :]
                else:
                    g, u, a_ref = gate
                    xh = self.xh[i]
                    add = (F.gelu(mv(xh, g), approximate="tanh") * mv(xh, u) / a_ref)[:, None] * vec[None, :]
                y = (y.float() + add).to(y.dtype)
            if i in self.cap_layers:
                self.cap[i] = y
            return y

        def proj(L, name, x, lora):
            y = orig_proj(L, name, x, lora)
            i = lidx.get(id(L))
            if name == "gate_up" and (i in self.neuron or i in self.vec):
                self.xh[i] = x.float()
            elif name == "down" and i in self.neuron:
                g, u, v = self.neuron[i]
                xh = self.xh[i]
                act = F.gelu(mv(xh, g), approximate="tanh") * mv(xh, u)
                y = (y.float() + act[:, None] * v[None, :]).to(y.dtype)
            return y

        model._layer, model._proj = layer, proj

    def clear(self):
        self.neuron, self.vec, self.cap, self.xh = {}, {}, {}, {}


def pair_stats(a, b, last):
    """a, b: residual deltas [T, H] fp32 (arm - base). Pooled over positions, mean per position, and last position."""
    dot, na, nb = (a * b).sum(-1), a.norm(dim=-1), b.norm(dim=-1)
    ok = (na > 0) & (nb > 0)
    sa, sb, sd = float((na ** 2).sum()), float((nb ** 2).sum()), float(((a - b) ** 2).sum())
    return {"cos_pooled": float(dot.sum()) / math.sqrt(sa * sb) if sa > 0 and sb > 0 else None,
            "cos_pos_mean": float((dot[ok] / (na[ok] * nb[ok])).mean()) if bool(ok.any()) else None,
            "norm_ratio_pooled": math.sqrt(sa / sb) if sb > 0 else None,
            "norm_ratio_pos_mean": float((na[ok] / nb[ok]).mean()) if bool(ok.any()) else None,
            "r2": 1 - sd / sa if sa > 0 else None, "frac_pos_nonzero": float(ok.float().mean()),
            "cos_last": float(dot[last] / (na[last] * nb[last])) if bool(ok[last]) else None,
            "norm_ratio_last": float(na[last] / nb[last]) if bool(ok[last]) else None,
            "r2_last": float(1 - ((a[last] - b[last]) ** 2).sum() / (na[last] ** 2)) if float(na[last]) > 0 else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="append to rows.jsonl, skipping points already there")
    a = p.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)
    from v2.steer.collect import build_model

    model = build_model(str(MODEL))
    dev, nL = model.dev, model.nL
    cap = sorted({18, 20, 22, 24, 28, 32, 36, nL - 1})
    base = safe_open(str(MODEL / "model.safetensors"), "pt")
    W = {}
    for li in WRITE:
        bp = f"model.language_model.layers.{li}."
        W[li] = (dequant(base, bp + "mlp.gate_proj"), dequant(base, bp + "mlp.up_proj"), dequant(base, bp + "mlp.down_proj"),
                 base.get_tensor(bp + "post_feedforward_layernorm.weight").float(),
                 float(base.get_tensor(bp + "layer_scalar").float()))
    zero_sd = load_file(str(E / "zero" / "adapter_model.safetensors"))
    assert all(float(t.float().abs().max()) == 0 for k, t in zero_sd.items() if ".lora_B" in k)  # base arm = no edit
    names = ["zero"] + [n for pr in PAIRS.values() for n in pr]
    specs = {n: edit_spec(n, base, zero_sd, W) for n in names[1:]}
    del W
    pads = {n: padded(model, n) for n in names}
    meta = {"cap_layers": cap, "pairs": PAIRS, "gelu4": G4, "specs": {
        n: {str(li): {k: s[k] for k in ("j", "m", "beta", "gap", "mode", "a_ref", "ls")} | {
            "v_norm": float(s["v"].norm()), "pred_shift_over_gap": s["m"] / s["gap"],
            "cos_d_unit_vs_pair_real": None} for li, s in sp.items()} for n, sp in specs.items()}}
    for real, ctrl in PAIRS.values():
        for li in WRITE:
            meta["specs"][ctrl][str(li)]["cos_d_unit_vs_pair_real"] = float(specs[ctrl][li]["d_unit"] @ specs[real][li]["d_unit"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["specs"])[:3000], flush=True)
    hooks = Hooks(model, cap)
    emb = model.embed.float()
    softcap = model.softcap

    def set_adapter(n):
        for pk, prm in model.lora_params.items():
            prm.copy_(pads[n][pk])

    def run(ids, adapter, neuron=None, vec=None, rows=None):
        hooks.clear()
        set_adapter(adapter)
        if neuron:
            hooks.neuron = {li: (s["g"].to(dev), s["u"].to(dev), s["v"].to(dev)) for li, s in specs[neuron].items()}
        if vec:
            hooks.vec = {li: ((s["m"] * s["d_unit"]).to(dev),
                              (s["g"].to(dev), s["u"].to(dev), s["a_ref"]) if s["mode"] == "gated" else None)
                         for li, s in specs[vec].items()}
        L = ids.numel()
        hid = model.forward_hidden(ids, [(0, L)], [[(0, L)]], lora=True)
        lps = []
        for c in range(0, rows.numel(), 512):
            z = hid.index_select(0, rows[c:c + 512]).float() @ emb.T
            if softcap:
                z = torch.tanh(z / softcap) * softcap
            lps.append(torch.log_softmax(z, -1))
        return {i: hooks.cap[i].float() for i in cap}, torch.cat(lps)

    def kl(lp, lq):
        return (lp.exp() * (lp - lq)).sum(-1)

    pts = [json.loads(x) for x in a.points.open()]
    rng = random.Random(a.seed)
    by = {lab: [x for x in pts if x["label"] == lab] for lab in (1, 0)}
    for v in by.values():
        rng.shuffle(v)
    order = [x for pr in zip(by[1], by[0]) for x in pr]  # label-balanced, seeded
    if a.limit:
        order = order[:a.limit]
    done = 0
    have = {(r["run_key"], r["call"]) for r in map(json.loads, (a.out / "rows.jsonl").open())} if a.resume else set()
    with (a.out / "rows.jsonl").open("a" if a.resume else "w") as fh:
        for pt in order:
            if (time.time() - t0) / 60 > a.minutes:
                break
            if (pt["run_key"], pt["call"]) in have:
                continue
            P = len(pt["prompt"])
            ids = torch.tensor(pt["prompt"] + pt["completion"], dtype=torch.long, device=dev)
            Lq = ids.numel()
            rows = torch.arange(P - 1, Lq - 1, device=dev)
            last = P - 1
            row = {"run_key": pt["run_key"], "call": pt["call"], "label": pt["label"], "P": P, "L": Lq, "pairs": {}}
            hb, lb = run(ids, "zero", rows=rows)
            for pname, (real, ctrl) in PAIRS.items():
                arms = {"A": (real, None, None), "N": ("zero", real, None), "V": ("zero", None, real),
                        "C": (ctrl, None, None), "CN": ("zero", ctrl, None), "CV": ("zero", None, ctrl)}
                dl, lp = {}, {}
                for arm, (ad, ne, ve) in arms.items():
                    h, lp[arm] = run(ids, ad, ne, ve, rows)
                    dl[arm] = {i: h[i] - hb[i] for i in cap}
                    del h
                res = {"resid": {}, "proj": {}, "kl": {}, "top1": {}}
                for x, y in (("A", "N"), ("A", "V"), ("C", "CN"), ("C", "CV"), ("A", "C"), ("N", "V")):
                    res["resid"][f"{x}|{y}"] = {str(i): pair_stats(dl[x][i], dl[y][i], last) for i in cap}
                for arm in arms:
                    res["proj"][arm] = {}
                    for li in WRITE:  # shift along the REAL edit's direction, in class gaps
                        s = specs[real][li]
                        pr = mv(dl[arm][li], s["d_unit"].to(dev)) / s["gap"]
                        res["proj"][arm][str(li)] = {"all": float(pr.mean()), "last": float(pr[last]),
                                                     "comp": float(pr[P:].mean()) if Lq > P else None}
                    k_ = kl(lb, lp[arm])
                    res["kl"][f"base|{arm}"] = {"mean": float(k_.mean()), "first": float(k_[0])}
                    res["top1"][f"base|{arm}"] = float((lb.argmax(-1) == lp[arm].argmax(-1)).float().mean())
                    res["top1"][f"logged|{arm}"] = float((lp[arm].argmax(-1) == ids[P:]).float().mean())
                for x, y in (("N", "A"), ("V", "A"), ("CN", "C"), ("CV", "C"), ("C", "A"), ("N", "V")):
                    k_ = kl(lp[x], lp[y])
                    res["kl"][f"{x}|{y}"] = {"mean": float(k_.mean()), "first": float(k_[0])}
                    res["top1"][f"{x}|{y}"] = float((lp[x].argmax(-1) == lp[y].argmax(-1)).float().mean())
                res["top1"]["logged|base"] = float((lb.argmax(-1) == ids[P:]).float().mean())
                row["pairs"][pname] = res
                del dl, lp
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            if done <= 3 or done % 10 == 0:
                print(done, round(time.time() - t0), Lq, json.dumps(row["pairs"]["explore_more_0.25"]["resid"]["A|N"]["18"])[:300],
                      flush=True)
    print("done", done, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
