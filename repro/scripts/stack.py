"""Round-3 edits: exact stacks of existing edits from different authors, repacked into one LoRA.

    python stack.py --out NAME --srcs editA[:scale] editB[:scale] ...   (run in $S/edits)
Every source is a rank-16 LoRA (alpha 32) that uses only a few slots per module. The stack's weight change is the
sum of the sources' weight changes (scale multiplies a source's down_proj B, i.e. what its neurons write). Non-silent slots are copied into one rank-32
LoRA with alpha 64, so alpha / r is unchanged and the sum is exact; the build fails if a module needs more than 32.
"""
import argparse, json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

R = 32
WRITE = "down_proj"     # scale applies to the neuron's write (down_proj) only; reads and gates are unchanged


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
    out, used, max_used, max_err = {}, 0, 0, 0.0
    for ka in sorted(k for k in ts[0] if ".lora_A" in k):
        kb = ka.replace(".lora_A", ".lora_B")
        As, Bs = [], []
        for t, (_, sc) in zip(ts, srcs):
            A, B = t[ka], t[kb]
            live = (A.float().abs().amax(1) > 0) & (B.float().abs().amax(0) > 0)
            As.append(A[live])
            Bs.append(B[:, live] * (sc if WRITE in ka else 1.0))
        A2, B2 = torch.cat(As, 0), torch.cat(Bs, 1)
        n = A2.shape[0]
        assert n <= R, (ka, n)
        A3 = torch.zeros(R, A2.shape[1], dtype=ts[0][ka].dtype)
        B3 = torch.zeros(B2.shape[0], R, dtype=ts[0][kb].dtype)
        A3[:n], B3[:, :n] = A2, B2
        out[ka], out[kb] = A3, B3
        used += n
        max_used = max(max_used, n)
        if n:
            ref = sum((sc if WRITE in ka else 1.0) * (t[kb].double() @ t[ka].double()) for t, (_, sc) in zip(ts, srcs))
            max_err = max(max_err, float((B3.double() @ A3.double() - ref).abs().max()))
    cfg["r"], cfg["lora_alpha"] = R, 2 * R
    od = Path(a.out)
    od.mkdir(exist_ok=True)
    (od / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    save_file(out, str(od / "adapter_model.safetensors"))
    rep = {"edit": a.out, "stack_of": [{"edit": n, "scale": s} for n, s in srcs], "live_slots": used,
           "max_slots_per_module": max_used, "max_abs_error_vs_sum": max_err}
    (od / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
