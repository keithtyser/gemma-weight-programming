"""Closed-form rescaling and extrapolation of LoRA updates (CPU, exact up to bf16 storage).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Two training-free edits of an RL-trained adapter, as baselines for the direction edits of v2.steer.fold:
    scale    Delta' = gamma * Delta_a                       (same rank; B' = gamma B). gamma < 1 moves back toward the
             base model, gamma > 1 extrapolates the whole RL update.
    extrap   Delta' = Delta_a + beta * (Delta_a - Delta_b)  (rank r_a + r_b; B' = [(1 + beta) B_a, -beta B_b],
             A' = [A_a; A_b]). With a = wave 5 and b = wave 4 this continues the last wave's step without rollouts.
    spectral Per module, Delta = U S V^T (exact rank-r SVD from the QR factors of B and A); keeps the top --keep
             singular components (0 = all) and replaces the kept S by S^power rescaled to the module's original
             Frobenius norm (power 1 = unchanged, < 1 flattens, > 1 sharpens). Same rank. With --layers LO:HI only
             modules of those decoder layers are edited (also applies to scale).
  Both adapters must use scaling alpha / r = 2 (the e4b recipe); the output keeps scaling 2.
  Writes <out>/adapter_config.json + adapter_model.safetensors (bf16) + mix_report.json.
  Usage: python -m v2.steer.lora_mix scale --adapter A --gamma 1.5 --out DIR
         python -m v2.steer.lora_mix extrap --adapter A --prev B --beta 1.0 --out DIR
         python -m v2.steer.lora_mix spectral --adapter A --keep 4 [--power 1.0] [--layers 0:21] --out DIR
SRP/DRY check: Pass - v2.steer.fold edits along residual directions; nothing else rescales or combines adapters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def load(ad: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    cfg = json.loads((ad / "adapter_config.json").read_text())
    if abs(float(cfg["lora_alpha"]) / int(cfg["r"]) - 2.0) > 1e-9:
        raise ValueError(f"{ad}: scaling alpha/r must be 2")
    return cfg, {k: v.float() for k, v in load_file(str(ad / "adapter_model.safetensors")).items()}


def layer_of(key: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> int:
    from safetensors.torch import save_file

    p = argparse.ArgumentParser(prog="python -m v2.steer.lora_mix")
    p.add_argument("cmd", choices=("scale", "extrap", "spectral"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--prev", type=Path, default=None)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--keep", type=int, default=0)
    p.add_argument("--power", type=float, default=1.0)
    p.add_argument("--layers", default="", help="LO:HI (inclusive) decoder layers to edit; default all")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.layers.split(":")) if a.layers else (0, 10**9)

    def sel(key: str) -> bool:
        li = layer_of(key)
        return li is not None and lo <= li <= hi if a.layers else True

    cfg, sd = load(a.adapter)
    r = int(cfg["r"])
    out: dict[str, torch.Tensor] = {}
    rep: dict = {"cmd": a.cmd, "adapter": str(a.adapter)}
    if a.cmd == "scale":
        for k, v in sd.items():
            out[k] = (v * a.gamma if ".lora_B." in k and sel(k) else v).to(torch.bfloat16).contiguous()
        r2 = r
        rep.update({"gamma": a.gamma, "layers": a.layers, "modules_edited": sum(".lora_B." in k and sel(k) for k in sd)})
    elif a.cmd == "spectral":
        r2 = r
        energy = torch.zeros(r, dtype=torch.float64)
        n_edit = 0
        for k, v in sd.items():
            if ".lora_B." not in k:
                continue
            ka = k.replace(".lora_B.", ".lora_A.")
            B, A = v.double(), sd[ka].double()
            qb, rb = torch.linalg.qr(B)
            qa, ra = torch.linalg.qr(A.T)
            u, s_, vh = torch.linalg.svd(rb @ ra.T)
            energy += (s_ ** 2) / (s_ ** 2).sum()
            if not sel(k):
                out[k], out[ka] = v.to(torch.bfloat16).contiguous(), sd[ka].to(torch.bfloat16).contiguous()
                continue
            n_edit += 1
            kk = a.keep or r
            s2 = torch.zeros_like(s_)
            s2[:kk] = s_[:kk] ** a.power
            s2 *= s_.norm() / s2.norm().clamp_min(1e-30) if a.power != 1.0 else 1.0
            out[k] = ((qb @ u) * s2.sqrt()).float().to(torch.bfloat16).contiguous()
            out[ka] = (s2.sqrt()[:, None] * (qa @ vh.T).T).float().to(torch.bfloat16).contiguous()
        n_mod = sum(".lora_B." in k for k in sd)
        rep.update({"keep": a.keep, "power": a.power, "layers": a.layers, "modules_edited": n_edit,
                    "mean_energy_by_component": [round(float(x) / n_mod, 4) for x in energy]})
    else:
        cfg_b, sd_b = load(a.prev)
        if set(sd) != set(sd_b):
            raise ValueError("adapters have different modules")
        rb = int(cfg_b["r"])
        r2 = r + rb
        num = den = 0.0
        for k, v in sd.items():
            if ".lora_B." in k:
                out[k] = torch.cat([(1 + a.beta) * v, -a.beta * sd_b[k]], 1).to(torch.bfloat16).contiguous()
                ka = k.replace(".lora_B.", ".lora_A.")
                step = v @ sd[ka] - sd_b[k] @ sd_b[ka]
                num += float(step.pow(2).sum())
                den += float((v @ sd[ka]).pow(2).sum())
            else:
                out[k] = torch.cat([v, sd_b[k]], 0).to(torch.bfloat16).contiguous()
        rep.update({"prev": str(a.prev), "beta": a.beta, "step_fro_over_delta_fro": (num / max(den, 1e-30)) ** 0.5})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "adapter_config.json").write_text(json.dumps(dict(cfg, r=r2, lora_alpha=2 * r2), indent=2))
    save_file(out, str(a.out / "adapter_model.safetensors"))
    rep["rank"] = r2
    (a.out / "mix_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
