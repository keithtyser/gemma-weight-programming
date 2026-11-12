"""Fold residual-direction ablations into an E4B PEFT LoRA adapter (CPU).

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: Directional ablation ("weight orthogonalization", Arditi et al. 2024; Weight-Space Programming) of unit
  residual directions r_1..r_k from every attention and MLP write of the chosen layers, expressed as a LoRA on the fixed
  INT4 base so the submission stays "base + one PEFT adapter" and exact RL can continue on top.

  Gemma 4 E4B block writes (v2/tii/trainer/aligned_forward.py `_attn_block` / `_mlp_block`, = vLLM gemma4.py):
      h += RMSNorm_w(o_proj(attn))        (w = post_attention_layernorm.weight, plain multiply)
      h += RMSNorm_w(down_proj(mlp))      (w = post_feedforward_layernorm.weight)
  RMSNorm_w(x) = w * x / rms(x), so r^T RMSNorm_w(x) = (w*r)^T x / rms(x). The write has NO component on r iff the
  projection output x is orthogonal to u = w*r. Hence per site we project the output onto the orthogonal complement of
  U = orth([w*r_1, ..., w*r_k]):  W' = (I - U U^T) W_eff with W_eff = W0 + s B A (W0 = the dequantized INT4 weight
  the engine multiplies, s = alpha / r). The edit is exact for the r-components of these writes; side effect (reported
  by the GPU validation, not here): rms(x) shrinks by the removed part, so the remaining components are scaled up by
  rms(x)/rms(x') >= 1. Writers we cannot edit inside a LoRA: token embeddings and the per-layer-input (PLE) branch.

  Target delta per site: D* = s B A - U U^T W_eff  (rank <= 16 + k). Three packings:
    exact   r' = 16 + k, alpha' = 2 r' (s' = 2 as in the source adapter): B' = [B, -U], A' = [A; U^T W_eff / 2], exact in fp32;
            vLLM rounds max_lora_rank up to its bucket (32). The aligned engine/learner run r16 today, so continued
            exact training on this variant needs lora_max_rank 32 there (a geometry change, not a kernel change).
    trunc16 best rank-16 approximation of D* (truncated SVD): smallest Frobenius error, but may leak part of the
            ablated component back (reported as leak).
    cons16  rank 16 with the ablation kept EXACT (leak 0 in fp32): D = U X + (I-P) M with X = -U^T W0 forced; the
            row space must contain rowspace(U^T W0) (k dims), so the original adapter is kept on 16-k free dims:
            D = U X + (I-P) sBA Pi_S + best rank-(16-k) approx of (I-P) sBA (I - Pi_S), S = rowspace(U^T W0).
  Errors per site (fp32, and after bf16 storage): rel_err = ||D - D*||_F / ||D*||_F, adapter_rel_err =
  ||(D - D*)||_F / ||sBA||_F, weight_rel_err = ||D - D*||_F / ||W_eff||_F, leak = ||U^T (W0 + D)||_F / ||U^T W_eff||_F (0 = fully ablated, 1 = untouched).
  Reads: base checkpoint (compressed-tensors pack-quantized, group 32), source adapter dir, a directions .pt
  ({"directions": [k, 2560] fp32 unit rows, "names": [...]}) from v2.steer.directions. Writes: <out>/<variant>/
  adapter_config.json + adapter_model.safetensors (bf16, PEFT keys of the source) and <out>/fold_report.json.
  Usage: python -m v2.steer.fold --base M --adapter A --directions D.pt --out DIR [--layers 0-41] [--variants ...]
SRP/DRY check: Pass - reuses v2.tii.trainer.ops.dequant_int4_bf16 (the trainer/Marlin dequant); no other LoRA
  merge/fold tool exists in the repo (v2.eval.lora_loop_probe only rescales lora_B).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

PFX = "base_model.model.model.language_model.layers."
SITES = {"self_attn.o_proj": "post_attention_layernorm", "mlp.down_proj": "post_feedforward_layernorm"}


def dequant_int4_bf16(qweight: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Same arithmetic as v2.tii.trainer.ops.dequant_int4_bf16 (imported there; re-stated here only when the trainer
    package cannot be imported, e.g. a CPU box without vLLM)."""
    try:
        from v2.tii.trainer.ops import dequant_int4_bf16 as f

        return f(qweight, scale, group)
    except Exception:  # noqa: BLE001 - vLLM-free fallback with identical math
        n, kp = qweight.shape
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        u4 = torch.bitwise_and(torch.bitwise_right_shift(qweight.unsqueeze(-1), shifts), 0xF).reshape(n, kp * 8)
        return ((u4.float() - 8.0) * scale.float().repeat_interleave(group, dim=1)).to(torch.bfloat16)


def factored_svd(L: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of L @ R (L [m, q], R [q, n], q small) without forming it: returns U [m, q], S [q], Vh [q, n]."""
    Q1, R1 = torch.linalg.qr(L)
    Q2, R2 = torch.linalg.qr(R.T)
    u, s, vh = torch.linalg.svd(R1 @ R2.T)
    return Q1 @ u, s, vh @ Q2.T


def fold_site(W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float, Uo: torch.Tensor, r: int,
              variants: list[str], alpha: float = 0.0) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """W0 [out, in] fp32, A [r, in], B [out, r] fp32, Uo [out, k] orthonormal. Returns per variant (A', B') to be used
    with scaling 2.0 (alpha' = 2 r') and the error report."""
    k = Uo.shape[1]
    Weff = W0 + s * (B @ A)
    UtW0 = Uo.T @ W0  # [k, in]
    UtWeff = Uo.T @ Weff
    # D* = s B A + (alpha - 1) U (U^T W_eff), factored as Lf @ Rf with inner dim r + k. alpha = 0 removes what the
    # matrix writes along U (ablation); alpha > 1 scales it up (amplification); W' = (I + (alpha - 1) U U^T) W_eff.
    g = alpha - 1.0
    Lf = torch.cat([s * B, g * Uo], 1)
    Rf = torch.cat([A, UtWeff], 0)
    Dstar_norm = torch.linalg.matrix_norm(Lf @ Rf).item()
    sBA_norm = torch.linalg.matrix_norm(s * B @ A).item()
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    Weff_norm = torch.linalg.matrix_norm(Weff).item()
    rep: dict[str, Any] = {"Dstar_fro": Dstar_norm, "sBA_fro": sBA_norm, "Weff_fro": Weff_norm,
                           "UtWeff_fro": torch.linalg.matrix_norm(UtWeff).item()}
    for v in variants:
        if v == "exact":
            # D*/2 = B A (s=2) - U U^T W_eff / 2 -> B' = [s/2 B, -U], A' = [A, U^T W_eff / 2]; rank r + k
            Bp = torch.cat([(s / 2.0) * B, g * Uo], 1)
            Ap = torch.cat([A, UtWeff / 2.0], 0)
        elif alpha != 0.0:
            raise ValueError("only the exact variant supports alpha != 0")
        elif v == "trunc16":
            u, sv, vh = factored_svd(Lf, Rf)
            Bp = u[:, :r] * (sv[:r] / 2.0)  # B' A' = U_r S_r V_r / 2 (scaling 2 restores U_r S_r V_r)
            Ap = vh[:r]
            rep["trunc16_dropped_sv"] = sv[r:].tolist()
        elif v == "cons16":
            P = Uo @ Uo.T
            I = torch.eye(W0.shape[0])
            VS, _ = torch.linalg.qr(UtW0.T)  # [in, k] orthonormal basis of S = rowspace(U^T W0)
            C = -(UtW0 @ VS)  # X = -U^T W0 = C VS^T
            Y = (I - P) @ (s * B) @ (A @ VS)  # (I-P) sBA Pi_S = Y VS^T
            Lz = (I - P) @ (s * B)
            Rz = A - (A @ VS) @ VS.T
            u, sv, vh = factored_svd(Lz, Rz)
            m = r - k
            # D = (U C + Y) VS^T + u_m S_m vh_m ; with scaling 2: B' A' = D / 2
            Bp = torch.cat([(Uo @ C + Y) / 2.0, u[:, :m] * sv[:m] / 2.0], 1)
            Ap = torch.cat([VS.T, vh[:m]], 0)
            rep["cons16_dropped_sv"] = sv[m:].tolist()
        else:
            raise ValueError(v)
        for tag, (bb, aa) in (("fp32", (Bp, Ap)), ("bf16", (Bp.bfloat16().float(), Ap.bfloat16().float()))):
            D = 2.0 * bb @ aa
            err = torch.linalg.matrix_norm(D - (Lf @ Rf)).item()
            leak = torch.linalg.matrix_norm(Uo.T @ (W0 + D)).item() / max(1e-30, rep["UtWeff_fro"])
            rep[f"{v}_{tag}"] = {"rel_err": err / max(1e-30, Dstar_norm), "adapter_rel_err": err / max(1e-30, sBA_norm),
                                 "weight_rel_err": err / Weff_norm,
                                 "leak": leak, "rank": int(aa.shape[0])}
        out[v] = (Ap, Bp)
    return out, rep


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser(prog="python -m v2.steer.fold")
    p.add_argument("--base", type=Path, required=True, help="compressed-tensors checkpoint dir (model.safetensors)")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--directions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", default="0-41", help="layers whose o_proj/down_proj writes are ablated")
    p.add_argument("--sites", default="self_attn.o_proj,mlp.down_proj")
    p.add_argument("--variants", default="exact,trunc16,cons16")
    p.add_argument("--select", default=None, help="comma list of direction indices to ablate (default all)")
    p.add_argument("--names", default=None, help="comma list of direction names (instead of --select)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="scale of the component written along the directions: 0 ablates (default), >1 amplifies")
    a = p.parse_args(argv)
    variants = a.variants.split(",")
    dd = torch.load(a.directions, map_location="cpu")
    R = dd["directions"].float()
    if a.names:
        a.select = ",".join(str(dd["names"].index(n)) for n in a.names.split(","))
    if a.select:
        R = R[[int(i) for i in a.select.split(",")]]
    R = R / R.norm(dim=1, keepdim=True)
    k = R.shape[0]
    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    s = alpha / r
    if abs(s - 2.0) > 1e-9:
        raise ValueError("fold assumes the source scaling alpha/r == 2 (e4b r16/alpha32)")
    sd = {kk: v.float() for kk, v in load_file(str(a.adapter / "adapter_model.safetensors")).items()}
    base = safe_open(str(a.base / "model.safetensors"), "pt")
    layers = parse_layers(a.layers)
    report: dict[str, Any] = {"k": k, "names": dd.get("names"), "layers": layers, "sites": {}, "source": str(a.adapter),
                              "alpha": a.alpha, "select": a.select}
    new = {v: {} for v in variants}
    for li in layers:
        for site in a.sites.split(","):
            mod = f"{PFX}{li}.{site}"
            w = base.get_tensor(f"model.language_model.layers.{li}.{SITES[site]}.weight").float()
            Uo, _ = torch.linalg.qr((w[:, None] * R.T))  # orth([w*r_1..w*r_k])
            bp = f"model.language_model.layers.{li}.{site}"
            W0 = dequant_int4_bf16(base.get_tensor(bp + ".weight_packed"), base.get_tensor(bp + ".weight_scale")).float()
            A, B = sd[mod + ".lora_A.weight"], sd[mod + ".lora_B.weight"]
            res, rep = fold_site(W0, A, B, s, Uo, r, variants, a.alpha)
            report["sites"][f"{li}.{site}"] = rep
            for v, (Ap, Bp) in res.items():
                new[v][mod] = (Ap, Bp)
    summary: dict[str, Any] = {}
    for v in variants:
        rv = r + k if v == "exact" else r
        tensors: dict[str, torch.Tensor] = {}
        for key, t in sd.items():  # every source module, padded to rank rv where untouched
            mod, ab = key.rsplit(".lora_", 1)
            if mod in new[v]:
                Ap, Bp = new[v][mod]
                t2 = Ap if ab.startswith("A") else Bp
            else:
                # untouched module: identical product with scaling 2 at rank rv (zero-padded rows / columns)
                if ab.startswith("A"):
                    t2 = torch.cat([t, t.new_zeros(rv - r, t.shape[1])], 0)
                else:
                    t2 = torch.cat([t, t.new_zeros(t.shape[0], rv - r)], 1)
            tensors[key] = t2.to(torch.bfloat16).contiguous()
        od = a.out / v
        od.mkdir(parents=True, exist_ok=True)
        c2 = dict(cfg, r=rv, lora_alpha=2 * rv)
        (od / "adapter_config.json").write_text(json.dumps(c2, indent=2))
        save_file(tensors, str(od / "adapter_model.safetensors"))
        rows = [report["sites"][x][f"{v}_bf16"] for x in report["sites"]]
        summary[v] = {"rank": rv, "max_rel_err": max(x["rel_err"] for x in rows),
                      "mean_rel_err": sum(x["rel_err"] for x in rows) / len(rows),
                      "max_adapter_rel_err": max(x["adapter_rel_err"] for x in rows),
                      "mean_adapter_rel_err": sum(x["adapter_rel_err"] for x in rows) / len(rows),
                      "max_weight_rel_err": max(x["weight_rel_err"] for x in rows),
                      "max_leak": max(x["leak"] for x in rows), "mean_leak": sum(x["leak"] for x in rows) / len(rows)}
        rows32 = [report["sites"][x][f"{v}_fp32"] for x in report["sites"]]
        summary[v]["fp32_mean_adapter_rel_err"] = sum(x["adapter_rel_err"] for x in rows32) / len(rows32)
        summary[v]["fp32_max_leak"] = max(x["leak"] for x in rows32)
    report["summary"] = summary
    (a.out / "fold_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
