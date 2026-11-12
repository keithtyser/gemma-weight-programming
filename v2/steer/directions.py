"""Diff-in-means residual directions per contrast and layer, validated on held-out tasks (CPU).

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: Reads <acts>/acts_<adapter>.pt from `v2.steer.collect acts` (residual stream after each of the 42 layers at
  labelled positions) and for each contrast (budget = budget@384..1536 pooled, each depth too; loop; giveup):
    r_l = mean(h_l | label 1, train) - mean(h_l | label 0, train)     (Arditi et al. 2024 diff-in-means)
  Validation on the heldout split (disjoint instance_ids): AUC of the projection h_l . r_l / |r_l| separating label 1
  from 0, standardized separation d' = (m1 - m0) / pooled sd, and the AUC of a RANDOM unit direction as the floor.
  Also the per-depth AUC for budget (a direction that only separates at one depth is a depth artefact), cosines
  between contrasts at the chosen layers, and how much of the residual norm the direction carries (|proj| / |h|).
  Picks per contrast the layer with the best heldout AUC among layers 8..34 (ablation at the very first or last
  layers mostly edits token identity / output logits, which we do not want), and writes <out>/directions.pt =
  {"directions": [k, 2560] unit rows, "names": [...], "layers": [...], "random": [3, 2560] unit control rows} plus
  <out>/directions_report.json.
  Usage: python -m v2.steer.directions --acts F.pt --out DIR [--min-layer 8 --max-layer 34]
         python -m v2.steer.directions transfer --acts OTHER.pt --dirs DIR/directions.pt --name w3
SRP/DRY check: Pass - consumer of v2.steer.collect output; no other direction-finding code in the repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC = P(score(pos) > score(neg)) + 0.5 P(tie)."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    ranks = torch.empty_like(allv)
    order = allv.argsort()
    ranks[order] = torch.arange(1, allv.numel() + 1, dtype=allv.dtype)
    # average ranks for ties
    uniq, inv, cnt = torch.unique(allv, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(uniq).scatter_add_(0, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: pos.numel()].sum()
    return float((rp - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


def contrasts_of(tags: list[str]) -> dict[str, Any]:
    """One contrast per tag family (the part before '@', pooled over its variants) and one per variant tag."""
    fams = sorted({t.split("@")[0] for t in tags})
    out: dict[str, Any] = {f: (lambda t, f=f: t.split("@")[0] == f) for f in fams}
    for t in sorted({t for t in tags if "@" in t}):
        out[t] = lambda x, t=t: x == t
    return out


def paired_acc(score: torch.Tensor, lab: torch.Tensor, keys: list[Any]) -> tuple[float, int]:
    """Share of (pair, tag) groups in which the label-1 state scores above the label-0 state (0.5 = chance)."""
    g: dict[Any, dict[int, float]] = {}
    for sc, l, k in zip(score.tolist(), lab.tolist(), keys):
        g.setdefault(k, {})[l] = sc
    d = [v[1] - v[0] for v in g.values() if len(v) == 2]
    return (sum((x > 0) + 0.5 * (x == 0) for x in d) / len(d) if d else float("nan")), len(d)


def dprime(pos: torch.Tensor, neg: torch.Tensor) -> float:
    sd = ((pos.var() + neg.var()) / 2).sqrt()
    return float((pos.mean() - neg.mean()) / sd) if sd > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.directions")
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-layer", type=int, default=8)
    p.add_argument("--max-layer", type=int, default=34)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    d = torch.load(a.acts, map_location="cpu")
    H = d["h"].float()  # [N, L, D]
    meta = d["meta"]
    N, nL, D = H.shape
    tags = [m["tag"] for m in meta]
    lab = torch.tensor([m["label"] for m in meta])
    tr = torch.tensor([m["split"] == "train" for m in meta])
    pkeys = [(m.get("pair"), m["tag"]) for m in meta]
    g = torch.Generator().manual_seed(a.seed)
    rnd = torch.randn(3, D, generator=g)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    contrasts = contrasts_of(tags)
    rep: dict[str, Any] = {"N": N, "layers": nL, "receipt": d.get("receipt"), "contrasts": {}}
    chosen: dict[str, tuple[int, torch.Tensor]] = {}
    dirs_all: dict[str, torch.Tensor] = {}
    for cname, f in contrasts.items():
        sel = torch.tensor([f(t) for t in tags])
        m1, m0 = sel & tr & (lab == 1), sel & tr & (lab == 0)
        h1, h0 = sel & ~tr & (lab == 1), sel & ~tr & (lab == 0)
        if m1.sum() < 5 or m0.sum() < 5 or h1.sum() < 3 or h0.sum() < 3:
            rep["contrasts"][cname] = {"skipped": "too few states", "n": [int(x.sum()) for x in (m1, m0, h1, h0)]}
            continue
        R = H[m1].mean(0) - H[m0].mean(0)  # [L, D]
        dirs_all[cname] = R
        has_pair = all("pair" in meta[i] for i in sel.nonzero().flatten().tolist())
        per_layer = []
        for l in range(nL):
            u = R[l] / R[l].norm()
            sp, sn = H[h1, l] @ u, H[h0, l] @ u
            rp, rn = H[h1, l] @ rnd[0], H[h0, l] @ rnd[0]
            row = {"layer": l, "auc": auc(sp, sn), "dprime": dprime(sp, sn), "rand_auc": auc(rp, rn),
                   "train_auc": auc(H[m1, l] @ u, H[m0, l] @ u),
                   "proj_frac": float((H[sel, l] @ u).abs().mean() / H[sel, l].norm(dim=1).mean()),
                   "diff_norm": float(R[l].norm())}
            if has_pair:
                hh = h1 | h0
                row["paired_acc"], row["n_pairs"] = paired_acc(H[hh, l] @ u, lab[hh], [k for k, x in zip(pkeys, hh.tolist()) if x])
            if cname == "budget":
                for dep in (384, 768, 1152, 1536):
                    sd = torch.tensor([t == f"budget@{dep}" for t in tags])
                    row[f"auc@{dep}"] = auc(H[h1 & sd, l] @ u, H[h0 & sd, l] @ u)
            per_layer.append(row)
        cand = [r for r in per_layer if a.min_layer <= r["layer"] <= a.max_layer]
        best = max(cand, key=lambda r: r["auc"])
        chosen[cname] = (best["layer"], R[best["layer"]] / R[best["layer"]].norm())
        rep["contrasts"][cname] = {"n_train": [int(m1.sum()), int(m0.sum())], "n_heldout": [int(h1.sum()), int(h0.sum())],
                                   "best": best, "per_layer": per_layer}
    names = list(chosen)
    cos = {f"{x}|{y}": float(chosen[x][1] @ chosen[y][1]) for i, x in enumerate(names) for y in names[i + 1 :]}
    # the same contrast's direction across layers: is it one stable feature?
    stab = {c: [float((R[l] / R[l].norm()) @ chosen[c][1]) for l in range(nL)] for c, R in dirs_all.items() if c in chosen}
    rep.update({"chosen": {c: chosen[c][0] for c in names}, "cosines": cos, "layer_stability": stab})
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": torch.stack([chosen[c][1] for c in names]) if names else torch.empty(0, D),
                "names": names, "layers": [chosen[c][0] for c in names], "random": rnd,
                "all_layers": {c: R for c, R in dirs_all.items()}}, a.out / "directions.pt")
    torch.save({"directions": rnd, "names": ["rand0", "rand1", "rand2"]}, a.out / "random.pt")
    (a.out / "directions_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({c: rep["contrasts"][c].get("best", rep["contrasts"][c]) for c in rep["contrasts"]}, indent=1))
    print(json.dumps({"cosines": cos}, indent=1))
    return 0


def transfer(argv: list[str] | None = None) -> int:
    """Do directions fitted on one adapter's activations hold for another adapter? On --acts (e.g. e4b_wave3_final,
    heldout states): per contrast and layer, AUC of the --dirs direction (fitted on the other adapter's train split)
    and cosine between it and this adapter's own heldout diff-in-means. Writes <dirs dir>/transfer_<name>.json."""
    p = argparse.ArgumentParser(prog="python -m v2.steer.directions transfer")
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--dirs", type=Path, required=True, help="directions.pt from main()")
    p.add_argument("--name", required=True)
    a = p.parse_args(argv)
    d = torch.load(a.acts, map_location="cpu")
    H, meta = d["h"].float(), d["meta"]
    dd = torch.load(a.dirs, map_location="cpu")
    tags = [m["tag"] for m in meta]
    lab = torch.tensor([m["label"] for m in meta])
    ho = torch.tensor([m["split"] == "heldout" for m in meta])
    contrasts = contrasts_of(tags)
    out: dict[str, Any] = {}
    for c, R in dd["all_layers"].items():
        if c not in contrasts:
            continue
        sel = torch.tensor([contrasts[c](t) for t in tags]) & ho
        p1, p0 = sel & (lab == 1), sel & (lab == 0)
        own = H[p1].mean(0) - H[p0].mean(0)
        rows = []
        for l in range(H.shape[1]):
            u = R[l] / R[l].norm()
            rows.append({"layer": l, "auc": auc(H[p1, l] @ u, H[p0, l] @ u),
                         "cos_own_heldout_diff": float(u @ (own[l] / own[l].norm()))})
        li = dd["layers"][dd["names"].index(c)] if c in dd["names"] else None
        out[c] = {"chosen_layer": li, "at_chosen": rows[li] if li is not None else None, "per_layer": rows,
                  "n": [int(p1.sum()), int(p0.sum())]}
    (a.dirs.parent / f"transfer_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({c: {"n": v["n"], **(v["at_chosen"] or {})} for c, v in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "transfer":
        raise SystemExit(transfer(sys.argv[2:]))
    raise SystemExit(main())
