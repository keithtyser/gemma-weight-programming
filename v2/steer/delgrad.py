"""One-shot weight deletion from logged rollouts, folded into the LoRA (after Deletion Tuning, arXiv 2611.10866).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Deletion Tuning learns, with RL, a binary mask over the gate projection of every MLP block. Here the mask is
  chosen WITHOUT any optimisation step, from one exact REINFORCE gradient over rollouts that already exist, and is
  restricted so that it can be served as a LoRA:
    rows     16 output rows (MLP neurons) of gate_proj per layer; default = the rows where the RL adapter's own update
             to gate_proj is largest (CPU, from the adapter).
    grad     G = dJ/dW on those rows, J = sum over rollouts of advantage * log pi (T = 1, advantage = reward - task
             mean, the wave recipe's "mean_only"). Read exactly through the spare LoRA slots of the rank-32 padded
             model: slot j of gate_proj gets B_j = e_{r_j}, A_j = 0 (function unchanged), and then
             dJ/dA_j = scaling * G[r_j, :]. Same aligned forward / backward as the learner (score_calls_subtrees).
             Rollouts come only from the "train" task split; gradients are kept separately for two disjoint halves of
             the tasks (split-half reliability). Also saves the base rows W[R, :] (identity through the frozen linear).
    fold     deleting entry (r, c) changes J by about -W[r, c] * G[r, c] (first order). Modes: gain (all entries with
             positive predicted gain), agree (positive in both halves), anti (negative gain; count-matched control),
             rand (random entries of the same rows, same count per row), rows (the whole rows). The deletion is the
             rank-16 update B = E_R / scaling, A = -(W[R, :] * mask), stacked on the rank-16 adapter -> rank 32.
  Reads <mine>/rollouts.jsonl + the rollouts' request logs. Writes <out>/delgrad.pt + delgrad_report.json (grad) and
  <out>/adapter_config.json + adapter_model.safetensors + del_report.json (fold).
  Usage: python -m v2.steer.delgrad grad --model M --adapter A --mine DIR --srcs a,b --out DIR [--max-rollouts 1200]
         python -m v2.steer.delgrad fold --grads DIR/delgrad.pt --adapter A --mode gain --out DIR [--frac 1.0]
SRP/DRY check: Pass - model loading / padding from v2.steer.collect, split from v2.steer.states, scoring from the
  aligned trainer; nothing else computes weight-space gradients outside the learner.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

ROWS = 16
GATE = "model.language_model.layers.{i}.mlp.gate_proj"


def half_of(iid: str) -> int:
    return int(hashlib.sha1(("half:" + iid).encode()).hexdigest(), 16) % 2


def pick_units(mine: Path, srcs: set[str], split: str, dev: set[str], cap: int, seed: int) -> list[dict[str, Any]]:
    from v2.steer.states import split_of

    by: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in map(json.loads, (mine / "rollouts.jsonl").open()):
        if r["src"] in srcs and not r["infra"] and r.get("resolved") is not None and r.get("request_log_path") \
                and Path(r["request_log_path"]).exists():
            by[(r["src"], r["instance_id"])].append(r)
    tasks = {h: [] for h in (0, 1)}
    for (src, iid), rs in sorted(by.items()):
        m = sum(bool(r["resolved"]) for r in rs) / len(rs)
        if split_of(iid) != split or iid in dev or m in (0.0, 1.0):
            continue
        tasks[half_of(iid)].append([{"src": src, "run_key": r["run_key"], "instance_id": iid, "half": half_of(iid),
                                     "adv": float(bool(r["resolved"])) - m, "log": r["request_log_path"]} for r in rs])
    rng = random.Random(seed)
    units: list[dict[str, Any]] = []
    for h in (0, 1):
        rng.shuffle(tasks[h])
        n = 0
        for g in tasks[h]:
            if n >= cap // 2:
                break
            units += g
            n += len(g)
    rng.shuffle(units)  # both halves progress together, so an interrupted run is still balanced
    return units


def read_calls(log: str, max_seq_len: int) -> list[tuple[list[int], list[int]]]:
    cs = []
    with open(log) as fh:
        for line in fh:
            c = json.loads(line)
            p, g = c.get("prompt_token_ids"), c.get("completion_token_ids")
            if p and g and len(p) + len(g) <= max_seq_len:
                cs.append(([int(x) for x in p], [int(x) for x in g]))
    return cs


def cmd_grad(a: argparse.Namespace) -> int:
    import torch

    from v2.steer.collect import build_model, load_padded
    from v2.tii.trainer import ops

    dev_ids = set(json.loads(a.dev_manifest.read_text())["instance_ids"]) if a.dev_manifest.exists() else set()
    units = pick_units(a.mine, set(a.srcs.split(",")), a.split, dev_ids, a.max_rollouts, a.seed)
    print(json.dumps({"units": len(units), "tasks": len({u["instance_id"] for u in units}),
                      "by_half": collections.Counter(u["half"] for u in units)}), flush=True)
    model = build_model(a.model)
    load_padded(model, a.adapter)
    model.adaptive_ckpt, model.no_ckpt_tokens, model.long_ckpt_policy = True, a.no_ckpt_tokens, "mlp"
    s = float(model.lora_scaling)
    nl = len(model.layers)
    H = model.H
    rows, W = {}, {}
    with torch.no_grad():
        eye = torch.eye(H, dtype=torch.bfloat16, device=model.dev)
        xt = torch.randn(8, H, device=model.dev).to(torch.bfloat16)
        werr = 0.0
        for i in range(nl):
            A = model.lora_params[model._pkey(GATE.format(i=i), "A")]
            B = model.lora_params[model._pkey(GATE.format(i=i), "B")]
            inter = B.shape[0]
            if a.row_select == "lora":
                rn = ((B[:, :ROWS] @ (A[:ROWS] @ A[:ROWS].T)) * B[:, :ROWS]).sum(1)
                R = torch.topk(rn, ROWS).indices.sort().values
            else:
                R = torch.randperm(inter, generator=torch.Generator().manual_seed(a.seed * 1000 + i))[:ROWS].sort().values.to(model.dev)
            lin = model.layers[i].lin["gate_up"]
            Wi = ops.frozen_linear(eye, lin)[:, :inter].index_select(1, R).T.float().contiguous()  # [ROWS, H]
            y = ops.frozen_linear(xt, lin)[:, :inter].index_select(1, R).float()
            werr = max(werr, float((y - xt.float() @ Wi.T).abs().max() / y.abs().max().clamp_min(1e-9)))
            if float(A[ROWS:].abs().max()) != 0.0 or float(B[:, ROWS:].abs().max()) != 0.0:
                raise ValueError("spare LoRA slots are not empty")
            B[R, torch.arange(ROWS, 2 * ROWS, device=model.dev)] = 1.0  # A stays 0: the function is unchanged
            rows[i], W[i] = R.cpu(), Wi.cpu()
    model.refresh_lora()
    print(json.dumps({"layers": nl, "H": H, "scaling": s, "w_row_check_rel_err": werr}), flush=True)
    params = dict(model.lora_params)
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    acc = {h: {k: torch.zeros_like(p) for k, p in params.items()} for h in (0, 1)}
    stat = {h: collections.Counter() for h in (0, 1)}
    done: list[str] = []
    a.out.mkdir(parents=True, exist_ok=True)
    part = a.out / "delgrad.part.pt"
    if part.exists():
        st = torch.load(part, map_location=model.dev)
        if any(not torch.equal(st["rows"][i].cpu(), rows[i]) for i in range(nl)):
            raise ValueError("partial file was made with other rows")
        acc, done = st["acc"], st["done"]
        stat = {h: collections.Counter(st["stat"][h]) for h in (0, 1)}
        print(f"resumed {len(done)} rollouts", flush=True)
    seen = set(done)

    def save(final: bool) -> None:
        out: dict[str, Any] = {"rows": rows, "W": W, "scaling": s, "done": done, "stat": {h: dict(stat[h]) for h in (0, 1)}}
        if not final:
            torch.save(dict(out, acc=acc), part)
            return
        # G = dJ/dW on the selected rows; the loss below is -J, and dloss/dA_spare = scaling * dloss/dW[R, :]
        out["G"] = {h: {i: (-acc[h][model._pkey(GATE.format(i=i), "A")][ROWS:] / s).cpu() for i in range(nl)} for h in (0, 1)}
        out["lora_grad"] = {h: {k: (-v).cpu() for k, v in acc[h].items()} for h in (0, 1)}  # dJ/d(masters), all modules
        torch.save(out, a.out / "delgrad.pt")

    t0 = time.time()
    n_new = 0
    for u in units:
        key = f'{u["src"]}|{u["run_key"]}'
        if key in seen:
            continue
        calls = read_calls(u["log"], a.max_seq_len)
        if not calls:
            stat[u["half"]]["no_calls"] += 1
            continue
        adv = float(u["adv"])
        tok = [0]

        def loss_fn(gidx: list[int], lps: list[Any]) -> Any:
            lp = torch.cat(lps)
            tok[0] += int(lp.numel())
            return -(adv * lp).sum()

        model.score_calls_subtrees(calls, 1.0, budget=a.budget or None, lora=True, loss_fn=loss_fn)
        h = u["half"]
        with torch.no_grad():
            for k, p in params.items():
                if p.grad is not None:
                    acc[h][k] += p.grad
                    p.grad = None
        stat[h]["rollouts"] += 1
        stat[h]["tokens"] += tok[0]
        stat[h]["pos" if adv > 0 else "neg"] += 1
        done.append(key)
        seen.add(key)
        n_new += 1
        if n_new % 10 == 0:
            el = time.time() - t0
            print(f"{len(done)}/{len(units)} rollouts, {el / n_new:.1f} s/rollout, eta {el / n_new * (len(units) - len(done)) / 60:.0f} min, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if n_new % 25 == 0:
            save(False)
    save(False)
    save(True)
    # checks: (1) chain rule identity dJ/dB[R, k] = dJ/dA_spare . A_k for the adapter's own slots k < 16;
    # (2) split-half agreement of the predicted deletion gains and of the ordinary LoRA gradient
    rep: dict[str, Any] = {"rollouts": len(done), "stat": {h: dict(stat[h]) for h in (0, 1)}, "w_row_check_rel_err": werr}
    ident = 0.0
    g0, g1 = [], []
    for i in range(nl):
        ka, kb = model._pkey(GATE.format(i=i), "A"), model._pkey(GATE.format(i=i), "B")
        ga = acc[0][ka][ROWS:] + acc[1][ka][ROWS:]
        gb = (acc[0][kb] + acc[1][kb])[rows[i].to(model.dev), :ROWS]
        pred = ga @ params[ka][:ROWS].T
        ident = max(ident, float((gb - pred).abs().max() / gb.abs().max().clamp_min(1e-30)))
        for h, dst in ((0, g0), (1, g1)):
            dst.append((W[i].to(model.dev) * acc[h][ka][ROWS:] / s).flatten())  # = -W * G = predicted gain
    v0, v1 = torch.cat(g0), torch.cat(g1)
    l0 = torch.cat([acc[0][k][:ROWS].flatten() if k.endswith("A") else acc[0][k][:, :ROWS].flatten() for k in sorted(params)])
    l1 = torch.cat([acc[1][k][:ROWS].flatten() if k.endswith("A") else acc[1][k][:, :ROWS].flatten() for k in sorted(params)])
    cos = lambda x, y: float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))  # noqa: E731
    rep.update({"chain_rule_identity_rel_err": ident, "gain_split_half_cos": cos(v0, v1),
                "gain_sign_agreement": float(((v0 > 0) == (v1 > 0))[(v0 != 0) & (v1 != 0)].float().mean()),
                "lora_grad_split_half_cos": cos(l0, l1), "seconds": time.time() - t0})
    (a.out / "delgrad_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    part.unlink(missing_ok=True)
    return 0


def cmd_fold(a: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    st = torch.load(a.grads, map_location="cpu")
    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r = int(cfg["r"])
    if r != ROWS or abs(float(cfg["lora_alpha"]) / r - 2.0) > 1e-9:
        raise ValueError("adapter must be rank 16 with scaling 2")
    sd = {k: v.float() for k, v in load_file(str(a.adapter / "adapter_model.safetensors")).items()}
    gen = torch.Generator().manual_seed(a.seed)
    out: dict[str, torch.Tensor] = {}
    n_del = n_cand = 0
    pred = 0.0
    fro_del = 0.0
    for k, v in sd.items():  # zero-pad every module to rank 32
        if ".lora_A." in k:
            out[k] = torch.cat([v, torch.zeros(ROWS, v.shape[1])], 0)
        else:
            out[k] = torch.cat([v, torch.zeros(v.shape[0], ROWS)], 1)
    for i, R in st["rows"].items():
        W = st["W"][i]
        gain = [-(W * st["G"][h][i]) for h in (0, 1)]
        g = gain[0] + gain[1]
        if a.mode == "gain":
            M = g > 0
            if a.frac < 1.0:
                M = g >= (torch.quantile(g[M], 1.0 - a.frac) if int(M.sum()) else 0.0)
        elif a.mode == "agree":
            M = (gain[0] > 0) & (gain[1] > 0)
        elif a.mode == "anti":
            M = g < 0
        elif a.mode == "rows":
            M = torch.ones_like(g, dtype=torch.bool)
        elif a.mode == "rand":
            ref = g > 0
            if a.frac < 1.0:
                ref = g >= (torch.quantile(g[ref], 1.0 - a.frac) if int(ref.sum()) else 0.0)
            M = torch.zeros_like(ref)
            for j in range(ROWS):  # same count per row, among the row's non-zero base weights
                nz = torch.nonzero(W[j] != 0).flatten()
                M[j, nz[torch.randperm(len(nz), generator=gen)[: int(ref[j].sum())]]] = True
        else:
            raise ValueError(a.mode)
        M &= W != 0
        ka = f"base_model.model.{GATE.format(i=i)}.lora_A.weight"
        kb = ka.replace(".lora_A.", ".lora_B.")
        out[ka][ROWS:] = -(W * M)
        out[kb][R, torch.arange(ROWS, 2 * ROWS)] = 0.5  # scaling 2 * 0.5 = 1
        n_del += int(M.sum())
        n_cand += M.numel()
        pred += float(g[M].sum())
        fro_del += float((W * M).pow(2).sum())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "adapter_config.json").write_text(json.dumps(dict(cfg, r=2 * ROWS, lora_alpha=4 * ROWS), indent=2))
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, str(a.out / "adapter_model.safetensors"))
    rep = {"mode": a.mode, "frac": a.frac, "deleted": n_del, "candidates": n_cand, "layers": len(st["rows"]),
           "deleted_fro": fro_del ** 0.5, "predicted_gain_first_order": pred, "rollouts": len(st["done"]), "rank": 2 * ROWS}
    (a.out / "del_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.delgrad")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grad")
    g.add_argument("--model", required=True)
    g.add_argument("--adapter", type=Path, required=True)
    g.add_argument("--mine", type=Path, required=True)
    g.add_argument("--srcs", required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--split", default="train")
    g.add_argument("--max-rollouts", type=int, default=1200)
    g.add_argument("--row-select", choices=("lora", "random"), default="lora")
    g.add_argument("--budget", type=int, default=0, help="unique tokens per tree forward; 0 = the trainer default")
    g.add_argument("--max-seq-len", type=int, default=32768)
    g.add_argument("--no-ckpt-tokens", type=int, default=20000, help="forwards above this recompute the MLP in backward")
    g.add_argument("--dev-manifest", type=Path, default=Path("/path/to/gemma-v2/eval_data/dev_v1/dev_v1_manifest.json"))
    g.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fold")
    f.add_argument("--grads", type=Path, required=True)
    f.add_argument("--adapter", type=Path, required=True)
    f.add_argument("--mode", choices=("gain", "agree", "anti", "rand", "rows"), required=True)
    f.add_argument("--frac", type=float, default=1.0, help="gain mode: keep this top fraction of positive-gain entries")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    return cmd_grad(a) if a.cmd == "grad" else cmd_fold(a)


if __name__ == "__main__":
    raise SystemExit(main())
