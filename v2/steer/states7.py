"""Labelled states for a within-task success contrast: suc (this episode goes on to solve the task) (CPU).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Companion of v2.steer.states5 (same row format, read by v2.steer.collect). v2.steer.within shows which
  behaviours separate solved from unsolved episodes of the same task; this builds the matching activation contrast.
  For every task with both outcomes, a solved and an unsolved episode are paired and one agent call at the same
  call index k >= 1 is taken from each (call 0 has the same prompt in both, so it carries no contrast).
    suc  label 1 = the episode was resolved, 0 = it was not. Positions @0 (last forced token before thinking) and
         @te (end-of-thinking token). The pair shares task and call index, so task identity and depth cancel.
    bg   one random prompt position and one random thinking position per picked call (label 0), as in states3.
  Split by instance_id hash as in v2.steer.states (70/30); dev_v1 tasks excluded.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl. Writes <out>/states.jsonl + states_summary.json.
  Usage: python -m v2.steer.states7 --mine DIR --out DIR [--srcs a,b] [--pairs 400 --pairs-heldout 150 --per-task 2]
SRP/DRY check: Pass - reuses split_of from v2.steer.states; other states* modules label single-call behaviours.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
from pathlib import Path
from typing import Any

from v2.steer.states import split_of


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states7")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--srcs", default="")
    p.add_argument("--pairs", type=int, default=400)
    p.add_argument("--pairs-heldout", type=int, default=150)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--max-prompt", type=int, default=30000)
    p.add_argument("--dev-manifest", type=Path, default=Path("/path/to/gemma-v2/eval_data/dev_v1/dev_v1_manifest.json"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    dev = set(json.loads(a.dev_manifest.read_text())["instance_ids"])
    srcs = set(a.srcs.split(",")) if a.srcs else None
    ros = {(r["src"], r["run_key"]): r for r in map(json.loads, (a.mine / "rollouts.jsonl").open())}
    calls: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    with gzip.open(a.mine / "calls.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            r = ros.get((c["src"], c["run_key"]))
            if r is None or r["infra"] or r["instance_id"] in dev or c["is_agent"] is False:
                continue
            if srcs and c["src"] not in srcs:
                continue
            if not (r.get("request_log_path") and Path(r["request_log_path"]).exists()):
                continue
            calls[(c["src"], c["run_key"])].append(c)
    by_task: dict[str, dict[bool, list[tuple[str, str]]]] = collections.defaultdict(lambda: {True: [], False: []})
    for key in sorted(calls):
        calls[key].sort(key=lambda c: c["call"])
        by_task[ros[key]["instance_id"]][str(ros[key]["resolved"]) == "True"].append(key)

    def usable(c: dict[str, Any]) -> bool:
        return (c["think_start"] is not None and c["think_end"] is not None and not c["forced"]
                and c["n_prompt"] <= a.max_prompt and c["n_think"] >= 8)

    def row(c: dict[str, Any], r: dict[str, Any], i: int, sp: str, label: int, pid: int) -> dict[str, Any]:
        base = c["n_prompt"] + c["think_start"]
        pos = [{"abs": base - 1, "tag": "suc@0", "label": label, "pair": pid, "idx": i},
               {"abs": c["n_prompt"] + c["think_end"], "tag": "suc@te", "label": label, "pair": pid, "idx": i},
               {"abs": rng.randrange(8, c["n_prompt"]), "tag": "bg@prompt", "label": 0},
               {"abs": base + rng.randrange(0, c["n_think"]), "tag": "bg@think", "label": 0}]
        return {"src": c["src"], "run_key": c["run_key"], "instance_id": r["instance_id"], "call": c["call"],
                "n_prompt": c["n_prompt"], "n_comp": c["n_comp"], "split": sp, "positions": pos,
                "request_log_path": r.get("request_log_path") or ""}

    cand: dict[str, list[tuple[Any, ...]]] = {"train": [], "heldout": []}
    mixed = 0
    for task in sorted(by_task):
        S, U = by_task[task][True], by_task[task][False]
        if not S or not U:
            continue
        mixed += 1
        rng.shuffle(S)
        rng.shuffle(U)
        for s, u in list(zip(S, U))[: a.per_task]:
            cs, cu = calls[s], calls[u]
            ks = [k for k in range(1, min(len(cs), len(cu))) if usable(cs[k]) and usable(cu[k])]
            if not ks:
                continue
            k = rng.choice(ks)
            cand[split_of(task)].append((cs[k], ros[s], cu[k], ros[u], k))
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {"mixed_tasks": mixed}
    pid = 0
    for sp, cap in (("train", a.pairs), ("heldout", a.pairs_heldout)):
        rng.shuffle(cand[sp])
        for cs_k, rs, cu_k, ru, k in cand[sp][:cap]:
            picked.append(row(cs_k, rs, k, sp, 1, pid))
            picked.append(row(cu_k, ru, k, sp, 0, pid))
            pid += 1
        counts[f"suc|{sp}"] = min(cap, len(cand[sp]))
        counts[f"suc_tasks|{sp}"] = len({x[1]["instance_id"] for x in cand[sp][:cap]})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for r_ in picked:
            fh.write(json.dumps(r_) + "\n")
    tok = sum(max(q["abs"] for q in r_["positions"]) + 1 for r_ in picked)
    summ = {"calls": len(picked), "forward_tokens": tok, "counts": counts}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
