"""Labelled residual-stream states for two more contrasts: end-of-thinking and within-task success (CPU).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Companion of v2.steer.states (budget / loop / giveup). Same row format, so v2.steer.collect reads it.
    stop     "thinking is about to end": positions k tokens before the natural end-of-thinking token (k in 1, 8, 32;
             tags stop@1, stop@8, stop@32; label 1) of calls that end thinking on their own, vs positions at the SAME
             thinking depth in calls that go on thinking for at least 128 more tokens (label 0). Depth-matched, so the
             contrast is "wrapping up" rather than "how long has it been thinking".
    success  "this rollout will be resolved": for a task with both resolved and unresolved rollouts under the same
             policy, pairs one resolved with one unresolved rollout and takes the end-of-thinking position (where the
             call decides what to emit) of the calls at the same agent-call index in both (indices 0, 2, 5, 9, 14,
             20, 27 that exist in both; tags success@<i>; label = resolved). Every pair contributes the same indices
             to both classes, so task, repository and call index cancel in the mean difference; what is left is what
             differs between a solving and a failing attempt at the same task.
  Split by instance_id hash as in v2.steer.states (70/30); dev_v1 tasks excluded.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl, dev_v1 manifest. Writes <out>/states.jsonl + states_summary.json.
  Usage: python -m v2.steer.states2 --mine DIR --out DIR --srcs a,b [--stop 300 --stop-heldout 120 --pairs 400
         --pairs-heldout 160 --pairs-per-task 2]
SRP/DRY check: Pass - reuses split_of from v2.steer.states; labels come from the miner.
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

STOP_K = (1, 8, 32)
CALL_IDX = (0, 2, 5, 9, 14, 20, 27)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states2")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--srcs", required=True)
    p.add_argument("--stop", type=int, default=300, help="train calls per class for the stop contrast")
    p.add_argument("--stop-heldout", type=int, default=120)
    p.add_argument("--pairs", type=int, default=400, help="train (resolved, unresolved) rollout pairs")
    p.add_argument("--pairs-heldout", type=int, default=160)
    p.add_argument("--pairs-per-task", type=int, default=2)
    p.add_argument("--max-prompt", type=int, default=60000, help="skip calls with a longer prompt (forward cost)")
    p.add_argument("--dev-manifest", type=Path, default=Path("/path/to/gemma-v2/eval_data/dev_v1/dev_v1_manifest.json"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    dev = set(json.loads(a.dev_manifest.read_text())["instance_ids"])
    srcs = set(a.srcs.split(","))
    ros = {(r["src"], r["run_key"]): r for r in map(json.loads, (a.mine / "rollouts.jsonl").open())}
    calls: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    with gzip.open(a.mine / "calls.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            r = ros.get((c["src"], c["run_key"]))
            if r is None or r["infra"] or r["instance_id"] in dev or c["is_agent"] is False or c["src"] not in srcs:
                continue
            calls[(c["src"], c["run_key"])].append(c)
    for cs in calls.values():
        cs.sort(key=lambda c: c["call"])
    picked: dict[tuple[str, str, int], dict[str, Any]] = {}

    def add(c: dict[str, Any], abs_pos: int, tag: str, label: int, extra: dict[str, Any] | None = None) -> None:
        key = (c["src"], c["run_key"], c["call"])
        r = ros[(c["src"], c["run_key"])]
        row = picked.setdefault(key, {"src": c["src"], "run_key": c["run_key"], "instance_id": r["instance_id"],
                                      "call": c["call"], "n_prompt": c["n_prompt"], "n_comp": c["n_comp"],
                                      "split": split_of(r["instance_id"]), "positions": [],
                                      "request_log_path": r.get("request_log_path") or ""})
        row["positions"].append({"abs": abs_pos, "tag": tag, "label": label, **(extra or {})})

    counts: dict[str, int] = collections.Counter()
    # ---- stop
    flat = [c for cs in calls.values() for c in cs if c["n_prompt"] <= a.max_prompt and c["think_start"] is not None]
    natural = [c for c in flat if c["think_end"] is not None and not c["forced"] and c["max_ng"] <= 2
               and c["think_end"] - c["think_start"] > STOP_K[-1] + 16]
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in natural:
        by_split[split_of(ros[(c["src"], c["run_key"])]["instance_id"])].append(c)
    for sp, cap in (("train", a.stop), ("heldout", a.stop_heldout)):
        pool = by_split[sp]
        rng.shuffle(pool)
        used: set[tuple[str, str, int]] = set()
        n = 0
        for c in pool:
            if n >= cap:
                break
            key = (c["src"], c["run_key"], c["call"])
            if key in used:
                continue
            d1 = c["think_end"] - 1 - c["think_start"]  # depth of the last thinking token
            cands = [x for x in rng.sample(pool, min(200, len(pool)))
                     if (x["src"], x["run_key"], x["call"]) not in used and x is not c
                     and x["think_end"] - x["think_start"] >= d1 + 1 + 128]
            if not cands:
                continue
            x = cands[0]
            used.update({key, (x["src"], x["run_key"], x["call"])})
            for k in STOP_K:
                d = d1 - (k - 1)
                add(c, c["n_prompt"] + c["think_start"] + d, f"stop@{k}", 1, {"depth": d})
                add(x, x["n_prompt"] + x["think_start"] + d, f"stop@{k}", 0, {"depth": d})
            n += 1
        counts[f"stop|{sp}"] = n
    # ---- success
    by_task: dict[tuple[str, str], dict[int, list[tuple[str, str]]]] = collections.defaultdict(lambda: {0: [], 1: []})
    for key, cs in calls.items():
        r = ros[key]
        if r.get("resolved") is None:
            continue
        by_task[(key[0], r["instance_id"])][int(bool(r["resolved"]))].append(key)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    pid = 0
    for sp, cap in (("train", a.pairs), ("heldout", a.pairs_heldout)):
        n = 0
        for rnd in range(a.pairs_per_task):  # one pair per task first, then second pairs: spreads pairs over tasks
            for t in tasks:
                if n >= cap or split_of(t[1]) != sp:
                    continue
                g = by_task[t]
                if rnd == 0:
                    rng.shuffle(g[0])
                    rng.shuffle(g[1])
                if len(g[0]) <= rnd or len(g[1]) <= rnd:
                    continue
                ca, cb = calls[g[1][rnd]], calls[g[0][rnd]]
                idx = [i for i in CALL_IDX if i < min(len(ca), len(cb))
                       and all(x[i]["think_end"] is not None and x[i]["n_prompt"] <= a.max_prompt for x in (ca, cb))]
                if not idx:
                    continue
                for i in idx:
                    for cs, label in ((ca, 1), (cb, 0)):
                        c = cs[i]
                        add(c, c["n_prompt"] + c["think_end"], f"success@{i}", label, {"pair": pid})
                pid += 1
                n += 1
        counts[f"success_pairs|{sp}"] = n
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for row in picked.values():
            fh.write(json.dumps(row) + "\n")
    tok = sum(max(q["abs"] for q in r["positions"]) + 1 for r in picked.values())
    tags = collections.Counter((q["tag"], q["label"], r["split"]) for r in picked.values() for q in r["positions"])
    summ = {"calls": len(picked), "positions": sum(len(r["positions"]) for r in picked.values()), "forward_tokens": tok,
            "counts": dict(counts), "rollouts": len({(r["src"], r["run_key"]) for r in picked.values()}),
            "tags": {f"{t}|{l}|{s}": n for (t, l, s), n in sorted(tags.items())}}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
