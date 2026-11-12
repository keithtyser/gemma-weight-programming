"""Select labelled residual-stream states (token positions) for diff-in-means directions (CPU).

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: From the miner output (v2.steer.mine) picks model calls and token positions for three contrasts, each with
  a positive and a negative class matched on the confounder that would otherwise dominate the mean difference:
    budget  "thinking will run into the 2,560-token budget": positions at thinking depth d in {384, 768, 1152, 1536}
            of calls whose end was budget-forced (label 1) vs calls that think long (n_think >= 1,600 + 64) but end
            NATURALLY before the budget (label 0). Same depths on both sides, and the control is "long but
            self-terminating", so the contrast is "runs away" rather than "thinks long". One call contributes all 4
            depths (fewer forwards).
    loop    "inside a verbatim repetition loop": positions loop_onset + {0, 64, 128} of calls with a 16-gram repeated
            >= 5 times (label 1) vs positions at the same absolute thinking depth of calls with no 16-gram repeated
            more than twice (label 0).
    giveup  "final decision to stop without submitting": the '<channel|>' position (end of thinking, where the call
            decides what to emit) of the LAST call of rollouts that ended without submit_patch (label 1) vs the same
            position of the last call of rollouts whose last call submitted (label 0).
  A position p means the residual stream at token p (the state that predicts token p+1); `abs` = prompt length + index
  into the completion. Calls of dev_v1 tasks are excluded (dev_v1 measures the edit), and each instance_id is assigned
  to train or heldout by a stable hash (70/30), so heldout validation never sees a task used for fitting.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl, eval_data/dev_v1/dev_v1_manifest.json. Writes <out>/states.jsonl (one
  row per call: request_log_path, call index, n_prompt, positions [{abs, tag, label}], split) + states_summary.json.
  Usage: python -m v2.steer.states --mine DIR --out DIR [--srcs screen_w2f,wave3,wave2] [--per-class 200 --per-class-heldout 120]
SRP/DRY check: Pass - consumer of v2.steer.mine; position labels are taken from the miner, never re-derived.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DEPTHS = (384, 768, 1152, 1536)


def split_of(instance_id: str) -> str:
    return "heldout" if int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16) % 10 < 3 else "train"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--srcs", default="screen_w2f,wave3,wave2")
    p.add_argument("--giveup-srcs", default="screen_w2f,wave3,wave2,wave1,screen_w1v6",
                   help="give-up is rare (<1 pct of rollouts): use every source for it")
    p.add_argument("--per-class", type=int, default=200, help="max train calls per (contrast, label)")
    p.add_argument("--per-class-heldout", type=int, default=120, help="max heldout calls per (contrast, label)")
    p.add_argument("--dev-manifest", type=Path, default=Path("/path/to/gemma-v2/eval_data/dev_v1/dev_v1_manifest.json"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    dev = set(json.loads(a.dev_manifest.read_text())["instance_ids"])
    srcs, gsrcs = set(a.srcs.split(",")), set(a.giveup_srcs.split(","))
    ros = {(r["src"], r["run_key"]): r for r in map(json.loads, (a.mine / "rollouts.jsonl").open())}
    calls = collections.defaultdict(list)
    with gzip.open(a.mine / "calls.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            r = ros.get((c["src"], c["run_key"]))
            if r is None or r["infra"] or r["instance_id"] in dev or c["is_agent"] is False:
                continue
            calls[(c["src"], c["run_key"])].append(c)
    # request_log_path per rollout from task_results (the miner keeps run_key; the path is under the source dir)
    paths: dict[tuple[str, str], str] = {}
    for (src, rk), r in ros.items():
        paths[(src, rk)] = r.get("request_log_path") or ""
    picked: dict[tuple[str, str, int], dict[str, Any]] = {}

    def add(c: dict[str, Any], abs_pos: int, tag: str, label: int) -> None:
        key = (c["src"], c["run_key"], c["call"])
        r = ros[(c["src"], c["run_key"])]
        row = picked.setdefault(key, {"src": c["src"], "run_key": c["run_key"], "instance_id": r["instance_id"],
                                      "call": c["call"], "n_prompt": c["n_prompt"], "n_comp": c["n_comp"],
                                      "split": split_of(r["instance_id"]), "positions": []})
        row["positions"].append({"abs": abs_pos, "tag": tag, "label": label})

    flat = [c for k, cs in calls.items() if k[0] in srcs for c in cs]
    pools: dict[tuple[str, int, str], list[tuple[dict[str, Any], int]]] = collections.defaultdict(list)
    for c in flat:
        sp = split_of(ros[(c["src"], c["run_key"])]["instance_id"])
        ts = c["think_start"]
        if c["forced"]:
            pools[("budget", 1, sp)].append((c, [c["n_prompt"] + ts + d for d in DEPTHS]))
        elif c["think_end"] is not None and c["n_think"] >= DEPTHS[-1] + 64 + 64:
            pools[("budget", 0, sp)].append((c, [c["n_prompt"] + ts + d for d in DEPTHS]))
        if c["max_ng"] >= 5 and c["loop_onset"] is not None:
            for off in (0, 64, 128):
                if c["loop_onset"] + off < c["n_think"]:
                    pools[("loop", 1, sp)].append((c, c["n_prompt"] + ts + c["loop_onset"] + off))
    # loop negatives: same absolute thinking depths as the positives, from calls without repetition
    loop_depths = [pos - c["n_prompt"] - c["think_start"] for (t, l, s), v in pools.items() if t == "loop" for c, pos in v]
    clean = [c for c in flat if c["max_ng"] <= 2 and c["think_end"] is not None]
    for d in loop_depths:
        cands = [c for c in rng.sample(clean, min(64, len(clean))) if c["n_think"] > d]
        if cands:
            c = cands[0]
            pools[("loop", 0, split_of(ros[(c["src"], c["run_key"])]["instance_id"]))].append(
                (c, c["n_prompt"] + c["think_start"] + d))
    # give-up: last call of each rollout
    for (src, rk), cs in calls.items():
        if src not in gsrcs or not cs:
            continue
        r, last = ros[(src, rk)], cs[-1]
        if last["think_end"] is None:
            continue
        sp = split_of(r["instance_id"])
        if r["no_submit"]:
            pools[("giveup", 1, sp)].append((last, last["n_prompt"] + last["think_end"]))
        elif "submit_patch" in (last["tools"] or []) and src in srcs:
            pools[("giveup", 0, sp)].append((last, last["n_prompt"] + last["think_end"]))
    counts: dict[str, int] = {}
    for (tag, label, sp), items in sorted(pools.items(), key=lambda kv: kv[0]):
        rng.shuffle(items)
        cap = a.per_class if sp == "train" else a.per_class_heldout
        keep = items[:cap]
        for c, pos in keep:
            if tag == "budget":
                for d, pp in zip(DEPTHS, pos):
                    add(c, pp, f"budget@{d}", label)
            else:
                add(c, pos, tag, label)
        counts[f"{tag}|{label}|{sp}"] = len(keep)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for row in picked.values():
            row["request_log_path"] = paths[(row["src"], row["run_key"])]
            fh.write(json.dumps(row) + "\n")
    tok = sum(max(p["abs"] for p in r["positions"]) + 1 for r in picked.values())
    summ = {"calls": len(picked), "positions": sum(len(r["positions"]) for r in picked.values()),
            "forward_tokens": tok, "counts": counts}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
