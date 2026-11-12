"""Labelled states for the "commit to an edit" contrast, plus background positions (CPU).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Companion of v2.steer.states / states2 (same row format, read by v2.steer.collect).
    edit   "this call will make the first edit": agent calls up to and including the rollout's first edit_file call,
           at agent-call index >= 2. Label 1 = the call's only tool is edit_file (the first edit); label 0 = its only
           tool is run_command or read_file (more exploration). Every positive is paired with a negative at the SAME
           agent-call index (other rollout), so "how far into the episode" cancels in the mean difference. Positions:
           edit@0 (last forced token before free thinking: the state before any thinking text commits to an action),
           edit@32 (32 thinking tokens in, if the call thinks >= 48 tokens), edit@te (the end-of-thinking token).
    bg     one uniformly random prompt position (bg@prompt) and one random thinking position (bg@think) per picked
           call, label 0: used for the mean MLP-input direction and for firing statistics of hand-written neurons.
  Split by instance_id hash as in v2.steer.states (70/30); dev_v1 tasks excluded.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl, dev_v1 manifest. Writes <out>/states.jsonl + states_summary.json.
  Usage: python -m v2.steer.states3 --mine DIR --out DIR --srcs a,b [--pairs 300 --pairs-heldout 120]
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

EDIT, EXPLORE = ("edit_file",), (("run_command",), ("read_file",))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states3")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--srcs", required=True)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--pairs-heldout", type=int, default=120)
    p.add_argument("--max-prompt", type=int, default=30000)
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
            if "_has_log" not in r:  # only rollouts whose request log is on this machine can be replayed
                r["_has_log"] = Path(r.get("request_log_path") or "/nonexistent").exists()
            if not r["_has_log"]:
                continue
            calls[(c["src"], c["run_key"])].append(c)
    pool: dict[tuple[str, int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for key, cs in calls.items():
        cs.sort(key=lambda c: c["call"])
        r = ros[key]
        sp = split_of(r["instance_id"])
        for i, c in enumerate(cs):
            tools = tuple(c.get("tools") or ())
            ok = (i >= 2 and c["think_start"] is not None and c["think_end"] is not None and not c["forced"]
                  and c["n_prompt"] <= a.max_prompt and c["n_think"] >= 8)
            if tools == EDIT:
                if ok:
                    pool[(sp, i, 1)].append((c, r))
                break  # only calls up to the first edit
            if "edit_file" in tools:
                break
            if ok and tools in EXPLORE:
                pool[(sp, i, 0)].append((c, r))
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    pid = 0
    for sp, cap in (("train", a.pairs), ("heldout", a.pairs_heldout)):
        idxs = sorted({k[1] for k in pool if k[0] == sp and k[2] == 1})
        for i in idxs:
            rng.shuffle(pool[(sp, i, 1)])
            rng.shuffle(pool[(sp, i, 0)])
        n, depth = 0, 0
        while n < cap:  # round-robin over call indices so no single index dominates
            took = False
            for i in idxs:
                P, N = pool[(sp, i, 1)], pool[(sp, i, 0)]
                if n >= cap or depth >= len(P) or depth >= len(N):
                    continue
                for (c, r), label in ((P[depth], 1), (N[depth], 0)):
                    base = c["n_prompt"] + c["think_start"]
                    pos = [{"abs": base - 1, "tag": "edit@0", "label": label, "pair": pid, "idx": i}]
                    if c["n_think"] >= 48:
                        pos.append({"abs": base + 31, "tag": "edit@32", "label": label, "pair": pid, "idx": i})
                    pos.append({"abs": c["n_prompt"] + c["think_end"], "tag": "edit@te", "label": label, "pair": pid, "idx": i})
                    pos.append({"abs": rng.randrange(8, c["n_prompt"]), "tag": "bg@prompt", "label": 0})
                    pos.append({"abs": base + rng.randrange(0, c["n_think"]), "tag": "bg@think", "label": 0})
                    picked.append({"src": c["src"], "run_key": c["run_key"], "instance_id": r["instance_id"],
                                   "call": c["call"], "n_prompt": c["n_prompt"], "n_comp": c["n_comp"], "split": sp,
                                   "positions": pos, "request_log_path": r.get("request_log_path") or ""})
                pid += 1
                n += 1
                took = True
            depth += 1
            if not took:
                break
        counts[f"pairs|{sp}"] = n
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for row in picked:
            fh.write(json.dumps(row) + "\n")
    tok = sum(max(q["abs"] for q in r["positions"]) + 1 for r in picked)
    idx = collections.Counter(q["idx"] for r in picked for q in r["positions"] if q["tag"] == "edit@0")
    summ = {"calls": len(picked), "forward_tokens": tok, "counts": counts,
            "pool": {f"{k[0]}|{k[2]}": sum(len(v) for kk, v in pool.items() if kk[0] == k[0] and kk[2] == k[2]) for k in pool},
            "call_index_hist": dict(sorted(idx.items()))}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
