"""Labelled states for the "submit" decision: is the patch right (ok), and submit vs keep testing (sv) (CPU).

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Companion of v2.steer.states3 (same row format, read by v2.steer.collect). Two contrasts at the call that
  submits the patch, both taken after the rollout's first edit:
    ok   "the patch being submitted is right": the first submit_patch-only call of a rollout. Label 1 = the rollout
         was resolved, 0 = it was not. Every positive is paired with a negative of the SAME task (other rollout,
         same policy), so task and repository cancel in the mean difference; what is left is what differs between
         submitting a right and a wrong patch for the same task. With --ok-unpaired the remaining submits are
         added class-balanced without the task pairing (heldout tasks stay disjoint from train). Positions: ok@0 (last forced token before free
         thinking), ok@te (the end-of-thinking token).
    sv   "this call submits": label 1 = the submit call above (any rollout), label 0 = a run_command-only call after
         the first edit at the SAME agent-call index in another rollout (keep testing). Positions: sv@0, sv@te.
    bg   one random prompt position and one random thinking position per picked call (label 0), as in states3.
  Split by instance_id hash as in v2.steer.states (70/30); dev_v1 tasks excluded.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl, dev_v1 manifest. Writes <out>/states.jsonl + states_summary.json.
  Usage: python -m v2.steer.states4 --mine DIR --out DIR --srcs a,b [--pairs 300 --pairs-heldout 120]
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

SUBMIT, RUN, EDITS = ("submit_patch",), ("run_command",), ("edit_file", "write_file")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states4")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--srcs", required=True)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--pairs-heldout", type=int, default=120)
    p.add_argument("--pairs-per-task", type=int, default=3)
    p.add_argument("--ok-unpaired", action="store_true")
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
            if "_has_log" not in r:
                r["_has_log"] = Path(r.get("request_log_path") or "/nonexistent").exists()
            if not r["_has_log"]:
                continue
            calls[(c["src"], c["run_key"])].append(c)
    # per rollout: the first submit-only call after an edit, and the run-only calls after the first edit
    sub: dict[tuple[str, str, bool], list[tuple[dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
    sub_i: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
    run_i: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
    for key, cs in calls.items():
        cs.sort(key=lambda c: c["call"])
        r = ros[key]
        if r.get("resolved") is None:
            continue
        sp = split_of(r["instance_id"])
        edited = False
        for i, c in enumerate(cs):
            tools = tuple(c.get("tools") or ())
            ok = (c["think_start"] is not None and c["think_end"] is not None and not c["forced"]
                  and c["n_prompt"] <= a.max_prompt and c["n_think"] >= 8)
            if edited and tools == SUBMIT:
                if ok:
                    sub[(sp, r["instance_id"], bool(r["resolved"]))].append((c, r, i))
                    sub_i[(sp, i)].append((c, r, i))
                break
            if edited and ok and tools == RUN:
                run_i[(sp, i)].append((c, r, i))
            if any(t in EDITS for t in tools):
                edited = True

    def row(c: dict[str, Any], r: dict[str, Any], i: int, sp: str, tag: str, label: int, pid: int) -> dict[str, Any]:
        base = c["n_prompt"] + c["think_start"]
        pos = [{"abs": base - 1, "tag": f"{tag}@0", "label": label, "pair": pid, "idx": i},
               {"abs": c["n_prompt"] + c["think_end"], "tag": f"{tag}@te", "label": label, "pair": pid, "idx": i},
               {"abs": rng.randrange(8, c["n_prompt"]), "tag": "bg@prompt", "label": 0},
               {"abs": base + rng.randrange(0, c["n_think"]), "tag": "bg@think", "label": 0}]
        return {"src": c["src"], "run_key": c["run_key"], "instance_id": r["instance_id"], "call": c["call"],
                "n_prompt": c["n_prompt"], "n_comp": c["n_comp"], "split": sp, "positions": pos,
                "request_log_path": r.get("request_log_path") or ""}

    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    pid = 0
    for sp, cap in (("train", a.pairs), ("heldout", a.pairs_heldout)):
        # ok: within-task pairs, round-robin over tasks
        tasks = sorted({k[1] for k in sub if k[0] == sp and (sp, k[1], True) in sub and (sp, k[1], False) in sub})
        rng.shuffle(tasks)
        for t in tasks:
            rng.shuffle(sub[(sp, t, True)])
            rng.shuffle(sub[(sp, t, False)])
        n = 0
        for depth in range(a.pairs_per_task):
            for t in tasks:
                P, N = sub[(sp, t, True)], sub[(sp, t, False)]
                if n >= cap or depth >= len(P) or depth >= len(N):
                    continue
                picked.append(row(*P[depth], sp, "ok", 1, pid))
                picked.append(row(*N[depth], sp, "ok", 0, pid))
                pid += 1
                n += 1
        if a.ok_unpaired:  # too few tasks with both outcomes: all remaining submits, classes balanced, tasks differ
            used = {(r_["src"], r_["run_key"]) for r_ in picked}
            P = [x for k, v in sorted(sub.items()) if k[0] == sp and k[2] for x in v if (x[0]["src"], x[0]["run_key"]) not in used]
            N = [x for k, v in sorted(sub.items()) if k[0] == sp and not k[2] for x in v if (x[0]["src"], x[0]["run_key"]) not in used]
            rng.shuffle(P)
            rng.shuffle(N)
            for x, y in zip(P[: cap - n], N):
                picked.append(row(*x, sp, "ok", 1, pid))
                picked.append(row(*y, sp, "ok", 0, pid))
                pid += 1
                n += 1
        counts[f"ok|{sp}"] = n
        counts[f"ok_tasks|{sp}"] = len(tasks)
        # sv: submit vs run at the same call index, round-robin over indices
        idxs = sorted({k[1] for k in sub_i if k[0] == sp and (sp, k[1]) in run_i})
        for i in idxs:
            rng.shuffle(sub_i[(sp, i)])
            rng.shuffle(run_i[(sp, i)])
        n, depth = 0, 0
        while n < cap:
            took = False
            for i in idxs:
                P, N = sub_i[(sp, i)], run_i[(sp, i)]
                if n >= cap or depth >= len(P) or depth >= len(N):
                    continue
                picked.append(row(*P[depth], sp, "sv", 1, pid))
                picked.append(row(*N[depth], sp, "sv", 0, pid))
                pid += 1
                n += 1
                took = True
            depth += 1
            if not took:
                break
        counts[f"sv|{sp}"] = n
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for r_ in picked:
            fh.write(json.dumps(r_) + "\n")
    tok = sum(max(q["abs"] for q in r_["positions"]) + 1 for r_ in picked)
    summ = {"calls": len(picked), "forward_tokens": tok, "counts": counts,
            "submit_pool": {f"{sp}|{res}": sum(len(v) for k, v in sub.items() if k[0] == sp and k[2] == res)
                            for sp in ("train", "heldout") for res in (True, False)}}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
