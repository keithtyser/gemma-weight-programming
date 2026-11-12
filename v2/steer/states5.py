"""Labelled states around failed edit_file calls: ef (last edit failed), rf (re-read vs retry), eok (edit lands) (CPU).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Companion of v2.steer.states4 (same row format, read by v2.steer.collect). v2.steer.editfail shows that a
  third of the base model's edit_file calls fail ("old_string not found") and that an episode with a failed edit is
  about four times less likely to solve the same task. Three contrasts, labelled from the tool results that the
  request logs carry in `messages` (the miner does not keep them):
    ef   "the edit I just made failed": an agent call whose previous call was edit_file. Label 1 = that edit
         returned status error, 0 = status ok. This is the READ side of a conditional neuron.
    rf   "after a failed edit, read the file again": calls right after a failed edit. Label 1 = this call is
         read_file only, 0 = edit_file only (a retry). This is the WRITE side.
    eok  "this edit will land": edit_file-only calls. Label 1 = its result was ok, 0 = error.
    bg   one random prompt position and one random thinking position per picked call (label 0), as in states3.
  Positions per contrast: @0 (last forced token before free thinking) and @te (the end-of-thinking token).
  Classes are balanced within split; negatives are drawn at the same agent-call index where possible.
  Split by instance_id hash as in v2.steer.states (70/30); dev_v1 tasks excluded.
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl + request logs. Writes <out>/states.jsonl + states_summary.json.
  Usage: python -m v2.steer.states5 --mine DIR --out DIR --srcs a,b [--pairs 300 --pairs-heldout 120]
SRP/DRY check: Pass - reuses split_of from v2.steer.states; states4 labels submits, this labels edit outcomes.
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

EDIT, READ = ("edit_file",), ("read_file",)


def results(path: str) -> dict[int, tuple[str, str]]:
    """Per request-log line: (name, status) of the tool result that ends its prompt, '' if none."""
    out: dict[int, tuple[str, str]] = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            m = json.loads(line)["messages"]
            if len(m) < 2 or m[-1].get("role") != "tool" or not m[-2].get("tool_calls"):
                continue
            try:
                st = json.loads(m[-1]["content"]).get("status") or ""
            except (ValueError, AttributeError):
                st = ""
            out[i] = (m[-2]["tool_calls"][-1]["function"]["name"], st)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states5")
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
            calls[(c["src"], c["run_key"])].append(c)
    # pools[(contrast, split, label)] -> [(call, rollout, agent-call index)]
    pools: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
    stats: collections.Counter[str] = collections.Counter()
    for key, cs in sorted(calls.items()):
        r = ros[key]
        path = r.get("request_log_path") or ""
        if not path or not Path(path).exists():
            stats["no_log"] += 1
            continue
        res = results(path)
        cs.sort(key=lambda c: c["call"])
        sp = split_of(r["instance_id"])
        for i, c in enumerate(cs):
            tools = tuple(c.get("tools") or ())
            prev = res.get(c["call"], ("", ""))
            nxt = res.get(cs[i + 1]["call"], ("", "")) if i + 1 < len(cs) else ("", "")
            if prev[0] == "edit_file" and prev[1] in ("ok", "error"):
                stats[f"prev_edit_{prev[1]}"] += 1
            ok = (c["think_start"] is not None and c["think_end"] is not None and not c["forced"]
                  and c["n_prompt"] <= a.max_prompt and c["n_think"] >= 8)
            if not ok:
                continue
            if prev[0] == "edit_file" and prev[1] in ("ok", "error"):
                pools[("ef", sp, int(prev[1] == "error"))].append((c, r, i))
                if prev[1] == "error" and tools in (READ, EDIT):
                    pools[("rf", sp, int(tools == READ))].append((c, r, i))
            if tools == EDIT and nxt[0] == "edit_file" and nxt[1] in ("ok", "error"):
                pools[("eok", sp, int(nxt[1] == "ok"))].append((c, r, i))

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
    for tag in ("ef", "rf", "eok"):
        for sp, cap in (("train", a.pairs), ("heldout", a.pairs_heldout)):
            P, N = pools[(tag, sp, 1)], pools[(tag, sp, 0)]
            rng.shuffle(P)
            rng.shuffle(N)
            by_idx: dict[int, list[tuple[dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
            for x in N:
                by_idx[x[2]].append(x)
            n = same = 0
            for x in P:
                if n >= cap:
                    break
                cand = by_idx.get(x[2])
                if cand:
                    y = cand.pop()
                    same += 1
                else:  # nearest call index that still has a negative
                    ks = [k for k, v in by_idx.items() if v]
                    if not ks:
                        break
                    y = by_idx[min(ks, key=lambda k: abs(k - x[2]))].pop()
                picked.append(row(*x, sp, tag, 1, pid))
                picked.append(row(*y, sp, tag, 0, pid))
                pid += 1
                n += 1
            counts[f"{tag}|{sp}"] = n
            counts[f"{tag}_same_idx|{sp}"] = same
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "states.jsonl").open("w") as fh:
        for r_ in picked:
            fh.write(json.dumps(r_) + "\n")
    tok = sum(max(q["abs"] for q in r_["positions"]) + 1 for r_ in picked)
    summ = {"calls": len(picked), "forward_tokens": tok, "counts": counts, "stats": dict(stats),
            "pool": {f"{k[0]}|{k[1]}|{k[2]}": len(v) for k, v in sorted(pools.items())}}
    (a.out / "states_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
