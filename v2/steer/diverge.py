"""Divergence mining: where do a solved and an unsolved greedy run of the same task first part ways?

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Evaluation is greedy, so two runs of one task by one model share an identical prefix until numeric noise
  flips one token. For every task with mixed outcomes across runs of the same arm, pair each solved run with each
  unsolved run, find the first agent call whose completion differs (prompts identical up to there), and record
  what each branch did at that call: tool chosen, thinking length, whether the split is inside thinking, the
  log-probability of the token each branch took. The table of "solved branch did X, unsolved branch did Y" is the
  diagnosis step for the next weight edit; the divergence calls are its decision points.
  Reads <eval>/<run>/**/task_results.jsonl and **/requests/<task>.jsonl. Writes <out>/diverge.jsonl, prints a summary.
  Usage: python -m v2.steer.diverge --eval results/steer_eval --runs b3_base,b4_base,... --out DIR
SRP/DRY check: Pass - v2.steer.within contrasts whole episodes; this locates the first differing decision.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

END_THINK = 101


def load_run(root: Path) -> tuple[dict[str, bool], dict[str, Path]]:
    res, logs = {}, {}
    for f in root.rglob("task_results.jsonl"):
        for line in f.open():
            d = json.loads(line)
            res[d["instance_id"]] = bool(d["resolved"])
    for f in root.rglob("requests/*.jsonl"):
        logs[f.stem] = f
    return res, logs


def calls(path: Path) -> list[dict]:
    out = []
    for line in path.open():
        r = json.loads(line)
        if not r.get("is_agent_call", True):
            continue
        tc = ((r.get("response_message") or {}).get("tool_calls") or [{}])[0].get("function", {})
        toks = r.get("completion_token_ids") or []
        out.append({"p": r["prompt_token_ids"], "c": toks, "lp": r.get("completion_logprobs") or [],
                    "tool": tc.get("name", "none"), "args": (tc.get("arguments") or "")[:300],
                    "think": toks.index(END_THINK) if END_THINK in toks else len(toks)})
    return out


def first_split(a: list[dict], b: list[dict]) -> dict | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x["p"] != y["p"]:
            return {"call": i, "kind": "prompt"}  # tool output differed (environment noise), not a model decision
        if x["c"] != y["c"]:
            j = next((k for k, (s, t) in enumerate(zip(x["c"], y["c"])) if s != t), min(len(x["c"]), len(y["c"])))
            return {"call": i, "kind": "think" if j < min(x["think"], y["think"]) else "action", "tok": j,
                    "lp_a": x["lp"][j] if j < len(x["lp"]) else None, "lp_b": y["lp"][j] if j < len(y["lp"]) else None}
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.diverge")
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    runs = {r: load_run(a.eval / r) for r in a.runs.split(",")}
    tasks = sorted(set().union(*(set(v[0]) for v in runs.values())))
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        have = [(r, v[0][t]) for r, v in runs.items() if t in v[0] and t in v[1]]
        win = [r for r, ok in have if ok]
        lose = [r for r, ok in have if not ok]
        if not win or not lose:
            continue
        cs = {r: calls(runs[r][1][t]) for r, _ in have}
        for w in win:
            for l in lose:
                s = first_split(cs[w], cs[l])
                if s is None:
                    continue
                i = s["call"]
                x, y = cs[w][i], cs[l][i]
                rows.append({"task": t, "win": w, "lose": l, **s, "n_win": len(cs[w]), "n_lose": len(cs[l]),
                             "tool_win": x["tool"], "tool_lose": y["tool"], "think_win": x["think"],
                             "think_lose": y["think"], "len_win": len(x["c"]), "len_lose": len(y["c"]),
                             "args_win": x["args"], "args_lose": y["args"],
                             "prior": [c["tool"] for c in cs[w][:i]][-6:]})
    with (a.out / "diverge.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    print("pairs", n, "tasks", len({r["task"] for r in rows}))
    print("kind", dict(collections.Counter(r["kind"] for r in rows)))
    m = [r for r in rows if r["kind"] != "prompt"]
    if not m:
        return 0
    mean = lambda k: round(sum(r[k] for r in m) / len(m), 1)
    print("call index of split: mean", mean("call"), "| episode calls win/lose", mean("n_win"), mean("n_lose"))
    print("thinking tokens at split: win", mean("think_win"), "lose", mean("think_lose"),
          "| win shorter in", sum(r["think_win"] < r["think_lose"] for r in m), "longer in",
          sum(r["think_win"] > r["think_lose"] for r in m))
    print("split position (call / win calls):", dict(sorted(collections.Counter(
        min(4, int(5 * r["call"] / max(1, r["n_win"]))) for r in m).items())))
    print("tool at split (win -> lose):")
    for (x, y), c in collections.Counter((r["tool_win"], r["tool_lose"]) for r in m).most_common(16):
        print(f"  {x:14s} {y:14s} {c}")
    tw, tl = collections.Counter(r["tool_win"] for r in m), collections.Counter(r["tool_lose"] for r in m)
    print("marginals win", dict(tw), "lose", dict(tl))
    lp = [(r["lp_a"], r["lp_b"]) for r in m if r["lp_a"] is not None and r["lp_b"] is not None]
    if lp:
        print("logprob of split token: win", round(sum(x for x, _ in lp) / len(lp), 2), "lose",
              round(sum(y for _, y in lp) / len(lp), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
