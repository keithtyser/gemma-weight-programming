"""Decision replay: what does an edited adapter do at logged decision points? (client of a vLLM server)

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: Fast inner-loop check of a weight edit before a full agent eval. For each logged agent call picked by
  v2.steer.states3 (the exact prompt_token_ids the engine saw), every adapter served by the endpoint generates the
  call again greedily (T = 0, harness thinking budget, stop at the end of the first tool call). Read-outs per
  adapter, split by what the logged wave-5 call did (label 1 = first edit, 0 = more exploration):
      frac_edit / frac_run / frac_read / frac_submit / frac_none   first tool of the regenerated call
      think   mean thinking tokens (tokens before <channel|>)
      old_ok  of the regenerated edit_file calls, the share whose old_string occurs verbatim in a tool result the
              prompt already holds (a necessary condition for the edit to land; a proxy for copy fidelity)
  Rows are paired across adapters (same prompts), so differences vs the reference adapter are paired.
  Reads <states>/states.jsonl + request logs. Writes <out>/replay_<adapter>.jsonl and prints / writes summary.json.
  Usage: python -m v2.steer.replay --states DIR --out DIR --url http://127.0.0.1:18001/v1 --adapters w5,a,b
             [--split heldout] [--limit 160] [--workers 8]
SRP/DRY check: Pass - v2.steer.confirm runs whole episodes; this replays single logged calls.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

END_THINK, END_CALL = 101, 49
NAME = re.compile(r"<\|tool_call>call:([a-z_]+)")
OLD = re.compile(r'old_string:<\|"\|>(.*?)<\|"\|>', re.S)


def read_prompt(path: str, call: int) -> tuple[list[int], list[str]]:
    """Prompt token ids of a logged call and the text of every tool result in its prompt."""
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == call:
                row = json.loads(line)
                seen = []
                for m in row["messages"]:
                    if m.get("role") != "tool":
                        continue
                    try:
                        seen += [v for v in json.loads(m["content"]).values() if isinstance(v, str)]
                    except (ValueError, AttributeError):
                        seen.append(str(m.get("content")))
                return row["prompt_token_ids"], seen
    raise IndexError(f"{path}: no call {call}")


def gen(url: str, model: str, prompt: list[int], budget: int, max_tokens: int, seen: list[str]) -> dict[str, Any]:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "skip_special_tokens": False, "stop_token_ids": [END_CALL], "thinking_token_budget": budget,
            "logprobs": 0, "return_tokens_as_token_ids": True}
    req = urllib.request.Request(url + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        ch = json.loads(r.read())["choices"][0]
    toks = [int(t.split(":")[1]) for t in ch["logprobs"]["tokens"]]
    m = NAME.search(ch["text"])
    o = OLD.search(ch["text"]) if m and m.group(1) == "edit_file" else None
    return {"tool": m.group(1) if m else "none", "n_tokens": len(toks),
            "old_ok": None if o is None else any(o.group(1) in t for t in seen), "old_len": len(o.group(1)) if o else 0,
            "think": toks.index(END_THINK) if END_THINK in toks else len(toks), "finish": ch["finish_reason"],
            "text_head": ch["text"][:200], "text_tail": ch["text"][-200:]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.replay")
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18001/v1")
    p.add_argument("--adapters", required=True)
    p.add_argument("--tag", default="edit@0")
    p.add_argument("--split", default="heldout")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget", type=int, default=2560)
    p.add_argument("--max-tokens", type=int, default=3072)
    a = p.parse_args(argv)
    rows = [r for r in map(json.loads, (a.states / "states.jsonl").open()) if a.split in ("all", r["split"])]
    if a.limit:
        rows = rows[: a.limit]
    items = []
    for r in rows:
        q = next((x for x in r["positions"] if x["tag"] == a.tag), None)
        if q is None:
            continue
        prompt, seen = read_prompt(r["request_log_path"], r["call"])
        if len(prompt) + a.max_tokens > 32768:
            continue
        items.append({"run_key": r["run_key"], "call": r["call"], "label": q["label"], "idx": q["idx"], "prompt": prompt,
                      "seen": seen})
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"n": len(items), "split": a.split}
    for name in a.adapters.split(","):
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda it: gen(a.url, name, it["prompt"], a.budget, a.max_tokens, it["seen"]), items))
        with (a.out / f"replay_{name.rsplit('/', 1)[-1]}.jsonl").open("w") as fh:  # base is served under its path
            for it, rr in zip(items, res):
                fh.write(json.dumps({**{k: it[k] for k in ("run_key", "call", "label", "idx")}, **rr}) + "\n")
        s: dict[str, Any] = {}
        for lab in (1, 0, "all"):
            rs = [rr for it, rr in zip(items, res) if lab == "all" or it["label"] == lab]
            cnt = collections.Counter(rr["tool"] for rr in rs)
            eds = [rr["old_ok"] for rr in rs if rr["old_ok"] is not None]
            s[str(lab)] = {"n": len(rs), "edit": round(cnt["edit_file"] / len(rs), 3),
                           "run": round(cnt["run_command"] / len(rs), 3), "read": round(cnt["read_file"] / len(rs), 3),
                           "submit": round(cnt["submit_patch"] / len(rs), 3), "none": round(cnt["none"] / len(rs), 3),
                           "think": round(sum(rr["think"] for rr in rs) / len(rs), 1),
                           "old_ok": round(sum(eds) / len(eds), 3) if eds else None, "n_old": len(eds),
                           "tokens": round(sum(rr["n_tokens"] for rr in rs) / len(rs), 1)}
        summary[name] = s
        print(name, json.dumps(s), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
