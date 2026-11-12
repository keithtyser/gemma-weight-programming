"""Prompt-rule contrast: the same logged decisions with and without a rule written into the system prompt.

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Step "locate" for compiling a natural-language rule into weights. For a sample of logged agent calls,
  decode the prompt, swap the base harness system prompt for a variant that carries one extra rule, re-encode, and
  emit both versions as single-call request logs plus states rows for v2.steer.collect. The labelled position is
  the last forced token before free thinking (prompt length + 2), tag "<rule>@0", label 1 = rule present, 0 = absent.
  The class-mean difference at that tag is the direction the rule writes into the residual stream at a decision;
  v2.steer.neuron turns it into a neuron. Calls whose prompt does not survive a tokenizer round trip, or whose
  tokens after the system prompt change under the swap, are dropped and counted.
  Reads <states>/states.jsonl only for its set of request logs. Writes <out>/states.jsonl, <out>/logs/<version>.jsonl.
  Usage: python -m v2.steer.states8 --states DIR --tokenizer tokenizer.json --base-harness H0
           --rule pedit=H1 --rule pexp=H2 --episodes 200 --per-episode 2 --out DIR
SRP/DRY check: Pass - states5/states7 label logged behaviour; this builds counterfactual prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

FORCED = 3  # "<|channel>", "thought", newline: identical in every completion


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.states8")
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base-harness", type=Path, required=True)
    p.add_argument("--rule", action="append", required=True, help="name=harness_dir (repeatable)")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--per-episode", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    s0 = (a.base_harness / "prompts/system.md").read_text().strip()
    rules = {}
    for spec in a.rule:
        name, d = spec.split("=", 1)
        rules[name] = (Path(d) / "prompts/system.md").read_text().strip()
        assert rules[name] != s0, name
    eps: dict[str, dict] = {}
    for line in (a.states / "states.jsonl").open():
        r = json.loads(line)
        eps.setdefault(r["request_log_path"], {k: r[k] for k in ("src", "run_key", "instance_id")})
    rng = random.Random(a.seed)
    paths = sorted(eps)
    rng.shuffle(paths)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logs = {v: (a.out / "logs" / f"{v}.jsonl").open("w") for v in ["plain", *rules]}
    rows, n, drop = [], 0, {"roundtrip": 0, "no_system": 0, "suffix": 0}
    for path in paths[: a.episodes]:
        calls = [(i, json.loads(l)) for i, l in enumerate(open(path))]
        calls = [(i, c) for i, c in calls if c.get("is_agent_call", True) and len(c.get("completion_token_ids") or []) > FORCED]
        for i, c in rng.sample(calls, min(a.per_episode, len(calls))):
            ids = c["prompt_token_ids"]
            text = tok.decode(ids, skip_special_tokens=False)
            if enc(text) != ids:
                drop["roundtrip"] += 1
                continue
            if text.count(s0) != 1:
                drop["no_system"] += 1
                continue
            tail = len(enc(text[text.index(s0) + len(s0):])) - 4  # tokens after the system prompt, minus a seam margin
            alt = {name: enc(text.replace(s0, s1)) for name, s1 in rules.items()}
            if any(v[-tail:] != ids[-tail:] for v in alt.values()):
                drop["suffix"] += 1
                continue
            meta = eps[path]
            split = "heldout" if int(hashlib.md5(meta["instance_id"].encode()).hexdigest(), 16) % 4 == 0 else "train"
            comp = c["completion_token_ids"][:FORCED]
            for v, pids in [("plain", ids), *alt.items()]:
                logs[v].write(json.dumps({"prompt_token_ids": pids, "completion_token_ids": comp}) + "\n")
                tags = list(rules) if v == "plain" else [v]
                rows.append({**meta, "src": f"{meta['src']}:{v}", "call": n, "orig_call": i, "n_prompt": len(pids),
                             "n_comp": FORCED, "split": split, "request_log_path": str(a.out / "logs" / f"{v}.jsonl"),
                             "positions": [{"abs": len(pids) + FORCED - 1, "tag": f"{t}@0", "label": int(v != "plain"),
                                            "pair": n} for t in tags]})
            n += 1
    for fh in logs.values():
        fh.close()
    with (a.out / "states.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"calls": n, "rows": len(rows), "dropped": drop,
                      "train": sum(r["split"] == "train" for r in rows), "rules": list(rules)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
