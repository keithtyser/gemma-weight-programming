"""Behavioural probe of edited adapters on logged agent contexts (GPU, vLLM offline): does thinking still run away?

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: A fast filter before dev_v1 (never a decision metric). Re-generates single model calls from logged prompts
  (exact prompt token ids from the request logs) under every --adapter in one vLLM process and measures the call-level
  pathologies with the same definitions as v2.steer.mine:
    - think tokens (no thinking budget is applied here, max_tokens 4096, so we see where thinking would end),
      share >= 2,560 (would be budget-forced), share unclosed at max_tokens;
    - verbatim loop (a 16-gram repeated >= 5 times in thinking), Wait per 1k thinking tokens;
    - share of calls that close thinking and then emit a tool call (Gemma 4 tool-call token present).
  Prompt sets (heldout instance_ids only, dev_v1 tasks excluded, from e4b_wave2_final's own screen rollouts):
    runaway  prompts whose logged call was budget-forced;  normal  random prompts whose logged call ended naturally.
  Decoding: greedy (the submission) and T=1 x --samples (the training sampler).
  Reads <mine>/calls.jsonl.gz + rollouts.jsonl, request logs, adapters. Writes <out>/probe.json + samples.jsonl.
  Usage: CUDA_VISIBLE_DEVICES=1 python -m v2.steer.probe --mine DIR --adapter ref=A --adapter ed=B --out DIR
SRP/DRY check: Pass - mirrors v2.eval.lora_loop_probe (offline vLLM, several LoRAs in one process) but on exact
  logged prompt ids with thinking-channel metrics; reuses v2.steer.mine.repetition/_think_span/wait_stats.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from v2.steer.mine import BUDGET, _think_span, repetition, wait_stats
from v2.steer.states import split_of


def pick_prompts(mine: Path, n: int, seed: int, src: str, dev_manifest: Path) -> dict[str, list[dict[str, Any]]]:
    dev = set(json.loads(dev_manifest.read_text())["instance_ids"])
    ros = {(r["src"], r["run_key"]): r for r in map(json.loads, (mine / "rollouts.jsonl").open())}
    run, norm = [], []
    with gzip.open(mine / "calls.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            r = ros.get((c["src"], c["run_key"]))
            if c["src"] != src or r is None or r["infra"] or r["instance_id"] in dev or c["is_agent"] is False:
                continue
            if split_of(r["instance_id"]) != "heldout" or c["n_prompt"] > 28000:
                continue
            c["request_log_path"] = r["request_log_path"]
            (run if c["forced"] else norm).append(c)
    rng = random.Random(seed)
    rng.shuffle(run)
    rng.shuffle(norm)
    return {"runaway": run[:n], "normal": [c for c in norm if c["think_end"] is not None][:n]}


def call_metrics(ids: list[int], tool_tok: int | None) -> dict[str, Any]:
    s, e = _think_span(ids)
    think = ids[s : (e if e is not None else len(ids))]
    rep = repetition(think)
    nw, wmax, _ = wait_stats(think)
    after = ids[e + 1 :] if e is not None else []
    return {"n_think": len(think), "closed": e is not None, "would_force": (e is None) or (e - 1 >= BUDGET),
            "loop5": rep["max_ng"] >= 5, "max_ng": rep["max_ng"], "wait": nw,
            "tool_call": bool(tool_tok is not None and tool_tok in after)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.probe")
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--adapter", action="append", required=True, help="NAME=DIR")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct")
    p.add_argument("--n", type=int, default=100, help="prompts per set")
    p.add_argument("--samples", type=int, default=2, help="T=1 samples per prompt (plus one greedy)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--src", default="screen_w2f")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev-manifest", type=Path, default=Path("/path/to/gemma-v2/eval_data/dev_v1/dev_v1_manifest.json"))
    a = p.parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    sets = pick_prompts(a.mine, a.n, a.seed, a.src, a.dev_manifest)
    tok = AutoTokenizer.from_pretrained(a.model)
    tool_ids = tok.encode("<|tool_call>", add_special_tokens=False)
    tool_tok = tool_ids[0] if len(tool_ids) == 1 else None
    prompts: list[tuple[str, dict[str, Any], list[int]]] = []
    for sname, cs in sets.items():
        for c in cs:
            with open(c["request_log_path"]) as fh:
                row = json.loads(fh.readlines()[c["call"]])
            prompts.append((sname, c, row["prompt_token_ids"]))
    arms = {}
    max_r = 16
    for i, spec in enumerate(a.adapter, start=1):
        name, ad = spec.split("=", 1)
        max_r = max(max_r, int(json.loads((Path(ad) / "adapter_config.json").read_text())["r"]))
        arms[name] = LoRARequest(name, i, str(Path(ad).resolve()))
    bucket = next(b for b in (16, 32, 64, 128) if b >= max_r)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.85, enable_lora=True,
              max_lora_rank=bucket, max_loras=len(arms), seed=a.seed, limit_mm_per_prompt={"image": 0, "audio": 0})
    tp = [TokensPrompt(prompt_token_ids=x) for _, _, x in prompts]
    decs = {"greedy": SamplingParams(temperature=0.0, max_tokens=a.max_tokens),
            "t1": SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, n=a.samples, seed=a.seed)}
    a.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"n_prompts": {k: len(v) for k, v in sets.items()}, "max_tokens": a.max_tokens,
                              "tool_tok": tool_tok, "arms": {}}
    with (a.out / "samples.jsonl").open("w") as fs:
        for name, lr in arms.items():
            for dname, sp in decs.items():
                outs = llm.generate(tp, sp, lora_request=lr)
                agg: dict[str, list[dict[str, Any]]] = {"runaway": [], "normal": []}
                for (sname, c, _), o in zip(prompts, outs):
                    for comp in o.outputs:
                        m = call_metrics(list(comp.token_ids), tool_tok)
                        agg[sname].append(m)
                        fs.write(json.dumps({"arm": name, "dec": dname, "set": sname, "run_key": c["run_key"],
                                             "call": c["call"], **m}) + "\n")
                for sname, ms in agg.items():
                    n = max(1, len(ms))
                    report["arms"].setdefault(name, {})[f"{dname}/{sname}"] = {
                        "n": len(ms), "would_force": sum(m["would_force"] for m in ms) / n,
                        "unclosed": sum(not m["closed"] for m in ms) / n,
                        "mean_think": sum(m["n_think"] for m in ms) / n,
                        "loop5": sum(m["loop5"] for m in ms) / n,
                        "wait_per_1k": 1000 * sum(m["wait"] for m in ms) / max(1, sum(m["n_think"] for m in ms)),
                        "tool_call": sum(m["tool_call"] for m in ms) / n}
                print(name, dname, json.dumps({k: v for k, v in report["arms"][name].items() if k.startswith(dname)}),
                      flush=True)
    (a.out / "probe.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
