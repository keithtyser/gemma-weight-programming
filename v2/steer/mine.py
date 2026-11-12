"""Mine E4B agent rollouts for reasoning pathologies (CPU only).

Author: Claude Opus 5.5 (steer agent)
Date: 06-November-2026
PURPOSE: Reads rollout dirs written by the swe stream worker (task_results.jsonl + requests/<run_key>.jsonl, one JSON
  line per model call with prompt/completion token ids and engine logprobs) and labels every model call and rollout:
    - thinking length in tokens (completion starts with '<|channel>thought\\n', thinking ends at '<channel|>');
    - budget-forced end: the '<channel|>' token was forced by vLLM's ThinkingTokenBudgetLogitsProcessor, recognised
      exactly as in v2.tii.trainer.logits_processors (processed logprob exactly 0.0 after >= budget thinking tokens);
    - looping: verbatim repetition inside the thinking channel (max count of any 16-token n-gram, distinct-8-gram
      ratio of the thinking tail, duplicated-line fraction from v2.eval.lora_loop_probe);
    - 'Wait' tokens (ids of 'Wait' / ' Wait') and the densest 512-token Wait window;
    - per rollout: resolved, timeout (7.5-min session cap), no submit_patch, turns cap, totals.
  Writes <out>/calls.jsonl.gz (one row per call), <out>/rollouts.jsonl (one row per rollout) and <out>/summary.json.
  The call rows also carry the token positions used later as labelled states by v2.steer.collect (loop onset, depth
  checkpoints), so activation extraction never re-derives labels.
  Usage: python -m v2.steer.mine --src NAME=DIR [--src ...] --out DIR [--workers 12]
SRP/DRY check: Pass - reuses line_repeat_fraction (v2.eval.lora_loop_probe); no other pathology miner in the repo.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from v2.eval.lora_loop_probe import line_repeat_fraction

# Gemma 4 tokenizer ids (checked against the model tokenizer on 06-Nov-2026).
THINK_START, THINK_END, NL, THOUGHT = 100, 101, 107, 45518
WAIT_IDS = frozenset({20470, 29202})  # 'Wait', ' Wait'
BUDGET = 2560  # harness thinking_token_budget (docs/rl-training.md)
NGRAM = 16


def _think_span(comp: list[int]) -> tuple[int, int | None]:
    """(first thinking token index, index of '<channel|>' or None). Completions open '<|channel>thought\\n'."""
    if len(comp) >= 3 and comp[0] == THINK_START and comp[1] == THOUGHT:
        start = 3 if comp[2] == NL else 2
    else:
        return 0, None if THINK_END not in comp else comp.index(THINK_END)
    try:
        return start, comp.index(THINK_END, start)
    except ValueError:
        return start, None


def repetition(tokens: list[int]) -> dict[str, Any]:
    """Verbatim-repetition stats of a token span. loop_onset = position (in span) where some 16-gram occurs for the
    3rd time, the earliest point at which the span is provably in a verbatim loop."""
    if len(tokens) < NGRAM:
        return {"max_ng": 0, "loop_onset": None, "tail_d8": 1.0}
    seen: dict[tuple[int, ...], int] = collections.defaultdict(int)
    onset, mx = None, 0
    for i in range(len(tokens) - NGRAM + 1):
        g = tuple(tokens[i : i + NGRAM])
        seen[g] += 1
        if seen[g] > mx:
            mx = seen[g]
        if onset is None and seen[g] == 3:
            onset = i + NGRAM
    tail = tokens[-512:]
    g8 = [tuple(tail[i : i + 8]) for i in range(len(tail) - 7)]
    d8 = len(set(g8)) / len(g8) if g8 else 1.0
    return {"max_ng": mx, "loop_onset": onset, "tail_d8": round(d8, 4)}


def wait_stats(tokens: list[int]) -> tuple[int, int, list[int]]:
    pos = [i for i, t in enumerate(tokens) if t in WAIT_IDS]
    best, j = 0, 0
    for i in range(len(pos)):  # densest 512-token window
        while pos[i] - pos[j] >= 512:
            j += 1
        best = max(best, i - j + 1)
    return len(pos), best, pos


def mine_rollout(args: tuple[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src, tr = args
    rk = tr["run_key"]
    path = Path(tr["request_log_path"])
    calls: list[dict[str, Any]] = []
    tool_names: list[str] = []
    if path.exists():
        with path.open() as fh:
            lines = fh.readlines()
        for ci, line in enumerate(lines):
            r = json.loads(line)
            comp = r.get("completion_token_ids") or []
            lps = r.get("completion_logprobs") or []
            msg = r.get("response_message") or {}
            names = [tc.get("function", {}).get("name") for tc in (msg.get("tool_calls") or [])]
            tool_names += names
            s, e = _think_span(comp)
            think = comp[s : (e if e is not None else len(comp))]
            n_think = len(think)
            # The processor counts every token after '<|channel>' (so 'thought' and the newline too): forced iff the end token sits
            # at >= BUDGET tokens after the start token and its processed logprob is exactly 0.0.
            forced = bool(e is not None and s > 0 and e - 1 >= BUDGET and e < len(lps) and lps[e] == 0.0)
            rep = repetition(think)
            nw, wmax, wpos = wait_stats(think)
            text = msg.get("reasoning_content") or msg.get("reasoning") or ""
            calls.append({
                "src": src, "run_key": rk, "call": ci, "n_calls": len(lines), "is_agent": r.get("is_agent_call"),
                "n_prompt": len(r.get("prompt_token_ids") or []), "n_comp": len(comp),
                "finish": r.get("finish_reason"), "think_start": s, "think_end": e, "n_think": n_think,
                "forced": forced, "unclosed": e is None and s > 0, **rep,
                "line_rep": round(line_repeat_fraction(text), 4), "wait": nw, "wait512": wmax,
                "wait_pos": wpos[:200], "tools": names, "resolved": bool(tr.get("resolved")),
                "mean_lp": round(sum(lps[s : s + n_think]) / n_think, 4) if n_think and lps else None,
            })
    err = str(tr.get("agent_error") or "")
    ro = {
        "src": src, "run_key": rk, "instance_id": tr.get("instance_id"), "resolved": bool(tr.get("resolved")),
        "infra": bool(tr.get("early_stopped")), "duration": tr.get("duration_seconds"),
        "timeout": "session timeout" in err, "no_submit": "without calling submit_patch" in err,
        "turns_cap": "turns budget" in err, "agent_error": err or None, "patch_chars": tr.get("patch_chars"),
        "submit_called": "submit_patch" in tool_names, "n_calls": len(calls), "log_found": path.exists(),
        "think_tokens": sum(c["n_think"] for c in calls), "n_forced": sum(c["forced"] for c in calls),
        "n_unclosed": sum(c["unclosed"] for c in calls), "max_ng": max([c["max_ng"] for c in calls], default=0),
        "wait": sum(c["wait"] for c in calls), "wait512": max([c["wait512"] for c in calls], default=0),
        "comp_tokens": sum(c["n_comp"] for c in calls), "request_log_path": str(path),
    }
    return ro, calls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.mine")
    p.add_argument("--src", action="append", required=True, help="NAME=DIR containing task_results.jsonl files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, dict[str, Any]]] = []
    for spec in a.src:
        name, d = spec.split("=", 1)
        for tf in sorted(Path(d).rglob("task_results.jsonl")):
            for line in tf.open():
                tr = json.loads(line)
                if tr.get("request_log_path") and tr.get("run_key"):
                    jobs.append((name, tr))
    # task_results can hold the same run_key twice (append after a worker restart): keep the last row.
    dedup = {(n, t["run_key"]): (n, t) for n, t in jobs}
    jobs = list(dedup.values())
    with Pool(a.workers) as pool, gzip.open(a.out / "calls.jsonl.gz", "wt") as fc, \
            (a.out / "rollouts.jsonl").open("w") as fr:
        for ro, calls in pool.imap_unordered(mine_rollout, jobs, chunksize=4):
            fr.write(json.dumps(ro) + "\n")
            for c in calls:
                fc.write(json.dumps(c) + "\n")
    print(f"mined {len(jobs)} rollouts -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
