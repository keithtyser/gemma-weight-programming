"""Engine logits processors that change PROCESSED logprobs, replayed exactly in the trainer's scorer.

Author: Claude Opus 5.5 (TII trainer track)
Date: 05-November-2026
PURPOSE: vLLM 0.19.1 Sampler.forward: logits.float() -> apply_logits_processors -> sample (temperature, then
  processed logprobs = log_softmax). The builtin ThinkingTokenBudgetLogitsProcessor (enabled by the server's
  --reasoning-config and a request's thinking_token_budget) sets logits[req, end_token] = 1e9 while a thinking
  section exceeds its budget, so the sampled end token's processed logprob is exactly 0.0 (fp32: 1e9/T - lse with
  lse == 1e9/T) and every other token has probability exactly 0. The model's own logprob of that token can be -20 nats:
  the smoke_round1 lag-0 mismatches (30 tokens, all '<channel|>' at completion index 2561, engine logprob 0.0).
  thinking_budget_forced() replays the processor on token ids and returns, per completion position, the token the
  engine forced (or -1). The scorer then applies the same 1e9 logit before temperature and log_softmax (ops.py), so
  the forced logprob is the engine's bytes and its gradient is exactly zero.

  Replay rule (per-token form of the processor's state machine):
    state from the prompt as _init_state_entry: in_think = last start sequence after last end sequence;
    before sampling completion token i: if in_end -> token i is forced to end_ids[end_count];
    after token i: in end mode end_count += 1 (leave end mode after the whole end sequence); otherwise re-derive
    in_think / think_count from the last start / end sequence and enter end mode when in_think and
    think_count >= budget.
  The processor re-parses only every check_count_down engine steps, but its count-down is set to remaining-1 at each
  check, so the end-mode transition lands on the same token as the per-token rule once the first check has happened;
  the first check happens after `budget` engine steps, before think_count can reach the budget, unless the PROMPT
  already ends inside a thinking section (then the count-down depends on the engine's prefill step count, which the
  trainer cannot see): that case raises.
SRP/DRY check: Pass - the only replay of engine logits processors in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThinkingBudget:
    budget: int  # request thinking_token_budget
    start_ids: tuple[int, ...]  # tokenized reasoning_start_str
    end_ids: tuple[int, ...]  # tokenized reasoning_end_str


def _last(seq: Sequence[int], pat: Sequence[int]) -> int:
    n, m = len(seq), len(pat)
    if m == 0:
        return -1
    for i in range(n - m, -1, -1):
        if list(seq[i : i + m]) == list(pat):
            return i
    return -1


def thinking_budget_replayable(prompt: Sequence[int], tb: ThinkingBudget | None) -> bool:
    """False when the processor's forced positions for a completion of `prompt` cannot be derived from token ids: the
    prompt ends inside a thinking section (an earlier turn opened <|channel> and never closed it, and the template
    re-rendered it). Then the processor starts in thinking mode with its first check after `budget` ENGINE STEPS
    (prefill chunks included), so whether / where it forces depends on the engine's prefill schedule (chunked prefill,
    prefix-cache hits), which the trainer does not see. Such calls are excluded from loss, IS weights and the bitwise
    metric by the learner (async_rl Learner.drop_unreplayable)."""
    if tb is None:
        return True
    return not (_last(prompt, list(tb.start_ids)) > _last(prompt, list(tb.end_ids)))


def thinking_budget_forced(prompt: Sequence[int], output: Sequence[int], tb: ThinkingBudget) -> list[int]:
    """Per completion position: the token id the ThinkingTokenBudgetLogitsProcessor forced, or -1."""
    s, e, B = list(tb.start_ids), list(tb.end_ids), int(tb.budget)
    ls, le = _last(prompt, s), _last(prompt, e)
    in_think = ls > le
    if in_think:
        raise ValueError("prompt ends inside a thinking section: the processor's first check depends on the engine's "
                         "prefill step count; not replayable from token ids")
    think_count, in_end, end_count = 0, False, 0
    forced = [-1] * len(output)
    out: list[int] = []
    for i, tok in enumerate(output):
        if in_end:
            forced[i] = e[end_count]
        out.append(int(tok))
        if in_end:
            end_count += 1
            if end_count >= len(e):
                in_end, end_count = False, 0
            continue
        w = max(len(s), len(e))
        lo = max(0, len(out) - w)
        rs, re_ = _last(out[lo:], s), _last(out[lo:], e)
        if rs >= 0 and (re_ < 0 or rs > re_):
            in_think = True
            think_count = len(out) - (lo + rs + len(s))
        elif re_ >= 0:
            in_think, think_count = False, 0
        elif in_think:
            think_count += 1
        if in_think and think_count >= B:
            in_think, in_end, end_count = False, True, 0
    return forced


def forced_targets(calls: Sequence[tuple[Sequence[int], Sequence[int]]], tb: ThinkingBudget | None) -> list[int]:
    """Concatenated per-sampled-token forced ids (-1 = not forced) of calls [(prompt, completion)]. Raises if a
    replayed forced token differs from the sampled token (the data would not come from this processor config)."""
    out: list[int] = []
    for p, g in calls:
        if tb is None:
            out += [-1] * len(g)
            continue
        f = thinking_budget_forced(p, g, tb)
        for i, (fi, gi) in enumerate(zip(f, g)):
            if fi >= 0 and fi != gi:
                raise ValueError(f"replayed thinking-budget force {fi} at completion index {i} != sampled token {gi}")
        out += f
    return out


REASONING_START_STR, REASONING_END_STR = "<|channel>", "<channel|>"  # aligned_engine.aligned_server_cmd --reasoning-config


def thinking_budget_from_submission(submission: str | None, model: str, budget: int | None = None) -> ThinkingBudget | None:
    """The rollout server's thinking-budget processor config: budget = the harness's
    generate_content_config.thinking_config.thinking_budget (every agent of the submission must agree) unless given;
    start/end ids = tokenizer.encode(reasoning str, add_special_tokens=False) as vLLM ReasoningConfig does.
    None when no budget applies."""
    import glob
    import os

    if budget is None or budget < 0:
        vals = set()
        for f in glob.glob(os.path.join(submission or "", "**", "*.yaml"), recursive=True):
            import yaml

            class _L(yaml.SafeLoader):
                pass

            _L.add_constructor("!include", lambda loader, node: None)
            try:
                doc = yaml.load(open(f, encoding="utf-8"), Loader=_L)
            except Exception:
                continue
            if isinstance(doc, dict):
                tb = ((doc.get("generate_content_config") or {}).get("thinking_config") or {}).get("thinking_budget")
                if tb is not None:
                    vals.add(int(tb))
        if not vals:
            return None
        if len(vals) > 1:
            raise ValueError(f"submission agents disagree on thinking_budget: {sorted(vals)}")
        budget = vals.pop()
    if not budget or budget <= 0:
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return ThinkingBudget(int(budget), tuple(tok.encode(REASONING_START_STR, add_special_tokens=False)),
                          tuple(tok.encode(REASONING_END_STR, add_special_tokens=False)))


# ------------------------------------------------------------------------------------------------------------------
# Exact resolution with the engine's async-scheduling blind spots (smoke_round2 step-2 verify mismatch, 06-Nov-2026)
# ------------------------------------------------------------------------------------------------------------------
# vLLM 0.19.1 runs logits-processor update_state in refresh_metadata at the START of a step (gpu_model_runner
# _update_states -> input_batch.refresh_metadata), but with async scheduling the previous step's sampled token is
# still a -1 placeholder in output_tok_ids then (it is filled in _sample by update_async_output_token_ids). The
# processor only re-parses at count-down checks, from prev_output_length on, so the token at the LAST index of every
# check window is never examined. The first check happens after `budget` update calls counted from the request's
# admission, i.e. it depends on the number of prefill steps P (chunked prefill, prefix-cache hits, batch load), which
# the request log does not record. Example (caikit__caikit-233__p0__r7, call 2): thinking ended naturally with
# <channel|> at completion index 2557 = the placeholder slot of the first check (P = 3); the processor kept counting
# and forced a second <channel|> at 2561 (engine logprob 0.0). The per-token rule cannot see this.
#
# What IS determined by token ids: the processor can force only at k = s + len(start) + budget for a start-sequence
# occurrence s in the completion (its think_count is always current_len - (s + len(start)) for some start it saw,
# and the count-down lands every transition exactly on think_count == budget; see the module docstring), and a forced
# token's processed logprob is exactly +0.0 with token == the end token. At a candidate k the two hypotheses predict
# different engine bytes unless the model's own logprob is +0.0: forced -> +0.0; not forced -> the model's logprob
# (bitwise, the TII invariant). So: forced(k) <=> k is a candidate AND engine logprob(k) is +0.0 AND token(k) == end.
# Where the model's own logprob is also +0.0 both hypotheses give the same logprob bytes; treating the token as forced
# gives it zero gradient, i.e. it is excluded from the gradient exactly like a masked token. Non-candidate positions
# are never forced. Requires single-token start/end sequences and a prompt that does not end inside a thinking section
# (thinking_budget_replayable); other calls are masked by the learner.

def _f32_is_pos_zero(x: float) -> bool:
    import struct

    return struct.unpack("<I", struct.pack("<f", float(x)))[0] == 0


def thinking_budget_candidates(output: Sequence[int], tb: ThinkingBudget) -> list[int]:
    """Completion indices where the processor can force (s + len(start) + budget for every start occurrence s)."""
    s = list(tb.start_ids)
    n = len(s)
    return sorted({i + n + int(tb.budget) for i in range(len(output) - n + 1) if list(output[i : i + n]) == s
                   and i + n + int(tb.budget) < len(output)})


def resolve_forced(prompt: Sequence[int], output: Sequence[int], engine_lp: Sequence[float], tb: ThinkingBudget | None) -> list[int]:
    """Per completion index: the forced token id or -1, resolved exactly from candidates + the engine's logprob bytes
    (see above). Raises for calls that are not replayable (the learner masks those first)."""
    if tb is None:
        return [-1] * len(output)
    if len(tb.start_ids) != 1 or len(tb.end_ids) != 1:
        raise ValueError("exact thinking-budget resolution implemented for single-token start/end sequences")
    if not thinking_budget_replayable(prompt, tb):
        raise ValueError("prompt ends inside a thinking section: not replayable")
    out = [-1] * len(output)
    end = int(tb.end_ids[0])
    for k in thinking_budget_candidates(output, tb):
        if int(output[k]) == end and _f32_is_pos_zero(engine_lp[k]):
            out[k] = end
    return out
