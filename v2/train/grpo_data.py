"""Teacher-free RL data from token-level rollouts (FrogNano / Rufus-Air style).

  python -m v2.train.grpo_data --rollout-dirs R1 [R2 ...] --out train_data/<run>/grpo.pt [--splits-json ...]

Inputs are rollout dirs from `run_rebench_v2` whose request logs carry `prompt_token_ids`, `completion_token_ids` and
`completion_logprobs` (see `v2/eval/request_capture.py`). Per task:

- reward: 1.0 if resolved else 0.0 (optional log length penalty on successes, `--length-penalty-alpha`);
- infra aborts (sandbox/worker/grader errors that are not the agent's doing) are dropped from the group;
- groups whose kept rewards all match carry no signal and are dropped;
- advantage = (R - mean) / (std + eps) by default, or mean-only (`--advantage mean_only`).

Each rollout's calls become training rows: consecutive calls are merged while the next prompt starts with the previous
prompt + completion (exact token prefix), so one row covers many calls and no prefix is re-processed. Labels are only
the sampled completion tokens; `old_lp` holds their sampling logprobs. Tool outputs and prompts are masked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

AGENT_OUTCOME_MARKERS = ("session timeout", "submit_patch", "contextwindowexceeded")
INFRA_ERROR_MARKERS = ("worker crash", "hard timeout before", "sandbox", "docker", "grader error", "image")


def is_infra_abort(rec: dict[str, Any]) -> bool:
    """True when the rollout failed for reasons outside the agent (kept out of its group)."""
    agent_err = str(rec.get("agent_error") or "").lower()
    if agent_err:
        # the agent's own outcomes (session cap, no submit, context overflow) are failures graded like any other
        # rollout, as the competition scores them; anything else the runner raised is infra
        return not any(m in agent_err for m in AGENT_OUTCOME_MARKERS)
    err = str(rec.get("error") or "").lower()
    if not err or "junit" in err or "test failures" in err or "session timeout" in err or "submit_patch" in err:
        return False
    return any(m in err for m in INFRA_ERROR_MARKERS)


def group_advantages(rewards: list[float], mode: str = "std", eps: float = 1e-6) -> list[float] | None:
    """Group-relative advantages; None when the group has no reward variance."""
    if len(rewards) < 2 or max(rewards) - min(rewards) < 1e-9:
        return None
    mean = sum(rewards) / len(rewards)
    if mode == "mean_only":
        return [r - mean for r in rewards]
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + eps) for r in rewards]


def rollout_reward(rec: dict[str, Any], *, alpha: float, n_ref: int) -> float:
    if not rec.get("resolved"):
        return 0.0
    n = int(rec.get("tool_calls") or 0)
    if alpha > 0.0 and n > n_ref > 0:
        return max(0.0, 1.0 - alpha * math.log(n / n_ref))
    return 1.0


def rows_from_calls(calls: list[dict[str, Any]], *, max_seq_len: int, merge: str = "align",
                    min_align: float = 0.9, with_versions: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge calls into rows of (input_ids, loss mask, old logprobs).

    merge="exact": extend a row only when the next prompt starts with prompt + completion token-for-token.
    merge="align" (default): the Gemma 4 template re-renders past assistant turns slightly differently from what was
    sampled (e.g. a "\\n" before `<channel|>`), so exact merging almost never applies. Instead extend when the next
    prompt starts with the previous *prompt*, locate the previous completion inside the re-rendered region with a
    token alignment, and label only the sampled tokens that survive (with their sampling logprobs). The last call of
    a row is exact; earlier calls differ from their sampling context by the template's inserted tokens.
    with_versions=True (async RL) adds "versions": per token, the call's `policy_version` (None on unlabeled tokens).
    """
    if merge == "align":
        return _rows_from_calls_align(calls, max_seq_len=max_seq_len, min_align=min_align, with_versions=with_versions)
    rows: list[dict[str, Any]] = []
    stats = {"calls": 0, "calls_missing_tokens": 0, "rows": 0, "prefix_breaks": 0, "too_long": 0}
    cur_ids: list[int] = []
    cur_mask: list[int] = []
    cur_lp: list[float] = []
    cur_ver: list[Any] = []

    def flush() -> None:
        if cur_ids and any(cur_mask):
            rows.append({"input_ids": list(cur_ids), "loss_mask": list(cur_mask), "old_lp": list(cur_lp)})
            if with_versions:
                rows[-1]["versions"] = list(cur_ver)

    for c in calls:
        stats["calls"] += 1
        p, g, lp = c.get("prompt_token_ids"), c.get("completion_token_ids"), c.get("completion_logprobs")
        if not p or not g or not lp or len(g) != len(lp):
            stats["calls_missing_tokens"] += 1
            flush()
            cur_ids, cur_mask, cur_lp, cur_ver = [], [], [], []
            continue
        if cur_ids and len(p) >= len(cur_ids) and p[: len(cur_ids)] == cur_ids:
            ext = p[len(cur_ids):]
        else:
            if cur_ids:
                stats["prefix_breaks"] += 1
                flush()
            cur_ids, cur_mask, cur_lp, cur_ver = [], [], [], []
            ext = p
        if len(cur_ids) + len(ext) + len(g) > max_seq_len:
            stats["too_long"] += 1
            flush()
            cur_ids, cur_mask, cur_lp, cur_ver = [], [], [], []
            if len(p) + len(g) > max_seq_len:
                continue
            ext = p
        cur_ids += ext + g
        cur_mask += [0] * len(ext) + [1] * len(g)
        cur_lp += [0.0] * len(ext) + [float(x) for x in lp]
        cur_ver += [None] * len(ext) + [c.get("policy_version")] * len(g)
    flush()
    stats["rows"] = len(rows)
    return rows, stats


def _rows_from_calls_align(calls: list[dict[str, Any]], *, max_seq_len: int,
                           min_align: float, with_versions: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import difflib

    rows: list[dict[str, Any]] = []
    stats = {"calls": 0, "calls_missing_tokens": 0, "rows": 0, "prefix_breaks": 0, "too_long": 0,
             "aligned_tokens": 0, "dropped_unaligned_tokens": 0}
    # Current row: the latest call's prompt (re-rendered history) with labels carried over from earlier calls
    ctx: list[int] = []
    mask: list[int] = []
    olp: list[float] = []
    ver: list[Any] = []
    pending: tuple[list[int], list[float]] | None = None  # latest call's own completion, not yet placed
    pending_ver: Any = None

    def flush() -> None:
        if pending is None:
            return
        g, lp = pending
        ids, m, o = ctx + g, mask + [1] * len(g), olp + list(lp)
        if len(ids) <= max_seq_len and any(m):
            rows.append({"input_ids": ids, "loss_mask": m, "old_lp": o})
            if with_versions:
                rows[-1]["versions"] = ver + [pending_ver] * len(g)

    for c in calls:
        stats["calls"] += 1
        p, g, lp = c.get("prompt_token_ids"), c.get("completion_token_ids"), c.get("completion_logprobs")
        if not p or not g or not lp or len(g) != len(lp):
            stats["calls_missing_tokens"] += 1
            flush()
            ctx, mask, olp, pending = [], [], [], None
            ver, pending_ver = [], None
            continue
        p = list(p)
        g = [int(x) for x in g]
        lp = [float(x) for x in lp]
        extended = False
        if pending is not None and len(p) >= len(ctx) and p[: len(ctx)] == ctx and len(p) + len(g) <= max_seq_len:
            region = p[len(ctx):]
            pg, plp = pending
            # The re-rendered completion sits at the start of the region; tool output follows and must never be
            # labeled, so align only inside a window just longer than the completion
            sm = difflib.SequenceMatcher(None, pg, region[: len(pg) + 32], autojunk=False)
            new_mask = [0] * len(region)
            new_lp = [0.0] * len(region)
            new_ver: list[Any] = [None] * len(region)
            matched = 0
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    new_mask[b + k] = 1
                    new_lp[b + k] = plp[a + k]
                    new_ver[b + k] = pending_ver
                matched += n
            if matched >= min_align * len(pg):
                stats["aligned_tokens"] += matched
                stats["dropped_unaligned_tokens"] += len(pg) - matched
                ctx, mask, olp = p, mask + new_mask, olp + new_lp
                ver = ver + new_ver
                extended = True
        if not extended:
            if pending is not None:
                if len(p) + len(g) > max_seq_len:
                    stats["too_long"] += 1
                else:
                    stats["prefix_breaks"] += 1
                flush()
            ctx, mask, olp = p, [0] * len(p), [0.0] * len(p)
            ver = [None] * len(p)
        pending = (g, lp)
        pending_ver = c.get("policy_version")
    flush()
    stats["rows"] = len(rows)
    return rows, stats


def _load_excluded(paths: list[Path] | None) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    repos: set[str] = set()
    for p in paths or []:
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        for key in ("unseen_cv_instance_ids", "eval_instance_ids", "firewall_instance_ids"):
            ids |= set(m.get(key) or [])
        for key in ("unseen_cv_repos", "firewall_repos"):
            repos |= set(m.get(key) or [])
        for part in ("comp", "rebench"):
            if isinstance(m.get(part), dict):
                ids |= set(m[part].get("instance_ids") or [])
                r = m[part].get("repos")
                repos |= set(r) if isinstance(r, list) else set()
    return ids, repos


def build(rollout_dirs: list[Path], *, max_seq_len: int, advantage: str, alpha: float, n_ref: int,
          excluded_ids: set[str], excluded_repos: set[str], merge: str = "align") -> dict[str, Any]:
    recs: list[dict[str, Any]] = []
    for d in rollout_dirs:
        for f in sorted(Path(d).glob("shard*/task_results.jsonl")) + sorted(Path(d).glob("*/task_results.jsonl")):
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    recs.append(json.loads(ln))
    seen: set[str] = set()
    uniq = []
    for r in recs:
        key = str(r.get("run_key") or f"{r.get('instance_id')}#{r.get('rollout_idx')}")
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    groups: dict[str, list[dict[str, Any]]] = {}
    summary = {"rollouts": len(uniq), "infra_aborts": 0, "excluded_golden": 0, "groups": 0,
               "groups_zero_variance": 0, "groups_kept": 0}
    for r in uniq:
        if r.get("instance_id") in excluded_ids or r.get("repo") in excluded_repos:
            summary["excluded_golden"] += 1
            continue
        if is_infra_abort(r):
            summary["infra_aborts"] += 1
            continue
        groups.setdefault(str(r["instance_id"]), []).append(r)
    summary["groups"] = len(groups)
    out_rows: list[dict[str, Any]] = []
    tok_stats: dict[str, int] = {}
    for iid, members in sorted(groups.items()):
        rewards = [rollout_reward(m, alpha=alpha, n_ref=n_ref) for m in members]
        advs = group_advantages(rewards, mode=advantage)
        if advs is None:
            summary["groups_zero_variance"] += 1
            continue
        summary["groups_kept"] += 1
        for m, rew, adv in zip(members, rewards, advs):
            log = m.get("request_log_path")
            if not log or not Path(log).is_file():
                continue
            calls = [json.loads(ln) for ln in Path(log).read_text(encoding="utf-8").splitlines() if ln.strip()]
            calls = [c for c in calls if c.get("is_agent_call", True)]
            rows, st = rows_from_calls(calls, max_seq_len=max_seq_len, merge=merge)
            for k, v in st.items():
                tok_stats[k] = tok_stats.get(k, 0) + v
            for row in rows:
                row.update({"instance_id": iid, "repo": m.get("repo"), "run_key": m.get("run_key"),
                            "reward": rew, "advantage": adv, "resolved": bool(m.get("resolved"))})
                out_rows.append(row)
    summary.update(tok_stats)
    summary["train_rows"] = len(out_rows)
    summary["trained_tokens"] = sum(sum(r["loss_mask"]) for r in out_rows)
    summary["processed_tokens"] = sum(len(r["input_ids"]) for r in out_rows)
    return {"rows": out_rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.train.grpo_data")
    p.add_argument("--rollout-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--advantage", choices=["std", "mean_only"], default="std")
    p.add_argument("--merge", choices=["align", "exact"], default="align")
    p.add_argument("--length-penalty-alpha", type=float, default=0.0, help="0 = off; FrogNano uses a log penalty")
    p.add_argument("--length-penalty-ref-calls", type=int, default=0)
    p.add_argument("--splits-json", type=Path, nargs="*", default=None, help="Manifests whose tasks/repos are excluded")
    args = p.parse_args(argv)

    import torch

    ex_ids, ex_repos = _load_excluded(args.splits_json)
    res = build(args.rollout_dirs, max_seq_len=args.max_seq_len, advantage=args.advantage,
                alpha=args.length_penalty_alpha, n_ref=args.length_penalty_ref_calls,
                excluded_ids=ex_ids, excluded_repos=ex_repos, merge=args.merge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res["rows"], args.out)
    args.out.with_suffix(".summary.json").write_text(json.dumps(res["summary"], indent=2), encoding="utf-8")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
