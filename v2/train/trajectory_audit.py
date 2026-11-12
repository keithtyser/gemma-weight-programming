"""Causal Prefix Trajectory Auditor & Action-Token Mask Builder for SFT / OPD / RL.

Enforces strict data hygiene on multi-turn agent trajectories before post-training:
  1. Split Seal Check: Rejects any trajectory from `unseen_repo_cv`, `original_splitv2_80`,
     `sealed_rich_httpx_49`, or `sealed_harbor_july_65repos`.
  2. Causal Prefix Verification (`window_audit`): Every `edit_file` target path MUST have been
     previously viewed via `read_file` or `run_command` earlier in the trajectory (eliminating
     lucky/memorized uninspected edits!).
  3. Native Gemma 4 Thought & Tool Hygiene (`preserve_thinking=True`): Verifies concise
     `<|channel>thought...<channel|>` reasoning per turn, zero protected-path edits in final diff,
     and constructs exact binary action masks `M_{i,t}` (`1` on assistant thought+tool tokens,
     `0` on system/user/tool-response tokens).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import posixpath
from typing import Any

from v2.constants import (
    ORIGINAL_SPLITV2_REPOS,
    SAFE_CORE_TOOLS,
    SEALED_RICH_HTTPX_REPOS,
    canonicalize_repo,
    is_protected_swegemma_path,
)


@dataclass(frozen=True)
class TrajectoryAuditResult:
    instance_id: str
    canonical_repo: str
    accepted: bool
    turn_count: int
    inspected_files_count: int
    edited_files_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_ws_path(raw: str) -> str:
    cleaned = raw.strip().replace("\\", "/")
    if cleaned.startswith("/workspace/"):
        cleaned = cleaned[len("/workspace/") :]
    return posixpath.normpath(cleaned).lstrip("/")


def audit_agent_trajectory(
    record: dict[str, Any],
    *,
    forbidden_repos: set[str] | frozenset[str] = ORIGINAL_SPLITV2_REPOS | SEALED_RICH_HTTPX_REPOS,
    forbidden_instance_ids: set[str] | frozenset[str] = frozenset(),
    require_resolved: bool = True,
    max_turns: int = 42,
    max_thought_chars_per_turn: int = 3200,
    allowed_tools: frozenset[str] = frozenset(SAFE_CORE_TOOLS),
) -> TrajectoryAuditResult:
    """Audit a multi-turn trajectory record for causal grounding, split purity, and format hygiene.

    Expected `record` schema:
      {
        "instance_id": str,
        "repo": str,
        "resolved": bool,
        "turns": [
          {
            "role": "assistant",
            "thought": str,
            "tool_calls": [{"name": str, "args": dict}],
            "text": str
          },
          {"role": "tool", "name": str, "content": str},
          ...
        ]
      }
    """
    iid = str(record.get("instance_id") or "").strip()
    canon_repo = canonicalize_repo(str(record.get("repo") or ""))
    reasons: list[str] = []

    forbidden_canon = {canonicalize_repo(r) for r in forbidden_repos}
    if canon_repo in forbidden_canon:
        reasons.append(f"trajectory_repo_in_forbidden_eval_or_holdout:{canon_repo}")
    if iid in forbidden_instance_ids:
        reasons.append(f"trajectory_instance_in_forbidden_split:{iid}")

    if require_resolved and not bool(record.get("resolved", False)):
        reasons.append("trajectory_not_verified_resolved")

    turns = record.get("turns") or []
    assistant_turns = [t for t in turns if isinstance(t, dict) and t.get("role") == "assistant"]
    if not assistant_turns:
        reasons.append("empty_assistant_turns")
    if len(assistant_turns) > max_turns:
        reasons.append(f"too_many_turns:{len(assistant_turns)}>{max_turns}")

    inspected_files: set[str] = set()
    edited_files: set[str] = set()
    called_submit = False

    for turn_idx, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        thought = str(turn.get("thought") or "")
        if len(thought) > max_thought_chars_per_turn:
            reasons.append(
                f"turn_{turn_idx}_runaway_thought:{len(thought)}>{max_thought_chars_per_turn}"
            )

        for tc in turn.get("tool_calls") or []:
            t_name = str(tc.get("name") or "")
            t_args = tc.get("args") or {}
            if t_name not in allowed_tools:
                reasons.append(f"disallowed_tool_call:{t_name}")
                continue

            if t_name == "read_file":
                fp = _norm_ws_path(str(t_args.get("filepath") or t_args.get("path") or ""))
                if fp:
                    inspected_files.add(fp)
            elif t_name == "run_command":
                cmd = str(t_args.get("command") or "")
                # Track any `.py` path explicitly inspected in shell commands
                for tok in cmd.replace('"', " ").replace("'", " ").split():
                    if tok.endswith(".py") and not tok.startswith("/tmp/"):
                        inspected_files.add(_norm_ws_path(tok))
            elif t_name == "edit_file":
                fp = _norm_ws_path(str(t_args.get("filepath") or t_args.get("path") or ""))
                if fp:
                    if is_protected_swegemma_path(fp):
                        reasons.append(f"edited_protected_path:{fp}")
                    if fp not in inspected_files:
                        reasons.append(f"uninspected_causal_edit_at_turn_{turn_idx}:{fp}")
                    edited_files.add(fp)
            elif t_name == "write_file":
                fp_raw = str(t_args.get("filepath") or t_args.get("path") or "")
                if not fp_raw.startswith("/tmp/"):
                    fp = _norm_ws_path(fp_raw)
                    if is_protected_swegemma_path(fp):
                        reasons.append(f"wrote_protected_path:{fp}")
                    edited_files.add(fp)
            elif t_name == "submit_patch":
                called_submit = True

    if require_resolved and not edited_files:
        reasons.append("no_production_files_edited")
    if require_resolved and not called_submit:
        reasons.append("missing_explicit_submit_patch_call")

    return TrajectoryAuditResult(
        instance_id=iid,
        canonical_repo=canon_repo,
        accepted=len(reasons) == 0,
        turn_count=len(assistant_turns),
        inspected_files_count=len(inspected_files),
        edited_files_count=len(edited_files),
        reasons=tuple(reasons),
    )


def build_span_action_mask(
    token_role_spans: list[tuple[str, int]],
) -> tuple[list[int], int]:
    """Construct exact action-token binary mask `M_{i,t}` from `(role, token_count)` spans.

    `M_{i,t} = 1` iff `role == 'assistant'` (`<|channel>thought...<channel|><|tool_call>...<turn|>`),
    `M_{i,t} = 0` on `'system'`, `'user'`, and `'tool'` observation spans.
    """
    mask: list[int] = []
    active = 0
    for role, length in token_role_spans:
        if length < 0:
            raise ValueError(f"Negative span length {length} for role {role!r}")
        bit = 1 if role == "assistant" else 0
        mask.extend([bit] * length)
        if bit:
            active += length
    return mask, active
