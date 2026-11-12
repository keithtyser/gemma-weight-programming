"""Zero-Compromise 6-Gate Task Quality Filter ('Garbage In, Garbage Out').

Every task admitted into `train_pool` or `unseen_repo_cv` MUST pass all 6 gates:
  1. clean_separation    — 5-way canonical repo disjointness + permissive license + family dedup
  2. clear_requirements  — substantive spec + no image-only stubs + AST hidden-contract audit
  3. verified_fixes      — non-empty source-only gold patch touching ZERO swegemma protected paths
  4. reproducible_bugs   — deterministic offline base failure (repeat=1,2) + editable /workspace check
  5. strong_tests        — real assertions + F2P & P2P regression coverage + fast non-flaky runtime
  6. trustworthy_grading — deterministic gold pass (repeat=1,2) + exact swegemma 0.2.11 JUnit parity
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any

from v2.constants import (
    ORIGINAL_SPLITV2_REPOS,
    PERMISSIVE_SPDX_LICENSES,
    SEALED_RICH_HTTPX_REPOS,
    canonicalize_repo,
    is_protected_swegemma_path,
)

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_URL_RE = re.compile(r"https?://\S+")
_IMAGE_OR_VIDEO_RE = re.compile(
    r"(?:!\[[^\]]*\]\([^)]+\)|user-attachments/assets/|\.(?:png|jpg|jpeg|gif|webp|mp4|mov)\b|imgur\.com|loom\.com)",
    re.IGNORECASE,
)
_CODE_ANCHOR_RE = re.compile(
    r"(?:```|`[A-Za-z_][A-Za-z0-9_.:/-]{1,}`|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b|"
    r"\b(?:Traceback|TypeError|ValueError|KeyError|AttributeError|AssertionError|RuntimeError|ImportError|SyntaxError)\b|"
    r"--[a-zA-Z0-9][a-zA-Z0-9_-]+|\b[\w/-]+\.py\b)"
)
_ASSERTION_LINE_RE = re.compile(
    r"(?:\bassert\b|\bassert_[a-zA-Z0-9_]+\s*\(|\.[a-zA-Z0-9_]*assert[a-zA-Z0-9_]*\s*\("
    r"|\b(?:pytest\.)?(?:raises|warns|deprecated_call)\s*\("
    r"|\bself\.validate_(?:identity|all)\s*\()"
)
_XFAIL_NONSTRICT_RE = re.compile(r"@pytest\.mark\.xfail\b(?![^\n]*strict\s*=\s*True)")
_CLI_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_])(--[a-zA-Z][a-zA-Z0-9_-]{2,})\b")
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
_ISSUE_REF_RE = re.compile(r"(?:(?:#|gh-|GH-)(\d{1,7})|/(?:issues|pull)/(\d{1,7})\b)")

# Standard Python / pytest / unittest vocabulary that is NOT a domain-specific hidden contract
_GENERIC_TEST_VOCAB: frozenset[str] = frozenset(
    {
        "True",
        "False",
        "None",
        "self",
        "cls",
        "args",
        "kwargs",
        "pytest",
        "unittest",
        "mock",
        "patch",
        "MagicMock",
        "AsyncMock",
        "monkeypatch",
        "tmp_path",
        "tmpdir",
        "capsys",
        "caplog",
        "capfd",
        "recwarn",
        "request",
        "fixture",
        "mark",
        "parametrize",
        "raises",
        "warns",
        "approx",
        "skip",
        "xfail",
        "result",
        "res",
        "out",
        "err",
        "output",
        "expected",
        "actual",
        "value",
        "val",
        "data",
        "items",
        "obj",
        "ret",
        "exc",
        "excinfo",
        "exc_info",
        "error",
        "msg",
        "message",
        "response",
        "resp",
        "client",
        "app",
        "cfg",
        "config",
        "path",
        "file",
        "filename",
        "text",
        "content",
        "lines",
        "row",
        "rows",
        "key",
        "keys",
        "name",
        "names",
        "idx",
        "index",
        "count",
        "total",
        "status",
        "code",
        "foo",
        "bar",
        "baz",
        "dummy",
        "sample",
        "example",
        "helper",
        "setup",
        "teardown",
        "main",
    }
)


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskQualityAudit:
    instance_id: str
    canonical_repo: str
    admitted: bool
    grading_tier: str  # "gold_raw_compatible" | "certified_junit_identity" | "rejected"
    gates: dict[str, GateVerdict]
    rejection_summary: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "canonical_repo": self.canonical_repo,
            "admitted": self.admitted,
            "grading_tier": self.grading_tier,
            "rejection_summary": list(self.rejection_summary),
            "gates": {k: asdict(v) for k, v in self.gates.items()},
        }


def parse_unified_diff(patch_text: str) -> dict[str, Any]:
    """Extract touched files, added/removed lines, context lines, binary & symlink flags."""
    files: list[str] = []
    seen_files: set[str] = set()
    added_lines: list[str] = []
    removed_lines: list[str] = []
    context_lines: list[str] = []
    has_binary = False
    has_symlink_mode = False

    for line in patch_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            for candidate in (m.group(1), m.group(2)):
                clean = candidate.strip().strip('"')
                if clean and clean != "/dev/null" and clean not in seen_files:
                    seen_files.add(clean)
                    files.append(clean)
            continue
        if line.startswith(("GIT binary patch", "Binary files ")):
            has_binary = True
            continue
        if line.startswith(("new file mode 120000", "old mode 120000", "new mode 120000")):
            has_symlink_mode = True
            continue
        if line.startswith(("+++ ", "--- ", "@@ ", "index ")):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
        elif line.startswith("-"):
            removed_lines.append(line[1:])
        elif line.startswith(" "):
            context_lines.append(line[1:])

    return {
        "files": files,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "context_lines": context_lines,
        "changed_line_count": len(added_lines) + len(removed_lines),
        "has_binary": has_binary,
        "has_symlink_mode": has_symlink_mode,
    }


def detect_hidden_test_contracts(
    problem_statement: str,
    gold_patch: str,
    test_patch: str,
    extra_known_symbols: set[str] | None = None,
) -> dict[str, list[str]]:
    """Detect undocumented APIs or CLI flags newly invented in gold_patch & asserted in test_patch.

    In SWE-bench audits, the #1 cause of unsolvable/noisy tasks is an issue that says
    'support X' without naming the exact new function/kwarg/CLI flag, while `test_patch`
    imports `new_func_name` or passes `--exact-new-flag`.

    Logic:
      1. Identify public symbols (`def`, `class`, public attributes/kwargs) and `--cli-flags`
         NEWLY ADDED in `gold_patch` (present in gold `+` lines, absent from gold `-`/context lines
         and `extra_known_symbols`).
      2. Intersect with symbols/flags directly referenced or asserted in `test_patch` `+` lines.
      3. Check whether each such required new contract token appears in `problem_statement`
         (case-insensitive or snake/kebab-normalized). Any missing token is a hidden contract!
    """
    gold_info = parse_unified_diff(gold_patch)
    test_info = parse_unified_diff(test_patch)

    pre_existing_text = "\n".join(gold_info["removed_lines"] + gold_info["context_lines"] + test_info["context_lines"])
    pre_existing_tokens = set(_IDENT_RE.findall(pre_existing_text))
    if extra_known_symbols:
        pre_existing_tokens |= extra_known_symbols
    pre_existing_flags = {f.lower() for f in _CLI_FLAG_RE.findall(pre_existing_text)}

    gold_added_text = "\n".join(gold_info["added_lines"])
    test_added_text = "\n".join(test_info["added_lines"])

    # Extract newly defined public API names, private helper locks, & kwarg names from gold_patch added lines
    gold_new_api_candidates: set[str] = set()
    gold_new_private_defs: set[str] = set()
    for line in gold_info["added_lines"]:
        stripped = line.strip()
        m_def = re.match(r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)", stripped)
        if m_def:
            fn_name = m_def.group(1)
            if not fn_name.startswith("_"):
                gold_new_api_candidates.add(fn_name)
            elif not fn_name.startswith("__"):
                gold_new_private_defs.add(fn_name)
        m_cls = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if m_cls:
            cls_name = m_cls.group(1)
            if not cls_name.startswith("_"):
                gold_new_api_candidates.add(cls_name)
            elif not cls_name.startswith("__"):
                gold_new_private_defs.add(cls_name)

    # Also parse test_patch added lines via AST where possible (line/block fallback if partial hunk)
    test_referenced_idents = set(_IDENT_RE.findall(test_added_text))
    try:
        tree = ast.parse(test_added_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg and not node.arg.startswith("_"):
                test_referenced_idents.add(node.arg)
                if node.arg in _IDENT_RE.findall(gold_added_text):
                    gold_new_api_candidates.add(node.arg)
    except SyntaxError:
        for kw_match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{2,})\s*=", test_added_text):
            kw = kw_match.group(1)
            if kw in _IDENT_RE.findall(gold_added_text):
                gold_new_api_candidates.add(kw)

    newly_invented_tested_symbols = (
        (gold_new_api_candidates & test_referenced_idents)
        - pre_existing_tokens
        - _GENERIC_TEST_VOCAB
    )
    # Exclude test_* helper names
    newly_invented_tested_symbols = {
        s for s in newly_invented_tested_symbols if not s.lower().startswith("test_")
    }

    # Private implementation helpers newly invented in gold_patch must NEVER be imported/mocked by test_patch
    private_implementation_locks = sorted(
        (gold_new_private_defs & test_referenced_idents) - pre_existing_tokens
    )

    gold_added_flags = {f.lower() for f in _CLI_FLAG_RE.findall(gold_added_text)}
    test_added_flags = {f.lower() for f in _CLI_FLAG_RE.findall(test_added_text)}
    newly_invented_tested_flags = (gold_added_flags & test_added_flags) - pre_existing_flags

    # Detect exact newly-invented exception message strings asserted via pytest.raises(..., match="...")
    gold_raise_msgs: set[str] = set()
    for m_raise in re.finditer(r"""\braise\s+[A-Za-z_][A-Za-z0-9_.]*\(\s*f?['"]([^'"\\]{10,})['"]""", gold_added_text):
        msg_lit = m_raise.group(1).strip()
        if msg_lit and msg_lit.lower() not in pre_existing_text.lower():
            gold_raise_msgs.add(msg_lit)

    test_match_msgs: list[str] = []
    for m_match in re.finditer(r"""\bmatch\s*=\s*r?['"]([^'"]{8,})['"]""", test_added_text):
        raw_pat = re.sub(r"\\[bBsSwWdD]|[\^$*+?()\[\]{}|]", " ", m_match.group(1))
        cleaned_phrase = " ".join(raw_pat.split()).strip()
        if len(cleaned_phrase) >= 8:
            test_match_msgs.append(cleaned_phrase)

    issue_lower = problem_statement.lower()
    issue_norm = issue_lower.replace("-", "_")

    hidden_symbols: list[str] = []
    for sym in sorted(newly_invented_tested_symbols):
        sym_low = sym.lower()
        if sym_low not in issue_lower and sym_low.replace("-", "_") not in issue_norm:
            hidden_symbols.append(sym)

    hidden_flags: list[str] = []
    for flag in sorted(newly_invented_tested_flags):
        bare = flag.lstrip("-").replace("-", "_")
        if flag not in issue_lower and bare not in issue_norm:
            hidden_flags.append(flag)

    hidden_exc_messages: list[str] = []
    for phrase in test_match_msgs:
        p_low = phrase.lower()
        if any(p_low in gm.lower() for gm in gold_raise_msgs) and p_low not in issue_lower:
            hidden_exc_messages.append(phrase)

    return {
        "hidden_symbols": hidden_symbols,
        "hidden_cli_flags": hidden_flags,
        "hidden_exception_messages": hidden_exc_messages,
        "private_implementation_locks": private_implementation_locks,
    }


def detect_exposed_solution_in_prompt(
    problem_statement: str,
    hints_text: str,
    gold_patch: str,
) -> list[str]:
    """Detect solution leakage in problem_statement or hints_text (Gate 1 Clean Separation)."""
    combined = f"{problem_statement}\n{hints_text or ''}"
    reasons: list[str] = []
    if "diff --git a/" in combined:
        reasons.append("exposed_git_diff_header_in_prompt")

    gold_info = parse_unified_diff(gold_patch)
    pre_existing_set = {ln.strip() for ln in (gold_info["removed_lines"] + gold_info["context_lines"]) if ln.strip()}
    novel_code_lines = [
        ln.strip()
        for ln in gold_info["added_lines"]
        if len(ln.strip()) >= 18
        and not ln.strip().startswith(("#", "import ", "from ", '"""', "'''"))
        and ln.strip() not in pre_existing_set
    ]
    if len(novel_code_lines) >= 3:
        norm_prompt = "\n".join(line.strip() for line in combined.splitlines())
        for idx in range(len(novel_code_lines) - 2):
            window = "\n".join(novel_code_lines[idx : idx + 3])
            if window in norm_prompt:
                reasons.append("exposed_verbatim_3line_gold_solution_in_prompt")
                break
    return reasons


def audit_task_quality(
    row: dict[str, Any],
    *,
    sealed_repos: set[str] | frozenset[str] = frozenset(),
    reserved_eval_repos: set[str] | frozenset[str] = ORIGINAL_SPLITV2_REPOS | SEALED_RICH_HTTPX_REPOS,
    control_report: dict[str, Any] | None = None,
    max_test_duration_sec: float = 120.0,
    min_problem_chars: int = 120,
    require_raw_swegemma_junit: bool = False,
    min_mutation_kill_score: float = 0.75,
    require_container_mutation_and_alt: bool = False,
    repo_rename_map: dict[str, str] | None = None,
) -> TaskQualityAudit:
    """Evaluate a task candidate against all 6 non-negotiable quality gates."""
    iid = str(row.get("instance_id") or "").strip()
    raw_repo = str(row.get("repo") or "").strip()
    canon_repo = canonicalize_repo(raw_repo)
    if repo_rename_map and canon_repo in repo_rename_map:
        canon_repo = canonicalize_repo(repo_rename_map[canon_repo])
    problem = str(row.get("problem_statement") or "")
    hints_text = str(row.get("hints_text") or "")
    gold_patch = str(row.get("patch") or "")
    test_patch = str(row.get("test_patch") or "")
    f2p = list(row.get("FAIL_TO_PASS") or [])
    p2p = list(row.get("PASS_TO_PASS") or [])

    def _canon_with_map(r: str) -> str:
        c = canonicalize_repo(r)
        if repo_rename_map and c in repo_rename_map:
            c = canonicalize_repo(repo_rename_map[c])
        return c

    sealed_canon = {_canon_with_map(r) for r in sealed_repos}
    reserved_canon = {_canon_with_map(r) for r in reserved_eval_repos}

    # ------------------------------------------------------------------
    # Gate 1: Clean Separation & Permissive License
    # ------------------------------------------------------------------
    g1_reasons: list[str] = []
    if not iid or not canon_repo or "/" not in canon_repo:
        g1_reasons.append("missing_instance_id_or_invalid_repo")
    if canon_repo in sealed_canon:
        g1_reasons.append(f"repo_in_sealed_holdout:{canon_repo}")
    if canon_repo in reserved_canon:
        g1_reasons.append(f"repo_in_reserved_eval_split:{canon_repo}")
    if "_interface" in iid:
        g1_reasons.append("synthetic_interface_variant_excluded")
    if bool(row.get("is_fork", False)):
        g1_reasons.append("fork_repository_excluded")

    license_spdx = row.get("license_spdx") or row.get("current_repository_license_spdx")
    if license_spdx is not None and str(license_spdx).strip() not in PERMISSIVE_SPDX_LICENSES:
        g1_reasons.append(f"non_permissive_or_unverified_license:{license_spdx}")

    exposed_reasons = detect_exposed_solution_in_prompt(problem, hints_text, gold_patch)
    g1_reasons.extend(exposed_reasons)

    gate1 = GateVerdict(
        passed=len(g1_reasons) == 0,
        reasons=tuple(g1_reasons),
        metrics={"canonical_repo": canon_repo, "license_spdx": license_spdx},
    )

    # ------------------------------------------------------------------
    # Gate 2: Clear Requirements & Zero Hidden Test Contracts
    # ------------------------------------------------------------------
    g2_reasons: list[str] = []
    prose_without_urls = _URL_RE.sub("", problem).strip()
    substantive_chars = len(" ".join(prose_without_urls.split()))
    if substantive_chars < min_problem_chars:
        g2_reasons.append(f"problem_statement_too_short:{substantive_chars}<{min_problem_chars}")

    has_media_link = bool(_IMAGE_OR_VIDEO_RE.search(problem))
    has_code_block_or_tb = ("```" in problem) or ("Traceback" in problem) or ("    " in problem)
    if has_media_link and not has_code_block_or_tb:
        g2_reasons.append("external_screenshot_or_video_dependent_without_text_repro")

    if not _CODE_ANCHOR_RE.search(problem):
        g2_reasons.append("missing_concrete_technical_anchor_in_problem_statement")

    hidden = detect_hidden_test_contracts(problem, gold_patch, test_patch)
    if hidden["hidden_symbols"]:
        g2_reasons.append(f"hidden_new_api_in_test_patch:{','.join(hidden['hidden_symbols'][:5])}")
    if hidden["hidden_cli_flags"]:
        g2_reasons.append(f"hidden_new_cli_flag_in_test_patch:{','.join(hidden['hidden_cli_flags'][:5])}")
    if hidden["hidden_exception_messages"]:
        g2_reasons.append(f"hidden_exception_message_in_test_patch:{','.join(hidden['hidden_exception_messages'][:3])}")

    llm_clarity = row.get("llm_spec_clarity_score")
    if llm_clarity is not None and float(llm_clarity) < 4.0:
        g2_reasons.append(f"low_llm_spec_clarity_score:{llm_clarity}")

    gate2 = GateVerdict(
        passed=len(g2_reasons) == 0,
        reasons=tuple(g2_reasons),
        metrics={
            "substantive_chars": substantive_chars,
            "hidden_symbols": hidden["hidden_symbols"],
            "hidden_cli_flags": hidden["hidden_cli_flags"],
            "hidden_exception_messages": hidden["hidden_exception_messages"],
        },
    )

    # ------------------------------------------------------------------
    # Gate 3: Verified Competition-Valid Gold Fix & Alternative Solution Acceptance
    # ------------------------------------------------------------------
    g3_reasons: list[str] = []
    gold_diff = parse_unified_diff(gold_patch)
    gold_files = gold_diff["files"]
    if not gold_patch.strip() or not gold_files:
        g3_reasons.append("empty_gold_patch")
    if len(gold_files) > 8:
        g3_reasons.append(f"gold_patch_touches_too_many_files:{len(gold_files)}>8")
    if gold_diff["changed_line_count"] == 0 or gold_diff["changed_line_count"] > 600:
        g3_reasons.append(f"gold_patch_changed_lines_out_of_bounds:{gold_diff['changed_line_count']}")
    if gold_diff["has_binary"]:
        g3_reasons.append("gold_patch_contains_binary_diff")
    if gold_diff["has_symlink_mode"]:
        g3_reasons.append("gold_patch_modifies_symlink_mode")

    protected_touched = [f for f in gold_files if is_protected_swegemma_path(f)]
    if protected_touched:
        g3_reasons.append(f"gold_patch_modifies_swegemma_protected_paths:{','.join(protected_touched)}")

    if hidden["private_implementation_locks"]:
        g3_reasons.append(
            f"test_patch_locks_private_gold_helpers:{','.join(hidden['private_implementation_locks'][:5])}"
        )

    alt_solution_status = "not_probed"
    if control_report is not None:
        if "alt_solution_passed" in control_report:
            if control_report["alt_solution_passed"] is False:
                g3_reasons.append("alternative_equivalent_solution_rejected_by_tests")
                alt_solution_status = "rejected"
            elif control_report["alt_solution_passed"] is True:
                alt_solution_status = "accepted"
            else:
                alt_solution_status = "not_applicable"
        elif require_container_mutation_and_alt:
            g3_reasons.append("missing_alternative_solution_probe")

    gate3 = GateVerdict(
        passed=len(g3_reasons) == 0,
        reasons=tuple(g3_reasons),
        metrics={
            "gold_file_count": len(gold_files),
            "gold_changed_lines": gold_diff["changed_line_count"],
            "protected_paths_touched": protected_touched,
            "alt_solution_status": alt_solution_status,
        },
    )

    # ------------------------------------------------------------------
    # Gate 4: Reproducible Bug at Base (Offline Container Controls)
    # ------------------------------------------------------------------
    g4_reasons: list[str] = []
    ctrl_metrics: dict[str, Any] = {}
    if control_report is None:
        g4_reasons.append("missing_container_control_execution_report")
    else:
        base_r1 = control_report.get("base_repeat_1") or {}
        base_r2 = control_report.get("base_repeat_2") or {}
        if base_r1.get("returncode") != 1 or base_r2.get("returncode") != 1:
            g4_reasons.append("base_control_did_not_exit_1_across_both_repeats")
        if not base_r1.get("f2p_all_failed", False) or not base_r2.get("f2p_all_failed", False):
            g4_reasons.append("f2p_did_not_deterministically_fail_on_base")
        if not base_r1.get("p2p_all_passed", True) or not base_r2.get("p2p_all_passed", True):
            g4_reasons.append("p2p_failed_on_unpatched_base")
        if base_r1.get("error_count", 0) > 0 or base_r2.get("error_count", 0) > 0:
            g4_reasons.append("pytest_collection_or_runtime_error_on_base")
        sig1 = base_r1.get("status_signature_sha256")
        sig2 = base_r2.get("status_signature_sha256")
        if not sig1 or sig1 != sig2:
            g4_reasons.append("non_deterministic_base_test_signature_across_repeats")
        if control_report.get("editable_workspace_verified") is False:
            g4_reasons.append("package_import_shadowed_outside_workspace")
        ctrl_metrics["base_signature"] = sig1

    gate4 = GateVerdict(passed=len(g4_reasons) == 0, reasons=tuple(g4_reasons), metrics=ctrl_metrics)

    # ------------------------------------------------------------------
    # Gate 5: Strong, Non-Trivial, Fast Tests (+ Mutation Kill Score)
    # ------------------------------------------------------------------
    g5_reasons: list[str] = []
    test_diff = parse_unified_diff(test_patch)
    if not test_patch.strip() or not test_diff["files"]:
        g5_reasons.append("empty_test_patch")
    if len(f2p) < 1:
        g5_reasons.append("missing_fail_to_pass_tests")
    if len(p2p) < 1 and len(f2p) < 2:
        g5_reasons.append("insufficient_regression_coverage_need_p2p_or_multi_f2p")

    added_test_src = "\n".join(test_diff["added_lines"])
    assertion_lines = sum(1 for ln in test_diff["added_lines"] if _ASSERTION_LINE_RE.search(ln))
    if assertion_lines < 1 and test_diff["added_lines"]:
        # Allow adding new cases to an existing @pytest.mark.parametrize block whose assertion is in diff context
        context_src = "\n".join(test_diff["context_lines"])
        if "parametrize" in context_src or "parametrize" in added_test_src:
            assertion_lines = sum(1 for ln in test_diff["context_lines"] if _ASSERTION_LINE_RE.search(ln))
    if assertion_lines < 1:
        g5_reasons.append("test_patch_contains_zero_explicit_assertions")
    if _XFAIL_NONSTRICT_RE.search(added_test_src):
        g5_reasons.append("ambiguous_non_strict_xfail_in_test_patch")

    max_dur = 0.0
    mut_kill_score: float | None = None
    mut_tested_count = 0
    if control_report is not None:
        durations = [
            float((control_report.get(k) or {}).get("duration_sec", 0.0))
            for k in ("base_repeat_1", "base_repeat_2", "ref_repeat_1", "ref_repeat_2")
        ]
        max_dur = max(durations, default=0.0)
        if max_dur > max_test_duration_sec:
            g5_reasons.append(f"test_suite_too_slow:{max_dur:.1f}s>{max_test_duration_sec:.1f}s")

        if "mutation_kill_score" in control_report:
            mut_tested_count = int(control_report.get("mutation_tested_count", 0))
            if control_report["mutation_kill_score"] is not None:
                mut_kill_score = float(control_report["mutation_kill_score"])
                if mut_tested_count > 0 and mut_kill_score < min_mutation_kill_score:
                    g5_reasons.append(
                        f"weak_tests_low_mutation_kill_score:{mut_kill_score:.2f}<{min_mutation_kill_score:.2f}"
                    )
            if require_container_mutation_and_alt and mut_tested_count < 1:
                g5_reasons.append("zero_plausible_mutants_tested_for_strong_tests_gate")
        elif require_container_mutation_and_alt:
            g5_reasons.append("missing_container_mutation_kill_probe")

    gate5 = GateVerdict(
        passed=len(g5_reasons) == 0,
        reasons=tuple(g5_reasons),
        metrics={
            "f2p_count": len(f2p),
            "p2p_count": len(p2p),
            "added_assertion_lines": assertion_lines,
            "max_control_duration_sec": round(max_dur, 2),
            "mutation_tested_count": mut_tested_count,
            "mutation_kill_score": mut_kill_score,
        },
    )

    # ------------------------------------------------------------------
    # Gate 6: Trustworthy Grading & Exact swegemma 0.2.11 JUnit Parity
    # ------------------------------------------------------------------
    g6_reasons: list[str] = []
    grading_tier = "rejected"
    if control_report is None:
        g6_reasons.append("missing_reference_grading_report")
    else:
        ref_r1 = control_report.get("ref_repeat_1") or {}
        ref_r2 = control_report.get("ref_repeat_2") or {}
        if ref_r1.get("returncode") != 0 or ref_r2.get("returncode") != 0:
            g6_reasons.append("reference_gold_patch_did_not_exit_0")
        if ref_r1.get("failed_count", 0) > 0 or ref_r2.get("failed_count", 0) > 0:
            g6_reasons.append("reference_gold_patch_left_failed_tests")
        if ref_r1.get("error_count", 0) > 0 or ref_r2.get("error_count", 0) > 0:
            g6_reasons.append("reference_gold_patch_left_error_tests")
        rsig1 = ref_r1.get("status_signature_sha256")
        rsig2 = ref_r2.get("status_signature_sha256")
        if not rsig1 or rsig1 != rsig2:
            g6_reasons.append("non_deterministic_reference_test_signature_across_repeats")

        raw_junit_ok = bool(control_report.get("swegemma_raw_junit_valid", False))
        adapted_junit_ok = bool(control_report.get("swegemma_certified_identity_valid", False))
        if raw_junit_ok:
            grading_tier = "gold_raw_compatible"
        elif adapted_junit_ok and not require_raw_swegemma_junit:
            grading_tier = "certified_junit_identity"
        else:
            g6_reasons.append("failed_swegemma_0_2_11_junit_xml_validation")

    gate6 = GateVerdict(
        passed=len(g6_reasons) == 0,
        reasons=tuple(g6_reasons),
        metrics={"grading_tier": grading_tier},
    )

    gates = {
        "1_clean_separation": gate1,
        "2_clear_requirements": gate2,
        "3_verified_fixes": gate3,
        "4_reproducible_bugs": gate4,
        "5_strong_tests": gate5,
        "6_trustworthy_grading": gate6,
    }
    all_passed = all(g.passed for g in gates.values())
    if not all_passed:
        grading_tier = "rejected"

    rejection_summary = tuple(
        f"{g_name}:{r}" for g_name, g_val in gates.items() for r in g_val.reasons
    )
    return TaskQualityAudit(
        instance_id=iid,
        canonical_repo=canon_repo,
        admitted=all_passed,
        grading_tier=grading_tier,
        gates=gates,
        rejection_summary=rejection_summary,
    )


def _norm_sha(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def deduplicate_task_families(
    rows: list[dict[str, Any]],
    repo_rename_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Transitive union-find family deduplication so each underlying bug appears at most once."""
    if not rows:
        return []

    parent = list(range(len(rows)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    key_owner: dict[tuple[str, ...], int] = {}
    for idx, row in enumerate(rows):
        repo = str(row.get("_canonical_repo") or canonicalize_repo(str(row["repo"])))
        if repo_rename_map and repo in repo_rename_map:
            repo = canonicalize_repo(repo_rename_map[repo])
        repo_id = str(row.get("github_repo_id") or "").strip()
        repo_family_key = f"id:{repo_id}" if repo_id else repo
        base = str(row.get("base_commit") or "").strip().lower()
        keys: list[tuple[str, ...]] = [
            ("base", repo_family_key, base),
            ("issue", _norm_sha(str(row.get("problem_statement") or ""))),
            ("patch", _norm_sha(str(row.get("patch") or ""))),
            ("test", _norm_sha(str(row.get("test_patch") or ""))),
        ]
        combined_text = f"{row.get('problem_statement', '')}\n{row.get('hints_text', '')}"
        for m in _ISSUE_REF_RE.finditer(combined_text):
            num = m.group(1) or m.group(2)
            if num:
                keys.append(("issue_num", repo_family_key, num))
        for k in keys:
            if k in key_owner:
                _union(key_owner[k], idx)
            else:
                key_owner[k] = idx

    groups: dict[int, list[dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(_find(idx), []).append(row)

    # Deterministic representative: prefer gold_raw_compatible, then lexicographically smallest instance_id
    representatives: list[dict[str, Any]] = []
    for members in groups.values():
        members_sorted = sorted(
            members,
            key=lambda r: (
                0 if r.get("_grading_tier") == "gold_raw_compatible" else 1,
                str(r["instance_id"]),
            ),
        )
        representatives.append(members_sorted[0])

    representatives.sort(key=lambda r: str(r["instance_id"]))
    return representatives


def build_repo_disjoint_splits(
    rows: list[dict[str, Any]],
    control_reports_by_id: dict[str, dict[str, Any]],
    *,
    sealed_repos: set[str] | frozenset[str] = frozenset(),
    reserved_eval_repos: set[str] | frozenset[str] = ORIGINAL_SPLITV2_REPOS | SEALED_RICH_HTTPX_REPOS,
    cv_repo_fraction: float = 0.25,
    max_tasks_per_cv_repo: int = 3,
    split_seed: str = "gemma4-v2-purity-seal-20261101",
    max_test_duration_sec: float = 120.0,
    min_mutation_kill_score: float = 0.75,
    require_container_mutation_and_alt: bool = False,
    repo_rename_map: dict[str, str] | None = None,
    pinned_train_repos: set[str] | frozenset[str] = frozenset(),
    pinned_unseen_cv_repos: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Run all 6 quality gates, deduplicate bug families, and partition into repo-disjoint splits.

    Guarantees:
      - ZERO repository overlap between `train_pool`, `unseen_repo_cv`, `original_splitv2_80`,
        `sealed_rich_httpx_49`, and `sealed_harbor_july_65repos`.
      - `unseen_repo_cv` ONLY admits `gold_raw_compatible` tasks so CV grading is 100% raw-identical
        to Kaggle's hidden scorer without any JUnit adapter shim.
    """
    audits: list[TaskQualityAudit] = []
    admitted_rows: list[dict[str, Any]] = []

    for row in rows:
        iid = str(row.get("instance_id") or "")
        audit = audit_task_quality(
            row,
            sealed_repos=sealed_repos,
            reserved_eval_repos=reserved_eval_repos,
            control_report=control_reports_by_id.get(iid),
            max_test_duration_sec=max_test_duration_sec,
            min_mutation_kill_score=min_mutation_kill_score,
            require_container_mutation_and_alt=require_container_mutation_and_alt,
            repo_rename_map=repo_rename_map,
        )
        audits.append(audit)
        if audit.admitted:
            enriched = dict(row)
            enriched["_canonical_repo"] = audit.canonical_repo
            enriched["_grading_tier"] = audit.grading_tier
            admitted_rows.append(enriched)

    deduped = deduplicate_task_families(admitted_rows, repo_rename_map=repo_rename_map)

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in deduped:
        by_repo.setdefault(row["_canonical_repo"], []).append(row)

    def _canon_any(r: str) -> str:
        c = canonicalize_repo(r)
        if repo_rename_map and c in repo_rename_map:
            c = canonicalize_repo(repo_rename_map[c])
        return c

    canon_pin_train = {_canon_any(r) for r in pinned_train_repos if r}
    # Never allow a historically trained repo to enter unseen_repo_cv
    canon_pin_cv = {_canon_any(r) for r in pinned_unseen_cv_repos if r} - canon_pin_train

    # Only repositories with >=1 gold_raw_compatible task are eligible for unseen_repo_cv
    eligible_cv_repos = sorted(
        repo
        for repo, repo_rows in by_repo.items()
        if any(r["_grading_tier"] == "gold_raw_compatible" for r in repo_rows)
    )
    ranked_cv_repos = sorted(
        eligible_cv_repos,
        key=lambda repo: hashlib.sha256(f"{split_seed}::{repo}".encode("utf-8")).hexdigest(),
    )
    if canon_pin_train or canon_pin_cv:
        cv_repos = {r for r in eligible_cv_repos if r in canon_pin_cv}
        free_eligible_cv = [
            r for r in ranked_cv_repos if r not in canon_pin_cv and r not in canon_pin_train
        ]
        for r in free_eligible_cv:
            h_hex = hashlib.sha256(f"{split_seed}::{r}".encode("utf-8")).hexdigest()
            u01 = int(h_hex[:16], 16) / float(1 << 64)
            if u01 < cv_repo_fraction:
                cv_repos.add(r)
        if not cv_repos and free_eligible_cv and cv_repo_fraction > 0.0:
            cv_repos.add(free_eligible_cv[0])
    else:
        target_cv_repo_count = max(1, round(len(by_repo) * cv_repo_fraction)) if eligible_cv_repos else 0
        cv_repos = set(ranked_cv_repos[:target_cv_repo_count])
    train_repos = set(by_repo.keys()) - cv_repos

    # Double-assert strict pairwise repository disjointness and historical pin integrity!
    all_sealed_and_reserved = {_canon_any(r) for r in (set(sealed_repos) | set(reserved_eval_repos))}
    assert cv_repos.isdisjoint(train_repos), "FATAL: repo overlap between CV and Train!"
    assert cv_repos.isdisjoint(canon_pin_train), "FATAL: historically trained repo leaked into unseen CV!"
    assert (train_repos & set(eligible_cv_repos)).isdisjoint(canon_pin_cv), (
        "FATAL: historically pinned unseen CV repo leaked into Train!"
    )
    assert cv_repos.isdisjoint(all_sealed_and_reserved), "FATAL: repo overlap between CV and Sealed!"
    assert train_repos.isdisjoint(all_sealed_and_reserved), "FATAL: repo overlap between Train and Sealed!"

    cv_ids: list[str] = []
    train_ids: list[str] = []
    for repo in sorted(cv_repos):
        raw_ok_rows = sorted(
            (r for r in by_repo[repo] if r["_grading_tier"] == "gold_raw_compatible"),
            key=lambda r: str(r["instance_id"]),
        )
        for r in raw_ok_rows[:max_tasks_per_cv_repo]:
            cv_ids.append(str(r["instance_id"]))

    for repo in sorted(train_repos):
        for r in sorted(by_repo[repo], key=lambda item: str(item["instance_id"])):
            train_ids.append(str(r["instance_id"]))

    split_payload = {
        "split_seed": split_seed,
        "train_repos": sorted(train_repos),
        "unseen_cv_repos": sorted(cv_repos),
        "train_instance_ids": sorted(train_ids),
        "unseen_cv_instance_ids": sorted(cv_ids),
    }
    seal_sha256 = hashlib.sha256(
        repr(split_payload).encode("utf-8")
    ).hexdigest()

    return {
        "seal_sha256": seal_sha256,
        "total_candidates_audited": len(rows),
        "total_passed_all_6_gates": len(admitted_rows),
        "total_after_family_dedup": len(deduped),
        "train_repo_count": len(train_repos),
        "unseen_cv_repo_count": len(cv_repos),
        "train_task_count": len(train_ids),
        "unseen_cv_task_count": len(cv_ids),
        **split_payload,
        "audits": [a.to_dict() for a in audits],
    }
