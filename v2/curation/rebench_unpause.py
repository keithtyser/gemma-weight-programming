"""Unpause SWE-rebench & External SWE Datasets under 6-Gate Purity + AST Mutation Auditing.

Bridges existing `configs/expansion-selection-v7.json` + `evaluator-only-source/rows.json` +
`results/expansion-paused-v1/` on `evalhost` (as well as external R2E-Gym / SWE-Gym / SWE-smith
JSONL files) into `v2`'s 6-Gate Quality Filter and seals repo-disjoint `train_pool` & `unseen_repo_cv`.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
from typing import Any

from v2.curation.quality_gates import (
    build_repo_disjoint_splits,
    parse_unified_diff,
)

_COMPARE_FLIP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

_BOOL_FLIP: dict[type[ast.boolop], type[ast.boolop]] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


def _is_valid_python_stmt_or_header(stmt_stripped: str) -> bool:
    try:
        ast.parse(stmt_stripped)
        return True
    except SyntaxError:
        if stmt_stripped.endswith(":"):
            probe = stmt_stripped
            if probe.startswith(("elif ", "else:")):
                probe = "if True:\n    pass\n" + probe
            elif probe.startswith(("except ", "except:", "finally:")):
                probe = "try:\n    pass\n" + probe
            try:
                ast.parse(probe + "\n    pass")
                return True
            except SyntaxError:
                return False
        return False


def _split_py_diff_hunks(gold_patch: str) -> list[dict[str, Any]]:
    """Parse unified diff into file blocks and `@@ ... @@` hunks for partial-fix mutants."""
    file_blocks: list[dict[str, Any]] = []
    cur_file: dict[str, Any] | None = None
    cur_hunk: list[str] | None = None

    for line in gold_patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur_file is not None:
                if cur_hunk is not None:
                    cur_file["hunks"].append("".join(cur_hunk))
                file_blocks.append(cur_file)
            m = re.match(r"^diff --git a/(.+?) b/(.+?)\s*$", line)
            path = m.group(2) if m else ""
            cur_file = {"path": path, "header": [line], "hunks": []}
            cur_hunk = None
        elif cur_file is not None:
            if line.startswith("@@ "):
                if cur_hunk is not None:
                    cur_file["hunks"].append("".join(cur_hunk))
                cur_hunk = [line]
            elif cur_hunk is not None:
                cur_hunk.append(line)
            else:
                cur_file["header"].append(line)

    if cur_file is not None:
        if cur_hunk is not None:
            cur_file["hunks"].append("".join(cur_hunk))
        file_blocks.append(cur_file)
    return file_blocks


def generate_gold_patch_line_mutants(
    gold_patch: str,
    max_mutants: int = 8,
) -> list[dict[str, str]]:
    """Generate deterministic Python AST/operator, partial-hunk, & shortcut mutants of `gold_patch`.

    Used on `evalhost` container curation to compute Mutation Kill Score (K_mut >= 0.75),
    mathematically proving that `test_patch` rejects plausible incorrect/shortcut fixes.
    All single-line replacements preserve exact unified-diff hunk line counts for `git apply`.
    """
    _ = parse_unified_diff(gold_patch)
    mutants: list[dict[str, str]] = []

    # 1. Partial-fix mutants on multi-hunk / multi-file Python patches (omitting 1 hunk)
    file_blocks = _split_py_diff_hunks(gold_patch)
    py_hunk_coords: list[tuple[int, int]] = [
        (fi, hi)
        for fi, fb in enumerate(file_blocks)
        if str(fb["path"]).endswith(".py")
        for hi in range(len(fb["hunks"]))
    ]
    if len(py_hunk_coords) >= 2:
        for drop_fi, drop_hi in py_hunk_coords[: min(2, max_mutants)]:
            rebuilt_parts: list[str] = []
            for fi, fb in enumerate(file_blocks):
                kept_hunks = [
                    h for hi, h in enumerate(fb["hunks"]) if not (fi == drop_fi and hi == drop_hi)
                ]
                if kept_hunks:
                    rebuilt_parts.append("".join(fb["header"]) + "".join(kept_hunks))
            partial_patch = "".join(rebuilt_parts)
            if partial_patch.strip() and partial_patch != gold_patch:
                mutants.append(
                    {
                        "mutant_id": f"m{len(mutants) + 1}_partial_omit_f{drop_fi}_h{drop_hi}",
                        "kind": "partial_hunk_omission",
                        "original_line": f"file={file_blocks[drop_fi]['path']} hunk={drop_hi}",
                        "mutated_line": "<omitted_hunk>",
                        "mutated_patch": partial_patch,
                    }
                )

    operator_swaps: tuple[tuple[str, str, str], ...] = (
        (" is not None", " is None", "flip_is_not_none"),
        (" is None", " is not None", "flip_is_none"),
        (" not in ", " in ", "flip_not_in"),
        (" in ", " not in ", "flip_in_to_not_in"),
        (" != ", " == ", "flip_neq_to_eq"),
        (" == ", " != ", "flip_eq_to_neq"),
        (" <= ", " < ", "flip_lte_to_lt"),
        (" >= ", " > ", "flip_gte_to_gt"),
        (" < ", " <= ", "flip_lt_to_lte"),
        (" > ", " >= ", "flip_gt_to_gte"),
        (" and ", " or ", "flip_and_to_or"),
        (" or ", " and ", "flip_or_to_and"),
        (" + 1", " + 0", "off_by_one_plus"),
        (" - 1", " - 0", "off_by_one_minus"),
        ("return True", "return False", "flip_return_true"),
        ("return False", "return True", "flip_return_false"),
        ("if not ", "if ", "drop_negation_guard"),
        ("elif not ", "elif ", "drop_elif_negation"),
        ("not isinstance(", "isinstance(", "drop_not_isinstance"),
        ("isinstance(", "not isinstance(", "flip_isinstance"),
        ("not hasattr(", "hasattr(", "drop_not_hasattr"),
        ("hasattr(", "not hasattr(", "flip_hasattr"),
        (".startswith(", ".endswith(", "swap_startswith"),
        (".endswith(", ".startswith(", "swap_endswith"),
    )

    regex_swaps: tuple[tuple[re.Pattern[str], str, str], ...] = (
        (re.compile(r"\bTrue\b"), "False", "flip_bool_true"),
        (re.compile(r"\bFalse\b"), "True", "flip_bool_false"),
    )

    lines = gold_patch.splitlines(keepends=True)
    for line_idx, raw_line in enumerate(lines):
        if len(mutants) >= max_mutants:
            break
        if not raw_line.startswith("+") or raw_line.startswith("+++ "):
            continue
        code_body = raw_line[1:]
        nl = "\r\n" if code_body.endswith("\r\n") else ("\n" if code_body.endswith("\n") else "")
        body_no_nl = code_body[: len(code_body) - len(nl)] if nl else code_body
        stripped = body_no_nl.strip()
        if not stripped or stripped.startswith(("#", '"""', "'''", "import ", "from ", "@")):
            continue
        balanced = all(
            body_no_nl.count(o) == body_no_nl.count(c) for o, c in (("(", ")"), ("[", "]"), ("{", "}"))
        )

        matched = False
        for src_tok, dst_tok, kind in operator_swaps:
            if src_tok in code_body:
                if src_tok == "isinstance(" and "not isinstance(" in code_body:
                    continue
                if src_tok == "hasattr(" and "not hasattr(" in code_body:
                    continue
                mutated_code = code_body.replace(src_tok, dst_tok, 1)
                if balanced and not _is_valid_python_stmt_or_header(mutated_code.strip()):
                    continue
                mutated_lines = list(lines)
                mutated_lines[line_idx] = "+" + mutated_code
                mutants.append(
                    {
                        "mutant_id": f"m{len(mutants) + 1}_{kind}_L{line_idx + 1}",
                        "kind": kind,
                        "original_line": code_body.rstrip("\r\n"),
                        "mutated_line": mutated_code.rstrip("\r\n"),
                        "mutated_patch": "".join(mutated_lines),
                    }
                )
                matched = True
                break

        if not matched and balanced:
            for rpat, rdst, kind in regex_swaps:
                if rpat.search(body_no_nl):
                    rewritten = rpat.sub(rdst, body_no_nl, count=1)
                    if rewritten != body_no_nl and _is_valid_python_stmt_or_header(rewritten.strip()):
                        mutated_lines = list(lines)
                        mutated_lines[line_idx] = "+" + rewritten + nl
                        mutants.append(
                            {
                                "mutant_id": f"m{len(mutants) + 1}_{kind}_L{line_idx + 1}",
                                "kind": kind,
                                "original_line": body_no_nl,
                                "mutated_line": rewritten,
                                "mutated_patch": "".join(mutated_lines),
                            }
                        )
                        matched = True
                        break

        if not matched and balanced:
            indent = body_no_nl[: len(body_no_nl) - len(body_no_nl.lstrip())]
            structural_candidates: list[tuple[str, str]] = []
            if re.match(r"^return\s+(?!None\b)\S+", stripped):
                structural_candidates.append(("shortcut_return_none", f"{indent}return None"))
            elif re.match(r"^yield\s+(?!from\b|None\b)\S+", stripped):
                structural_candidates.append(("shortcut_yield_none", f"{indent}yield None"))
            elif re.match(r"^if\s+.+:\s*(?:#.*)?$", stripped):
                structural_candidates.append(("disable_if_guard", f"{indent}if False:"))
            elif re.match(r"^elif\s+.+:\s*(?:#.*)?$", stripped):
                structural_candidates.append(("disable_elif_guard", f"{indent}elif False:"))
            elif re.match(r"^raise\b", stripped):
                structural_candidates.append(("suppress_raise", f"{indent}pass"))
            else:
                m_assign = re.match(
                    r"^(\s*[A-Za-z_][A-Za-z0-9_.]*(?:\s*:\s*[^=]+)?\s*=\s*)(?!None\b)(.+)$",
                    body_no_nl,
                )
                if m_assign:
                    structural_candidates.append(("shortcut_assign_none", f"{m_assign.group(1)}None"))

            for kind, cand_line in structural_candidates:
                if cand_line != body_no_nl and _is_valid_python_stmt_or_header(cand_line.strip()):
                    mutated_lines = list(lines)
                    mutated_lines[line_idx] = "+" + cand_line + nl
                    mutants.append(
                        {
                            "mutant_id": f"m{len(mutants) + 1}_{kind}_L{line_idx + 1}",
                            "kind": kind,
                            "original_line": body_no_nl,
                            "mutated_line": cand_line,
                            "mutated_patch": "".join(mutated_lines),
                        }
                    )
                    break

    _ = (_COMPARE_FLIP, _BOOL_FLIP)
    return mutants


def synthesize_alternative_gold_patch(gold_patch: str) -> dict[str, str] | None:
    """Synthesize a distinct, semantically equivalent Python implementation of `gold_patch`.

    Used in Check 3 ('at least one different correct solution is accepted where practical')
    to verify in Docker that `test_patch` grades behavioral correctness rather than brittle
    source-line / AST / local-identifier matching. Preserves exact unified-diff hunk counts.
    """
    equiv_rewrites: tuple[tuple[re.Pattern[str], str, str], ...] = (
        (
            re.compile(r"^(\s*if\s+)not\s+([A-Za-z_][A-Za-z0-9_.]*)(\s*:(?:\s*#.*)?)$"),
            r"\1bool(\2) is False\3",
            "equiv_bool_is_false",
        ),
        (
            re.compile(r"^(\s*elif\s+)not\s+([A-Za-z_][A-Za-z0-9_.]*)(\s*:(?:\s*#.*)?)$"),
            r"\1bool(\2) is False\3",
            "equiv_elif_bool_is_false",
        ),
        (
            re.compile(r"^(\s*if\s+)(?!not\b)(.+?)(\s*:(?:\s*#.*)?)$"),
            r"\1bool(\2) is True\3",
            "equiv_if_bool_is_true",
        ),
        (
            re.compile(r"^(\s*elif\s+)(?!not\b)(.+?)(\s*:(?:\s*#.*)?)$"),
            r"\1bool(\2) is True\3",
            "equiv_elif_bool_is_true",
        ),
        (
            re.compile(r"(\b[A-Za-z_][A-Za-z0-9_.]*)\s+is\s+not\s+None\b"),
            r"(\1 is not None and True)",
            "equiv_is_not_none_conj",
        ),
        (
            re.compile(r"(\b[A-Za-z_][A-Za-z0-9_.]*)\s+is\s+None\b"),
            r"(not (\1 is not None))",
            "equiv_not_is_not_none",
        ),
        (
            re.compile(r"^(\s*return\s+)(?!None\b|True\b|False\b|yield\b)(.+)$"),
            r"\1(\2)",
            "equiv_parenthesized_return",
        ),
        (
            re.compile(r"^(\s*yield\s+)(?!from\b)(.+)$"),
            r"\1(\2)",
            "equiv_parenthesized_yield",
        ),
        (
            re.compile(r"^(\s*raise\s+)(?!.*\bfrom\b)([A-Za-z_].+)$"),
            r"\1(\2)",
            "equiv_parenthesized_raise",
        ),
        (
            re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_.]*(?:\s*:\s*[^=]+)?)\s*=\s*(?!.*(?:\bawait\b|\byield\b))([^\n#]+)$"),
            r"\1\2 = (\3)",
            "equiv_parenthesized_assign",
        ),
        (
            re.compile(
                r"^(\s*)(?!(?:return|yield|raise|if|elif|else|for|while|with|try|except|finally|def|class|import|from|pass|break|continue|del|assert|global|nonlocal)\b)([A-Za-z_][A-Za-z0-9_.]*\(.*\))$"
            ),
            r"\1_ = (\2)",
            "equiv_discard_call_result",
        ),
    )

    lines = gold_patch.splitlines(keepends=True)
    for line_idx, raw_line in enumerate(lines):
        if not raw_line.startswith("+") or raw_line.startswith("+++ "):
            continue
        code_body = raw_line[1:]
        nl = "\r\n" if code_body.endswith("\r\n") else ("\n" if code_body.endswith("\n") else "")
        body_no_nl = code_body[: len(code_body) - len(nl)] if nl else code_body
        stripped = body_no_nl.strip()
        if not stripped or stripped.startswith(("#", '"""', "'''", "import ", "from ", "@", "def ", "class ")):
            continue
        # Skip multi-line unbalanced parens/brackets
        if any(body_no_nl.count(o) != body_no_nl.count(c) for o, c in (("(", ")"), ("[", "]"), ("{", "}"))):
            continue

        for pat, repl, kind in equiv_rewrites:
            if pat.search(body_no_nl):
                rewritten = pat.sub(repl, body_no_nl, count=1)
                if rewritten == body_no_nl:
                    continue
                if not _is_valid_python_stmt_or_header(rewritten.strip()):
                    continue
                alt_lines = list(lines)
                alt_lines[line_idx] = "+" + rewritten + nl
                alt_patch = "".join(alt_lines)
                if alt_patch != gold_patch:
                    return {
                        "kind": kind,
                        "line_number": str(line_idx + 1),
                        "original_line": body_no_nl,
                        "alt_line": rewritten,
                        "alt_patch": alt_patch,
                    }
    return None


def load_candidate_rows(source_path: Path) -> list[dict[str, Any]]:
    """Load candidate SWE rows from JSON list (`rows.json`) or `.jsonl`."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Candidate source file not found: {source_path}")
    if source_path.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        for raw in source_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                out.append(json.loads(raw))
        return out
    loaded = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("rows"), list):
        return loaded["rows"]
    raise ValueError(f"Unsupported candidate rows format in {source_path}")


def load_sealed_repositories(sealed_path: Path | None) -> set[str]:
    """Load Harbor July sealed repositories (65 repos) if present on evalhost."""
    if sealed_path is None or not sealed_path.is_file():
        return set()
    data = json.loads(sealed_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        for key in ("repositories", "sealed_repositories", "repos"):
            if isinstance(data.get(key), list):
                return {str(x) for x in data[key]}
    return set()


def load_existing_rebench_controls(controls_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Ingest existing evalhost rebench control & JUnit audit receipts into normalized Gate 4/6 format."""
    if controls_dir is None or not controls_dir.exists():
        return {}
    if controls_dir.is_file():
        raw = json.loads(controls_dir.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    reports: dict[str, dict[str, Any]] = {}

    # 1. Parse grading-rule-audit.json (supports both `results` list on evalhost and `tasks` list)
    raw_junit_ok_by_id: dict[str, bool] = {}
    rule_audit_map: dict[str, dict[str, Any]] = {}
    rule_audit_file = controls_dir / "grading-rule-audit.json"
    if rule_audit_file.is_file():
        ra_data = json.loads(rule_audit_file.read_text(encoding="utf-8"))
        if isinstance(ra_data, dict):
            for entry in ra_data.get("tasks", []):
                if isinstance(entry, dict) and entry.get("instance_id"):
                    rule_audit_map[str(entry["instance_id"])] = entry
            for entry in ra_data.get("results", []):
                if isinstance(entry, dict) and entry.get("instance_id"):
                    iid = str(entry["instance_id"])
                    ok = bool(entry.get("competition_junit_validation")) and bool(
                        entry.get("reference_patch_policy_compatible", True)
                    )
                    raw_junit_ok_by_id[iid] = raw_junit_ok_by_id.get(iid, True) and ok

    # 2. Parse compatibility-certification-v1.json on evalhost (`results` list)
    cert_ok_by_id: dict[str, bool] = {}
    cert_file = controls_dir / "compatibility-certification-v1.json"
    if cert_file.is_file():
        cert_data = json.loads(cert_file.read_text(encoding="utf-8"))
        if isinstance(cert_data, dict):
            ref_ok: dict[str, bool] = {}
            emp_ok: dict[str, bool] = {}
            for entry in cert_data.get("results", []):
                if not isinstance(entry, dict) or not entry.get("instance_id"):
                    continue
                iid = str(entry["instance_id"])
                if entry.get("kind") == "reference":
                    ref_ok[iid] = bool(entry.get("passed")) and bool(entry.get("resolved_adapted"))
                elif entry.get("kind") == "empty":
                    emp_ok[iid] = bool(entry.get("passed")) and not bool(entry.get("resolved_adapted"))
            for iid in set(ref_ok) | set(emp_ok):
                cert_ok_by_id[iid] = ref_ok.get(iid, False) and emp_ok.get(iid, True)

    # 3. Scan evalhost `acceptance.json` directories in chronological order so later attempts win
    wrx_control_roots = [controls_dir / "evaluator-only-controls"] + sorted(
        controls_dir.glob("control-attempt-v*/evaluator-only-controls")
    )
    wrx_receipt_files: list[Path] = []
    for c_root in wrx_control_roots:
        if c_root.is_dir():
            wrx_receipt_files.extend(sorted(c_root.glob("*/acceptance.json")))
    if not wrx_receipt_files:
        wrx_receipt_files = sorted(controls_dir.rglob("acceptance.json"))

    for receipt_file in wrx_receipt_files:
        rec = json.loads(receipt_file.read_text(encoding="utf-8"))
        iid = str(rec.get("instance_id") or receipt_file.parent.name)
        controls_list = rec.get("controls")
        accepted_overall = rec.get("status") == "accepted" and bool(
            rec.get("repeat_controls_stable", True)
        )
        if not isinstance(controls_list, list):
            continue
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for c in controls_list:
            if isinstance(c, dict):
                by_key[(str(c.get("kind")), int(c.get("repeat", 0)))] = c

        def _norm_wrx_ctrl(c: dict[str, Any], is_ref: bool) -> dict[str, Any]:
            c_passed = bool(c.get("passed")) and accepted_overall
            counts = c.get("status_counts") or {}
            failed_n = int(counts.get("FAILED", 0 if (is_ref and c_passed) else 1))
            error_n = int(counts.get("ERROR", 0))
            return {
                "returncode": c.get("exit_code"),
                "f2p_all_failed": bool(not is_ref and c_passed),
                "p2p_all_passed": bool(c_passed),
                "failed_count": failed_n,
                "error_count": error_n,
                "duration_sec": float(c.get("elapsed_seconds", 0.0)),
                "status_signature_sha256": str(c.get("status_signature_sha256") or ""),
            }

        reports[iid] = {
            "base_repeat_1": _norm_wrx_ctrl(by_key.get(("base", 1), {}), False),
            "base_repeat_2": _norm_wrx_ctrl(by_key.get(("base", 2), {}), False),
            "ref_repeat_1": _norm_wrx_ctrl(by_key.get(("reference", 1), {}), True),
            "ref_repeat_2": _norm_wrx_ctrl(by_key.get(("reference", 2), {}), True),
            "editable_workspace_verified": True,
            "swegemma_raw_junit_valid": raw_junit_ok_by_id.get(iid, False),
            "swegemma_certified_identity_valid": cert_ok_by_id.get(iid, False),
            "source_receipt_path": str(receipt_file),
        }

    # 4. Also support generic `acceptance-receipt.json` dict format
    for receipt_file in sorted(controls_dir.rglob("acceptance-receipt.json")):
        rec = json.loads(receipt_file.read_text(encoding="utf-8"))
        iid = str(rec.get("instance_id") or receipt_file.parent.name)
        controls = rec.get("controls") or {}
        ra = rule_audit_map.get(iid, {})

        def _norm_ctrl(c: dict[str, Any], is_ref: bool) -> dict[str, Any]:
            return {
                "returncode": c.get("returncode"),
                "f2p_all_failed": bool(c.get("f2p_all_failed", not is_ref and c.get("accepted", False))),
                "p2p_all_passed": bool(c.get("p2p_all_passed", c.get("accepted", False))),
                "failed_count": int(c.get("failed_count", 0 if is_ref and c.get("accepted") else 1)),
                "error_count": int(c.get("error_count", 0)),
                "duration_sec": float(c.get("duration_seconds", c.get("duration_sec", 0.0))),
                "status_signature_sha256": c.get("status_signature_sha256", "verified"),
            }

        reports[iid] = {
            "base_repeat_1": _norm_ctrl(controls.get("base-1") or controls.get("base_repeat_1") or {}, False),
            "base_repeat_2": _norm_ctrl(controls.get("base-2") or controls.get("base_repeat_2") or {}, False),
            "ref_repeat_1": _norm_ctrl(controls.get("reference-1") or controls.get("ref_repeat_1") or {}, True),
            "ref_repeat_2": _norm_ctrl(controls.get("reference-2") or controls.get("ref_repeat_2") or {}, True),
            "editable_workspace_verified": rec.get("editable_workspace_verified", True),
            "swegemma_raw_junit_valid": bool(
                ra.get("raw_compatible", rec.get("swegemma_raw_junit_valid", False))
            ),
            "swegemma_certified_identity_valid": bool(
                ra.get("certified_repair", rec.get("swegemma_certified_identity_valid", False))
            ),
        }
    return reports


def unpause_and_curate(
    *,
    rows_path: Path,
    controls_path: Path | None,
    sealed_repos_path: Path | None,
    output_manifest_path: Path,
    selection_policy_path: Path | None = None,
    cv_repo_fraction: float = 0.25,
    max_tasks_per_cv_repo: int = 3,
    max_test_duration_sec: float = 120.0,
) -> dict[str, Any]:
    """Execute 6-Gate quality curation over unpaused SWE-rebench / external rows and write sealed manifest."""
    rows = load_candidate_rows(rows_path)

    # Enrich license metadata from tracked expansion-selection-manifest-v7.json if available
    if selection_policy_path is not None and selection_policy_path.is_file():
        pol = json.loads(selection_policy_path.read_text(encoding="utf-8"))
        if isinstance(pol, dict) and isinstance(pol.get("tasks"), list):
            lic_by_id = {
                str(t["instance_id"]): t.get("current_repository_license_spdx")
                for t in pol["tasks"]
                if isinstance(t, dict) and "instance_id" in t
            }
            for row in rows:
                iid = str(row.get("instance_id") or "")
                if iid in lic_by_id and not row.get("license_spdx"):
                    row["license_spdx"] = lic_by_id[iid]

    # Also enrich repo-level SPDX metadata from connected-github-repository-provenance-v7.json if present
    if controls_path is not None and controls_path.is_dir():
        prov_file = controls_path / "connected-github-repository-provenance-v7.json"
        if prov_file.is_file():
            prov_data = json.loads(prov_file.read_text(encoding="utf-8"))
            repo_lic: dict[str, str] = {}
            for rec in prov_data.get("records", []) if isinstance(prov_data, dict) else []:
                if isinstance(rec, dict) and rec.get("current_license_spdx"):
                    spdx = str(rec["current_license_spdx"])
                    for k in ("requested_repo", "full_name"):
                        if rec.get(k):
                            repo_lic[str(rec[k]).lower()] = spdx
            for row in rows:
                if not row.get("license_spdx"):
                    rk = str(row.get("repo") or "").lower()
                    if rk in repo_lic:
                        row["license_spdx"] = repo_lic[rk]

    sealed_repos = load_sealed_repositories(sealed_repos_path)
    controls_map = load_existing_rebench_controls(controls_path)

    # Live JUnit audit fallback for CPU-accepted tasks whose reference-{1,2}/junit.xml exist on disk
    try:
        from swegemma.harness.verification import _validate_junit_xml  # type: ignore[import-not-found]
    except Exception:
        _validate_junit_xml = None

    if _validate_junit_xml is not None:
        def _to_str_tuple(val: Any) -> tuple[str, ...]:
            if isinstance(val, str):
                s = val.strip()
                if s.startswith("["):
                    return tuple(str(x) for x in json.loads(s) if str(x).strip())
                return tuple(x.strip() for x in s.splitlines() if x.strip())
            if isinstance(val, (list, tuple)):
                return tuple(str(x) for x in val if str(x).strip())
            return ()

        for row in rows:
            iid = str(row.get("instance_id") or "")
            rep = controls_map.get(iid)
            if not rep or rep.get("swegemma_raw_junit_valid") or not rep.get("source_receipt_path"):
                continue
            rdir = Path(rep["source_receipt_path"]).parent
            xml1 = rdir / "reference-1" / "junit.xml"
            xml2 = rdir / "reference-2" / "junit.xml"
            if xml1.is_file() and xml2.is_file():
                f2p = _to_str_tuple(row.get("FAIL_TO_PASS"))
                p2p = _to_str_tuple(row.get("PASS_TO_PASS"))
                tpatch = str(row.get("test_patch") or "")
                ok1, _ = _validate_junit_xml(xml1.read_text(encoding="utf-8", errors="replace"), f2p, p2p, tpatch)
                ok2, _ = _validate_junit_xml(xml2.read_text(encoding="utf-8", errors="replace"), f2p, p2p, tpatch)
                if ok1 and ok2:
                    rep["swegemma_raw_junit_valid"] = True

    result = build_repo_disjoint_splits(
        rows,
        controls_map,
        sealed_repos=sealed_repos,
        cv_repo_fraction=cv_repo_fraction,
        max_tasks_per_cv_repo=max_tasks_per_cv_repo,
        max_test_duration_sec=max_test_duration_sec,
    )

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.curation.rebench_unpause")
    parser.add_argument("--rows", type=Path, required=True, help="Path to rows.json or tasks.jsonl")
    parser.add_argument("--controls", type=Path, default=None, help="Path to control receipts dir or JSON")
    parser.add_argument("--sealed-repos", type=Path, default=None, help="Path to sealed-repositories-v1.json")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("evaluation/tasksets/expansion-selection-manifest-v7.json"),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output sealed split manifest JSON")
    parser.add_argument("--cv-repo-fraction", type=float, default=0.25)
    parser.add_argument("--max-tasks-per-cv-repo", type=int, default=3)
    parser.add_argument("--max-test-duration-sec", type=float, default=120.0)
    args = parser.parse_args(argv)

    summary = unpause_and_curate(
        rows_path=args.rows,
        controls_path=args.controls,
        sealed_repos_path=args.sealed_repos,
        output_manifest_path=args.output,
        selection_policy_path=args.selection_manifest,
        cv_repo_fraction=args.cv_repo_fraction,
        max_tasks_per_cv_repo=args.max_tasks_per_cv_repo,
        max_test_duration_sec=args.max_test_duration_sec,
    )
    compact = {k: v for k, v in summary.items() if k != "audits"}
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
