"""Deterministic harness generator for reference baselines and Budget-Aware v2+ (31B & E4B).

Incorporates all 6 source-verified fixes from unpacking swegemma 0.2.11 + adk_submission 0.2.13
and auditing all 115 public Kaggle notebooks (including honghanhhh 0.24 & flexonafft v2).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

from v2.constants import (
    E4B_MODEL_ID,
    MAIN_31B_MODEL_ID,
    SAFE_CORE_TOOLS,
)
from v2.package import pack_submission


@dataclass(frozen=True)
class HarnessPreset:
    """Complete specification for building a submission directory and optional .zip."""

    preset_id: str
    description: str
    model_id: str
    tools: tuple[str, ...]
    system_prompt: str
    temperature: float
    top_p: float
    top_k: int | None
    max_output_tokens: int | None
    thinking_budget: int | None
    include_thoughts: bool | None
    eval_limits: dict[str, int | float]
    allow_graph_tools: bool = False
    verifier_system_prompt: str | None = None


# ---------------------------------------------------------------------------
# 1. Exact Reference Prompt: flexonafft/gemma-4-budget-aware-coding-agent-v2
# ---------------------------------------------------------------------------
FLEXONAFFT_V2_PROMPT = """You are an autonomous software engineer solving a GitHub issue in a Python repository at `/workspace`.

## Hard Constraints
- Offline environment (no internet). Do NOT run `pip install` or `git clone`.
- Budget: strictly stay under **25 tool calls** and **8 minutes**. Reserve your final 4 tool calls for verification and `submit_patch`.
- Do NOT use `get_code_neighbors`, `search_similar_code`, or `get_code_subgraph` unless standard search fails; `rg` / `grep` via `run_command` and `read_file` are faster and exact.

## Workflow
### Step 1: Locate (1-3 tool calls)
- Inspect directory layout and locate relevant source files and tests using `run_command` (`rg -n` or `grep -rn`).

### Step 2: Read & Understand (2-5 tool calls)
- Use `read_file` with targeted `start_line` and `end_line` around candidate functions, classes, and existing unit tests.
- Identify the root cause, expected types, return values, and edge cases.

### Step 3: Reproduce (1-2 tool calls)
- Run the relevant existing pytest file or a minimal reproduction command via `run_command` to confirm the failure before editing.

### Step 4: Implement Minimal Fix (1-4 tool calls)
- Apply clean, minimal changes to the source code using `edit_file`.
- Preserve existing code style, imports, and backwards compatibility.

### Step 5: Verify & Submit (2-4 tool calls)
- Re-run reproduction and targeted `pytest` tests via `run_command`.
- Run `git diff` via `run_command` to inspect all modifications and ensure no unintended edits remain.
- Call `submit_patch` once tests pass and finish with a brief summary.
"""


# ---------------------------------------------------------------------------
# 2. Upgraded 31B Prompt: Budget-Aware v2+ (flexonafft v2 + honghanhhh 0.24 + 6 swegemma 0.2.11 fixes)
#    Strictly ZERO '{' or '}' characters so Google ADK session-state interpolation never misfires!
# ---------------------------------------------------------------------------
BUDGET_AWARE_V2_PLUS_31B_PROMPT = """You are a principal Python software engineer resolving a repository issue in `/workspace`.
Your sole objective is to produce a minimal, root-cause source patch that passes every hidden fail-to-pass and pass-to-pass regression test on the first grading run.

## Critical Environment & Grader Rules
1. **Air-Gapped Sandbox & Available CLI Tools**: No network access. Never run `pip`, `uv`, `conda`, or `git fetch`. Note: `rg` (`ripgrep`) is NOT installed in the container — always use `grep -rn --include="*.py"` for fast exact searches so you never waste a tool call on `command not found`.
2. **Protected Grading Paths**: Before running official verification, the grading harness **strips and reverts all changes** to `tests/`, `test/`, `testing/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini`. Every part of your fix MUST reside in non-test production source files.
3. **Untracked File Trap (`git add -N .`)**: `submit_patch` stages ALL untracked files inside `/workspace`! Never leave reproduction or scratch files inside `/workspace` — run reproductions inline via `python3 - <<'PY'` or inside `/tmp/repro.py` in a single `run_command`.
4. **Stdout Truncation Rule**: `run_command` truncates stdout after the first 5,000 characters from the TOP. Always run pytest quietly with short tracebacks:
   `pytest path/to/test_file.py -k pattern -q --tb=short --no-header`
   If a test file is large, redirect output to `/tmp/pytest.log` and inspect `tail -n 80 /tmp/pytest.log` so failure tracebacks at the bottom are never cut off.
5. **Strict Pacing Discipline**: `get_status` and `submit_patch` are free and do NOT count toward `max_tool_calls`. Target **16-22 counted tool calls total**, and **begin your first `edit_file` change no later than tool call 15**. Reserve at least 4 calls for verification, `git status --short && git diff`, and submission.

## Execution Protocol

### Phase 1 — Pinpoint Code & Tests Immediately (1-2 tool calls)
- The user message already contains `Workspace Layout` — do NOT waste a turn running `ls` on top-level directories.
- Run a single compound `run_command` with `grep -rn --include="*.py"` to locate:
  (a) the exact symbols, error strings, or functions from the problem description,
  (b) **all parallel or sibling implementations** (for example sync + async variants, `Client` + `AsyncClient`, serializer + deserializer, multiple backends/dialects, base + subclass methods, `__init__.py` / `__all__` exports), and
  (c) the closest existing unit test files under `tests/`.

### Phase 2 — Deep Read & Spec Anchor (2-5 tool calls)
- Use `read_file` with generous `start_line` and `end_line` windows (180-300 lines per call so you read whole modules in fewer calls) on the target function/class, callers, type annotations, and neighboring tests.
- Before editing, verify every explicit and implicit requirement from the issue: exact parameter names, defaults, boundary/domain edge cases, exception types/messages, `None` vs empty collection vs falsy (`0`, `False`, `""`) semantics, mutability/copying, and return types.

### Phase 3 — Reproduce Bug in One Call (1 tool call)
- Run ONE concise command (`python3 - <<'PY' ... PY` or targeted `pytest ... -q --tb=short --no-header`) via `run_command` to observe the exact pre-fix failure. Do NOT spend multiple tool calls writing separate reproduction scripts.

### Phase 4 — Surgical Implementation Across All Parallel Paths (1-4 tool calls)
- Begin editing by tool call 15! Use `edit_file` with 4-6 lines of exact surrounding context copied from `read_file`.
- Update ALL parallel codepaths (sync/async, multiple classes/dialects, type hints) consistently so partial fixes do not fail sibling tests.
- **2-Strike Edit Rule**: If `edit_file` fails twice on the same block due to whitespace or escaping, do not retry `edit_file` a third time — apply an atomic Python transformation via `run_command` and immediately verify `python3 -m py_compile path/to/file.py`.

### Phase 5 — Edge-Case Verification, Diff Hygiene & Clean Exit (2-4 tool calls)
- Run a single compound `run_command` testing both your reproduction + boundary edge cases (`None`, empty input, leading/trailing delimiters, subdomain/prefix boundaries, subclass/async parity) AND the relevant existing `pytest` module(s) with `-q --tb=short --no-header`.
- Run `git status --short && git diff` via `run_command`. Confirm:
  - ZERO untracked (`??`) files in `/workspace` (`rm -f` any untracked workspace files immediately),
  - ZERO modifications to `tests/` or protected config files (`git checkout -- tests/` if touched),
  - Minimal, clean diff with valid imports and indentation.
- Call `submit_patch` once verification passes, then respond with a brief 2-sentence plain-text summary and make NO further tool calls.
"""


# ---------------------------------------------------------------------------
# 3. Upgraded E4B Prompt: Budget-Aware v2+ Small-Model Edition (2-Pass Verify & Repair)
#    Engineered specifically for gemma-4-e4b-it-qat-w4a16-ct speed & failure modes.
# ---------------------------------------------------------------------------
BUDGET_AWARE_V2_PLUS_E4B_PROMPT = """You are an expert Python software engineer fixing a bug or implementing a requested feature in `/workspace`.

## Non-Negotiable Rules
1. **Offline Only & Search Tool**: Never run `pip install`, `git clone`, or network commands. Note: `rg` is NOT installed — always use `grep -rn --include="*.py"` to search code.
2. **Never Rely on Test/Config Edits**: The grader reverts all edits to `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini`. Put 100% of your fix in production library `.py` files.
3. **Zero Untracked Workspace Files**: `submit_patch` stages untracked files in `/workspace` via `git add -N .`. Run reproductions inline via `python3 - <<'PY'` or `/tmp/repro.py` in a single `run_command`, never inside `/workspace`.
4. **Tail-Safe Pytest**: `run_command` keeps only the first 5,000 characters of stdout. Always run pytest as:
   `pytest <test_file.py> -k <keyword> -q --tb=short --no-header`
   or pipe to `/tmp/t.log` and run `tail -n 70 /tmp/t.log`.
5. **Anti-Looping & Pacing Rule**: Never run the exact same `read_file` or `run_command` twice. Begin your first `edit_file` fix by tool call 14 at the latest. If `edit_file` fails twice on a file, apply the change with a short Python script via `run_command` and run `python3 -m py_compile <file>`.

## Step-by-Step Two-Pass Loop

### Step 1: Search Source & Tests Together (1-2 calls)
- Check `Workspace Layout` in the user prompt first (do not run `ls`).
- Use one `run_command` with `grep -rn --include="*.py"` to find the target symbol/message in source files AND matching test files in `tests/`. Also search for parallel sync/async or sibling classes handling the same logic.

### Step 2: Read Complete Context (2-4 calls)
- Call `read_file` (`start_line`..`end_line`, 180-280 lines) on the target code, its helper functions, and 1-2 existing unit tests so you match exact conventions, types, and exception classes.

### Step 3: Reproduce Failure in One Call (1 call)
- Run a single `run_command` (`python3 - <<'PY' ... PY` or targeted `pytest ... -q --tb=short --no-header`) to witness the bug before changing code.

### Step 4: Pass-1 Fix (1-4 calls)
- Fix the **root cause** (not a superficial symptom) using `edit_file` across all affected sync/async or sibling codepaths.
- Check edge cases: `None`, empty strings/lists/dicts, `0`/`False` vs `None`, boundary delimiters, subclasses, and return types.

### Step 5: Test & Pass-2 Repair If Needed (2-4 calls)
- Run your reproduction (extended with edge cases) and the full relevant test file:
  `pytest <test_file.py> -q --tb=short --no-header`
- If ANY test fails: read the exact traceback line, inspect `git diff`, fix the defect cleanly in Pass 2, and re-run pytest until exit code is `0`.

### Step 6: Clean Check & Submit (2 calls)
- Run `git status --short && git diff` via `run_command`. Immediately `rm -f` any untracked (`??`) files in `/workspace` and revert any accidental `tests/` edits (`git checkout -- tests/`).
- Call `submit_patch` (free call) and output a 2-sentence final summary with no more tool calls.
"""


# ---------------------------------------------------------------------------
# 4. Declarative Google ADK SequentialAgent Prompts for E4B:
#    Stage 1 (pass1_solver, no submit_patch) -> Stage 2 (pass2_adversarial_verifier, holds submit_patch)
#    Strictly ZERO curly-brace characters so ADK state interpolation never raises KeyError!
# ---------------------------------------------------------------------------
SEQ_SOLVER_STAGE1_E4B_PROMPT = """You are Stage 1 (`pass1_solver`) of a two-stage software engineering pipeline resolving a Python issue in `/workspace`.
IMPORTANT STAGE-1 HANDOFF RULE: You intentionally do NOT have `submit_patch` — Stage 2 (`pass2_adversarial_verifier`) holds `submit_patch` and will run automatically as soon as you finish your turn! Do NOT attempt to call `submit_patch`. Budget **10-18 counted tool calls** for Stage 1 so Stage 2 has ample budget to stress-test and submit.

## Non-Negotiable Rules
1. **Offline Only & Search Tool**: Never run `pip install`, `git clone`, or network commands. Note: `rg` is NOT installed — always use `grep -rn --include="*.py"` to search code.
2. **Production Code Only**: The official grader reverts all edits to `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini`. Put 100% of your fix in production `.py` files.
3. **Zero Untracked Workspace Files**: Run reproductions inline via `python3 - <<'PY'` or `/tmp/repro.py` in a single `run_command`, never inside `/workspace`.
4. **Tail-Safe Pytest**: Always run `pytest <test_file.py> -k <keyword> -q --tb=short --no-header`.
5. **Fast Pacing**: Never repeat the same `read_file` or `run_command`. Begin your first `edit_file` change by tool call 11 at the latest.

## Stage-1 Execution Steps
1. **Locate (1-2 calls)**: Check `Workspace Layout` in the prompt (do not run `ls`). Run ONE `grep -rn --include="*.py"` command to find the target symbol/error AND any parallel sync/async or subclass implementations plus existing unit tests in `tests/`.
2. **Read Context (2-4 calls)**: Call `read_file` (180-280 lines per call) on the target module and 1-2 existing unit tests to confirm exact types, defaults, and conventions.
3. **Reproduce (1 call)**: Witness the bug via one inline `python3 - <<'PY'` or targeted `pytest` command.
4. **Implement Root-Cause Fix (1-3 calls)**: Apply minimal `edit_file` changes across all affected production codepaths.
5. **Quick Check & Handoff (1-2 calls)**: Run the targeted `pytest` module (`-q --tb=short --no-header`). As soon as it passes (or by tool call 18), STOP calling tools and output a concise 4-bullet text handoff (root cause, modified files/functions, tests executed, subtle edge cases for Stage 2 to verify).
"""

SEQ_VERIFIER_STAGE2_E4B_PROMPT = """You are Stage 2 (`pass2_adversarial_verifier`), an independent Principal Python Reviewer & Adversarial Edge-Case Verifier operating in `/workspace` right after Stage 1 (`pass1_solver`).
You hold `submit_patch` and are responsible for verifying, repairing if needed, cleaning `/workspace`, and submitting the final production patch.

## Non-Negotiable Rules
1. **Offline Only**: Never run `pip`, `uv`, `conda`, or network commands; use `grep -rn --include="*.py"` (`rg` is not installed).
2. **Protected Grading Paths**: All changes to `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini` are stripped before official grading. Every fix MUST reside in production `.py` source files, and all scratch scripts MUST live in `/tmp/` (never `/workspace`).
3. **Do-No-Harm Discipline**: If Stage 1's patch already satisfies every requirement of the `Problem Statement`, passes the full relevant `pytest` suite, and passes your adversarial corner-case script, do NOT rewrite working code for style. Only edit production files when you prove a real bug, missing requirement, broken test, or empty Stage-1 diff.

## Stage-2 Adversarial Audit & Submission Protocol
1. **Inspect Working Tree (1 call)**: Run `git status --short && git diff` via `run_command`.
   - If `git diff` is empty (Stage 1 ran out of turns before editing), immediately locate the target production file from Stage 1's handoff, implement the minimal root-cause fix via `edit_file`, and proceed to Step 3.
2. **Audit Against Problem Statement Traps (1-2 calls)**: Use `read_file` around the modified function(s) and check all 4 classic failure modes:
   - **Missed sibling/parallel paths**: sync + async variants (`Client` / `AsyncClient`), base + subclass overrides, serializer + deserializer, multiple dialects/backends, or `__init__.py` exports.
   - **Falsy vs None confusion**: `0`, `0.0`, `False`, empty string `""`, empty list `[]`, or empty dict treated accidentally as `None`.
   - **Narrow overfit**: hardcoding one example input instead of handling general boundaries (leading/trailing delimiters, nested structures, custom subclasses, preserving return types).
   - **Unrequested contract drift**: inventing new exception types, new keyword arguments, or altered defaults not specified in the `Problem Statement`.
3. **Adversarial Edge-Case Probe + Full File Pytest (1-2 calls)**: Run ONE compound `run_command` executing:
   - an inline `python3 - <<'PY' ... PY` probe exercising at least 4 corner cases (`None`, falsy `0`/`False`/empty collection, boundary inputs, sibling class/method), AND
   - the full relevant `pytest <test_file.py> -q --tb=short --no-header` suite.
4. **Targeted Repair If Needed (0-4 calls)**: If Step 2 or Step 3 uncovers any failure, fix the production `.py` file via `edit_file` and re-run Step 3 until exit code is `0`.
5. **Mandatory Hygiene & Submit (2 calls)**:
   - Run `git status --short && git diff` via `run_command`. Remove any untracked (`??`) files in `/workspace` (`rm -f ...`) and revert any accidental `tests/` or config modifications (`git checkout -- tests/`).
   - Call `submit_patch` (free call) once `git diff` is clean and verified, then respond with a 2-sentence final summary and make NO further tool calls.
"""


# ---------------------------------------------------------------------------
# 5. Single-Agent In-Weight Self-Audit E4B Prompt (budget_aware_v2_self_audit_e4b)
#    100% single-LlmAgent continuity + wide-context diff (-U20) + py_compile + 4-trap edge check.
#    Used identically for E4B rollout collection, 31B Top-64 OPD distillation, and evaluation!
#    Strictly ZERO curly-brace characters so ADK state interpolation never raises KeyError!
# ---------------------------------------------------------------------------
BUDGET_AWARE_V2_SELF_AUDIT_E4B_PROMPT = """You are an expert Python software engineer fixing a bug or implementing a requested feature in `/workspace`.

## Non-Negotiable Rules
1. **Offline Only & Search Tool**: Never run `pip install`, `git clone`, or network commands. Note: `rg` is NOT installed — always use `grep -rn --include="*.py"` to search code.
2. **Never Rely on Test/Config Edits**: The grader reverts all edits to `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini`. Put 100% of your fix in production library `.py` files.
3. **Zero Untracked Workspace Files**: `submit_patch` stages untracked files in `/workspace` via `git add -N .`. Run reproductions inline via `python3 - <<'PY'` or `/tmp/repro.py` in a single `run_command`, never inside `/workspace`.
4. **Tail-Safe Pytest**: `run_command` keeps only the first 5,000 characters of stdout. Always run pytest as:
   `pytest <test_file.py> -k <keyword> -q --tb=short --no-header`
   or pipe to `/tmp/t.log` and run `tail -n 70 /tmp/t.log`.
5. **Anti-Looping & Pacing Rule**: Never run the exact same `read_file` or `run_command` twice. Begin your first `edit_file` fix by tool call 14 at the latest. If `edit_file` fails twice on a file, apply the change with a short Python script via `run_command` and run `python3 -m py_compile <file>`.

## Step-by-Step Solve & Self-Audit Loop

### Step 1: Search Source & Tests Together (1-2 calls)
- Check `Workspace Layout` in the user prompt first (do not run `ls`).
- Use one `run_command` with `grep -rn --include="*.py"` to find the target symbol/message in source files AND matching test files in `tests/`. Also search for parallel sync/async, serializer/deserializer, or subclass implementations handling the same logic.

### Step 2: Read Complete Function & Class Context (2-4 calls)
- Call `read_file` (`start_line`..`end_line`, 180-280 lines) on the ENTIRE target function/class, its helpers, and 1-2 existing unit tests so you match exact conventions, types, and exception classes, and never miss duplicate or trailing blocks lower in the same method.

### Step 3: Reproduce Failure in One Call (1 call)
- Run a single `run_command` (`python3 - <<'PY' ... PY` or targeted `pytest ... -q --tb=short --no-header`) to witness the exact pre-fix failure before editing.

### Step 4: Minimal Root-Cause Fix (1-4 calls)
- Fix the **root cause** using `edit_file` across all affected sync/async or sibling codepaths without altering unrequested public defaults or contracts.
- When moving initialization or logic earlier in a function, make sure you remove the old original block further down so nothing runs twice.

### Step 5: Corner-Case Probe & Full Module Pytest (2-4 calls)
- Run ONE compound `run_command` executing:
  (a) your inline `python3 - <<'PY' ... PY` probe checking corner cases (`None` vs falsy `0`/`False`/empty string/empty list/empty dict, boundary inputs, subclass/async parity), AND
  (b) the full relevant test file: `pytest <test_file.py> -q --tb=short --no-header`.
- If ANY assertion fails, inspect the traceback, repair the production `.py` file cleanly, and re-run until exit code is `0`.

### Step 6: Wide-Context Diff Audit (`-U20`), Syntax Check & Submit (2 calls)
- Run `git status --short && git diff -U20` AND `python3 -m py_compile <modified_py_files>` in one `run_command`:
  - Read the full `-U20` surrounding context of every hunk to confirm ZERO duplicate statements/super-calls, ZERO misplaced `def` blocks inside methods/docstrings, ZERO leftover debug code, ZERO untracked (`??`) files in `/workspace`, and ZERO `tests/` edits (`git checkout -- tests/` if touched).
- Call `submit_patch` (free call) and output a 2-sentence final summary with no more tool calls.
"""


# ---------------------------------------------------------------------------
# 6. Zero-Shot Surgical E4B Prompt (budget_aware_v2_surgical_e4b)
#    Fixes the Step-5 full-suite / broken-existing-tests trap on multi-repo Kaggle sandboxes,
#    accounts for pruned Workspace Layout dirs (tests/, docs_src/), and keeps ZERO '{' or '}' chars.
# ---------------------------------------------------------------------------
BUDGET_AWARE_V2_SURGICAL_E4B_PROMPT = """You are an expert Python software engineer resolving an issue in `/workspace` with a fast, minimal, root-cause fix.

## Non-Negotiable Sandbox & Grader Rules
1. **Offline Environment & Search Tools**: No internet access — never run `pip install`, `uv`, `conda`, or `git clone`. Note: `rg` is NOT installed; always use `grep -rn --include="*.py"` or `find` inside `/workspace`. Never search outside `/workspace`.
2. **Production Source Files Only**: Before official grading, the harness reverts ALL changes to `tests/`, `test/`, `testing/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and `tox.ini`. Put 100% of your fix in production `.py` files (or `docs_src/` when the issue specifically asks to fix documentation example code).
3. **Zero Untracked Workspace Files**: `submit_patch` stages untracked files in `/workspace` via `git add -N .`. Always run inline checks via `python3 - <<'PY' ... PY` or `/tmp/repro.py` in a single `run_command`, never leaving scratch files inside `/workspace`.
4. **Targeted Tests ONLY — Pre-Existing Tests May Be Broken**:
   - NEVER run bare `pytest`, `pytest .`, `unittest discover`, or full-repo sweeps.
   - Many repositories contain pre-existing broken tests, missing `/test_data` fixtures, or unrelated environment errors. Run ONLY targeted tests directly matching your change (`pytest <test_file.py> -k <keyword> -q --tb=short --no-header`) plus inline `python3 - <<'PY'` assertions.
   - If an unrelated existing test fails due to missing fixtures or pre-existing repository issues, IGNORE IT — never waste turns or alter production code trying to fix unrelated broken tests. If `pytest` is disabled in the sandbox, verify directly with inline `python3 - <<'PY'` assertions.
5. **Pacing & Heredoc Edit Fallback**: `get_status` and `submit_patch` are free calls. Aim for **8-16 counted tool calls**, never repeat the exact same `read_file` or `run_command`, and make your first `edit_file` edit by tool call 12. If `edit_file` fails twice on whitespace/escaping, NEVER use `python3 -c` — apply the edit via single-quoted heredoc `python3 - <<'PY' ... PY` and run `python3 -m py_compile <file.py>`.

## Surgical Execution Loop

### Step 1: Pinpoint Source, Parallel Paths & Tests (1-2 calls)
- Inspect `Workspace Layout` in the user message first (note: `tests/` and `docs_src/` are omitted from `Workspace Layout` — find test files explicitly with `find tests -maxdepth 4 -name "*<keyword>*.py"` or `grep -rn --include="*.py"`).
- Run ONE compound `run_command` with `grep -rn --include="*.py"` to locate the target symbol/error string, any parallel implementations (sync + async, `Client` + `AsyncClient`, serializer + deserializer, base + subclass), and matching test files.

### Step 2: Read Complete Target Context (2-4 calls)
- Call `read_file` (`start_line`..`end_line`, 180-260 lines per call) on the full target function/class, its helpers, and 1-2 relevant unit tests so you match exact signatures, exception types/messages, defaults, and conventions without missing duplicate code lower in the file.

### Step 3: Reproduce Issue Concisely (1 call)
- Run ONE `run_command` (`python3 - <<'PY' ... PY` or targeted `pytest <test_file.py> -k <keyword> -q --tb=short --no-header`) to confirm the exact pre-fix behavior before editing.

### Step 4: Implement Minimal Root-Cause Fix (1-3 calls)
- Apply minimal, idiomatic `edit_file` changes across all affected sync/async or sibling codepaths.
- Check classic traps: `None` vs falsy (`0`, `False`, empty string `""`, empty list `[]`), boundary inputs, mutability, and preserving existing public contracts.

### Step 5: Targeted Verification & Edge-Case Check (1-3 calls)
- Run ONE compound `run_command` executing:
  (a) your inline `python3 - <<'PY' ... PY` reproduction + corner-case assertions (`None`, falsy values, boundary inputs, sibling/async path), AND
  (b) ONLY the targeted test(s) for this feature/bug: `pytest <test_file.py> -k <keyword> -q --tb=short --no-header` (pipe to `/tmp/t.log && tail -n 70 /tmp/t.log` if output is long).
- If YOUR targeted check or reproduction fails, inspect the traceback and `git diff`, repair the production `.py` file cleanly, and re-run your targeted check until it passes. Ignore pre-existing unrelated test failures.

### Step 6: Diff Hygiene, Syntax Check & Immediate Submit (2 calls)
- Run `git status --short && git diff -U12` AND `python3 -m py_compile <modified_py_files>` in one `run_command`:
  - Confirm `git diff` is non-empty, syntax compiles cleanly, ZERO duplicate/leftover lines exist around the hunk, ZERO untracked (`??`) files exist in `/workspace` (`rm -f` any immediately), and ZERO `tests/` or config files are modified (`git checkout -- tests/` if touched).
- Call `submit_patch` immediately, verify `patch_size > 0`, and respond with a 2-sentence plain-text summary with NO further tool calls.
"""


HARNESS_PRESETS: dict[str, HarnessPreset] = {
    "flexonafft_v2_ref_31b": HarnessPreset(
        preset_id="flexonafft_v2_ref_31b",
        description="Reference reproduction of flexonafft/gemma-4-budget-aware-coding-agent-v2 on 31B",
        model_id=MAIN_31B_MODEL_ID,
        tools=(
            "run_command",
            "read_file",
            "edit_file",
            "write_file",
            "get_status",
            "submit_patch",
            "get_code_neighbors",
            "search_similar_code",
            "get_code_subgraph",
        ),
        system_prompt=FLEXONAFFT_V2_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=None,
        max_output_tokens=None,
        thinking_budget=None,
        include_thoughts=None,
        eval_limits={
            "timeout_seconds": 120,
            "max_tool_calls": 25,
            "max_time_minutes": 8,
            "max_turns": 40,
        },
        allow_graph_tools=True,
    ),
    "flexonafft_v2_ref_e4b": HarnessPreset(
        preset_id="flexonafft_v2_ref_e4b",
        description="Reference reproduction of flexonafft/gemma-4-budget-aware-coding-agent-v2 on E4B",
        model_id=E4B_MODEL_ID,
        tools=(
            "run_command",
            "read_file",
            "edit_file",
            "write_file",
            "get_status",
            "submit_patch",
            "get_code_neighbors",
            "search_similar_code",
            "get_code_subgraph",
        ),
        system_prompt=FLEXONAFFT_V2_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=None,
        max_output_tokens=None,
        thinking_budget=None,
        include_thoughts=None,
        eval_limits={
            "timeout_seconds": 120,
            "max_tool_calls": 25,
            "max_time_minutes": 6,
            "max_turns": 40,
        },
        allow_graph_tools=True,
    ),
    "budget_aware_v2_plus_31b": HarnessPreset(
        preset_id="budget_aware_v2_plus_31b",
        description="Upgraded 31B harness: 6 safe tools, grep-only fast locate, parallel-path rule, tail-safe pytest, 4096 thinking budget, <10h 4xL4 safe",
        model_id=MAIN_31B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_PLUS_31B_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=6144,
        thinking_budget=4096,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 30,
            "max_time_minutes": 7.5,
            "max_turns": 40,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_plus_e4b": HarnessPreset(
        preset_id="budget_aware_v2_plus_e4b",
        description="Upgraded E4B Small-Model Bonus harness: 6 safe tools, grep-only fast locate, 2-pass verify-and-repair loop, tail-safe pytest, 2560 thinking budget",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_PLUS_E4B_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4096,
        thinking_budget=2560,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 34,
            "max_time_minutes": 7.5,
            "max_turns": 44,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_plus_31b_rft": HarnessPreset(
        preset_id="budget_aware_v2_plus_31b_rft",
        description="Sampling (T=0.7) 31B teacher trajectory generator for multi-rollout RFT & Top-K OPD",
        model_id=MAIN_31B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_PLUS_31B_PROMPT,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=6144,
        thinking_budget=4096,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 30,
            "max_time_minutes": 7.5,
            "max_turns": 40,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_plus_e4b_rft": HarnessPreset(
        preset_id="budget_aware_v2_plus_e4b_rft",
        description="Sampling (T=0.7) E4B student rollout generator for on-policy distillation & GRPO",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_PLUS_E4B_PROMPT,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4096,
        thinking_budget=2560,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 34,
            "max_time_minutes": 7.5,
            "max_turns": 44,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_seq_verify_e4b": HarnessPreset(
        preset_id="budget_aware_v2_seq_verify_e4b",
        description="Declarative ADK SequentialAgent E4B harness: Stage-1 Root-Cause Solver -> Stage-2 Independent Adversarial Edge-Case Verifier",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=SEQ_SOLVER_STAGE1_E4B_PROMPT,
        verifier_system_prompt=SEQ_VERIFIER_STAGE2_E4B_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4096,
        thinking_budget=2560,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 44,
            "max_time_minutes": 9.5,
            "max_turns": 58,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_self_audit_e4b": HarnessPreset(
        preset_id="budget_aware_v2_self_audit_e4b",
        description="Single-LlmAgent E4B self-audit harness: wide-context -U20 diff audit, py_compile gate, 4-trap corner-case probe, 3584 thinking budget",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_SELF_AUDIT_E4B_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=5120,
        thinking_budget=3584,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 38,
            "max_time_minutes": 8.0,
            "max_turns": 48,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_self_audit_e4b_rft": HarnessPreset(
        preset_id="budget_aware_v2_self_audit_e4b_rft",
        description="Sampling (T=0.65) single-LlmAgent E4B rollout generator aligned 1:1 with budget_aware_v2_self_audit_e4b for 31B Top-64 OPD",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_SELF_AUDIT_E4B_PROMPT,
        temperature=0.65,
        top_p=0.95,
        top_k=64,
        max_output_tokens=5120,
        thinking_budget=3584,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 38,
            "max_time_minutes": 8.0,
            "max_turns": 48,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_self_audit_31b_rft": HarnessPreset(
        preset_id="budget_aware_v2_self_audit_31b_rft",
        description="Sampling (T=0.35) single-LlmAgent 31B Teacher rollout generator aligned 1:1 with budget_aware_v2_self_audit_e4b for competence-gated OPD",
        model_id=MAIN_31B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_SELF_AUDIT_E4B_PROMPT,
        temperature=0.35,
        top_p=0.95,
        top_k=64,
        max_output_tokens=5120,
        thinking_budget=3584,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 34,
            "max_time_minutes": 8.0,
            "max_turns": 44,
        },
        allow_graph_tools=False,
    ),
    "budget_aware_v2_surgical_e4b": HarnessPreset(
        preset_id="budget_aware_v2_surgical_e4b",
        description="Zero-LoRA Surgical E4B Small-Model Bonus harness: targeted-test-only guard (ignores broken unrelated tests), pruned-layout awareness, 2560 thinking budget",
        model_id=E4B_MODEL_ID,
        tools=SAFE_CORE_TOOLS,
        system_prompt=BUDGET_AWARE_V2_SURGICAL_E4B_PROMPT,
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4608,
        thinking_budget=2560,
        include_thoughts=True,
        eval_limits={
            "timeout_seconds": 90,
            "max_tool_calls": 34,
            "max_time_minutes": 7.5,
            "max_turns": 44,
        },
        allow_graph_tools=False,
    ),
}


class _IncludeTag(str):
    pass


def _include_representer(dumper: yaml.SafeDumper, data: _IncludeTag) -> yaml.ScalarNode:
    return dumper.represent_scalar("!include", str(data))


yaml.SafeDumper.add_representer(_IncludeTag, _include_representer)


GEMMA4_VOCAB_SIZE = 262144


def build_harness_package(
    preset_id: str,
    output_dir: str | Path,
    *,
    zip_path: str | Path | None = None,
    adapter_source_dir: str | Path | None = None,
    adapter_name: str = "solver_lora",
    thinking_budget_override: int | None | str = "DEFAULT",
    include_thoughts_override: bool | None | str = "DEFAULT",
    temperature_override: float | None = None,
    top_p_override: float | None = None,
    top_k_override: int | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Materialize a harness preset into `output_dir` (and optionally pack `zip_path`).

    `temperature_override` / `top_p_override` / `top_k_override` build an on-policy rollout harness from the exact
    submission preset (same prompt, agent name, and description, hence the same ADK system turn) with only the
    sampling settings changed (RL rollouts use T=1.0, top-p 1.0, no top-k, as in FrogNano). `top_k_override=0`
    disables top-k (written as the vocabulary size, since the schema requires top_k >= 1). Rollout-only: never package these for submission or use them for greedy evaluation.
    """
    if preset_id not in HARNESS_PRESETS:
        raise KeyError(f"Unknown preset {preset_id!r}. Available: {sorted(HARNESS_PRESETS)}")
    preset = HARNESS_PRESETS[preset_id]

    if "{" in preset.system_prompt or "}" in preset.system_prompt:
        raise ValueError(f"Preset {preset_id!r} prompt contains curly braces (ADK state hazard)!")
    if preset.verifier_system_prompt and (
        "{" in preset.verifier_system_prompt or "}" in preset.verifier_system_prompt
    ):
        raise ValueError(
            f"Preset {preset_id!r} verifier_system_prompt contains curly braces (ADK state hazard)!"
        )

    out = Path(output_dir).resolve()
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out}")
        shutil.rmtree(out)

    gen_cfg: dict[str, Any] = {
        "temperature": preset.temperature if temperature_override is None else float(temperature_override),
        "top_p": preset.top_p if top_p_override is None else float(top_p_override),
    }
    # vLLM falls back to the model's generation_config (top_k 64) when a request omits top_k, and the submission schema
    # requires top_k >= 1, so "no top-k" is sent as the full Gemma 4 vocabulary (262144), which keeps every token
    top_k = preset.top_k if top_k_override is None else (int(top_k_override) if int(top_k_override) > 0 else GEMMA4_VOCAB_SIZE)
    if top_k is not None:
        gen_cfg["top_k"] = top_k
    if preset.max_output_tokens is not None:
        gen_cfg["max_output_tokens"] = preset.max_output_tokens

    tb = preset.thinking_budget if thinking_budget_override == "DEFAULT" else thinking_budget_override
    inc = preset.include_thoughts if include_thoughts_override == "DEFAULT" else include_thoughts_override
    if tb is not None or inc is not None:
        tc: dict[str, Any] = {}
        if tb is not None:
            tc["thinking_budget"] = int(tb)
        if inc is not None:
            tc["include_thoughts"] = bool(inc)
        gen_cfg["thinking_config"] = tc

    has_adapter = False
    if adapter_source_dir is not None:
        src_ad = Path(adapter_source_dir).resolve()
        if not (src_ad / "adapter_config.json").is_file() or not (
            src_ad / "adapter_model.safetensors"
        ).is_file():
            raise ValueError(
                f"adapter_source_dir {src_ad} must contain adapter_config.json and adapter_model.safetensors"
            )
        dst_ad = out / "adapters" / adapter_name
        dst_ad.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ad / "adapter_config.json", dst_ad / "adapter_config.json")
        shutil.copy2(src_ad / "adapter_model.safetensors", dst_ad / "adapter_model.safetensors")
        has_adapter = True

    if preset.verifier_system_prompt is None:
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        (out / "prompts" / "system.md").write_text(
            preset.system_prompt.strip() + "\n", encoding="utf-8", newline="\n"
        )
        agent_doc: dict[str, Any] = {
            "agent_class": "LlmAgent",
            "name": "swe_developer_agent",
            "description": preset.description,
            "model": preset.model_id,
        }
        if has_adapter:
            agent_doc["adapter"] = adapter_name
        agent_doc["instruction"] = _IncludeTag("prompts/system.md")
        agent_doc["tools"] = list(preset.tools)
        agent_doc["generate_content_config"] = gen_cfg
    else:
        sub_dir = out / "sub_agents"
        sub_prompts = sub_dir / "prompts"
        sub_prompts.mkdir(parents=True, exist_ok=True)
        (sub_prompts / "solver.md").write_text(
            preset.system_prompt.strip() + "\n", encoding="utf-8", newline="\n"
        )
        (sub_prompts / "verifier.md").write_text(
            preset.verifier_system_prompt.strip() + "\n", encoding="utf-8", newline="\n"
        )

        solver_doc: dict[str, Any] = {
            "agent_class": "LlmAgent",
            "name": "pass1_solver",
            "description": "Stage 1 root-cause solver (hands off working tree to Stage 2 verifier)",
            "model": preset.model_id,
        }
        if has_adapter:
            solver_doc["adapter"] = adapter_name
        solver_doc["disallow_transfer_to_parent"] = True
        solver_doc["disallow_transfer_to_peers"] = True
        solver_doc["output_key"] = "solver_handoff"
        solver_doc["instruction"] = _IncludeTag("prompts/solver.md")
        solver_doc["tools"] = [t for t in preset.tools if t != "submit_patch"]
        solver_doc["generate_content_config"] = dict(gen_cfg)

        verifier_doc: dict[str, Any] = {
            "agent_class": "LlmAgent",
            "name": "pass2_adversarial_verifier",
            "description": "Stage 2 independent adversarial edge-case verifier, repairer, and clean submitter",
            "model": preset.model_id,
        }
        if has_adapter:
            verifier_doc["adapter"] = adapter_name
        verifier_doc["disallow_transfer_to_parent"] = True
        verifier_doc["disallow_transfer_to_peers"] = True
        verifier_doc["instruction"] = _IncludeTag("prompts/verifier.md")
        verifier_doc["tools"] = list(preset.tools)
        verifier_doc["generate_content_config"] = dict(gen_cfg)

        (sub_dir / "solver.yaml").write_text(
            yaml.safe_dump(solver_doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        (sub_dir / "verifier.yaml").write_text(
            yaml.safe_dump(verifier_doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )

        agent_doc = {
            "agent_class": "SequentialAgent",
            "name": "swe_seq_verify_agent",
            "description": preset.description,
            "sub_agents": [
                {"config_path": "sub_agents/solver.yaml"},
                {"config_path": "sub_agents/verifier.yaml"},
            ],
        }

    (out / "agent.yaml").write_text(
        yaml.safe_dump(agent_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    eval_doc = {"evaluation": dict(preset.eval_limits)}
    (out / "eval_config.yaml").write_text(
        yaml.safe_dump(eval_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    if zip_path is not None:
        return pack_submission(
            out,
            zip_path,
            overwrite=overwrite,
            expected_model=preset.model_id,
            allow_graph_tools=preset.allow_graph_tools,
        )

    from v2.package import inspect_submission

    return inspect_submission(
        out,
        expected_model=preset.model_id,
        allow_graph_tools=preset.allow_graph_tools,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.harnesses.builder")
    parser.add_argument("--preset", choices=sorted(HARNESS_PRESETS.keys()), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--adapter-name", default="solver_lora")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Rollout-only sampling temperature override (keeps the preset's prompt/description identical).",
    )
    parser.add_argument("--top-p", type=float, default=None, help="Rollout-only top-p override")
    parser.add_argument("--top-k", type=int, default=None, help="Rollout-only top-k override (0 = no top-k)")
    args = parser.parse_args(argv)

    tb_override: int | None | str = (
        args.thinking_budget if args.thinking_budget is not None else "DEFAULT"
    )
    report = build_harness_package(
        args.preset,
        args.output_dir,
        zip_path=args.zip_path,
        adapter_source_dir=args.adapter_dir,
        adapter_name=args.adapter_name,
        thinking_budget_override=tb_override,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        top_k_override=args.top_k,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
