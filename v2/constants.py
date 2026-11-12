"""Canonical constants, model profiles, scorer rules, and repository alias tables."""

from __future__ import annotations

from dataclasses import dataclass
import posixpath


@dataclass(frozen=True)
class ModelSpec:
    """Hardware, architecture, and competition serving profile for an allowed Gemma 4 model."""

    model_id: str
    track: str  # "small_bonus" | "main" | "ablation"
    num_hidden_layers: int
    independent_kv_layers: tuple[int, int]  # inclusive (start, end)
    logit_softcap: float
    # Empirical throughput on Kaggle 4x NVIDIA L4 (TP=4, vLLM 0.19.1, QAT-W4A16)
    est_seconds_per_turn_no_lora: float
    est_seconds_per_turn_with_lora: float
    recommended_max_lora_rank: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "gemma-4-e2b-it-qat-w4a16-ct": ModelSpec(
        model_id="gemma-4-e2b-it-qat-w4a16-ct",
        track="small_bonus",
        num_hidden_layers=35,
        independent_kv_layers=(0, 19),
        logit_softcap=30.0,
        est_seconds_per_turn_no_lora=3.2,
        est_seconds_per_turn_with_lora=3.7,
        recommended_max_lora_rank=32,
    ),
    "gemma-4-e4b-it-qat-w4a16-ct": ModelSpec(
        model_id="gemma-4-e4b-it-qat-w4a16-ct",
        track="small_bonus",
        num_hidden_layers=42,
        independent_kv_layers=(0, 23),  # Layers 24..41 share KV projections!
        logit_softcap=30.0,
        est_seconds_per_turn_no_lora=4.6,
        est_seconds_per_turn_with_lora=5.2,
        recommended_max_lora_rank=32,
    ),
    "gemma-4-12b-it-qat-w4a16-ct": ModelSpec(
        model_id="gemma-4-12b-it-qat-w4a16-ct",
        track="ablation",
        num_hidden_layers=48,
        independent_kv_layers=(0, 47),
        logit_softcap=30.0,
        est_seconds_per_turn_no_lora=8.5,
        est_seconds_per_turn_with_lora=9.6,
        recommended_max_lora_rank=32,
    ),
    "gemma-4-26b-a4b-it-qat-w4a16-ct": ModelSpec(
        model_id="gemma-4-26b-a4b-it-qat-w4a16-ct",
        track="main",
        num_hidden_layers=46,
        independent_kv_layers=(0, 45),
        logit_softcap=30.0,
        est_seconds_per_turn_no_lora=5.8,
        est_seconds_per_turn_with_lora=6.7,
        recommended_max_lora_rank=32,
    ),
    "gemma-4-31b-it-qat-w4a16-ct": ModelSpec(
        model_id="gemma-4-31b-it-qat-w4a16-ct",
        track="main",
        num_hidden_layers=60,
        independent_kv_layers=(0, 59),
        logit_softcap=30.0,
        est_seconds_per_turn_no_lora=12.8,
        est_seconds_per_turn_with_lora=14.4,
        recommended_max_lora_rank=32,
    ),
}

ALLOWED_COMPETITION_MODELS: frozenset[str] = frozenset(MODEL_SPECS.keys())

E4B_MODEL_ID = "gemma-4-e4b-it-qat-w4a16-ct"
MAIN_31B_MODEL_ID = "gemma-4-31b-it-qat-w4a16-ct"

# Live Kaggle evaluation wheelhouse versions (verified Oct/Nov 2026)
LIVE_HARNESS_VERSIONS: dict[str, str] = {
    "swegemma": "0.2.11",
    "adk-submission": "0.2.13",
    "adk-eval-core": "0.1.0",
    "google-adk": "1.36.1",
    "vllm": "0.19.1",
}

# Kaggle evaluation runtime limits
KAGGLE_TOTAL_TASKS_ESTIMATE = 120
KAGGLE_HARD_WALL_CLOCK_HOURS = 12.0
KAGGLE_SAFE_WALL_CLOCK_HOURS = 10.5
KAGGLE_MAX_SUBMISSION_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB

# Core workspace & execution tools vs hazardous/bloated graph tools
SAFE_CORE_TOOLS: tuple[str, ...] = (
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
)

GRAPH_INTELLIGENCE_TOOLS: frozenset[str] = frozenset(
    {
        "get_code_neighbors",
        "search_similar_code",
        "get_code_subgraph",
    }
)

# Tool known to trigger vLLM #753973 infinite hang on list[str] tool arguments
VLLM_LIST_HANG_TOOLS: frozenset[str] = frozenset({"get_code_subgraph"})

# Local grader controls (val15_empty_control_v3/v4, val15_gold_control_v4) showed these tasks are uninformative:
# an empty patch "resolved" requests_7427 / requests_7309 / fastapi_14851 in at least one run, and the gold patch
# failed requests_7433 / fastapi_14186. Promotion decisions exclude them.
FLAKY_GRADER_TASK_IDS: frozenset[str] = frozenset(
    {"requests_7427", "requests_7309", "fastapi_14851", "requests_7433", "fastapi_14186"}
)

# KnownADK session.state keys populated by swegemma.harness.agent_runner
KNOWN_ADK_SESSION_STATE_KEYS: frozenset[str] = frozenset({"problem_description"})

# Valid vLLM max_lora_rank buckets (from adk_submission/server.py:59)
VALID_VLLM_LORA_RANKS: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256, 320, 512)

# Canonical GitHub repository rename & lowercase alias map to prevent split leakage
REPO_RENAME_ALIASES: dict[str, str] = {
    "tiangolo/fastapi": "fastapi/fastapi",
    "kennethreitz/requests": "psf/requests",
    "textualize/rich": "textualize/rich",
    "willmcgugan/rich": "textualize/rich",
    "encode/httpx": "encode/httpx",
    "delgan/loguru": "delgan/loguru",
    "pypa/setuptools_scm": "pypa/setuptools-scm",
}

# Repositories permanently reserved in original competition / holdout splits
ORIGINAL_SPLITV2_REPOS: frozenset[str] = frozenset(
    {"fastapi/fastapi", "psf/requests"}
)
SEALED_RICH_HTTPX_REPOS: frozenset[str] = frozenset(
    {"textualize/rich", "encode/httpx"}
)

PERMISSIVE_SPDX_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSD",
        "ISC",
        "PSF-2.0",
        "0BSD",
        "Unlicense",
    }
)

# Exact protected test & config filenames stripped before grading in swegemma 0.2.11
# Mirrors swegemma/harness/verification.py:62-86 (_is_protected_test_or_config_path)
PROTECTED_CONFIG_BASENAMES: frozenset[str] = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
        "sitecustomize.py",
        "usercustomize.py",
        "_swegemma_stubs.py",
    }
)

PROTECTED_TEST_DIRS: frozenset[str] = frozenset({"tests", "test", "testing"})


def canonicalize_repo(repo: str) -> str:
    """Normalize repository owner/name to lowercase and resolve known GitHub renames."""
    cleaned = repo.strip().strip("/").lower()
    return REPO_RENAME_ALIASES.get(cleaned, cleaned)


def is_protected_swegemma_path(rel_path: str) -> bool:
    """Exact replica of swegemma 0.2.11 verification._is_protected_test_or_config_path.

    Any changes made by an agent (or required by an external dataset's gold patch)
    to these paths are stripped/reverted before Phase 2 pytest grading on Kaggle.
    """
    norm = posixpath.normpath(rel_path.replace("\\", "/")).lstrip("/")
    if not norm or norm == ".":
        return False
    parts = [p.lower() for p in norm.split("/") if p and p != "."]
    if not parts:
        return False
    basename = parts[-1]
    if basename in PROTECTED_CONFIG_BASENAMES or basename.endswith(".pth"):
        return True
    if any(seg in PROTECTED_TEST_DIRS for seg in parts[:-1]) and basename.endswith(".py"):
        return True
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    if basename.endswith("_test.py"):
        return True
    return False
