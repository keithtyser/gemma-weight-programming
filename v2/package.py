"""Multi-model deterministic submission packager & preflight validator (swegemma 0.2.11 / adk-submission 0.2.13).

Supports both the Small Model Bonus track (gemma-4-e4b-it-qat-w4a16-ct) and the
Main Competition track (gemma-4-31b-it-qat-w4a16-ct) while adding 6 competition-specific
preflight hazard gates on top of structural archive validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any
import zipfile

import yaml

from v2.constants import (
    ALLOWED_COMPETITION_MODELS,
    GRAPH_INTELLIGENCE_TOOLS,
    KAGGLE_HARD_WALL_CLOCK_HOURS,
    KAGGLE_MAX_SUBMISSION_BYTES,
    KAGGLE_SAFE_WALL_CLOCK_HOURS,
    KAGGLE_TOTAL_TASKS_ESTIMATE,
    KNOWN_ADK_SESSION_STATE_KEYS,
    MODEL_SPECS,
    VALID_VLLM_LORA_RANKS,
    VLLM_LIST_HANG_TOOLS,
)

# Exact allowed extensions from adk_submission 0.2.13 (adk_submission/limits.py:103-135)
ADK_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".rst",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ggml",
        ".model",
        ".tiktoken",
        ".jinja",
        ".json",
        ".jsonl",
        ".toml",
        ".cfg",
        ".ini",
        ".py",
        ".sh",
        ".sql",
        ".csv",
        ".tsv",
        ".patch",
        ".diff",
        ".html",
        ".xml",
    }
)

FORBIDDEN_ARCHIVE_FILENAMES: frozenset[str] = frozenset(
    {"tasks.jsonl", "manifest.json", "results.jsonl", "rows.json", "solution.parquet"}
)

EVAL_LIMIT_KEYS: tuple[str, ...] = (
    "timeout_seconds",
    "max_tool_calls",
    "max_time_minutes",
    "max_turns",
)

# Google ADK instructions_utils.inject_session_state interpolates {var_name} patterns
_ADK_STATE_BRACE_RE = re.compile(r"\{+([a-zA-Z_][a-zA-Z0-9_.]*)\}+")
_LAYER_KV_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.(k_proj|v_proj)(?:\.|$)")

_SAFETENSOR_DTYPES: dict[str, tuple[int, str | None]] = {
    "F64": (8, "d"),
    "F32": (4, "f"),
    "F16": (2, "e"),
    "BF16": (2, None),
    "I64": (8, "q"),
    "I32": (4, "i"),
    "I16": (2, "h"),
    "I8": (1, "b"),
    "U64": (8, "Q"),
    "U32": (4, "I"),
    "U16": (2, "H"),
    "U8": (1, "B"),
    "BOOL": (1, "?"),
}


def inspect_safetensors_cpu(path: Path) -> dict[str, Any]:
    """Pure-Python CPU binary inspector for PEFT .safetensors files (supports BF16/F16/F32)."""
    size = path.stat().st_size
    if size < 16:
        raise ValueError(f"Safetensors file too small ({size} bytes): {path}")

    with path.open("rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        if header_size < 2 or header_size > min(8 * 1024 * 1024, size - 8):
            raise ValueError(f"Invalid safetensors header size {header_size} in {path}")
        header = json.loads(handle.read(header_size).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"Safetensors header must be a JSON object in {path}")

        data_start = 8 + header_size
        spans: list[tuple[int, int, str, dict[str, Any]]] = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"Invalid tensor entry {name!r} in {path}")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if dtype not in _SAFETENSOR_DTYPES:
                raise ValueError(f"Unsupported tensor dtype {dtype!r} for {name!r}")
            if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
                raise ValueError(f"Invalid shape {shape!r} for {name!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(x, int) and x >= 0 for x in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"Invalid data_offsets {offsets!r} for {name!r}")
            itemsize, _ = _SAFETENSOR_DTYPES[dtype]
            count = math.prod(shape) if shape else 1
            if offsets[1] - offsets[0] != count * itemsize:
                raise ValueError(f"Byte-length mismatch for tensor {name!r}")
            spans.append((offsets[0], offsets[1], name, meta))

        spans.sort(key=lambda item: item[0])
        cursor = 0
        tensors_summary: list[dict[str, Any]] = []
        total_nonfinite = 0
        lora_a_nonzero = 0
        lora_b_nonzero = 0
        inferred_ranks: list[int] = []

        for start, end, name, meta in spans:
            if start != cursor:
                raise ValueError(f"Non-contiguous safetensors layout at {name!r}")
            cursor = end
            dtype = meta["dtype"]
            shape = meta["shape"]
            itemsize, fmt = _SAFETENSOR_DTYPES[dtype]

            handle.seek(data_start + start)
            remaining = end - start
            nonzero = 0
            nonfinite = 0
            max_abs = 0.0

            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    raise ValueError(f"Unexpected EOF reading tensor {name!r}")
                remaining -= len(chunk)
                n_items = len(chunk) // itemsize
                if dtype == "BF16":
                    for (bits,) in struct.iter_unpack("<H", chunk):
                        mag = bits & 0x7FFF
                        if mag:
                            nonzero += 1
                        if (bits & 0x7F80) == 0x7F80:
                            nonfinite += 1
                        else:
                            # Convert BF16 to FP32 by left-shifting 16 bits
                            val = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                            abs_v = abs(val)
                            if abs_v > max_abs:
                                max_abs = abs_v
                else:
                    assert fmt is not None
                    for (val,) in struct.iter_unpack(f"<{fmt}", chunk):
                        if isinstance(val, float):
                            if not math.isfinite(val):
                                nonfinite += 1
                            else:
                                if val != 0.0:
                                    nonzero += 1
                                abs_v = abs(val)
                                if abs_v > max_abs:
                                    max_abs = abs_v
                        elif val != 0:
                            nonzero += 1

            total_nonfinite += nonfinite
            low_name = name.lower()
            if "lora_a" in low_name:
                lora_a_nonzero += nonzero
            if "lora_b" in low_name:
                lora_b_nonzero += nonzero
            if ("lora_a" in low_name or "lora_b" in low_name) and len(shape) == 2 and min(shape) > 0:
                inferred_ranks.append(min(shape[0], shape[1]))

            tensors_summary.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "nonzero_elements": nonzero,
                    "nonfinite_elements": nonfinite,
                    "max_abs": round(max_abs, 6),
                }
            )

        if data_start + cursor != size:
            raise ValueError(f"Trailing bytes detected in {path}")

    return {
        "tensor_count": len(tensors_summary),
        "total_nonfinite": total_nonfinite,
        "lora_a_nonzero": lora_a_nonzero,
        "lora_b_nonzero": lora_b_nonzero,
        "is_dummy_zero_lora": (lora_a_nonzero == 0 or lora_b_nonzero == 0),
        "inferred_rank": max(inferred_ranks) if inferred_ranks else None,
        "tensors": tensors_summary,
    }


def find_unknown_adk_state_placeholders(
    instruction: str,
    allowed_keys: frozenset[str] = KNOWN_ADK_SESSION_STATE_KEYS,
) -> list[str]:
    """Detect `{identifier}` substrings that ADK `inject_session_state` will interpolate.

    In Google ADK, single `{var_name}` in an LlmAgent instruction triggers lookup in
    `session.state[var_name]`, raising `KeyError` if `var_name` is missing!
    """
    unknown: list[str] = []
    seen: set[str] = set()
    for match in _ADK_STATE_BRACE_RE.finditer(instruction):
        full = match.group(0)
        key = match.group(1)
        # Double braces {{foo}} are literal-escaped in some templates, single {foo} interpolates
        if full.startswith("{{") and full.endswith("}}"):
            continue
        if key not in allowed_keys and key not in seen:
            seen.add(key)
            unknown.append(key)
    return unknown


def estimate_kaggle_wall_clock(
    model_id: str,
    limits: dict[str, int | float],
    has_active_lora: bool,
    *,
    has_thinking_budget_cap: bool = False,
    has_graph_tools: bool = False,
    total_tasks: int = KAGGLE_TOTAL_TASKS_ESTIMATE,
    effective_parallel_slots: float = 1.25,
) -> dict[str, float | bool]:
    """Estimate 120-task Kaggle evaluation wall-clock hours on 4x NVIDIA L4 (TP=4).

    Calibrated to empirical 4xL4 competition benchmarks (where unbounded 31B w/ graph tools at
    `max_tool_calls=25, max_time_minutes=8` averages ~11.9h/120 tasks, while capping
    `thinking_budget <= 2048` and dropping graph tools reduces per-turn decode time by ~22%).
    Uses Central Limit Theorem variance aggregation (`P95_sum = N*mu + 1.645*sqrt(N)*sigma`)
    rather than assuming all 120 tasks simultaneously hit single-task worst case.
    """
    spec = MODEL_SPECS.get(model_id)
    sec_per_turn = (
        (spec.est_seconds_per_turn_with_lora if has_active_lora else spec.est_seconds_per_turn_no_lora)
        if spec is not None
        else 14.0
    )
    if has_thinking_budget_cap:
        sec_per_turn *= 0.80
    if has_graph_tools:
        sec_per_turn *= 1.12

    max_tool_calls = int(limits["max_tool_calls"])
    max_turns = int(limits["max_turns"])
    max_time_sec = float(limits["max_time_minutes"]) * 60.0

    # Typical coding agent consumes ~0.62 of max_tool_calls on average (+2 final/status turns)
    expected_turns = min(max_turns, max(6.0, 0.62 * max_tool_calls + 2.0))
    container_overhead_sec = 28.0
    expected_agent_sec = min(max_time_sec, expected_turns * sec_per_turn + 18.0)
    mean_task_sec = expected_agent_sec + container_overhead_sec

    # Single-task p90 vs mean gives task-level std dev sigma; aggregate across N independent tasks via CLT
    single_task_cap_sec = max_time_sec + container_overhead_sec
    sigma_task_sec = max(30.0, 0.45 * (single_task_cap_sec - mean_task_sec))

    startup_hours = 0.10
    slots = max(1.0, effective_parallel_slots)
    expected_hours = startup_hours + (total_tasks * mean_task_sec) / (3600.0 * slots)
    p95_sum_sec = total_tasks * mean_task_sec + 1.645 * math.sqrt(total_tasks) * sigma_task_sec
    p90_hours = startup_hours + p95_sum_sec / (3600.0 * slots)
    hard_cap_hours = startup_hours + (total_tasks * single_task_cap_sec) / (3600.0 * slots)

    return {
        "expected_hours_120_tasks": round(expected_hours, 2),
        "p90_hours_120_tasks": round(p90_hours, 2),
        "theoretical_ceiling_hours": round(hard_cap_hours, 2),
        "within_safe_budget": expected_hours <= KAGGLE_SAFE_WALL_CLOCK_HOURS
        and p90_hours <= (KAGGLE_HARD_WALL_CLOCK_HOURS - 0.5),
    }


def _load_yaml_sandboxed(root: Path, rel_path: str, files: dict[str, Path]) -> Any:
    """Parse YAML supporting !include with cycle, depth, duplicate-key, and alias guards."""

    def _parse(current_rel: str, stack: tuple[str, ...]) -> Any:
        if current_rel in stack:
            raise ValueError(f"YAML !include cycle detected: {' -> '.join((*stack, current_rel))}")
        if len(stack) >= 10:
            raise ValueError("YAML !include depth exceeds maximum (10)")
        target_file = files[current_rel]
        raw_text = target_file.read_text(encoding="utf-8")

        class _SandboxedLoader(yaml.SafeLoader):
            pass

        def _include_ctor(loader: yaml.Loader, node: yaml.Node) -> Any:
            if not isinstance(node, yaml.ScalarNode):
                raise ValueError("!include expects a scalar relative path")
            inc_raw = loader.construct_scalar(node)
            inc_posix = PurePosixPath(inc_raw)
            if inc_posix.is_absolute() or ".." in inc_posix.parts or not inc_raw:
                raise ValueError(f"Unsafe !include path: {inc_raw!r}")
            resolved = (root / PurePosixPath(current_rel).parent / inc_posix).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"!include escapes submission root: {inc_raw!r}")
            inc_rel = resolved.relative_to(root).as_posix()
            if inc_rel not in files:
                raise ValueError(f"Missing !include target: {inc_rel}")
            suffix = PurePosixPath(inc_rel).suffix.lower()
            if suffix in {".yaml", ".yml"}:
                return _parse(inc_rel, (*stack, current_rel))
            if suffix in {".md", ".txt", ".jinja", ".rst"}:
                return files[inc_rel].read_text(encoding="utf-8")
            raise ValueError(f"Unsupported !include target extension: {inc_rel}")

        def _unique_mapping(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
            out: dict[Any, Any] = {}
            for k_node, v_node in node.value:
                k = loader.construct_object(k_node, deep=deep)
                if k in out:
                    raise ValueError(f"Duplicate YAML key {k!r} in {current_rel}")
                out[k] = loader.construct_object(v_node, deep=deep)
            return out

        _SandboxedLoader.add_constructor("!include", _include_ctor)
        _SandboxedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
        )
        for ev in yaml.scan(raw_text, Loader=_SandboxedLoader):
            if isinstance(ev, yaml.AliasEvent):
                raise ValueError(f"YAML aliases are forbidden in {current_rel}")
        return yaml.load(raw_text, Loader=_SandboxedLoader)

    return _parse(rel_path, ())


def inspect_submission(
    submission_dir: str | Path,
    *,
    expected_model: str | None = None,
    allow_graph_tools: bool = False,
    allow_dummy_lora: bool = False,
    strict_wall_clock: bool = True,
) -> dict[str, Any]:
    """Comprehensive structural + competition-hazard preflight audit for a submission directory."""
    raw_root = Path(submission_dir)
    if raw_root.is_symlink():
        raise ValueError(f"Submission root cannot be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Submission directory does not exist: {root}")

    files: dict[str, Path] = {}
    total_bytes = 0
    for entry in sorted(root.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"Symlink prohibited inside submission: {entry.relative_to(root)}")
        if entry.is_file():
            rel = entry.relative_to(root).as_posix()
            parts = PurePosixPath(rel).parts
            if any(p.startswith(".") for p in parts):
                raise ValueError(f"Hidden file/directory prohibited: {rel}")
            if parts[-1] in FORBIDDEN_ARCHIVE_FILENAMES:
                raise ValueError(f"Forbidden evaluator filename inside submission: {rel}")
            ext = PurePosixPath(rel).suffix.lower()
            if ext not in ADK_ALLOWED_EXTENSIONS:
                raise ValueError(f"Disallowed file extension {ext!r}: {rel}")
            files[rel] = entry
            total_bytes += entry.stat().st_size

    if len(files) > 10_000:
        raise ValueError(f"Too many files in submission ({len(files)} > 10,000)")
    if total_bytes > KAGGLE_MAX_SUBMISSION_BYTES:
        raise ValueError(f"Submission exceeds 3 GiB limit ({total_bytes:,} bytes)")

    root_configs = [k for k in ("agent.yaml", "root_agent.yaml") if k in files]
    if not root_configs:
        raise ValueError("Missing root 'agent.yaml' (or 'root_agent.yaml')")
    if len(root_configs) > 1:
        raise ValueError(f"Ambiguous root agent configs: {root_configs}")
    primary_config_rel = root_configs[0]

    yaml_rels = [k for k in files if PurePosixPath(k).suffix.lower() in {".yaml", ".yml"}]
    if len(yaml_rels) > 1_000:
        raise ValueError(f"Too many YAML files ({len(yaml_rels)} > 1,000)")

    declared_models: set[str] = set()
    referenced_adapters: set[str] = set()
    declared_tools: set[str] = set()
    warnings: list[str] = []
    issues: list[str] = []

    def _walk_agent_dict(obj: Any, source_rel: str) -> None:
        if isinstance(obj, dict):
            if "model" in obj and obj["model"] is not None:
                m = str(obj["model"]).strip()
                if m not in ALLOWED_COMPETITION_MODELS:
                    raise ValueError(
                        f"Unsupported competition model {m!r} in {source_rel}. "
                        f"Allowed: {sorted(ALLOWED_COMPETITION_MODELS)}"
                    )
                declared_models.add(m)

            if "adapter" in obj and obj["adapter"] is not None:
                ad_name = str(obj["adapter"]).strip()
                if not ad_name or "/" in ad_name or "\\" in ad_name or ".." in ad_name:
                    raise ValueError(f"Invalid adapter identifier {ad_name!r} in {source_rel}")
                referenced_adapters.add(ad_name)

            if "instruction" in obj and isinstance(obj["instruction"], str):
                unknown_placeholders = find_unknown_adk_state_placeholders(obj["instruction"])
                if unknown_placeholders:
                    issues.append(
                        f"{source_rel}: instruction contains unknown ADK state brace placeholder(s) "
                        f"{unknown_placeholders} which will raise KeyError at runtime"
                    )

            if "tools" in obj and isinstance(obj["tools"], list):
                for tool_item in obj["tools"]:
                    if isinstance(tool_item, str):
                        declared_tools.add(tool_item)
                        if tool_item in VLLM_LIST_HANG_TOOLS and not allow_graph_tools:
                            issues.append(
                                f"{source_rel}: tool {tool_item!r} triggers vLLM #753973 infinite hang on list[str] args"
                            )
                        elif tool_item in GRAPH_INTELLIGENCE_TOOLS and not allow_graph_tools:
                            warnings.append(
                                f"{source_rel}: graph tool {tool_item!r} adds prompt bloat and 259k-char context-overflow risk"
                            )

            for v in obj.values():
                _walk_agent_dict(v, source_rel)
        elif isinstance(obj, list):
            for elem in obj:
                _walk_agent_dict(elem, source_rel)

    parsed_yamls: dict[str, Any] = {}
    for y_rel in yaml_rels:
        if files[y_rel].stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"YAML file exceeds 50 MiB limit: {y_rel}")
        parsed = _load_yaml_sandboxed(root, y_rel, files)
        parsed_yamls[y_rel] = parsed
        if y_rel != "eval_config.yaml":
            _walk_agent_dict(parsed, y_rel)

    if not declared_models:
        raise ValueError("No competition model declared in agent YAML configuration")
    if len(declared_models) > 1:
        raise ValueError(
            f"Multiple distinct base models declared across submission ({sorted(declared_models)}); "
            "Kaggle mounts a single base model per submission"
        )

    active_model = next(iter(declared_models))
    if expected_model is not None and active_model != expected_model:
        raise ValueError(f"Declared model {active_model!r} != expected_model {expected_model!r}")

    model_spec = MODEL_SPECS[active_model]

    # Inspect discovered adapters
    discovered_adapters: dict[str, dict[str, Any]] = {}
    for rel, file_path in files.items():
        parts = PurePosixPath(rel).parts
        if len(parts) == 3 and parts[0] == "adapters" and parts[2] == "adapter_model.safetensors":
            ad_name = parts[1]
            cfg_rel = f"adapters/{ad_name}/adapter_config.json"
            if cfg_rel not in files:
                raise ValueError(f"Adapter {ad_name!r} missing {cfg_rel}")
            cfg_json = json.loads(files[cfg_rel].read_text(encoding="utf-8"))
            if cfg_json.get("peft_type") != "LORA":
                raise ValueError(f"Adapter {ad_name!r} peft_type must be 'LORA'")
            rank = cfg_json.get("r")
            if isinstance(rank, bool) or not isinstance(rank, int) or not (1 <= rank <= 128):
                raise ValueError(f"Adapter {ad_name!r} rank r={rank!r} must be int in [1, 128]")
            if rank not in VALID_VLLM_LORA_RANKS:
                warnings.append(
                    f"Adapter {ad_name!r} rank={rank} will be rounded up by vLLM to next bucket in {VALID_VLLM_LORA_RANKS}"
                )
            alpha = float(cfg_json.get("lora_alpha", rank))
            scale_ratio = alpha / float(rank)
            if scale_ratio > 2.0:
                warnings.append(
                    f"Adapter {ad_name!r} has high lora_alpha/r={scale_ratio:.2f} (>2.0); "
                    "on QAT-W4A16 this risks logit-softcap saturation and tool-call syntax drift"
                )

            st_report = inspect_safetensors_cpu(file_path)
            if st_report["total_nonfinite"] > 0:
                issues.append(f"Adapter {ad_name!r} contains NaN/Inf weights!")
            if st_report["is_dummy_zero_lora"] and not allow_dummy_lora:
                issues.append(
                    f"Adapter {ad_name!r} has all-zero LoRA weights (dummy LoRA)! "
                    "Mounting dummy LoRAs forces vLLM --enable-lora Punica overhead (~12% slower) for zero gain."
                )

            # Check forbidden vision/audio tower (Gemma4ClippableLinear) & PLE tensors!
            forbidden_module_tensors = [
                t_meta["name"]
                for t_meta in st_report["tensors"]
                if any(
                    bad in t_meta["name"]
                    for bad in (
                        "vision_tower",
                        "audio_tower",
                        "multi_modal_projector",
                        "per_layer_projection",
                        "per_layer_model_projection",
                    )
                )
            ]
            if forbidden_module_tensors:
                issues.append(
                    f"Adapter {ad_name!r} targets {len(forbidden_module_tensors)} unsupported multimodal/PLE "
                    f"tensors (e.g. {forbidden_module_tensors[0]}) which fail or drop in vLLM Punica!"
                )

            # Check E4B/E2B shared KV layer hazard!
            kv_start, kv_end = model_spec.independent_kv_layers
            if kv_end < model_spec.num_hidden_layers - 1:
                dead_kv_tensors: list[str] = []
                for t_meta in st_report["tensors"]:
                    m = _LAYER_KV_RE.search(t_meta["name"])
                    if m:
                        layer_idx = int(m.group(1))
                        if not (kv_start <= layer_idx <= kv_end):
                            dead_kv_tensors.append(t_meta["name"])
                if dead_kv_tensors:
                    issues.append(
                        f"Adapter {ad_name!r} targets {len(dead_kv_tensors)} shared-KV layer tensors on {active_model} "
                        f"(independent KV is layers {kv_start}..{kv_end} only; e.g. {dead_kv_tensors[0]})"
                    )

            # Check Gemma 4 attention_k_eq_v hazard (e.g. 31B full-attention layers):
            # Every self_attn.k_proj LoRA tensor MUST have a matching self_attn.v_proj LoRA tensor
            # so vLLM QKVParallelLinear preserves V == K!
            tensor_name_set = {t_meta["name"] for t_meta in st_report["tensors"]}
            missing_keqv_vproj = [
                t_name
                for t_name in sorted(tensor_name_set)
                if ".self_attn.k_proj.lora_" in t_name
                and t_name.replace(".self_attn.k_proj.", ".self_attn.v_proj.") not in tensor_name_set
            ]
            if missing_keqv_vproj:
                issues.append(
                    f"Adapter {ad_name!r} has {len(missing_keqv_vproj)} k_proj LoRA tensor(s) without matching v_proj "
                    f"(Gemma 4 attention_k_eq_v hazard in vLLM QKVParallelLinear; e.g. {missing_keqv_vproj[0]}). "
                    "Run patch_keqv_adapter_safetensors()!"
                )

            discovered_adapters[ad_name] = {
                "rank": rank,
                "lora_alpha": alpha,
                "scale_ratio": round(scale_ratio, 4),
                "size_bytes": file_path.stat().st_size,
                "tensor_count": st_report["tensor_count"],
                "is_dummy_zero_lora": st_report["is_dummy_zero_lora"],
            }

    missing_adapters = referenced_adapters - set(discovered_adapters.keys())
    if missing_adapters:
        issues.append(f"Agent config references missing adapter(s): {sorted(missing_adapters)}")
    unreferenced_adapters = set(discovered_adapters.keys()) - referenced_adapters
    if unreferenced_adapters:
        warnings.append(
            f"Unreferenced adapter(s) in adapters/ will still be mounted by vLLM: {sorted(unreferenced_adapters)}"
        )

    if len(discovered_adapters) > 8:
        issues.append(f"Too many adapters discovered ({len(discovered_adapters)} > 8)")

    # Inspect eval_config.yaml & 4xL4 wall-clock projection
    eval_limits: dict[str, int | float] = {
        "timeout_seconds": 120,
        "max_tool_calls": 30,
        "max_time_minutes": 7.0,
        "max_turns": 40,
    }
    if "eval_config.yaml" in parsed_yamls and isinstance(parsed_yamls["eval_config.yaml"], dict):
        raw_eval = parsed_yamls["eval_config.yaml"]
        section = raw_eval.get("evaluation", raw_eval)
        if not isinstance(section, dict):
            issues.append("eval_config.yaml 'evaluation' section must be a mapping")
        else:
            extra_keys = set(section.keys()) - set(EVAL_LIMIT_KEYS)
            if extra_keys:
                issues.append(f"eval_config.yaml contains unrecognized keys: {sorted(extra_keys)}")
            for key in EVAL_LIMIT_KEYS:
                val = section.get(key)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                    issues.append(f"eval_config.yaml '{key}' must be a positive finite number, got {val!r}")
                elif key in {"max_tool_calls", "max_turns"} and not isinstance(val, int):
                    issues.append(f"eval_config.yaml '{key}' must be an integer, got {val!r}")
                else:
                    eval_limits[key] = val

    # Detect whether all LlmAgent configs set a bounded thinking_budget and greedy temperature
    has_tb_cap = False
    llm_tb_flags: list[bool] = []
    max_sampling_temperature = 0.0
    for y_rel, y_doc in parsed_yamls.items():
        if y_rel == "eval_config.yaml" or not isinstance(y_doc, dict):
            continue
        if (y_doc.get("agent_class") or "LlmAgent") == "LlmAgent":
            gcc = y_doc.get("generate_content_config")
            tb_ok = False
            if isinstance(gcc, dict):
                temp_val = gcc.get("temperature")
                if isinstance(temp_val, (int, float)) and not isinstance(temp_val, bool):
                    max_sampling_temperature = max(max_sampling_temperature, float(temp_val))
                if isinstance(gcc.get("thinking_config"), dict):
                    tb_val = gcc["thinking_config"].get("thinking_budget")
                    if isinstance(tb_val, int) and 0 < tb_val <= 4096:
                        tb_ok = True
            llm_tb_flags.append(tb_ok)
    if llm_tb_flags and all(llm_tb_flags):
        has_tb_cap = True
    if max_sampling_temperature > 0.0:
        warnings.append(
            f"Agent YAML uses non-zero sampling temperature ({max_sampling_temperature}); "
            "rollout presets (*_rft) must NOT be used for greedy evaluation or Kaggle submission!"
        )

    wall_clock = estimate_kaggle_wall_clock(
        model_id=active_model,
        limits=eval_limits,
        has_active_lora=bool(discovered_adapters),
        has_thinking_budget_cap=has_tb_cap,
        has_graph_tools=bool(declared_tools & GRAPH_INTELLIGENCE_TOOLS),
    )
    if not wall_clock["within_safe_budget"]:
        msg = (
            f"Projected 120-task 4xL4 runtime ({wall_clock['expected_hours_120_tasks']}h expected, "
            f"{wall_clock['p90_hours_120_tasks']}h p90) risks Kaggle 12.0h timeout!"
        )
        if strict_wall_clock:
            issues.append(msg)
        else:
            warnings.append(msg)

    return {
        "preflight_passed": len(issues) == 0,
        "model": active_model,
        "track": model_spec.track,
        "root_config": primary_config_rel,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "declared_tools": sorted(declared_tools),
        "adapters": discovered_adapters,
        "eval_limits": eval_limits,
        "max_sampling_temperature": max_sampling_temperature,
        "kaggle_4xl4_projection": wall_clock,
        "issues": issues,
        "warnings": warnings,
    }


def pack_submission(
    submission_dir: str | Path,
    destination_zip: str | Path,
    *,
    overwrite: bool = False,
    expected_model: str | None = None,
    allow_graph_tools: bool = False,
    allow_dummy_lora: bool = False,
    strict_wall_clock: bool = True,
    allow_sampling_temperature: bool = False,
) -> dict[str, Any]:
    """Validate and deterministically pack a submission directory into a Kaggle .zip archive."""
    report = inspect_submission(
        submission_dir,
        expected_model=expected_model,
        allow_graph_tools=allow_graph_tools,
        allow_dummy_lora=allow_dummy_lora,
        strict_wall_clock=strict_wall_clock,
    )
    if report.get("max_sampling_temperature", 0.0) > 0.0 and not allow_sampling_temperature:
        report["issues"].append(
            f"Refusing to pack Kaggle submission .zip with non-zero sampling temperature "
            f"(temperature={report['max_sampling_temperature']}); use greedy evaluation preset "
            f"(e.g. budget_aware_v2_self_audit_e4b, NOT *_rft) or pass allow_sampling_temperature=True"
        )
        report["preflight_passed"] = False
    if not report["preflight_passed"]:
        raise ValueError(
            "Submission failed preflight validation:\n  - " + "\n  - ".join(report["issues"])
        )

    root = Path(submission_dir).resolve()
    dest = Path(destination_zip).resolve()
    if dest == root or dest.is_relative_to(root):
        raise ValueError("Destination zip must reside outside submission_dir")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination archive already exists (pass overwrite=True): {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    fixed_dt = (2026, 11, 1, 0, 0, 0)

    all_rel_files = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )
    with zipfile.ZipFile(dest, mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in all_rel_files:
            info = zipfile.ZipInfo(rel, fixed_dt)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            zf.writestr(info, (root / rel).read_bytes())
        corrupt_member = zf.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Corrupt zip entry detected: {corrupt_member}")

    archive_bytes = dest.read_bytes()
    return {
        **report,
        "archive_path": str(dest),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.package")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Audit a submission directory")
    p_pre.add_argument("submission_dir", type=Path)
    p_pre.add_argument("--expected-model", default=None)
    p_pre.add_argument("--allow-graph-tools", action="store_true")
    p_pre.add_argument("--allow-dummy-lora", action="store_true")
    p_pre.add_argument("--no-strict-wall-clock", action="store_true")

    p_pack = sub.add_parser("pack", help="Validate and build deterministic submission .zip")
    p_pack.add_argument("submission_dir", type=Path)
    p_pack.add_argument("destination_zip", type=Path)
    p_pack.add_argument("--overwrite", action="store_true")
    p_pack.add_argument("--expected-model", default=None)
    p_pack.add_argument("--allow-graph-tools", action="store_true")
    p_pack.add_argument("--allow-dummy-lora", action="store_true")
    p_pack.add_argument("--no-strict-wall-clock", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        rep = inspect_submission(
            args.submission_dir,
            expected_model=args.expected_model,
            allow_graph_tools=args.allow_graph_tools,
            allow_dummy_lora=args.allow_dummy_lora,
            strict_wall_clock=not args.no_strict_wall_clock,
        )
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["preflight_passed"] else 1

    rep = pack_submission(
        args.submission_dir,
        args.destination_zip,
        overwrite=args.overwrite,
        expected_model=args.expected_model,
        allow_graph_tools=args.allow_graph_tools,
        allow_dummy_lora=args.allow_dummy_lora,
        strict_wall_clock=not args.no_strict_wall_clock,
    )
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
