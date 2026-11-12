"""Dual RTX PRO 6000 Blackwell (`evalhost`) Server & Parallel Evaluation Orchestrator.

Upgrades the legacy single-GPU `swegemma 0.2.7` runner to full `swegemma 0.2.11` +
`adk-submission 0.2.13` parity across BOTH 96 GiB GPUs (`GPU0` + `GPU1` = 192 GiB VRAM):
  1. Fixes `setup_gemma_model_registry` removal in `swegemma 0.2.11` via native
     `adk_submission 0.2.13` `ModelRegistry` + `install_litellm_reasoning_patch()`.
  2. Supports 0, 1, or N LoRA adapters cleanly (no hardcoded `{'main_lora', 'tool_lora'}`!).
  3. Enforces Gemma 4 W4A16 Marlin + Punica vLLM 0.19.1 invariants (`--dtype bfloat16`,
     `--kv-cache-dtype auto`, `--no-scheduler-reserve-full-isl`, integer `CUDA_VISIBLE_DEVICES`).
  4. Orchestrates 3 dual-GPU topologies:
       - `dual_replica_sharded`: 2x independent replicas (`shard 0/2` + `shard 1/2`) w/ concurrency
       - `ab_head_to_head`: simultaneous Anchor (`GPU0`) vs Candidate (`GPU1`) paired benchmark
       - `opd_teacher_student_cohost`: `GPU0` co-serving E4B student + 31B logprob teacher, `GPU1` training
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

from v2.constants import VALID_VLLM_LORA_RANKS


def round_up_vllm_lora_rank(rank: int) -> int:
    for r_bucket in VALID_VLLM_LORA_RANKS:
        if r_bucket >= rank:
            return r_bucket
    return VALID_VLLM_LORA_RANKS[-1]


@dataclass(frozen=True)
class VllmWorkerSpec:
    worker_id: str
    gpu_index: int
    port: int
    model_path: str
    declared_model_alias: str
    gpu_memory_utilization: float
    max_model_len: int
    lora_modules: dict[str, str]
    max_lora_rank: int | None
    command: list[str]
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class DualGpuTopologyPlan:
    topology: str
    workers: tuple[VllmWorkerSpec, ...]
    eval_commands: tuple[list[str], ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_lora_modules_and_rank(submission_dir: Path | None) -> tuple[dict[str, str], int | None]:
    """Inspect `submission_dir/adapters/` identically to adk_submission 0.2.13 `discover_adapters`."""
    if submission_dir is None:
        return {}, None
    adapters_root = submission_dir.resolve() / "adapters"
    if not adapters_root.is_dir():
        return {}, None

    modules: dict[str, str] = {}
    ranks: list[int] = []
    for cfg_file in sorted(adapters_root.glob("*/adapter_config.json")):
        peft_dir = cfg_file.parent
        sf_file = peft_dir / "adapter_model.safetensors"
        if not sf_file.is_file():
            continue
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        r_val = int(cfg.get("r", 16))
        modules[peft_dir.name] = str(peft_dir)
        ranks.append(r_val)

    max_rank = round_up_vllm_lora_rank(max(ranks)) if ranks else None
    return modules, max_rank


def build_vllm_0213_command(
    *,
    python_bin: str,
    model_path: str,
    port: int,
    host: str = "127.0.0.1",
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    lora_modules: dict[str, str] | None = None,
    max_lora_rank: int | None = None,
    enable_prefix_caching: bool = True,
) -> list[str]:
    """Construct exact `adk_submission 0.2.13` `VllmServer.build_cmd()` CLI arguments for Gemma 4.

    Crucial safety invariants:
      - Omits `--enable-lora` when `lora_modules` is empty so prompt-only harnesses run pure
        Marlin W4A16 kernels without Punica dispatch overhead.
      - Never sets `--kv-cache-dtype fp8` (prevents Gemma 4 hybrid sliding/global + LoRA loop bug).
      - Always sets `--no-scheduler-reserve-full-isl` so hybrid sliding-window requests never stall.
    """
    reasoning_cfg = json.dumps(
        {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}
    )
    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--reasoning-config",
        reasoning_cfg,
        "--no-scheduler-reserve-full-isl",
        # Returned logprobs (only when a client asks) are of the sampling distribution (after temperature), so
        # captured rollout logprobs are the exact old-policy logprobs for RL; sampling itself is unchanged.
        "--logprobs-mode",
        "processed_logprobs",
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if lora_modules:
        eff_rank = round_up_vllm_lora_rank(max_lora_rank or 64)
        cmd.extend(
            [
                "--enable-lora",
                "--max-loras",
                str(len(lora_modules)),
                "--max-lora-rank",
                str(eff_rank),
                "--lora-modules",
            ]
        )
        for name, path in sorted(lora_modules.items()):
            cmd.append(f"{name}={path}")

    return cmd


def build_swegemma_0211_model_registry(
    *,
    api_base: str,
    declared_model_alias: str,
    served_model_id: str,
    submission_dir: Path,
    api_key: str = "EMPTY",
    num_retries: int = 5,
) -> Any:
    """Drop-in replacement for removed `swegemma.models.setup_gemma_model_registry` on swegemma 0.2.11.

    Installs `adk_submission 0.2.13` LiteLLM reasoning + nullable-schema patches and binds
    `declared_model_alias` plus all discovered LoRA routes (`name` and `adapter:name`).
    """
    from adk_submission import ModelRegistry, discover_adapters
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from google.adk.models.lite_llm import LiteLlm

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    install_litellm_reasoning_patch()

    registry = ModelRegistry()
    target_base = (
        served_model_id
        if served_model_id.startswith(("openai/", "hosted_vllm/", "custom/"))
        else f"openai/{served_model_id}"
    )
    registry.register(
        declared_model_alias,
        LiteLlm(
            model=target_base,
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
        ),
    )

    manifest = discover_adapters(submission_dir)
    for name in manifest.adapters:
        lora_client = LiteLlm(
            model=f"openai/{name}",
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
        )
        registry.register(name, lora_client)
        registry.register(f"adapter:{name}", lora_client)

    return registry


def build_dual_gpu_topology_plan(
    topology: str,
    *,
    python_bin: str = "python3",
    primary_model_path: str,
    primary_declared_alias: str,
    primary_submission_dir: Path | None = None,
    secondary_model_path: str | None = None,
    secondary_declared_alias: str | None = None,
    secondary_submission_dir: Path | None = None,
    tasks_path: Path = Path("tasks.jsonl"),
    snapshots_dir: Path = Path("snapshots"),
    results_root: Path = Path("results/v2_run"),
    sandbox_image: str = "swebench-sandbox:latest",
    concurrency_per_gpu: int = 4,
    base_port: int = 18860,
) -> DualGpuTopologyPlan:
    """Create ready-to-launch server + evaluator commands for both RTX PRO 6000 GPUs on evalhost."""
    base_env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
        "OTEL_SDK_DISABLED": "true",
    }

    if topology == "dual_replica_sharded":
        loras, max_r = discover_lora_modules_and_rank(primary_submission_dir)
        workers: list[VllmWorkerSpec] = []
        eval_cmds: list[list[str]] = []
        sub_dir_str = str(primary_submission_dir or Path("submission"))

        for shard_idx, gpu_idx in enumerate((0, 1)):
            port = base_port + shard_idx
            cmd = build_vllm_0213_command(
                python_bin=python_bin,
                model_path=primary_model_path,
                port=port,
                gpu_memory_utilization=0.90,
                lora_modules=loras,
                max_lora_rank=max_r,
            )
            workers.append(
                VllmWorkerSpec(
                    worker_id=f"replica_gpu{gpu_idx}",
                    gpu_index=gpu_idx,
                    port=port,
                    model_path=primary_model_path,
                    declared_model_alias=primary_declared_alias,
                    gpu_memory_utilization=0.90,
                    max_model_len=32768,
                    lora_modules=loras,
                    max_lora_rank=max_r,
                    command=cmd,
                    env_overrides={**base_env, "CUDA_VISIBLE_DEVICES": str(gpu_idx)},
                )
            )
            eval_cmds.append(
                [
                    python_bin,
                    "-m",
                    "swegemma.cli",
                    "eval",
                    "--tasks",
                    str(tasks_path),
                    "--snapshots-dir",
                    str(snapshots_dir),
                    "--results-dir",
                    str(results_root / f"shard_{shard_idx}"),
                    "--submission-dir",
                    sub_dir_str,
                    "--image",
                    sandbox_image,
                    "--concurrency",
                    str(concurrency_per_gpu),
                    "--shard-index",
                    str(shard_idx),
                    "--num-shards",
                    "2",
                    "--display",
                    "quiet",
                ]
            )

        return DualGpuTopologyPlan(
            topology=topology,
            workers=tuple(workers),
            eval_commands=tuple(eval_cmds),
            notes=(
                f"Runs 2 independent replicas across GPU0 + GPU1 ({2 * concurrency_per_gpu} concurrent sandboxes).",
                "Set OPENAI_BASE_URL=http://127.0.0.1:<port>/v1 per shard process.",
            ),
        )

    if topology == "ab_head_to_head":
        sec_model = secondary_model_path or primary_model_path
        sec_alias = secondary_declared_alias or primary_declared_alias
        sec_sub = secondary_submission_dir or primary_submission_dir

        loras_a, max_ra = discover_lora_modules_and_rank(primary_submission_dir)
        loras_b, max_rb = discover_lora_modules_and_rank(sec_sub)

        w0 = VllmWorkerSpec(
            worker_id="anchor_gpu0",
            gpu_index=0,
            port=base_port,
            model_path=primary_model_path,
            declared_model_alias=primary_declared_alias,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            lora_modules=loras_a,
            max_lora_rank=max_ra,
            command=build_vllm_0213_command(
                python_bin=python_bin,
                model_path=primary_model_path,
                port=base_port,
                lora_modules=loras_a,
                max_lora_rank=max_ra,
            ),
            env_overrides={**base_env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        w1 = VllmWorkerSpec(
            worker_id="candidate_gpu1",
            gpu_index=1,
            port=base_port + 1,
            model_path=sec_model,
            declared_model_alias=sec_alias,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            lora_modules=loras_b,
            max_lora_rank=max_rb,
            command=build_vllm_0213_command(
                python_bin=python_bin,
                model_path=sec_model,
                port=base_port + 1,
                lora_modules=loras_b,
                max_lora_rank=max_rb,
            ),
            env_overrides={**base_env, "CUDA_VISIBLE_DEVICES": "1"},
        )
        eval_a = [
            python_bin,
            "-m",
            "swegemma.cli",
            "eval",
            "--tasks",
            str(tasks_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--results-dir",
            str(results_root / "anchor"),
            "--submission-dir",
            str(primary_submission_dir or Path("anchor_sub")),
            "--image",
            sandbox_image,
            "--concurrency",
            str(concurrency_per_gpu),
            "--display",
            "quiet",
        ]
        eval_b = [
            python_bin,
            "-m",
            "swegemma.cli",
            "eval",
            "--tasks",
            str(tasks_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--results-dir",
            str(results_root / "candidate"),
            "--submission-dir",
            str(sec_sub or Path("candidate_sub")),
            "--image",
            sandbox_image,
            "--concurrency",
            str(concurrency_per_gpu),
            "--display",
            "quiet",
        ]
        return DualGpuTopologyPlan(
            topology=topology,
            workers=(w0, w1),
            eval_commands=(eval_a, eval_b),
            notes=(
                "Simultaneous paired A/B evaluation: GPU0 runs Anchor while GPU1 runs Candidate on identical tasks.",
            ),
        )

    if topology == "opd_teacher_student_cohost":
        if not secondary_model_path:
            raise ValueError("opd_teacher_student_cohost requires secondary_model_path (31B teacher)")
        student_loras, student_r = discover_lora_modules_and_rank(primary_submission_dir)
        teacher_loras, teacher_r = discover_lora_modules_and_rank(secondary_submission_dir)

        # Both E4B (~9.92 GiB PLE+W4A16) and 31B (~22 GiB W4A16) fit comfortably on GPU0 (96 GiB)!
        w_student = VllmWorkerSpec(
            worker_id="e4b_rollout_gpu0",
            gpu_index=0,
            port=base_port,
            model_path=primary_model_path,
            declared_model_alias=primary_declared_alias,
            gpu_memory_utilization=0.34,  # ~32.6 GiB (9.92 GiB weights + ~22.7 GiB KV cache)
            max_model_len=32768,
            lora_modules=student_loras,
            max_lora_rank=student_r or 64,
            command=build_vllm_0213_command(
                python_bin=python_bin,
                model_path=primary_model_path,
                port=base_port,
                gpu_memory_utilization=0.34,
                lora_modules=student_loras,
                max_lora_rank=student_r or 64,
            ),
            env_overrides={**base_env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        w_teacher = VllmWorkerSpec(
            worker_id="31b_teacher_prefill_gpu0",
            gpu_index=0,
            port=base_port + 1,
            model_path=secondary_model_path,
            declared_model_alias=secondary_declared_alias or "gemma-4-31b-it-qat-w4a16-ct",
            gpu_memory_utilization=0.58,  # ~55.7 GiB (18.5 GiB weights + ~37 GiB prefill KV cache)
            max_model_len=32768,
            lora_modules=teacher_loras,
            max_lora_rank=teacher_r,
            command=build_vllm_0213_command(
                python_bin=python_bin,
                model_path=secondary_model_path,
                port=base_port + 1,
                gpu_memory_utilization=0.58,
                lora_modules=teacher_loras,
                max_lora_rank=teacher_r,
            ),
            env_overrides={**base_env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        trainer_cmd = [
            python_bin,
            "-m",
            "v2.train.distill_and_rl",
            "--mode",
            "opd_topk",
            "--student-endpoint",
            f"http://127.0.0.1:{base_port}/v1",
            "--teacher-endpoint",
            f"http://127.0.0.1:{base_port + 1}/v1",
            "--train-device",
            "cuda:0",  # Under CUDA_VISIBLE_DEVICES=1
        ]
        return DualGpuTopologyPlan(
            topology=topology,
            workers=(w_student, w_teacher),
            eval_commands=(trainer_cmd,),
            notes=(
                "GPU0 (96 GiB) co-hosts E4B rollout engine (30% VRAM) + 31B-RL teacher prefill engine (58% VRAM).",
                "GPU1 (96 GiB, CUDA_VISIBLE_DEVICES=1) runs PyTorch W_dequant BF16 + LoRA (r=64) student updates.",
            ),
        )

    raise ValueError(f"Unknown topology {topology!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.eval.gpu_runner")
    parser.add_argument(
        "--topology",
        choices=("dual_replica_sharded", "ab_head_to_head", "opd_teacher_student_cohost"),
        required=True,
    )
    parser.add_argument("--primary-model-path", required=True)
    parser.add_argument("--primary-alias", default="gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--primary-submission", type=Path, default=None)
    parser.add_argument("--secondary-model-path", default=None)
    parser.add_argument("--secondary-alias", default=None)
    parser.add_argument("--secondary-submission", type=Path, default=None)
    parser.add_argument("--concurrency-per-gpu", type=int, default=4)
    args = parser.parse_args(argv)

    plan = build_dual_gpu_topology_plan(
        args.topology,
        primary_model_path=args.primary_model_path,
        primary_declared_alias=args.primary_alias,
        primary_submission_dir=args.primary_submission,
        secondary_model_path=args.secondary_model_path,
        secondary_declared_alias=args.secondary_alias,
        secondary_submission_dir=args.secondary_submission,
        concurrency_per_gpu=args.concurrency_per_gpu,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
