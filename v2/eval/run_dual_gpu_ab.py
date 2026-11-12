"""Live Dual-GPU (`GPU0` + `GPU1`) `swegemma 0.2.11` Runner with Exact Kaggle `inference.py` Parity.

Manages two isolated `vLLM` servers across `GPU0` and `GPU1` on `evalhost` (`2 x RTX PRO 6000`),
configures exact Kaggle `scripts/inference.py` session compaction & `eval_config.yaml` budgets,
and executes either:
- `ab_head_to_head`: simultaneous Anchor (`GPU0`) vs Candidate (`GPU1`) on identical tasks + paired McNemar/bootstrap gate.
- `dual_replica_sharded`: Shard `0/2` (`GPU0`) + Shard `1/2` (`GPU1`) for 2x throughput benchmark / trajectory collection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import yaml

from v2.eval.governance import evaluate_paired_promotion, summarize_failure_modes
from v2.eval.gpu_launch_lock import server_popen
from v2.eval.gpu_runner import (
    build_swegemma_0211_model_registry,
    build_vllm_0213_command,
    discover_lora_modules_and_rank,
)


def load_submission_eval_budgets(submission_dir: Path) -> dict[str, Any]:
    """Replicate Kaggle `scripts/inference.py` `load_submission_eval_config()` defaults + overrides."""
    defaults: dict[str, Any] = {
        "timeout_seconds": 300,
        "max_tool_calls": 100,
        "max_time_minutes": 60.0,
        "max_turns": 500,
    }
    cfg_path = submission_dir / "eval_config.yaml"
    if not cfg_path.is_file():
        return defaults
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    ev = raw.get("evaluation") if isinstance(raw, dict) else None
    if isinstance(ev, dict):
        if ev.get("timeout_seconds") is not None:
            defaults["timeout_seconds"] = int(ev["timeout_seconds"])
        if ev.get("max_tool_calls") is not None:
            defaults["max_tool_calls"] = int(ev["max_tool_calls"])
        if ev.get("max_time_minutes") is not None:
            defaults["max_time_minutes"] = float(ev["max_time_minutes"])
        if ev.get("max_turns") is not None:
            defaults["max_turns"] = int(ev["max_turns"])
    return defaults


def build_kaggle_adk_session_configs() -> dict[str, Any]:
    """Return exact Kaggle `scripts/inference.py` ContextCacheConfig & EventsCompactionConfig."""
    out: dict[str, Any] = {}
    try:
        from google.adk.agents.context_cache_config import ContextCacheConfig

        out["context_cache_config"] = ContextCacheConfig(
            min_tokens=2048,
            ttl_seconds=1800,
            cache_intervals=10,
        )
    except Exception:
        pass

    EventsCompactionConfig = None
    try:
        from google.adk.apps._configs import EventsCompactionConfig as _ECC

        EventsCompactionConfig = _ECC
    except ImportError:
        try:
            from google.adk.apps.app import EventsCompactionConfig as _ECC2

            EventsCompactionConfig = _ECC2
        except ImportError:
            pass
    if EventsCompactionConfig is not None:
        try:
            out["events_compaction_config"] = EventsCompactionConfig(
                compaction_interval=5,
                overlap_size=2,
                token_threshold=14336,
                event_retention_size=5,
            )
        except Exception:
            pass
    return out


def wait_for_vllm_health(port: int, proc: subprocess.Popen[Any], timeout_sec: float = 900.0) -> float:
    """Poll `http://127.0.0.1:<port>/health` until HTTP 200 or server exit (then release the GPU launch lock that
    `gpu_launch_lock.server_popen` may have attached to `proc`)."""
    from v2.eval.gpu_launch_lock import release_launch_lock

    try:
        return _wait_for_vllm_health(port, proc, timeout_sec)
    finally:
        release_launch_lock(proc)


def _wait_for_vllm_health(port: int, proc: subprocess.Popen[Any], timeout_sec: float) -> float:
    t0 = time.monotonic()
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() - t0 < timeout_sec:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"vLLM server on port {port} exited prematurely with code {rc}")
        try:
            with urllib.request.urlopen(url, timeout=2.5) as resp:
                if resp.status == 200:
                    return round(time.monotonic() - t0, 2)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server on port {port} did not become healthy within {timeout_sec}s")


def terminate_process_group(proc: subprocess.Popen[Any] | None) -> None:
    """Cleanly terminate a `start_new_session=True` server process group."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass


async def run_single_swegemma_worker(
    *,
    tasks_path: Path,
    snapshots_dir: Path,
    wheels_dir: Path | None,
    results_dir: Path,
    submission_dir: Path,
    sandbox_image: str,
    api_base: str,
    served_model_id: str,
    declared_model_alias: str,
    concurrency: int,
    task_ids: list[str] | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict[str, Any]:
    """Run an in-process `swegemma 0.2.11` `Evaluator` against a dedicated vLLM endpoint."""
    from adk_submission import discover_adapters
    from swegemma.evaluate import EvalConfig, Evaluator

    results_dir.mkdir(parents=True, exist_ok=True)
    budgets = load_submission_eval_budgets(submission_dir)
    session_cfgs = build_kaggle_adk_session_configs()
    adapters = discover_adapters(submission_dir)
    models = build_swegemma_0211_model_registry(
        api_base=api_base,
        served_model_id=served_model_id,
        declared_model_alias=declared_model_alias,
        submission_dir=submission_dir,
    )

    cfg = EvalConfig(
        tasks_path=tasks_path,
        snapshots_dir=snapshots_dir,
        wheels_dir=wheels_dir,
        results_dir=results_dir,
        submission_dir=submission_dir,
        models=models,
        image=sandbox_image,
        adapter_manifest=adapters,
        concurrency=concurrency,
        shard_index=shard_index,
        num_shards=num_shards,
        task_ids=task_ids,
        display_mode="quiet",
        **budgets,
        **session_cfgs,
    )
    t0 = time.monotonic()
    evaluator = Evaluator(cfg)
    ev_res = await evaluator.run()
    elapsed = round(time.monotonic() - t0, 2)
    return {
        "results_dir": str(results_dir),
        "resolved": int(ev_res.resolved),
        "total": int(ev_res.total),
        "resolution_rate": round(float(ev_res.resolution_rate), 4),
        "elapsed_seconds": elapsed,
        "budgets": budgets,
    }


def _run_worker_entry(kwargs_json: str) -> int:
    """Subprocess entrypoint so each GPU evaluator has isolated LiteLLM/env state."""
    kw = json.loads(kwargs_json)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "OTEL_SDK_DISABLED": "true",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_TELEMETRY": "False",
            "TOKENIZERS_PARALLELISM": "false",
            "OPENAI_BASE_URL": kw["api_base"],
            "OPENAI_API_KEY": "EMPTY",
        }
    )
    summary = asyncio.run(
        run_single_swegemma_worker(
            tasks_path=Path(kw["tasks_path"]),
            snapshots_dir=Path(kw["snapshots_dir"]),
            wheels_dir=Path(kw["wheels_dir"]) if kw.get("wheels_dir") else None,
            results_dir=Path(kw["results_dir"]),
            submission_dir=Path(kw["submission_dir"]),
            sandbox_image=kw["sandbox_image"],
            api_base=kw["api_base"],
            served_model_id=kw["served_model_id"],
            declared_model_alias=kw["declared_model_alias"],
            concurrency=int(kw["concurrency"]),
            task_ids=kw.get("task_ids"),
            shard_index=kw.get("shard_index"),
            num_shards=kw.get("num_shards"),
        )
    )
    (Path(kw["results_dir"]) / "worker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


def load_task_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Read raw task result rows from `task_results.jsonl`."""
    rows: list[dict[str, Any]] = []
    jsonl = results_dir / "task_results.jsonl"
    if not jsonl.is_file():
        return rows
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = str(row["instance_id"])
        rows.append(
            {
                "instance_id": iid,
                "repo": str(row.get("repo") or iid.split("_")[0]),
                "resolved": bool(row.get("resolved", False)),
                "error": row.get("error"),
                "tool_calls": int(row.get("tool_calls", 0) or 0),
                "total_llm_calls": int(row.get("total_llm_calls", 0) or 0),
                "duration_seconds": float(row.get("duration_seconds", 0.0) or 0.0),
            }
        )
    return rows


def verify_docker_sandbox_image_exists(image_tag: str) -> None:
    """Fail immediately before booting vLLM if the Docker sandbox image tag is missing locally."""
    res = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        raise RuntimeError(
            f"Eval preflight failed: local Docker sandbox image {image_tag!r} not found "
            f"(docker image inspect rc={res.returncode}: {err})"
        )


def verify_greedy_eval_temperature(submission_dir: Path) -> None:
    """Block accidental evaluation of *_rft rollout-sampling harnesses (temperature > 0.0)."""
    from v2.package import inspect_submission

    report = inspect_submission(
        submission_dir,
        allow_graph_tools=True,
        allow_dummy_lora=True,
        strict_wall_clock=False,
    )
    max_temp = float(report.get("max_sampling_temperature", 0.0) or 0.0)
    if max_temp > 0.0:
        raise RuntimeError(
            f"Eval preflight failed: submission {submission_dir} sets sampling temperature={max_temp} "
            f"(rollout *_rft preset)! Benchmark evaluations must use temperature=0.0 greedy presets "
            f"(or pass --allow-sampling-temperature explicitly)."
        )


def orchestrate_dual_gpu(args: argparse.Namespace) -> dict[str, Any]:
    """Start vLLM server(s) on GPU0/GPU1, run parallel swegemma workers, and clean up."""
    if not getattr(args, "skip_docker_check", False):
        verify_docker_sandbox_image_exists(args.image)
    pri_sub_check = args.primary_submission.resolve()
    sec_sub_check = (args.secondary_submission or args.primary_submission).resolve()
    if not getattr(args, "allow_sampling_temperature", False):
        verify_greedy_eval_temperature(pri_sub_check)
        if args.mode == "ab_head_to_head":
            verify_greedy_eval_temperature(sec_sub_check)
    out_root = args.results_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sec_model = args.secondary_model_path or args.primary_model_path
    sec_alias = args.secondary_alias or args.primary_alias
    sec_sub = (args.secondary_submission or args.primary_submission).resolve()
    pri_sub = args.primary_submission.resolve()

    loras_0, max_r0 = discover_lora_modules_and_rank(pri_sub)
    loras_1, max_r1 = discover_lora_modules_and_rank(sec_sub)

    cmd_0 = build_vllm_0213_command(
        python_bin=args.vllm_python,
        model_path=args.primary_model_path,
        port=args.base_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_modules=loras_0,
        max_lora_rank=max_r0,
    )
    cmd_1 = build_vllm_0213_command(
        python_bin=args.vllm_python,
        model_path=sec_model,
        port=args.base_port + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_modules=loras_1,
        max_lora_rank=max_r1,
    )

    base_server_env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SETUPTOOLS_USE_DISTUTILS": "stdlib",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OTEL_SDK_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.server_pythonpath:
        base_server_env["PYTHONPATH"] = str(args.server_pythonpath)
    if args.hf_home:
        base_server_env["HF_HOME"] = str(args.hf_home)
    if args.vllm_cache_root:
        base_server_env["VLLM_CACHE_ROOT"] = str(args.vllm_cache_root)

    proc_0: subprocess.Popen[Any] | None = None
    proc_1: subprocess.Popen[Any] | None = None
    try:
        common = {
            "tasks_path": str(args.tasks.resolve()),
            "snapshots_dir": str(args.snapshots_dir.resolve()),
            "wheels_dir": str(args.wheels_dir.resolve()) if args.wheels_dir else None,
            "sandbox_image": args.image,
            "concurrency": args.concurrency_per_gpu,
            "task_ids": args.task_ids,
        }

        if args.mode == "single_gpu":
            gpu_idx = int(getattr(args, "single_gpu_index", 0))
            log0 = (logs_dir / f"vllm_gpu{gpu_idx}.log").open("w", encoding="utf-8")
            proc_0 = server_popen(
                cmd_0,
                env={**base_server_env, "CUDA_VISIBLE_DEVICES": str(gpu_idx)},
                stdout=log0,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            startup_0 = wait_for_vllm_health(args.base_port, proc_0, timeout_sec=args.startup_timeout)
            w0_cfg = {
                **common,
                "results_dir": str(out_root / f"single_gpu{gpu_idx}"),
                "submission_dir": str(pri_sub),
                "api_base": f"http://127.0.0.1:{args.base_port}/v1",
                "served_model_id": args.primary_model_path,
                "declared_model_alias": args.primary_alias,
            }
            ew0_log = (logs_dir / f"eval_worker_gpu{gpu_idx}.log").open("w", encoding="utf-8")
            ew0 = subprocess.Popen(
                [sys.executable, "-m", "v2.eval.run_dual_gpu_ab", "--internal-worker-json", json.dumps(w0_cfg)],
                stdout=ew0_log,
                stderr=subprocess.STDOUT,
            )
            rc0 = ew0.wait()
            if rc0 != 0:
                raise RuntimeError(f"Single-GPU evaluator worker failed (rc0={rc0}); inspect {logs_dir}")
            s0 = json.loads((Path(w0_cfg["results_dir"]) / "worker_summary.json").read_text(encoding="utf-8"))
            rows_a = load_task_rows(Path(w0_cfg["results_dir"]))
            s0["worker_errors"] = sum(1 for r in rows_a if r.get("error"))
            s0["mean_tool_calls"] = (
                round(sum(r["tool_calls"] for r in rows_a) / len(rows_a), 2) if rows_a else 0.0
            )
            report_single: dict[str, Any] = {
                "mode": args.mode,
                "gpu_index": gpu_idx,
                "vllm_startup_seconds": {f"gpu{gpu_idx}": startup_0},
                "worker_0": s0,
                "combined": {
                    "resolved": s0["resolved"],
                    "total": s0["total"],
                    "resolution_rate": round(s0["resolved"] / s0["total"], 4) if s0["total"] else 0.0,
                },
            }
            (out_root / "dual_gpu_report.json").write_text(
                json.dumps(report_single, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            return report_single

        log0 = (logs_dir / "vllm_gpu0.log").open("w", encoding="utf-8")
        log1 = (logs_dir / "vllm_gpu1.log").open("w", encoding="utf-8")
        proc_0 = server_popen(
            cmd_0,
            env={**base_server_env, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=log0,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc_1 = server_popen(
            cmd_1,
            env={**base_server_env, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=log1,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        startup_0 = wait_for_vllm_health(args.base_port, proc_0, timeout_sec=args.startup_timeout)
        startup_1 = wait_for_vllm_health(args.base_port + 1, proc_1, timeout_sec=args.startup_timeout)

        if args.mode == "ab_head_to_head":
            w0_cfg = {
                **common,
                "results_dir": str(out_root / "anchor_gpu0"),
                "submission_dir": str(pri_sub),
                "api_base": f"http://127.0.0.1:{args.base_port}/v1",
                "served_model_id": args.primary_model_path,
                "declared_model_alias": args.primary_alias,
            }
            w1_cfg = {
                **common,
                "results_dir": str(out_root / "candidate_gpu1"),
                "submission_dir": str(sec_sub),
                "api_base": f"http://127.0.0.1:{args.base_port + 1}/v1",
                "served_model_id": sec_model,
                "declared_model_alias": sec_alias,
            }
        else:
            w0_cfg = {
                **common,
                "results_dir": str(out_root / "shard_0"),
                "submission_dir": str(pri_sub),
                "api_base": f"http://127.0.0.1:{args.base_port}/v1",
                "served_model_id": args.primary_model_path,
                "declared_model_alias": args.primary_alias,
                "shard_index": 0,
                "num_shards": 2,
            }
            w1_cfg = {
                **common,
                "results_dir": str(out_root / "shard_1"),
                "submission_dir": str(pri_sub),
                "api_base": f"http://127.0.0.1:{args.base_port + 1}/v1",
                "served_model_id": args.primary_model_path,
                "declared_model_alias": args.primary_alias,
                "shard_index": 1,
                "num_shards": 2,
            }

        ew0_log = (logs_dir / "eval_worker0.log").open("w", encoding="utf-8")
        ew1_log = (logs_dir / "eval_worker1.log").open("w", encoding="utf-8")
        ew0 = subprocess.Popen(
            [sys.executable, "-m", "v2.eval.run_dual_gpu_ab", "--internal-worker-json", json.dumps(w0_cfg)],
            stdout=ew0_log,
            stderr=subprocess.STDOUT,
        )
        ew1 = subprocess.Popen(
            [sys.executable, "-m", "v2.eval.run_dual_gpu_ab", "--internal-worker-json", json.dumps(w1_cfg)],
            stdout=ew1_log,
            stderr=subprocess.STDOUT,
        )
        rc0 = ew0.wait()
        rc1 = ew1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Evaluator worker failed (rc0={rc0}, rc1={rc1}); inspect {logs_dir}")

        s0 = json.loads((Path(w0_cfg["results_dir"]) / "worker_summary.json").read_text(encoding="utf-8"))
        s1 = json.loads((Path(w1_cfg["results_dir"]) / "worker_summary.json").read_text(encoding="utf-8"))

        report: dict[str, Any] = {
            "mode": args.mode,
            "vllm_startup_seconds": {"gpu0": startup_0, "gpu1": startup_1},
            "worker_0": s0,
            "worker_1": s1,
        }

        if args.mode == "ab_head_to_head":
            rows_a = load_task_rows(Path(w0_cfg["results_dir"]))
            rows_b = load_task_rows(Path(w1_cfg["results_dir"]))
            report["worker_0"]["worker_errors"] = sum(1 for r in rows_a if r.get("error"))
            report["worker_1"]["worker_errors"] = sum(1 for r in rows_b if r.get("error"))
            report["worker_0"]["mean_tool_calls"] = (
                round(sum(r["tool_calls"] for r in rows_a) / len(rows_a), 2) if rows_a else 0.0
            )
            report["worker_1"]["mean_tool_calls"] = (
                round(sum(r["tool_calls"] for r in rows_b) / len(rows_b), 2) if rows_b else 0.0
            )
            report["worker_0"]["failure_modes"] = summarize_failure_modes(rows_a)
            report["worker_1"]["failure_modes"] = summarize_failure_modes(rows_b)
            ids_a = {r["instance_id"] for r in rows_a}
            ids_b = {r["instance_id"] for r in rows_b}
            if rows_a and ids_a == ids_b:
                est_120h = round(
                    (float(s1["elapsed_seconds"]) * (120.0 / max(1, int(s1["total"])))) / 3600.0,
                    3,
                )
                promo = evaluate_paired_promotion(
                    anchor_runs=[rows_a],
                    candidate_runs=[rows_b],
                    projected_120task_hours=est_120h,
                    min_seeds=1,
                )
                report["paired_promotion"] = promo.to_dict()
        else:
            total = s0["total"] + s1["total"]
            resolved = s0["resolved"] + s1["resolved"]
            report["combined"] = {
                "resolved": resolved,
                "total": total,
                "resolution_rate": round(resolved / total, 4) if total else 0.0,
            }

        (out_root / "dual_gpu_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return report
    finally:
        terminate_process_group(proc_0)
        terminate_process_group(proc_1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.eval.run_dual_gpu_ab")
    parser.add_argument("--internal-worker-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=("ab_head_to_head", "dual_replica_sharded", "single_gpu"),
        default="ab_head_to_head",
    )
    parser.add_argument("--single-gpu-index", type=int, default=0)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--primary-model-path", required=False)
    parser.add_argument("--primary-alias", default="gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--primary-submission", type=Path, required=False)
    parser.add_argument("--secondary-model-path", default=None)
    parser.add_argument("--secondary-alias", default=None)
    parser.add_argument("--secondary-submission", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, required=False)
    parser.add_argument("--snapshots-dir", type=Path, required=False)
    parser.add_argument("--wheels-dir", type=Path, default=None)
    parser.add_argument("--image", default="swegemma-sandbox:v2-local")
    parser.add_argument("--skip-docker-check", action="store_true")
    parser.add_argument("--allow-sampling-temperature", action="store_true")
    parser.add_argument("--results-root", type=Path, required=False)
    parser.add_argument("--concurrency-per-gpu", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=18900)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--vllm-cache-root", type=Path, default=None)
    default_py_shim = "/path/to/gemma-v2/py_shim" if Path("/path/to/gemma-v2/py_shim").is_dir() else None
    parser.add_argument("--server-pythonpath", type=str, default=default_py_shim)
    parser.add_argument("--task-ids", nargs="*", default=None)
    args = parser.parse_args(argv)

    if args.internal_worker_json:
        return _run_worker_entry(args.internal_worker_json)

    for req_name in ("primary_model_path", "primary_submission", "tasks", "snapshots_dir", "results_root"):
        if getattr(args, req_name) is None:
            parser.error(f"--{req_name.replace('_', '-')} is required")

    report = orchestrate_dual_gpu(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
