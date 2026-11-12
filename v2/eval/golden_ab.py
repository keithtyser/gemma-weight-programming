"""Paired A/B on a golden eval set (`golden_v1`, or `golden_v2` with `--golden v2`): competition + held-out SWE-rebench.

`golden_v1` has two frozen parts (manifest `v2/experiments/splits/golden_v1_manifest.json`):
  comp     public competition tasks (the leaderboard's own repos / pipeline) that pass empty+gold controls,
           graded by the official swegemma path via `run_dual_gpu_ab` (one invocation per seed)
  rebench  `evalset_v1`: sealed SWE-rebench tasks on repos disjoint from every training pool, graded by
           `run_rebench_v2 --mode ab_paired --num-rollouts <seeds>`

Anchor runs on GPU0, candidate on GPU1, both greedy. Promotion requires the pooled paired gate
(`evaluate_paired_promotion`: McNemar p <= 0.10, repo-cluster CI90 lower bound >= 0, flaky tasks excluded) AND a
non-negative mean delta on the comp part. Both parts and their failure modes are always reported.

  python -m v2.eval.golden_ab --anchor <harness_dir> --candidate <harness_dir> --out results/<name> --seeds 2
  python -m v2.eval.golden_ab --golden v2 ...        # golden_v2: manifest, data root and comp image from the manifest
  python -m v2.eval.golden_ab ... --report-only      # recompute the decision from existing outputs

Faster, cached form: produce each harness's results once on BOTH GPUs (an "arm"), then decide from two arms.
The base anchor arm is reused across candidates as long as its harness fingerprint and the golden manifest match.
  python -m v2.eval.golden_ab --arm <harness_dir> --out results/arms/<name> --seeds 2
  python -m v2.eval.golden_ab --anchor-arm results/arms/base --candidate-arm results/arms/<cand> --out results/<ab>
Pairing is by task and seed index exactly as in the paired run (greedy runs; seeds are reruns).
An arm can also run against an ALREADY-RUNNING server (e.g. the async RL inference server, adapter hot-loaded under
its own name), with no server start/stop:
  python -m v2.eval.golden_ab --arm <greedy harness> --external-api-base http://127.0.0.1:18990/v1 \
      --served-adapter eval-v12 --out results/arms/<name> --seeds 1 --concurrency-per-gpu 8 [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from v2.constants import E4B_MODEL_ID
from v2.eval.governance import evaluate_paired_promotion, summarize_failure_modes

G = Path("/path/to/gemma-v2")  # evalhost defaults; override with --golden-root / --python / --model-path elsewhere


SPLITS = Path(__file__).resolve().parents[1] / "experiments" / "splits"
GOLDEN_SETS = {"v1": SPLITS / "golden_v1_manifest.json", "v2": SPLITS / "golden_v2_manifest.json"}
_V1_DEFAULTS = {"golden_root": G / "eval_data" / "golden_v1", "comp_image": "swegemma-sandbox:comp129"}


def resolve_golden(args: argparse.Namespace) -> dict[str, Any]:
    """Fill --golden-manifest / --golden-root / --comp-image from the selected golden set's frozen manifest.

    Refuses to grade on a comp image whose local ID differs from the image the manifest's controls were run on, and on
    a comp tasks.jsonl whose sha256 differs from the manifest's (both only when something is about to be graded)."""
    args.golden_manifest = Path(args.golden_manifest or GOLDEN_SETS[args.golden]).resolve()
    man = json.loads(args.golden_manifest.read_text(encoding="utf-8"))
    comp = man.get("comp", {})
    if args.golden_root is None:
        args.golden_root = Path(man["data_root"]) if man.get("data_root") else _V1_DEFAULTS["golden_root"]
    if args.comp_image is None:
        args.comp_image = comp.get("image_tag") or _V1_DEFAULTS["comp_image"]
    grading = (not args.report_only and not (args.anchor_arm and args.candidate_arm) and not args.dry_run
               and not getattr(args, "noise_arm", None))
    if grading and comp.get("image_id"):
        got = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", args.comp_image],
                             capture_output=True, text=True, check=False).stdout.strip()
        if got != comp["image_id"]:
            raise SystemExit(f"comp image {args.comp_image} is {got or 'missing'}, but {args.golden_manifest.name} was "
                             f"control-verified on {comp['image_id']}; build it with v2.eval.assemble_golden_v1 "
                             f"--manifest {args.golden_manifest} --build-image")
    tasks = Path(args.golden_root) / "comp" / "tasks.jsonl"
    if grading and comp.get("tasks_jsonl_sha256") and tasks.is_file():
        if hashlib.sha256(tasks.read_bytes()).hexdigest() != comp["tasks_jsonl_sha256"]:
            raise SystemExit(f"{tasks} does not match {args.golden_manifest.name} comp.tasks_jsonl_sha256")
    return man


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _env_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for name in ("vllm_python", "server_pythonpath", "hf_home", "vllm_cache_root"):
        val = getattr(args, name, None)
        if val:
            flags += ["--" + name.replace("_", "-"), str(val)]
    return flags


def commands(args: argparse.Namespace) -> list[list[str]]:
    model = str(args.model_path)
    GOLDEN = args.golden_root
    EVAL_PY = args.python
    cmds = []
    for s in range(args.seeds):
        cmds.append([
            EVAL_PY, "-m", "v2.eval.run_dual_gpu_ab", "--mode", "ab_head_to_head",
            "--tasks", str(GOLDEN / "comp" / "tasks.jsonl"),
            "--snapshots-dir", str(GOLDEN / "comp" / "snapshots"),
            "--wheels-dir", str(GOLDEN / "comp" / "wheels"),
            "--image", args.comp_image,
            "--results-root", str(args.out / f"comp_s{s}"),
            "--primary-submission", str(args.anchor), "--secondary-submission", str(args.candidate),
            "--primary-model-path", model, "--secondary-model-path", model,
            "--primary-alias", args.alias, "--secondary-alias", args.alias,
            "--concurrency-per-gpu", str(args.concurrency_per_gpu),
            *_env_flags(args),
        ])
    cmds.append([
        EVAL_PY, "-m", "v2.eval.run_rebench_v2", "run", "--mode", "ab_paired",
        "--tasks-path", str(GOLDEN / "rebench" / "tasks.jsonl"),
        "--rebench-dir", str(GOLDEN / "rebench"),
        "--results-root", str(args.out / "rebench"),
        "--primary-submission", str(args.anchor), "--secondary-submission", str(args.candidate),
        "--primary-model-path", model, "--secondary-model-path", model,
        "--primary-alias", args.alias, "--secondary-alias", args.alias,
        "--num-rollouts", str(args.seeds),
        "--concurrency-per-gpu", str(args.concurrency_per_gpu), "--batch-size", "64",
        "--eval-python", EVAL_PY,
        *_env_flags(args),
    ])
    return cmds


def seed_range(args: argparse.Namespace) -> range:
    """Runs this invocation produces: --seed-offset K --seeds N -> runs K..K+N-1 (N runs added to an arm)."""
    off = int(getattr(args, "seed_offset", 0) or 0)
    return range(off, off + int(args.seeds))


def rebench_dirname(offset: int) -> str:
    """rebench/ holds runs 0..N-1 (rollout_idx = run); runs added with --seed-offset K go to rebench_x<K>/, where
    rollout_idx j means run K+j (the rebench runner itself is unchanged)."""
    return "rebench" if not offset else f"rebench_x{int(offset)}"


def manifest_compatible_shas(manifest: Path) -> set[str]:
    """The manifest's sha plus the shas of its earlier versions with the same task set and image (protocol edits)."""
    man = json.loads(manifest.read_text(encoding="utf-8"))
    return {hashlib.sha256(manifest.read_bytes()).hexdigest(), *man.get("compatible_manifest_sha256", [])}


def check_arm_extension(args: argparse.Namespace, harness: Path, out: Path) -> dict[str, Any] | None:
    """Checked before anything runs. --seed-offset K must extend an existing arm in --out with exactly K runs, the
    same harness fingerprint, model and serving target, and a compatible manifest, and must not touch its files.
    Without --seed-offset, an existing arm is never overwritten. Returns the existing arm.json (None for a new arm)."""
    meta_p = out / "arm.json"
    if not args.seed_offset:
        if meta_p.is_file():
            n = json.loads(meta_p.read_text(encoding="utf-8")).get("seeds")
            raise SystemExit(f"{out} already holds an arm with {n} runs; add runs with --seed-offset {n}")
        return None
    if not meta_p.is_file():
        raise SystemExit(f"--seed-offset {args.seed_offset}: {meta_p} is missing (nothing to extend)")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    problems = []
    if int(meta.get("seeds", -1)) != int(args.seed_offset):
        problems.append(f"arm has {meta.get('seeds')} runs but --seed-offset is {args.seed_offset}")
    if meta.get("harness_sha256") != harness_fingerprint(harness):
        problems.append("harness fingerprint differs from the arm's")
    if str(meta.get("model_path")) != str(args.model_path):
        problems.append(f"model_path {args.model_path} differs from the arm's {meta.get('model_path')}")
    if meta.get("golden_manifest_sha") not in manifest_compatible_shas(args.golden_manifest):
        problems.append("arm was run on a different golden task set/manifest")
    if meta.get("comp_image") not in (None, args.comp_image):
        problems.append(f"comp image {args.comp_image} differs from the arm's {meta.get('comp_image')}")
    if meta.get("engine", "production") != getattr(args, "engine", "production"):
        problems.append(f"engine: arm is {meta.get('engine', 'production')}, this invocation {args.engine}")
    for k in ("external_api_base", "served_adapter") if getattr(args, "engine", "production") == "production" else ():
        if meta.get(k) != getattr(args, k, None):
            problems.append(f"{k}: arm has {meta.get(k)!r}, this invocation {getattr(args, k, None)!r}")
    clash = [str(out / f"comp_s{s}") for s in seed_range(args) if (out / f"comp_s{s}").exists()]
    if (out / rebench_dirname(getattr(args, "seed_offset", 0))).exists():
        clash.append(str(out / rebench_dirname(getattr(args, "seed_offset", 0))))
    if clash:
        problems.append(f"would overwrite {clash}")
    if problems:
        raise SystemExit("cannot extend the arm: " + "; ".join(problems))
    return meta


def write_arm_json(args: argparse.Namespace, harness: Path, out: Path, prev: dict[str, Any] | None,
                   extra: dict[str, Any] | None = None) -> None:
    """arm.json: `seeds` = total runs; `runs` lists each invocation (offset, count, manifest sha)."""
    manifest = args.golden_manifest
    run = {"seed_offset": int(args.seed_offset), "seeds": int(args.seeds),
           "golden_manifest_sha": hashlib.sha256(manifest.read_bytes()).hexdigest(), **(extra or {})}
    if prev is None:
        meta = {"harness": str(harness), "harness_sha256": harness_fingerprint(harness), "seeds": int(args.seeds),
                "golden_manifest_sha": run["golden_manifest_sha"], "golden_manifest": manifest.name,
                "comp_image": args.comp_image, "golden_root": str(args.golden_root),
                "model_path": str(args.model_path), **(extra or {}), "runs": [run]}
    else:
        meta = dict(prev)
        meta["runs"] = list(prev.get("runs") or [{"seed_offset": 0, "seeds": int(prev["seeds"]),
                                                    "golden_manifest_sha": prev.get("golden_manifest_sha")}])
        meta["runs"].append(run)
        meta["seeds"] = int(args.seed_offset) + int(args.seeds)
    (out / "arm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def harness_fingerprint(harness: Path) -> str:
    """sha256 over every file of a harness dir (prompts, agent.yaml, adapters), path-sorted."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(harness).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(harness)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def arm_commands(args: argparse.Namespace, harness: Path, out: Path) -> list[list[str]]:
    """One harness on BOTH GPUs: comp split across GPUs per seed, rebench streamed with all seeds as rollouts."""
    model = str(args.model_path)
    GOLDEN = args.golden_root
    cmds = []
    for s in seed_range(args):
        cmds.append([
            args.python, "-m", "v2.eval.run_dual_gpu_ab", "--mode", "dual_replica_sharded",
            "--tasks", str(GOLDEN / "comp" / "tasks.jsonl"),
            "--snapshots-dir", str(GOLDEN / "comp" / "snapshots"),
            "--wheels-dir", str(GOLDEN / "comp" / "wheels"),
            "--image", args.comp_image,
            "--results-root", str(out / f"comp_s{s}"),
            "--primary-submission", str(harness), "--secondary-submission", str(harness),
            "--primary-model-path", model, "--secondary-model-path", model,
            "--primary-alias", args.alias, "--secondary-alias", args.alias,
            "--concurrency-per-gpu", str(args.concurrency_per_gpu),
            *_env_flags(args),
        ])
    cmds.append([
        args.python, "-m", "v2.eval.run_rebench_v2", "run", "--mode", "dual_replica_rollouts",
        "--tasks-path", str(GOLDEN / "rebench" / "tasks.jsonl"),
        "--rebench-dir", str(GOLDEN / "rebench"),
        "--results-root", str(out / rebench_dirname(getattr(args, "seed_offset", 0))),
        "--primary-submission", str(harness),
        "--primary-model-path", model, "--primary-alias", args.alias,
        "--num-rollouts", str(args.seeds), "--concurrency-per-gpu", str(args.concurrency_per_gpu), "--stream",
        "--eval-python", args.python,
        *_env_flags(args),
    ])
    return cmds


def external_arm_jobs(args: argparse.Namespace, harness: Path, out: Path) -> list[dict[str, Any]]:
    """An arm against an ALREADY-RUNNING vLLM server (no server start/stop): the same per-GPU workers the arm
    orchestrators launch, pointed at `--external-api-base` with every agent call served by adapter
    `--served-adapter` (the served model id the agent alias maps to). Output layout = a normal arm
    (comp_s<s>/shard_0, rebench/shard0_gpu0), so `load_arm` / `--anchor-arm` work unchanged. Jobs run one after
    another on the server's spare capacity at `--concurrency-per-gpu` agents."""
    from v2.eval.run_rebench_v2 import load_rebench_task_inputs

    GOLDEN = args.golden_root
    api = args.external_api_base.rstrip("/")
    jobs: list[dict[str, Any]] = []
    for s in seed_range(args):
        wcfg = {
            "tasks_path": str(GOLDEN / "comp" / "tasks.jsonl"), "snapshots_dir": str(GOLDEN / "comp" / "snapshots"),
            "wheels_dir": str(GOLDEN / "comp" / "wheels"), "sandbox_image": args.comp_image,
            "concurrency": args.concurrency_per_gpu, "task_ids": None,
            "results_dir": str(out / f"comp_s{s}" / "shard_0"), "submission_dir": str(harness), "api_base": api,
            "served_model_id": args.served_adapter, "declared_model_alias": args.alias,
        }
        jobs.append({"name": f"comp_s{s}",
                     "cmd": [args.python, "-m", "v2.eval.run_dual_gpu_ab", "--internal-worker-json", json.dumps(wcfg)]})
    _, tasks_path, snapshots_dir, routes_all, task_ids, raw_rows = load_rebench_task_inputs(
        GOLDEN / "rebench", GOLDEN / "rebench" / "tasks.jsonl")
    n = max(1, int(args.seeds))
    items = [{"instance_id": i, "rollout_idx": r, "num_rollouts": n} for i in task_ids for r in range(n)]
    reb_root = out / rebench_dirname(getattr(args, "seed_offset", 0))
    reb_dir = reb_root / "shard0_gpu0"
    payload = {
        "tasks_path": str(tasks_path), "snapshots_dir": str(snapshots_dir), "results_dir": str(reb_dir),
        "submission_dir": str(harness), "api_base": api, "served_model_id": args.served_adapter,
        "declared_model_alias": args.alias, "concurrency": args.concurrency_per_gpu,
        "routes": {i: routes_all[i] for i in task_ids}, "batch_images": {}, "items": items, "stream": True,
        "build_ctx_root": str(reb_root / "tmp_docker_contexts"), "raw_rows": {i: raw_rows.get(i) for i in task_ids},
        "cleanup_images": not getattr(args, "keep_images", False),
    }
    p = reb_root / "payload_external.json"
    jobs.append({"name": rebench_dirname(getattr(args, "seed_offset", 0)), "payload_path": str(p), "payload": payload, "routes": routes_all,
                 "cmd": [args.python, "-m", "v2.eval.run_rebench_v2", "_worker_batch", str(p)]})
    return jobs


def run_external_arm(args: argparse.Namespace, harness: Path, out: Path) -> int:
    from v2.eval.run_dual_gpu_ab import verify_greedy_eval_temperature
    from v2.eval.run_rebench_v2 import append_missing_records

    jobs = external_arm_jobs(args, harness, out)
    if not args.dry_run:
        verify_greedy_eval_temperature(harness)
        import urllib.request

        with urllib.request.urlopen(args.external_api_base.rstrip("/") + "/models", timeout=15) as resp:
            served = {m.get("id") for m in json.loads(resp.read().decode()).get("data", [])}
        if args.served_adapter not in served:
            print(f"adapter {args.served_adapter!r} is not served by {args.external_api_base} (serving {sorted(served)})",
                  file=sys.stderr)
            return 2
    for job in jobs:
        if "payload" in job:
            Path(job["payload_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["payload_path"]).write_text(json.dumps(job["payload"]), encoding="utf-8")
        print("== " + job["name"] + ": " + " ".join(job["cmd"][:5]) + " ...", flush=True)
        if args.dry_run:
            continue
        logs = out / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"{job['name']}.log").open("w", encoding="utf-8") as lf:
            rc = subprocess.run(job["cmd"], stdout=lf, stderr=subprocess.STDOUT, check=False).returncode
        if "payload" in job:  # crashes/timeouts never disappear from task_results.jsonl
            append_missing_records(Path(job["payload"]["results_dir"]), job["payload"]["items"], job["routes"], rc)
        if rc != 0:
            print(f"{job['name']} failed (rc={rc}); stopping", file=sys.stderr)
            return rc
    plan = {"external_api_base": args.external_api_base, "served_adapter": args.served_adapter, "dry_run": args.dry_run,
            "jobs": [{"name": j["name"], "cmd": j["cmd"][:5]} for j in jobs]}
    (out / "external_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return 0


def _comp_worker_job(args: argparse.Namespace, harness: Path, results_dir: Path, api: str, served: str,
                     task_ids: list[str] | None, name: str) -> dict[str, Any]:
    GOLDEN = args.golden_root
    wcfg = {
        "tasks_path": str(GOLDEN / "comp" / "tasks.jsonl"), "snapshots_dir": str(GOLDEN / "comp" / "snapshots"),
        "wheels_dir": str(GOLDEN / "comp" / "wheels"), "sandbox_image": args.comp_image,
        "concurrency": args.concurrency_per_gpu, "task_ids": task_ids, "results_dir": str(results_dir),
        "submission_dir": str(harness), "api_base": api, "served_model_id": served, "declared_model_alias": args.alias,
    }
    return {"name": name, "cmd": [args.python, "-m", "v2.eval.run_dual_gpu_ab", "--internal-worker-json", json.dumps(wcfg)]}


def run_aligned_arm(args: argparse.Namespace) -> int:
    """--engine aligned: one batch-invariant aligned server (wave-3 closure, v2.eval.aligned_arm) for this harness,
    then the normal external-server arm against it (or, with --determinism-check N, the first N comp tasks twice)."""
    from v2.eval.aligned_arm import compare_runs, make_server, server_record

    harness = args.arm.resolve()
    det = int(args.determinism_check or 0)
    prev = None if det else check_arm_extension(args, harness, args.out)
    server, loras = make_server(args.out / ("aligned_server" if not det else "aligned_server_det"),
                                Path(args.model_path), harness, gpu=str(args.aligned_gpu), port=int(args.aligned_port),
                                concurrency=int(args.concurrency_per_gpu))
    rec = server_record(server, loras)
    args.external_api_base = server.base + "/v1"
    args.served_adapter = str(args.model_path)  # like production workers: base model id; agent.yaml picks the adapter
    if args.dry_run:
        plan = {"engine": "aligned", "runs": [] if det else list(seed_range(args)), "determinism_check": det, **rec}
        if not det:
            plan["jobs"] = [{"name": j["name"], "cmd": j["cmd"][:5]} for j in external_arm_jobs(args, harness, args.out)]
        print(json.dumps(plan, indent=2))
        return 0
    from v2.eval.run_dual_gpu_ab import verify_greedy_eval_temperature

    verify_greedy_eval_temperature(harness)
    secs = server.launch(loras)
    print(f"aligned server ready on GPU {server.gpu} in {secs:.0f}s", flush=True)
    try:
        if det:
            comp_ids = [json.loads(x)["instance_id"] for x in
                        (args.golden_root / "comp" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][:det]
            logs = args.out / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            for tag in ("a", "b"):
                job = _comp_worker_job(args, harness, args.out / f"det_{tag}" / "shard_0", args.external_api_base,
                                       args.served_adapter, comp_ids, f"det_{tag}")
                with (logs / f"det_{tag}.log").open("w", encoding="utf-8") as lf:
                    rc = subprocess.run(job["cmd"], stdout=lf, stderr=subprocess.STDOUT, check=False).returncode
                if rc != 0:
                    print(f"determinism run {tag} failed (rc={rc})", file=sys.stderr)
                    return rc
            rep = {**compare_runs(args.out / "det_a" / "shard_0", args.out / "det_b" / "shard_0", comp_ids),
                   "harness": str(harness), "harness_sha256": harness_fingerprint(harness), **rec}
            (args.out / "determinism.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: rep[k] for k in ("tasks", "identical_trajectories", "identical_patches",
                                                  "identical_verdicts")}
                             | {"causes": [r["cause"] for r in rep["rows"]]}, indent=2))
            return 0
        rc = run_external_arm(args, harness, args.out)
    finally:
        server.stop()
    if rc != 0:
        return rc
    write_arm_json(args, harness, args.out, prev, rec)
    print(json.dumps({"arm": str(args.out), "engine": "aligned", "runs": list(seed_range(args))}, indent=2))
    return 0


def load_arm(arm: Path, seeds: int) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Runs 0..seeds-1 of an arm. comp run s = comp_s<s>/shard_*; rebench run s = rollout_idx s in rebench/, or
    rollout_idx s-K in rebench_x<K>/ (runs added later with --seed-offset K)."""
    comp = [_rows(arm / f"comp_s{s}" / "shard_0" / "task_results.jsonl")
            + _rows(arm / f"comp_s{s}" / "shard_1" / "task_results.jsonl") for s in range(seeds)]
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for d in sorted(arm.glob("rebench*")):
        if d.name == "rebench":
            off = 0
        elif d.name.startswith("rebench_x") and d.name[len("rebench_x"):].isdigit():
            off = int(d.name[len("rebench_x"):])
        else:
            continue
        for shard in ("shard0_gpu0", "shard1_gpu1"):
            for r in _rows(d / shard / "task_results.jsonl"):
                by_seed.setdefault(off + int(r.get("rollout_idx", 0)), []).append(r)
    reb = [by_seed.get(s, []) for s in range(seeds)]
    return comp, reb


def arm_engine(meta: dict[str, Any]) -> str:
    """Serving engine of an arm: 'production' (standard vLLM, the default) or 'aligned' (batch-invariant TII engine).
    Raises if one arm mixes engines across its runs."""
    engines = {meta.get("engine", "production")} | {r.get("engine", meta.get("engine", "production"))
                                                     for r in meta.get("runs", [])}
    if len(engines) != 1:
        raise SystemExit(f"arm mixes engines across runs: {sorted(engines)}")
    return engines.pop()


def read_arm_meta(arm: Path) -> dict[str, Any]:
    meta_p = arm / "arm.json"
    return json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}


def decide_arms(anchor_arm: Path, candidate_arm: Path, out: Path, seeds: int, rule_version: int = 1) -> dict[str, Any]:
    """Decision from two independently produced arms (e.g. a cached base arm vs a new candidate arm).
    seeds=0: use every run both arms have (the smaller run count)."""
    metas = [read_arm_meta(anchor_arm), read_arm_meta(candidate_arm)]
    shas = {m.get("golden_manifest_sha") for m in metas} | {r.get("golden_manifest_sha") for m in metas
                                                             for r in m.get("runs", [])}
    if len(shas) > 1 and not any(shas <= manifest_compatible_shas(m) for m in GOLDEN_SETS.values() if m.is_file()):
        raise SystemExit(f"arms were run on different golden manifests: {metas}")
    engines = [arm_engine(m) for m in metas]
    if engines[0] != engines[1]:
        raise SystemExit(f"never pool engines: anchor arm is {engines[0]}, candidate arm is {engines[1]}")
    if not seeds:
        seeds = min(int(m.get("seeds") or 0) for m in metas)
    comp_a, reb_a = load_arm(anchor_arm, seeds)
    comp_c, reb_c = load_arm(candidate_arm, seeds)
    rep = decide_rows(comp_a, comp_c, reb_a, reb_c, out, seeds, rule_version=rule_version)
    rep["engine"] = engines[0]
    if engines[0] == "aligned":
        rep["note"] = ("aligned (deterministic anchor) arms: report side by side with the production decision; the "
                       "promote rule applies to production arms only")
        rep["promote"] = False
        rep["reasons"] = ["aligned_engine_anchor_not_a_promotion_decision"] + list(rep.get("reasons", []))
    rep["arms"] = {"anchor": {"dir": str(anchor_arm), **metas[0]}, "candidate": {"dir": str(candidate_arm), **metas[1]}}
    (out / "golden_report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return rep


def decide(out: Path, seeds: int, rule_version: int = 1) -> dict[str, Any]:
    comp_a = [_rows(out / f"comp_s{s}" / "anchor_gpu0" / "task_results.jsonl") for s in range(seeds)]
    comp_c = [_rows(out / f"comp_s{s}" / "candidate_gpu1" / "task_results.jsonl") for s in range(seeds)]
    reb_a_all = _rows(out / "rebench" / "anchor_gpu0" / "task_results.jsonl")
    reb_c_all = _rows(out / "rebench" / "candidate_gpu1" / "task_results.jsonl")
    reb_a = [[r for r in reb_a_all if int(r.get("rollout_idx", 0)) == s] for s in range(seeds)]
    reb_c = [[r for r in reb_c_all if int(r.get("rollout_idx", 0)) == s] for s in range(seeds)]
    return decide_rows(comp_a, comp_c, reb_a, reb_c, out, seeds, rule_version=rule_version)


def decide_rows(comp_a: list[list[dict[str, Any]]], comp_c: list[list[dict[str, Any]]],
                reb_a: list[list[dict[str, Any]]], reb_c: list[list[dict[str, Any]]], out: Path,
                seeds: int, rule_version: int = 1) -> dict[str, Any]:
    """rule_version 1: per-seed McNemar gate (golden_v1). rule_version 2 (golden_v2): multi-run per-task means with a
    repo-clustered bootstrap (v2.eval.multirun_stats); the v1 gate is still computed and reported as legacy_v1."""

    def _norm(runs: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return [[{"instance_id": str(r["instance_id"]), "repo": str(r.get("repo") or ""), "resolved": bool(r.get("resolved"))} for r in run] for run in runs]

    report: dict[str, Any] = {"seeds": seeds}
    parts = {"comp": (comp_a, comp_c), "rebench": (reb_a, reb_c)}
    for name, (a, c) in parts.items():
        flat_a = [r for run in a for r in run]
        flat_c = [r for run in c for r in run]
        report[name] = {"anchor": summarize_failure_modes(flat_a), "candidate": summarize_failure_modes(flat_c)}
        if all(a) and all(c):
            report[name]["gate"] = evaluate_paired_promotion(_norm(a), _norm(c), projected_120task_hours=5.0, min_seeds=1).to_dict()
    if all(comp_a) and all(comp_c) and all(reb_a) and all(reb_c):
        pooled_a = [ca + ra for ca, ra in zip(_norm(comp_a), _norm(reb_a))]
        pooled_c = [cc + rc for cc, rc in zip(_norm(comp_c), _norm(reb_c))]
        pooled = evaluate_paired_promotion(pooled_a, pooled_c, projected_120task_hours=5.0, min_seeds=1).to_dict()
        comp_delta = report["comp"]["gate"]["mean_delta"]
        report["pooled_gate"] = pooled
        report["promote"] = bool(pooled["promote"]) and comp_delta >= 0.0
        report["reasons"] = list(pooled["reasons"]) + ([] if comp_delta >= 0.0 else [f"comp_delta_negative:{comp_delta}"])
    else:
        report["promote"] = False
        report["reasons"] = ["incomplete_results"]
    if rule_version >= 2:
        from v2.eval.multirun_stats import decide_v2

        report["legacy_v1"] = {"promote": report["promote"], "reasons": report["reasons"]}
        report["promote_rule"] = 2
        if all(comp_a) and all(comp_c) and all(reb_a) and all(reb_c):
            v2 = decide_v2(_norm(comp_a), _norm(comp_c), _norm(reb_a), _norm(reb_c))
            report["multirun"] = v2
            report["promote"], report["reasons"] = v2["promote"], v2["reasons"]
        else:
            report["promote"], report["reasons"] = False, ["incomplete_results"]
    (out / "golden_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def arm_noise_report(arm: Path, seeds: int) -> dict[str, Any]:
    """Run-to-run noise floor of ONE arm from existing results: per-run solve rate (comp, rebench, pooled) and SD."""
    import statistics

    meta = read_arm_meta(arm)
    seeds = seeds or int(meta.get("seeds") or 0)
    comp, reb = load_arm(arm, seeds)
    out: dict[str, Any] = {"arm": str(arm), "engine": arm_engine(meta), "runs": seeds,
                           "harness": meta.get("harness"), "served_adapter": meta.get("served_adapter")}
    for name, runs in (("comp", comp), ("rebench", reb), ("pooled", [c + r for c, r in zip(comp, reb)])):
        rates = [sum(bool(x.get("resolved")) for x in run) / len(run) if run else None for run in runs]
        ok = [r for r in rates if r is not None]
        out[name] = {"run_rates": rates, "rows_per_run": [len(run) for run in runs],
                     "mean": statistics.fmean(ok) if ok else None,
                     "sd": statistics.stdev(ok) if len(ok) > 1 else None}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.eval.golden_ab")
    p.add_argument("--anchor", type=Path, default=None, help="Greedy anchor harness dir (GPU0) for a paired run")
    p.add_argument("--candidate", type=Path, default=None, help="Greedy candidate harness dir (GPU1) for a paired run")
    p.add_argument("--arm", type=Path, default=None,
                   help="Run ONE harness on both GPUs into --out (an arm; reusable as a cached anchor)")
    p.add_argument("--anchor-arm", type=Path, default=None, help="Decide from a saved anchor arm dir ...")
    p.add_argument("--candidate-arm", type=Path, default=None, help="... and a saved candidate arm dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=2,
                   help="Runs per arm. Decisions: how many runs to use. With --arm: how many runs this invocation adds")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="With --arm: ADD --seeds runs to the arm already in --out, which must hold exactly this many "
                        "runs of the same harness/model/task set (writes comp_s<K>..comp_s<K+N-1> and rebench_x<K>/; "
                        "existing runs are never touched)")
    p.add_argument("--model-path", type=Path, default=G / "models" / "gemma-4-e4b-it-qat-w4a16-ct")
    p.add_argument("--alias", default=E4B_MODEL_ID)
    p.add_argument("--golden", choices=sorted(GOLDEN_SETS), default="v1",
                   help="Golden set: picks the frozen manifest and, unless given explicitly, --golden-root and "
                        "--comp-image from it (v2 = golden_v2_manifest.json, comp graded on swegemma-sandbox:comp129-v2)")
    p.add_argument("--golden-manifest", type=Path, default=None, help="Use this manifest file instead of --golden's")
    p.add_argument("--comp-image", default=None, help="Comp grader image (default: the manifest's comp.image_tag)")
    p.add_argument("--golden-root", type=Path, default=None,
                   help="Dir with comp/ and rebench/ (default: the manifest's data_root; golden_v1: eval_data/golden_v1)")
    p.add_argument("--python", default=str(G / "eval-venv" / "bin" / "python") if (G / "eval-venv").is_dir() else sys.executable,
                   help="Python of the swegemma/adk-submission eval environment")
    evalhost_vllm = Path("/path/to/gemma-devagent-20261001/server-venv/bin/python")
    p.add_argument("--vllm-python", default=str(evalhost_vllm) if evalhost_vllm.is_file() else None,
                   help="Python of the vLLM 0.19.1 environment (forwarded to runners)")
    p.add_argument("--server-pythonpath", default=None)
    p.add_argument("--hf-home", default=None)
    p.add_argument("--vllm-cache-root", default=None)
    p.add_argument("--concurrency-per-gpu", type=int, default=16,
                   help="Agents per GPU; 16 measured with no per-rollout slowdown on RTX PRO 6000 (2026-11-05)")
    p.add_argument("--report-only", action="store_true",
                   help="Paired run: recompute the decision from existing outputs. (--anchor-arm/--candidate-arm and "
                        "--noise-arm never run anything.) --seeds 0 = every run both arms have")
    p.add_argument("--engine", choices=("production", "aligned"), default="production",
                   help="With --arm: production = standard vLLM (Kaggle-faithful); aligned = deterministic anchor served "
                        "by the batch-invariant TII engine with the wave-3 closure (v2.eval.aligned_arm; 1 run)")
    p.add_argument("--aligned-gpu", default="0", help="--engine aligned: GPU of the aligned server")
    p.add_argument("--aligned-port", type=int, default=18995, help="--engine aligned: port of the aligned server")
    p.add_argument("--determinism-check", type=int, default=0, metavar="N",
                   help="--engine aligned: run the first N comp tasks twice on one aligned server and compare "
                        "verdicts, patches and trajectories (writes <out>/determinism.json; no arm)")
    p.add_argument("--noise-arm", type=Path, nargs="+", default=None,
                   help="Report-only: per-run solve rates and run-to-run SD of these saved arm dirs")
    p.add_argument("--external-api-base", default=None,
                   help="With --arm: evaluate against this ALREADY-RUNNING vLLM server (e.g. http://127.0.0.1:18990/v1) "
                        "instead of launching servers on both GPUs")
    p.add_argument("--served-adapter", default=None,
                   help="With --external-api-base: served model/adapter name every agent call goes to (e.g. eval-v12)")
    p.add_argument("--keep-images", action="store_true",
                   help="With --external-api-base: do not remove per-task agent/grader image tags after use (needed when "
                        "several external arms run at the same time, since the tags are shared per task)")
    p.add_argument("--dry-run", action="store_true", help="With --external-api-base: write the plan/payloads, run nothing; with a local --arm: print the "
                        "exact runner commands (golden set wiring) and run nothing")
    args = p.parse_args(argv)
    man = resolve_golden(args)
    rule_version = int(man.get("promote_rule", {}).get("version", 1))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.anchor_arm and args.candidate_arm:  # always report-only: recomputes from the arms' saved results
        rep = decide_arms(args.anchor_arm.resolve(), args.candidate_arm.resolve(), args.out, args.seeds,
                          rule_version=rule_version)
        print(json.dumps(decision_summary(rep), indent=2))
        return 0
    if args.noise_arm:  # report-only: run-to-run noise floor of saved arms
        rep = {"arms": [arm_noise_report(a.resolve(), args.seeds) for a in args.noise_arm]}
        (args.out / "arm_noise.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rep, indent=2))
        return 0
    if args.arm is not None and args.engine == "aligned":
        if args.external_api_base:
            p.error("--engine aligned starts its own aligned server; drop --external-api-base")
        return run_aligned_arm(args)
    if args.engine == "aligned" or args.determinism_check:
        p.error("--engine aligned / --determinism-check need --arm <harness>")
    if args.arm is not None and args.external_api_base:
        if not args.served_adapter:
            p.error("--external-api-base needs --served-adapter")
        prev = check_arm_extension(args, args.arm.resolve(), args.out)
        rc = run_external_arm(args, args.arm.resolve(), args.out)
        if rc != 0 or args.dry_run:
            return rc
        write_arm_json(args, args.arm.resolve(), args.out, prev,
                       {"external_api_base": args.external_api_base, "served_adapter": args.served_adapter})
        print(json.dumps({"arm": str(args.out), "runs": list(seed_range(args))}, indent=2))
        return 0
    if args.arm is not None and args.dry_run:
        check_arm_extension(args, args.arm.resolve(), args.out)
        print(json.dumps({"runs": list(seed_range(args)),
                          "golden_manifest": str(args.golden_manifest), "golden_root": str(args.golden_root),
                          "comp_image": args.comp_image,
                          "commands": [" ".join(c) for c in arm_commands(args, args.arm.resolve(), args.out)]}, indent=2))
        return 0
    if args.arm is not None:
        vllm_py = args.vllm_python or args.python
        if subprocess.run([vllm_py, "-c", "import vllm"], capture_output=True).returncode != 0:
            print(f"{vllm_py} cannot import vllm; pass --vllm-python <vLLM env python>", file=sys.stderr)
            return 2
        prev = check_arm_extension(args, args.arm.resolve(), args.out)
        for cmd in arm_commands(args, args.arm.resolve(), args.out):
            print("== " + " ".join(cmd[:6]) + " ...", flush=True)
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"command failed (rc={rc}); stopping", file=sys.stderr)
                return rc
        write_arm_json(args, args.arm.resolve(), args.out, prev)
        print(json.dumps({"arm": str(args.out), "runs": list(seed_range(args))}, indent=2))
        return 0
    if args.anchor is None or args.candidate is None:
        p.error("need --anchor and --candidate (paired run), --arm (one arm), or --anchor-arm and --candidate-arm")
    if not args.report_only:
        # Fail fast: the runners otherwise start vLLM from the eval env and die after the servers time out
        vllm_py = args.vllm_python or args.python
        if subprocess.run([vllm_py, "-c", "import vllm"], capture_output=True).returncode != 0:
            print(f"{vllm_py} cannot import vllm; pass --vllm-python <vLLM 0.19.1 env python>", file=sys.stderr)
            return 2
        for cmd in commands(args):
            print("== " + " ".join(cmd[:6]) + " ...", flush=True)
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"command failed (rc={rc}); stopping", file=sys.stderr)
                return rc
    rep = decide(args.out, args.seeds, rule_version=rule_version)
    print(json.dumps(decision_summary(rep), indent=2))
    return 0


def decision_summary(rep: dict[str, Any]) -> dict[str, Any]:
    """Console view: the decision, and for rule v2 the per-part mean deltas, CI90, sign test and noise floor."""
    out: dict[str, Any] = {k: rep.get(k) for k in ("promote", "reasons", "seeds", "engine", "promote_rule")}
    mr = rep.get("multirun")
    if mr:
        for part in ("pooled", "comp", "rebench"):
            x = mr[part]
            out[part] = {k: x[k] for k in ("tasks", "repos", "anchor_rate", "candidate_rate", "mean_delta",
                                           "ci90_low", "ci90_high")}
            out[part]["sign_test"] = x["sign_test"]
            out[part]["run_sd"] = {"anchor": x["noise_floor"]["anchor_run_sd"],
                                   "candidate": x["noise_floor"]["candidate_run_sd"]}
        out["legacy_v1"] = rep.get("legacy_v1")
    else:
        out["comp"], out["rebench"] = rep.get("comp"), rep.get("rebench")
    return out


if __name__ == "__main__":
    sys.exit(main())
