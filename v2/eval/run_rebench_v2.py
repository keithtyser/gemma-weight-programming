"""Rolling-Image Dual-GPU (`GPU0` + `GPU1`) Runner & Grader for 6-Gate `SWE-rebench` Tasks.

Supports:
1. `materialize`: Prepares sanitized agent-only `.tgz` snapshots, `train28.jsonl`, `unseen_cv8.jsonl`,
   `all36.jsonl`, and evaluator-only `evaluator_routes36.json` under `/path/to/gemma-v2/eval_data/rebench36/`
   without mutating any files in `/path/to/gemma-devagent-20261001`.
2. `run`: Launches two persistent `vLLM` servers (`GPU0` & `GPU1`) and executes `swegemma 0.2.11`
   in rolling Docker-image batches (`--batch-size 8`, keeping >110 GiB NVMe free at all times) in either:
   - `ab_paired`: simultaneous Anchor (`GPU0`) vs Candidate (`GPU1`) on identical `unseen_cv8` / `rebench` tasks
     with `evaluate_paired_promotion` statistical gating.
   - `dual_replica_rollouts`: sharded `(task, rollout_idx)` trajectory generation across `GPU0 + GPU1`
     (`G=4` or `G=6`, `T=0.7`) for Stage 1 RFT and Stage 2 On-Policy Distillation.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import io
import functools
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import uuid
import xml.etree.ElementTree as ET

from v2.eval.governance import evaluate_paired_promotion, summarize_failure_modes
from v2.eval.gpu_launch_lock import server_popen
from v2.eval.run_dual_gpu_ab import (
    build_kaggle_adk_session_configs,
    load_submission_eval_budgets,
    terminate_process_group,
    wait_for_vllm_health,
)
from v2.eval.gpu_runner import (
    build_swegemma_0211_model_registry,
    build_vllm_0213_command,
    discover_lora_modules_and_rank,
)

CONDA_PATH_ENV = (
    "/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:"
    "/opt/conda/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

ISOLATED_GRADER_LIMITS = (
    "--network",
    "none",
    "--cpus",
    "2",
    "--memory",
    "4g",
    "--pids-limit",
    "256",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges:true",
    "--env",
    "CUDA_VISIBLE_DEVICES=",
    "--env",
    "HIP_VISIBLE_DEVICES=",
    "--env",
    "OMP_NUM_THREADS=2",
    "--env",
    "OPENBLAS_NUM_THREADS=2",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raise_nofile_soft_limit(target: int = 65536) -> tuple[int, int] | None:
    """Raise POSIX RLIMIT_NOFILE soft limit so large rollout batches never hit Errno 24 under SSH."""
    try:
        import resource  # POSIX only

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(target, hard) if hard != resource.RLIM_INFINITY else target
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            soft = desired
        return int(soft), int(hard)
    except Exception:
        return None


def _is_vllm_responsive(port: int, proc: subprocess.Popen[Any] | None, timeout_sec: float = 12.0) -> bool:
    """Verify both OS process liveness and event-loop responsiveness (/v1/models HTTP 200)."""
    if proc is None or proc.poll() is not None:
        return False
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def _run_cmd(args: list[str], timeout: int = 900, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        res = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise
        # check=False callers inspect the return code; a timeout is a failure (124, like coreutils timeout), not an
        # exception (wave1: a 60 s `docker rm -f` timeout in grading cleanup escaped and killed the rollout worker)
        return subprocess.CompletedProcess(args, 124, b"", f"timeout after {timeout}s".encode())
    if check and res.returncode != 0:
        err_tail = res.stderr.decode("utf-8", errors="replace")[-1200:]
        out_tail = res.stdout.decode("utf-8", errors="replace")[-600:]
        raise RuntimeError(f"Command failed ({res.returncode}): {args[:4]} | stderr={err_tail} | stdout={out_tail}")
    return res


def _safe_tar_members(archive: tarfile.TarFile):
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if (
            member.name.startswith("/")
            or "\\" in member.name
            or ":" in member.name
            or ".." in parts
            or member.isdev()
            or member.isfifo()
            or member.islnk()
        ):
            raise ValueError(f"Unsafe tar member: {member.name}")
        if member.issym():
            import posixpath

            norm_target = posixpath.normpath(
                posixpath.join(posixpath.dirname(member.name), member.linkname)
            )
            if (
                "\\" in member.linkname
                or ":" in member.linkname
                or PurePosixPath(member.linkname).is_absolute()
                or norm_target == ".."
                or norm_target.startswith("../")
            ):
                raise ValueError(f"Unsafe symlink in tar archive: {member.name} -> {member.linkname}")
        yield member


def _decode_test_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or not all(isinstance(x, str) and x for x in parsed):
        raise ValueError(f"Malformed test ID list: {type(value)}")
    return list(dict.fromkeys(parsed))


def _junit_key(required: str) -> str:
    head, sep, param = required.partition("[")
    parts = head.split("::")
    path = parts[0]
    if path.endswith(".py"):
        path = path[:-3]
    return ".".join([path.replace("/", ".")] + parts[1:]) + sep + param


def _parse_junit_bytes(data: bytes) -> dict[str, Any]:
    if not data.strip():
        raise ValueError("Missing JUnit XML report")
    root = ET.fromstring(data)
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    cases = root.findall(".//testcase")
    if not suites or not cases:
        raise ValueError("Zero collected testcases in JUnit XML")
    statuses: dict[str, str] = {}
    errors: list[str] = []
    for case in cases:
        name = case.get("name", "")
        cls = case.get("classname", "")
        ident = f"{cls}.{name}"
        state = (
            "ERROR"
            if case.find("error") is not None
            else "FAILED"
            if case.find("failure") is not None
            else "SKIPPED"
            if case.find("skipped") is not None
            else "PASSED"
        )
        if ident in statuses and statuses[ident] != state:
            raise ValueError(f"Conflicting repeated JUnit testcase: {ident}")
        statuses[ident] = state
        if state == "ERROR":
            errors.append(ident)
    return {
        "statuses": statuses,
        "collected": len(cases),
        "counts": dict(collections.Counter(statuses.values())),
        "errors": errors,
    }


def _compare_required_tests(
    report: dict[str, Any],
    f2p: list[str],
    p2p: list[str],
) -> list[dict[str, Any]]:
    states = report["statuses"]
    issues: list[dict[str, Any]] = []
    for req in f2p + p2p:
        actual = states.get(_junit_key(req))
        if actual != "PASSED":
            issues.append({"test": req, "expected": "PASSED", "actual": actual or "MISSING"})
    if report["errors"]:
        issues.append({"infrastructure_errors": len(report["errors"])})
    if not f2p or not p2p:
        issues.append({"missing_required_set": "FAIL_TO_PASS or PASS_TO_PASS empty"})
    return issues


def _image_slug(iid: str) -> str:
    # REBENCH_IMAGE_NS namespaces the per-task agent/grader tags, so runs that share tasks (several eval arms at once,
    # or an eval next to a training run) never remove each other's tags.
    ns = re.sub(r"[^a-z0-9_.-]+", "-", os.environ.get("REBENCH_IMAGE_NS", "").lower()).strip("-")
    return re.sub(r"[^a-z0-9_.-]+", "-", iid.lower()).strip("-") + (f"-{ns}" if ns else "")


# ---------------------------------------------------------------------------
# 1. Materialize 36 6-Gate SWE-rebench Snapshots & Split Manifests
# ---------------------------------------------------------------------------
def materialize_rebench36(
    *,
    splits_json: Path,
    pilot_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build sanitized `.tgz` snapshots, public `.jsonl` task files, and evaluator routes."""
    splits = json.loads(splits_json.read_text(encoding="utf-8"))
    train_ids: list[str] = sorted(splits["train_instance_ids"])
    cv_ids: list[str] = sorted(splits["unseen_cv_instance_ids"])
    all_ids: list[str] = sorted(set(train_ids) | set(cv_ids))

    growth_root = pilot_root / "growth-ready-v1"
    rows_by_id: dict[str, dict[str, Any]] = {}
    for rows_path in (
        pilot_root / "evaluator-only-source" / "rows.json",
        growth_root / "evaluator-only-source" / "rows.json",
    ):
        if rows_path.is_file():
            for row in json.loads(rows_path.read_text(encoding="utf-8")):
                rows_by_id[str(row["instance_id"])] = row

    accepted_by_id: dict[str, Path] = {}
    for acc_path in sorted(
        list(pilot_root.glob("**/acceptance.json")) + list(growth_root.glob("**/acceptance.json"))
    ):
        doc = json.loads(acc_path.read_text(encoding="utf-8"))
        if doc.get("status") == "accepted":
            accepted_by_id[str(doc["instance_id"])] = acc_path

    prebuilt_tgz: dict[str, Path] = {}
    for tgz_path in sorted(
        list(pilot_root.glob("**/agent-inputs-*/snapshots/*.tgz"))
        + list(growth_root.glob("**/agent-inputs-*/snapshots/*.tgz"))
    ):
        prebuilt_tgz[tgz_path.stem] = tgz_path

    snaps_dir = output_dir / "snapshots"
    tmp_root = output_dir / "tmp_build"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    public_rows: dict[str, dict[str, Any]] = {}
    evaluator_routes: dict[str, dict[str, Any]] = {}
    reused_count = 0
    built_count = 0

    for iid in all_ids:
        if iid not in rows_by_id:
            raise KeyError(f"Missing source row for {iid}")
        if iid not in accepted_by_id:
            raise KeyError(f"Missing accepted CPU receipt for {iid}")

        row = rows_by_id[iid]
        acc_path = accepted_by_id[iid]
        acc_doc = json.loads(acc_path.read_text(encoding="utf-8"))
        env_doc = acc_doc["environment"]

        dst_tgz = snaps_dir / f"{iid}.tgz"
        if not dst_tgz.is_file():
            if iid in prebuilt_tgz:
                shutil.copy2(prebuilt_tgz[iid], dst_tgz)
                reused_count += 1
            else:
                base_tar = acc_path.parent / "base-snapshot.tar"
                if not base_tar.is_file():
                    raise FileNotFoundError(f"Missing base-snapshot.tar for {iid}: {base_tar}")
                if _sha256_bytes(base_tar.read_bytes()) != env_doc["base_snapshot_sha256"]:
                    raise ValueError(f"SHA256 mismatch on base-snapshot.tar for {iid}")
                with tempfile.TemporaryDirectory(prefix=f"snap-{iid[:20]}-", dir=tmp_root) as td:
                    work = Path(td)
                    with tarfile.open(base_tar) as archive:
                        archive.extractall(work, members=_safe_tar_members(archive), filter="data")
                    for git_cmd in (
                        ["git", "init", "--quiet", str(work)],
                        ["git", "-C", str(work), "config", "user.name", "Gemma Evaluation"],
                        ["git", "-C", str(work), "config", "user.email", "evaluation@localhost"],
                        ["git", "-C", str(work), "add", "--all"],
                        ["git", "-C", str(work), "commit", "--quiet", "--allow-empty", "-m", "Pinned task base snapshot"],
                    ):
                        _run_cmd(git_cmd, timeout=120)
                    with dst_tgz.open("wb") as stream:
                        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz:
                            with tarfile.open(fileobj=gz, mode="w") as out_tar:
                                for p in sorted(work.rglob("*")):
                                    out_tar.add(p, arcname=p.relative_to(work), recursive=False)
                built_count += 1

        install_cfg = row["install_config"]
        if isinstance(install_cfg, str):
            install_cfg = json.loads(install_cfg)

        public_rows[iid] = {
            "instance_id": iid,
            "repo": str(row.get("repo", "")),
            "base_commit": str(row.get("base_commit", "")),
            "problem_statement": str(row.get("problem_statement", "")),
            "hints_text": str(row.get("hints_text", "") or ""),
            "created_at": str(row.get("created_at", "") or ""),
        }
        evaluator_routes[iid] = {
            "instance_id": iid,
            "repo": str(row.get("repo", "")),
            "split": "train_pool" if iid in set(train_ids) else "unseen_repo_cv",
            "base_commit": str(row.get("base_commit", "")),
            "upstream_registry_digest": str(env_doc["source_registry_digest"]),
            "snapshot_sha256": _sha256_bytes(dst_tgz.read_bytes()),
            "acceptance_receipt_path": str(acc_path),
            "test_cmd": str(install_cfg["test_cmd"]),
            "test_patch": str(row["test_patch"]),
            "FAIL_TO_PASS": _decode_test_list(row["FAIL_TO_PASS"]),
            "PASS_TO_PASS": _decode_test_list(row["PASS_TO_PASS"]),
        }

    shutil.rmtree(tmp_root, ignore_errors=True)

    for fname, id_list in (
        ("train28.jsonl", train_ids),
        ("unseen_cv8.jsonl", cv_ids),
        ("all36.jsonl", all_ids),
    ):
        (output_dir / fname).write_text(
            "".join(json.dumps(public_rows[i], ensure_ascii=False) + "\n" for i in id_list),
            encoding="utf-8",
            newline="\n",
        )

    routes_path = output_dir / "evaluator_routes36.json"
    routes_path.write_text(
        json.dumps(evaluator_routes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(routes_path, 0o600)

    report = {
        "output_dir": str(output_dir),
        "total_tasks": len(all_ids),
        "train_tasks": len(train_ids),
        "unseen_cv_tasks": len(cv_ids),
        "reused_prebuilt_tgz": reused_count,
        "built_from_base_tar": built_count,
    }
    (output_dir / "materialize_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


# ---------------------------------------------------------------------------
# 2. Rolling Docker Batch Image Preparation & Cleanup
# ---------------------------------------------------------------------------
_FORBIDDEN_WORKDIRS = frozenset(
    {"", "/", "/workspace", "/root", "/tmp", "/usr", "/var", "/opt", "/etc", "/bin", "/sbin"}
)


def _resolve_image_workdir(route: dict[str, Any], image_ref: str) -> str:
    raw_wd = str(route.get("workdir") or "").strip().rstrip("/")
    if not raw_wd:
        insp = json.loads(_run_cmd(["docker", "image", "inspect", image_ref], timeout=30).stdout)[0]
        raw_wd = str((insp.get("Config") or {}).get("WorkingDir") or "/testbed").strip().rstrip("/") or "/testbed"
    if not raw_wd.startswith("/") or raw_wd in _FORBIDDEN_WORKDIRS:
        raise RuntimeError(f"Unsafe container workdir for {route.get('instance_id')}: {raw_wd!r}")
    return raw_wd


def _prepare_single_task_images(
    route: dict[str, Any],
    build_ctx_root: Path,
    raw_row: dict[str, Any] | None = None,
) -> dict[str, str]:
    iid = route["instance_id"]
    slug = _image_slug(iid)
    digest_ref = route["upstream_registry_digest"]
    grader_tag = f"gemma-v2-rebench-grader:{slug}"
    agent_tag = f"gemma-v2-rebench-agent:{slug}"

    base_no_digest = digest_ref.split("@", 1)[0]
    stripped = base_no_digest.removeprefix("docker.io/")
    cand_refs = [digest_ref, base_no_digest, stripped]
    if ":" not in stripped.split("/")[-1]:
        cand_refs.append(f"{stripped}:latest")
    if isinstance(raw_row, dict):
        for k in ("docker_image", "image_name"):
            v = str(raw_row.get(k) or "").strip()
            if v and v not in cand_refs:
                cand_refs.append(v)

    resolved_local: str | None = None
    for cref in cand_refs:
        if cref and _run_cmd(["docker", "image", "inspect", cref], timeout=15, check=False).returncode == 0:
            resolved_local = cref
            break

    if resolved_local is not None:
        _run_cmd(["docker", "tag", resolved_local, grader_tag], timeout=60)
    else:
        built_local = False
        if isinstance(raw_row, dict):
            from v2.curation.verify_rebench_containers import (
                _build_local_swerebenchv1_image,
                _build_local_swerebenchv2_image,
            )

            icfg = raw_row.get("install_config") or {}
            if isinstance(icfg, str):
                try:
                    icfg = json.loads(icfg)
                except Exception:
                    icfg = {}
            repo_str = str(raw_row.get("repo") or route.get("repo") or "").strip()
            b_commit = str(raw_row.get("base_commit") or route.get("base_commit") or "").strip()
            img_hint = str(raw_row.get("docker_image") or raw_row.get("image_name") or digest_ref)
            target_cache_tag = f"{stripped}:latest" if ":" not in stripped.split("/")[-1] else stripped
            if "swerebenchv2/" in img_hint:
                built_local = _build_local_swerebenchv2_image(
                    repo=repo_str,
                    base_commit=b_commit,
                    install_cfg=icfg if isinstance(icfg, dict) else {},
                    target_tag=target_cache_tag,
                )
            elif "swerebench/" in img_hint:
                built_local = _build_local_swerebenchv1_image(
                    repo=repo_str,
                    base_commit=b_commit,
                    install_cfg=icfg if isinstance(icfg, dict) else {},
                    requirements_txt=str(raw_row.get("requirements") or "").strip(),
                    target_tag=target_cache_tag,
                )
            if built_local:
                _run_cmd(["docker", "tag", target_cache_tag, grader_tag], timeout=60)
        if not built_local:
            _run_cmd(["docker", "pull", "--quiet", digest_ref], timeout=900)
            _run_cmd(["docker", "tag", digest_ref, grader_tag], timeout=60)
    workdir = _resolve_image_workdir(route, grader_tag)
    # Pin shared base layers across batches so only repo delta layers download
    chk = _run_cmd(
        ["docker", "image", "inspect", "gemma-v2-rebench-base-anchor:keep"],
        timeout=15,
        check=False,
    )
    if chk.returncode != 0:
        _run_cmd(["docker", "tag", grader_tag, "gemma-v2-rebench-base-anchor:keep"], timeout=30, check=False)

    extra_rm = f" {shlex.quote(workdir)}" if workdir != "/testbed" else ""
    extra_ln = (
        f" && mkdir -p {shlex.quote(str(PurePosixPath(workdir).parent))} && ln -s /workspace {shlex.quote(workdir)}"
        if workdir != "/testbed"
        else ""
    )

    ctx_dir = build_ctx_root / slug
    ctx_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = (
        f"FROM {grader_tag}\n"
        "USER root\n"
        f"RUN rm -rf /workspace /testbed{extra_rm} /root /tmp /home /var/cache/apt /var/lib/apt/lists "
        "&& mkdir -p /workspace /root /tmp /home && chmod 1777 /tmp "
        f"&& ln -s /workspace /testbed{extra_ln} && rm -f /.env /issue.md\n"
        "RUN find /opt /usr/local /var /tmp -name .git -prune -exec rm -rf '{}' '+'\n"
        "ENV PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH\n"
        "ENV PIP_CACHE_DIR=/tmp/.cache/pip PYTHONPYCACHEPREFIX=/tmp/.cache/pycache "
        "XDG_CACHE_HOME=/tmp/.cache XDG_CONFIG_HOME=/tmp/.config XDG_DATA_HOME=/tmp/.local/share\n"
        'ENV CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PIP_NO_INDEX=1\n'
        "WORKDIR /workspace\n"
        "ENTRYPOINT []\n"
        'CMD ["sleep", "infinity"]\n'
    )
    (ctx_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
    _run_cmd(["docker", "build", "--quiet", "--network", "none", "-t", agent_tag, str(ctx_dir)], timeout=600)
    return {
        "instance_id": iid,
        "agent_tag": agent_tag,
        "grader_tag": grader_tag,
        "digest_ref": digest_ref,
        "workdir": workdir,
    }


def prepare_batch_images(
    task_ids: list[str],
    routes: dict[str, dict[str, Any]],
    build_ctx_root: Path,
    max_pull_workers: int = 4,
    raw_rows_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    """Pull or locally build & sanitize Docker images for one rolling batch of tasks."""
    build_ctx_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_pull_workers) as pool:
        futures = {
            iid: pool.submit(
                _prepare_single_task_images,
                routes[iid],
                build_ctx_root,
                (raw_rows_by_id or {}).get(iid),
            )
            for iid in task_ids
        }
        return {iid: fut.result() for iid, fut in futures.items()}


def prefetch_task_images(task_ids: list[str], routes: dict[str, dict[str, Any]], workers: int = 8) -> threading.Thread:
    """Pull the next batch's upstream images in the background while the current batch runs.

    Uncached pool tasks otherwise make every batch wait on Docker Hub with both GPUs idle. Pulls are idempotent and
    failures are ignored here; `prepare_batch_images` still resolves/builds each image itself when its batch starts.
    """

    def _pull(ref: str) -> None:
        if _run_cmd(["docker", "image", "inspect", ref], timeout=15, check=False).returncode != 0:
            _run_cmd(["docker", "pull", "--quiet", ref], timeout=3600, check=False)

    def _run() -> None:
        refs = [str(routes[i]["upstream_registry_digest"]) for i in task_ids if i in routes]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_pull, refs))

    t = threading.Thread(target=_run, name="image-prefetch", daemon=True)
    t.start()
    return t


def cleanup_batch_images(batch_map: dict[str, dict[str, str]], prune: bool = True) -> None:
    """Remove per-batch agent/grader Docker tags while keeping cached base/task images on NVMe."""
    refs: list[str] = []
    for info in batch_map.values():
        refs.extend([info["agent_tag"], info["grader_tag"]])
    if refs:
        _run_cmd(["docker", "rmi", "-f", *refs], timeout=300, check=False)
    if prune:
        _run_cmd(["docker", "builder", "prune", "-f", "--keep-storage", "50GB"], timeout=120, check=False)


# ---------------------------------------------------------------------------
# 3. Isolated Per-Task Grader (Full SWE-rebench Suite + Competition JUnit Parity)
# ---------------------------------------------------------------------------
def _copy_bytes_into_container(cid: str, data: bytes, dest_dir: str, filename: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        info = tarfile.TarInfo(filename)
        info.size = len(data)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
    proc = subprocess.run(["docker", "cp", "-", f"{cid}:{dest_dir}"], input=buf.getvalue(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp failed: {proc.stderr.decode('utf-8', errors='replace')}")


def grade_rebench_patch(
    *,
    raw_patch: str,
    route: dict[str, Any],
    grader_tag: str,
    grade_out_dir: Path,
) -> dict[str, Any]:
    """Grade `raw_patch` in a fresh `--network none` container with dual competition + rebench checks."""
    from swegemma.harness.verification import (
        _is_protected_test_or_config_path,
        _validate_junit_xml,
        strip_protected_files_from_patch,
    )

    grade_out_dir.mkdir(parents=True, exist_ok=True)
    if not raw_patch or not raw_patch.strip():
        return {
            "resolved": False,
            "strict_clean_patch": False,
            "error": "Empty agent patch",
            "test_exit_code": -1,
        }

    cleaned_patch = strip_protected_files_from_patch(raw_patch)
    strict_clean = cleaned_patch.strip() == raw_patch.strip()
    if not cleaned_patch.strip():
        return {
            "resolved": False,
            "strict_clean_patch": False,
            "error": "Patch contained only protected test/config paths",
            "test_exit_code": -1,
        }

    cid: str | None = None
    try:
        workdir = _resolve_image_workdir(route, grader_tag)
        q_wd = shlex.quote(workdir)
        cid = (
            _run_cmd(
                [
                    "docker",
                    "run",
                    "-d",
                    *ISOLATED_GRADER_LIMITS,
                    "--entrypoint",
                    "/bin/bash",
                    grader_tag,
                    "-c",
                    "sleep infinity",
                ],
                timeout=120,
            )
            .stdout.decode("utf-8")
            .strip()
        )
        _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git reset --hard {shlex.quote(route['base_commit'])} && git clean -fd"],
            timeout=120,
        )
        _copy_bytes_into_container(cid, cleaned_patch.encode("utf-8"), "/tmp", "agent.patch")
        apply_chk = _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git apply --check /tmp/agent.patch && git apply /tmp/agent.patch"],
            timeout=120,
            check=False,
        )
        if apply_chk.returncode != 0:
            return {
                "resolved": False,
                "strict_clean_patch": strict_clean,
                "error": "git apply failed: " + apply_chk.stderr.decode("utf-8", errors="replace")[-500:],
                "test_exit_code": -1,
            }

        # Double-check git status for any remaining protected paths
        changed = (
            _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git diff --name-only"], timeout=60)
            .stdout.decode("utf-8", errors="replace")
            .splitlines()
        )
        prot = [p for p in changed if p.strip() and _is_protected_test_or_config_path(p.strip())]
        if prot:
            return {
                "resolved": False,
                "strict_clean_patch": False,
                "error": f"Protected path modified after strip: {prot}",
                "test_exit_code": -1,
            }

        _copy_bytes_into_container(cid, route["test_patch"].encode("utf-8"), "/tmp", "grader.patch")
        _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git apply --check /tmp/grader.patch && git apply /tmp/grader.patch"],
            timeout=120,
        )

        # Cryptographic per-run JUnit path prevents imported library code from pre-fabricating XML reports
        nonce_xml = f"/tmp/rebench-grade-{uuid.uuid4().hex}.xml"
        test_script = (
            "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH; "
            "export PYTHONPYCACHEPREFIX=/tmp/.cache/pycache XDG_CACHE_HOME=/tmp/.cache; "
            "export PIP_NO_INDEX=1; "
            f"cd {q_wd} && {route['test_cmd']} --junitxml={nonce_xml}"
        )
        test_res = _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", test_script],
            timeout=420,
            check=False,
        )
        xml_bytes = _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cat {nonce_xml} 2>/dev/null || true"],
            timeout=60,
            check=False,
        ).stdout

        (grade_out_dir / "stdout.log").write_bytes(test_res.stdout)
        (grade_out_dir / "stderr.log").write_bytes(test_res.stderr)
        (grade_out_dir / "junit.xml").write_bytes(xml_bytes)

        try:
            report = _parse_junit_bytes(xml_bytes)
        except Exception as parse_exc:
            return {
                "resolved": False,
                "strict_clean_patch": strict_clean,
                "test_exit_code": test_res.returncode,
                "error": f"Invalid JUnit XML: {parse_exc}",
            }

        f2p = list(route["FAIL_TO_PASS"])
        p2p = list(route["PASS_TO_PASS"])
        req_issues = _compare_required_tests(report, f2p, p2p)
        official_ok, official_err = _validate_junit_xml(
            xml_bytes.decode("utf-8", errors="replace"),
            tuple(f2p),
            tuple(p2p),
            route["test_patch"],
        )
        resolved = (
            test_res.returncode == 0
            and not req_issues
            and official_ok
            and not report["counts"].get("FAILED")
            and not report["errors"]
        )
        err_msg: str | None = None
        if not resolved:
            if official_err:
                err_msg = official_err
            elif req_issues:
                err_msg = f"Required test failures: {req_issues[:3]}"
            else:
                err_msg = f"Test exit code {test_res.returncode}, counts={report['counts']}"

        return {
            "resolved": bool(resolved),
            "strict_clean_patch": bool(strict_clean),
            "test_exit_code": int(test_res.returncode),
            "collected_tests": int(report["collected"]),
            "status_counts": report["counts"],
            "required_issues_count": len(req_issues),
            "competition_junit_ok": bool(official_ok),
            "error": err_msg,
        }
    except subprocess.TimeoutExpired:
        return {
            "resolved": False,
            "strict_clean_patch": strict_clean,
            "test_exit_code": -1,
            "error": "Grader pytest timed out (>420s)",
        }
    finally:
        if cid and _run_cmd(["docker", "rm", "-f", cid], timeout=60, check=False).returncode == 124:
            # slow dockerd: finish the removal in the background; the grade above stands
            subprocess.Popen(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)


# ---------------------------------------------------------------------------
# 4. Isolated Worker Subprocess for One GPU Across a Batch of Items
# ---------------------------------------------------------------------------
def verify_populated_workspace_command() -> str:
    """Shell check confirming `/workspace` has `.git` and at least one non-hidden file or directory."""
    return (
        'test -d /workspace/.git && '
        '[ "$(find /workspace -maxdepth 2 -mindepth 1 -not -name \'.*\' | head -n 5 | wc -l)" -ge 1 ]'
    )


# Sandbox pre-warm (async RL `prewarm_sandboxes`): containers whose workspace was prepared before their rollout got a
# slot. The setup steps run_agent_sandbox would repeat on them are skipped (same commands, run earlier); _SETUP_FNS
# holds the real steps for the pre-warm itself (filled by _task_native_swegemma_env).
_PREWARMED: set[str] = set()
_SETUP_FNS: dict[str, Any] = {}
# Every ContainerManager of a running worker, so SIGTERM (async_rl stopping the worker at the end of a run) can remove
# in-flight containers instead of leaking them (~47 per run before this; they keep running `sleep infinity`).
_LIVE_MANAGERS: set[Any] = set()


def _prewarm_sandbox(manager: Any, snapshot_path: Path) -> float:
    """Start a container and run run_agent_sandbox's workspace setup on it (verified snapshot extraction, git exclude,
    baseline commit), then park it as the manager's idle container, which ContainerManager.start() hands out first.
    Returns the setup seconds, which the caller adds back to the session clock so the time budget counts the setup
    exactly as before. The skipped steps in run_agent_sandbox are the ones _task_native_swegemma_env no-ops anyway
    (wheels, editable install, test deps, test config) plus these three."""
    t0 = time.perf_counter()
    cid = manager.start()
    try:
        _SETUP_FNS["extract_snapshot"](manager, cid, snapshot_path)
        _SETUP_FNS["setup_git_exclude"](manager, cid)
        _SETUP_FNS["setup_baseline_commit"](manager, cid, "baseline")
    except Exception:
        manager.cleanup_all()
        raise
    with manager._lock:
        manager._active_containers.discard(cid)
        if getattr(manager, "_default_container_id", None) == cid:
            manager._default_container_id = None
        manager._idle_containers.append(cid)
    _PREWARMED.add(cid)
    return time.perf_counter() - t0


def _cleanup_live_managers(timeout_s: float = 12.0) -> int:
    """Remove the containers of every live manager in parallel (bounded by `timeout_s`); returns how many."""
    mgrs = list(_LIVE_MANAGERS)
    if not mgrs:
        return 0
    pool = ThreadPoolExecutor(max_workers=16)
    futs = [pool.submit(m.cleanup_all) for m in mgrs]
    deadline = time.monotonic() + timeout_s
    for f in futs:
        try:
            f.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)
    return len(mgrs)


@contextmanager
def _task_native_swegemma_env():
    """Bypass generic public80 wheel/editable hooks and enforce verified non-empty snapshot extraction."""
    from swegemma.harness import agent_runner

    hook_names = (
        "setup_container_wheels",
        "install_editable_package",
        "install_test_dependencies",
        "setup_workspace_test_config",
    )
    originals = {name: getattr(agent_runner, name) for name in hook_names}
    orig_extract = agent_runner.extract_snapshot

    def _verified_extract_snapshot(
        docker: Any,
        container_id: str,
        snapshot_path: Path,
        *,
        patch_path: Path | None = None,
        base_snapshot_path: Path | None = None,
    ) -> None:
        tar_to_copy = base_snapshot_path if base_snapshot_path is not None else snapshot_path
        orig_extract(
            docker,
            container_id,
            snapshot_path,
            patch_path=patch_path,
            base_snapshot_path=base_snapshot_path,
        )
        chk = docker.exec(container_id, verify_populated_workspace_command())
        if chk.exit_code != 0:
            # Fallback for base images without `pigz` where pre-existing `/workspace/.cache` fooled `ls -A`
            _copy_bytes_into_container(container_id, tar_to_copy.read_bytes(), "/tmp", "_verified_snap.tgz")
            ext_res = docker.exec(
                container_id,
                "tar --no-same-owner --unlink-first -xzf /tmp/_verified_snap.tgz -C /workspace "
                "&& rm -f /tmp/_verified_snap.tgz",
            )
            if ext_res.exit_code != 0:
                raise RuntimeError(
                    f"Fallback tar -xzf failed for {tar_to_copy.name}: {ext_res.stderr or ext_res.stdout}"
                )
            if patch_path is not None and patch_path.exists():
                _copy_bytes_into_container(container_id, patch_path.read_bytes(), "/tmp", "_inc.patch")
                p_res = docker.exec(
                    container_id,
                    "cd /workspace && git apply /tmp/_inc.patch && rm -f /tmp/_inc.patch",
                )
                if p_res.exit_code != 0:
                    raise RuntimeError(f"Incremental patch failed after fallback extract: {p_res.stderr}")
            docker.exec(
                container_id,
                "python3 -S -c \"import pathlib; ws=pathlib.Path('/workspace').resolve(); "
                "[p.unlink() for p in list(ws.rglob('*')) if p.is_symlink() and "
                "(p.resolve() == ws or ws.is_relative_to(p.resolve()))]\" 2>/dev/null || true",
            )
            chk2 = docker.exec(container_id, verify_populated_workspace_command())
            if chk2.exit_code != 0:
                raise RuntimeError(
                    f"Preflight abort: /workspace is missing .git or source files after extracting {tar_to_copy.name}!"
                )

    orig_exclude, orig_baseline = agent_runner.setup_git_exclude, agent_runner.setup_baseline_commit

    def _unless_prewarmed(fn: Any) -> Any:
        def wrapped(docker: Any, container_id: str, *a: Any, **kw: Any) -> None:
            if container_id not in _PREWARMED:
                fn(docker, container_id, *a, **kw)
        return wrapped

    try:
        for name in hook_names:
            setattr(agent_runner, name, lambda *a, **kw: None)
        _SETUP_FNS.update(extract_snapshot=_verified_extract_snapshot, setup_git_exclude=orig_exclude,
                          setup_baseline_commit=orig_baseline)
        setattr(agent_runner, "extract_snapshot", _unless_prewarmed(_verified_extract_snapshot))
        setattr(agent_runner, "setup_git_exclude", _unless_prewarmed(orig_exclude))
        setattr(agent_runner, "setup_baseline_commit", _unless_prewarmed(orig_baseline))
        yield
    finally:
        for name, val in originals.items():
            setattr(agent_runner, name, val)
        setattr(agent_runner, "extract_snapshot", orig_extract)
        setattr(agent_runner, "setup_git_exclude", orig_exclude)
        setattr(agent_runner, "setup_baseline_commit", orig_baseline)
        _SETUP_FNS.clear()


EARLY_STOP_AGENT_ERROR = "early_stopped_screen: task already has a resolved and an unresolved graded rollout"


class MixedStopTracker:
    """Screen-only early stop (owner rule, 2026-11-06): once a task has >= 1 resolved AND >= 1 unresolved graded
    rollout it is mixed by construction, so its remaining rollouts are skipped (queued) or cancelled (in flight).
    Infra-aborted rollouts (v2.train.grpo_data.is_infra_abort) are not outcomes. Never used for training rollouts."""

    def __init__(self) -> None:
        self.outcomes: dict[str, set[bool]] = collections.defaultdict(set)
        self.events: dict[str, asyncio.Event] = {}

    def event(self, iid: str) -> asyncio.Event:
        if iid not in self.events:
            self.events[iid] = asyncio.Event()
        return self.events[iid]

    def is_mixed(self, iid: str) -> bool:
        return len(self.outcomes.get(iid, ())) == 2

    def record(self, rec: dict[str, Any]) -> bool:
        """Register a finished rollout record; returns True when this record makes the task mixed. Keyed by the unit
        (v2.eval.async_protocol.unit_key: the task id, plus the pass when the record carries one)."""
        from v2.eval.async_protocol import unit_key
        from v2.train.grpo_data import is_infra_abort

        if rec.get("early_stopped") or is_infra_abort(rec):
            return False
        iid = unit_key(rec)
        was = self.is_mixed(iid)
        self.outcomes[iid].add(bool(rec.get("resolved")))
        if self.is_mixed(iid) and not was:
            self.event(iid).set()
            return True
        return False


async def run_unless_mixed(coro: Any, stop: asyncio.Event) -> tuple[bool, Any]:
    """Await `coro` unless `stop` fires first; on stop the coroutine is cancelled (its own finally blocks run).
    Returns (stopped, result)."""
    work = asyncio.ensure_future(coro)
    if stop.is_set():
        work.cancel()
    waiter = asyncio.ensure_future(stop.wait())
    try:
        done, _ = await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if work in done and not work.cancelled():
        return False, work.result()
    work.cancel()
    try:
        await work
    except BaseException:  # noqa: BLE001 - cancellation of the agent session is the expected path
        pass
    return True, None


async def _run_gpu_batch_items(kw: dict[str, Any]) -> list[dict[str, Any]]:
    from adk_submission import discover_adapters
    from swegemma.config import EvalConfig
    from swegemma.context import SwegemmaContext
    from swegemma.harness import agent_runner
    from swegemma.harness.verification import save_trace_artifact
    from swegemma.models import Task
    from swegemma.sandbox.docker import ContainerConfig, ContainerManager

    submission_dir = Path(kw["submission_dir"])
    results_dir = Path(kw["results_dir"])
    snapshots_dir = Path(kw["snapshots_dir"])
    tasks_path = Path(kw["tasks_path"])
    routes: dict[str, dict[str, Any]] = kw["routes"]
    batch_images: dict[str, dict[str, str]] = kw["batch_images"]
    items: list[dict[str, Any]] = kw["items"]
    concurrency = int(kw["concurrency"])

    results_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = results_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    budgets = load_submission_eval_budgets(submission_dir)
    session_cfgs = build_kaggle_adk_session_configs()
    adapters = discover_adapters(submission_dir)
    models = build_swegemma_0211_model_registry(
        api_base=kw["api_base"],
        served_model_id=kw["served_model_id"],
        declared_model_alias=kw["declared_model_alias"],
        submission_dir=submission_dir,
    )
    from v2.eval.async_protocol import item_run_key, read_admission_doc_cached, unit_key, wait_admitted
    from v2.eval.async_protocol import Router
    from v2.eval.request_capture import CURRENT_REQUEST_LOG, CURRENT_ROUTE, install_request_capture

    # Log the exact LiteLLM messages/tools of every call (traces omit ADK compaction summaries)
    install_request_capture()

    public_tasks: dict[str, Task] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            public_tasks[str(row["instance_id"])] = Task(**row)

    stream = bool(kw.get("stream"))
    # Async RL (both optional; absent = unchanged behaviour): items may carry `pass_idx` (run_key and the result
    # record then encode the pass, and stream mode keys image lifetimes by (task, pass) unit with a per-task
    # refcount), and `admission_file` gates each item's start on the learner's admission limit (see async_protocol).
    admission_file = kw.get("admission_file")
    if stream:
        first_image = f"gemma-v2-rebench-agent:{_image_slug(str(items[0]['instance_id']))}"
    else:
        first_image = next(iter(batch_images.values()))["agent_tag"]
    cfg = EvalConfig(
        tasks_path=tasks_path,
        snapshots_dir=snapshots_dir,
        wheels_dir=None,
        results_dir=results_dir,
        submission_dir=submission_dir,
        models=models,
        image=first_image,
        adapter_manifest=adapters,
        concurrency=1,
        display_mode="quiet",
        **budgets,
        **session_cfgs,
    )

    sem = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    prewarm_n = int(kw.get("prewarm_sandboxes") or 0)
    prewarm_gate = asyncio.Semaphore(prewarm_n) if prewarm_n > 0 else None
    # Run-time controls from the learner's PipelineGuard in the admission file (absent = unchanged): `slot_cap` caps
    # the rollouts holding a slot below `concurrency` (tool-headroom fallback), `prewarm: false` stops pre-warming.
    in_slot = {"n": 0}

    def _control() -> dict[str, Any]:
        return read_admission_doc_cached(Path(admission_file), 0.5) if admission_file else {}

    async def _acquire_slot() -> None:
        await sem.acquire()
        while True:
            cap = _control().get("slot_cap")
            if cap is None or in_slot["n"] < int(cap):
                break
            await asyncio.sleep(0.5)
        in_slot["n"] += 1
    # Async RL replicas (optional): `routing_file` (async_protocol.write_routing) pins every rollout to one of several
    # identical rollout servers; `free_slot_before_grading` releases the rollout slot once the agent is done.
    router = Router(Path(kw["routing_file"])) if kw.get("routing_file") else None
    free_slot_before_grading = bool(kw.get("free_slot_before_grading"))

    # Stream mode: a bounded producer prepares task images just in time, a few tasks ahead of the agents, so no batch
    # waits on its slowest rollout and the GPUs never idle on Docker Hub pulls. Each task's tags are removed after its
    # last rollout; image failures become per-rollout infra errors (dropped from RL groups) instead of crashing.
    ready: dict[str, asyncio.Event] = {}
    img_err: dict[str, str] = {}
    remaining: dict[str, int] = {}
    order: list[str] = []
    unit_iid: dict[str, str] = {}
    img_refs: dict[str, int] = {}  # units of a task currently holding its prepared images
    unit_holds: set[str] = set()
    prep_started: set[str] = set()
    img_locks: dict[str, asyncio.Lock] = {}
    stopped = {"flag": False}
    if stream:
        for it in items:
            u_ = unit_key(it)
            if u_ not in ready:
                ready[u_] = asyncio.Event()
                order.append(u_)
                unit_iid[u_] = str(it["instance_id"])
                img_locks.setdefault(unit_iid[u_], asyncio.Lock())
            remaining[u_] = remaining.get(u_, 0) + 1
        # Lookahead = units whose images are prepared (or preparing) ahead of the agents. Default: a unit holds its
        # slot until its LAST rollout finishes, so straggling units can stall preparation and leave the GPUs idle at
        # batch transitions (smoke_round2: ~1.5-2 min at 0-15 running sequences). With `prep_ahead_units` the slot is
        # released once every rollout of the unit holds a rollout slot, so N units are always prepared ahead;
        # image lifetimes stay refcounted per task (img_refs), independent of the lookahead.
        prep_ahead = int(kw.get("prep_ahead_units") or 0)
        unit_total = dict(remaining)
        unit_started: dict[str, int] = {}
        la_released: set[str] = set()
        lookahead = asyncio.Semaphore(prep_ahead if prep_ahead > 0
                                      else max(4, concurrency // max(remaining.values()) + 4))

        def _release_lookahead(u_: str) -> None:
            if u_ not in la_released:
                la_released.add(u_)
                lookahead.release()

        def _unit_started(u_: str) -> None:
            if prep_ahead > 0:
                unit_started[u_] = unit_started.get(u_, 0) + 1
                if unit_started[u_] >= unit_total[u_]:
                    _release_lookahead(u_)
        prep_sem = asyncio.Semaphore(int(kw.get("prep_workers", 4)))
        build_root = Path(kw["build_ctx_root"])
        raw_rows: dict[str, Any] = kw.get("raw_rows") or {}

        async def _prep(u_: str) -> None:
            iid_ = unit_iid[u_]
            async with prep_sem:
                try:
                    if stopped["flag"]:
                        img_err[u_] = "admission stopped before image preparation"
                        return
                    async with img_locks[iid_]:
                        # A task's images are built once while any of its units (passes) holds them
                        if img_refs.get(iid_, 0) == 0 or iid_ not in batch_images:
                            batch_images[iid_] = await asyncio.to_thread(
                                _prepare_single_task_images, routes[iid_], build_root, raw_rows.get(iid_)
                            )
                        img_refs[iid_] = img_refs.get(iid_, 0) + 1
                        unit_holds.add(u_)
                except Exception as exc:
                    img_err[u_] = f"image preparation failed: {type(exc).__name__}: {exc}"[:500]
                finally:
                    ready[u_].set()

        async def _producer() -> None:
            for u_ in order:
                await lookahead.acquire()
                prep_started.add(u_)
                asyncio.create_task(_prep(u_))

        async def _task_done(u_: str) -> None:
            remaining[u_] -= 1
            if remaining[u_] == 0:
                iid_ = unit_iid[u_]
                if u_ in prep_started:  # a unit skipped at admission stop may still be preparing: release after it
                    await ready[u_].wait()
                async with img_locks[iid_]:
                    if u_ in unit_holds:
                        unit_holds.discard(u_)
                        img_refs[iid_] -= 1
                        if img_refs[iid_] == 0 and kw.get("cleanup_images") and iid_ in batch_images:
                            await asyncio.to_thread(cleanup_batch_images, {iid_: batch_images[iid_]}, False)
                            if unit_iid[u_] != u_:  # pass units re-prepare on the next pass
                                del batch_images[iid_]
                _release_lookahead(u_)

    async def _infra_record(item: dict[str, Any], msg: str) -> dict[str, Any]:
        """A rollout that failed outside the agent: recorded (dropped from its RL group as infra), never raised."""
        iid_ = str(item["instance_id"])
        rec_ = {
            "run_key": item_run_key(item),
            "instance_id": iid_, "rollout_idx": int(item.get("rollout_idx", 0)), "repo": public_tasks[iid_].repo,
            "resolved": False, "strict_clean_patch": False, "error": msg,
            "agent_error": msg, "tool_calls": 0, "total_llm_calls": 0, "phase1_seconds": 0.0,
            "duration_seconds": 0.0, "patch_chars": 0, "trace_path": None, "request_log_path": None,
            "grade": {"resolved": False, "error": msg},
        }
        if "pass_idx" in item:
            rec_["pass_idx"] = int(item["pass_idx"])
        if admission_file:
            rec_["t_end"] = time.time()
        async with file_lock:
            with (results_dir / "task_results.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec_, sort_keys=True) + "\n")
        return rec_

    # Early stop of SCREENING rollouts (payload `early_stop_mixed`, set only by screening payloads, never by training;
    # owner rule 2026-11-06, one contract with the pool agent): once a unit has >= 1 resolved and >= 1 unresolved
    # non-infra graded record (MixedStopTracker), its queued rollouts are skipped and its in-flight agents cancelled
    # (run_unless_mixed; containers removed, slots freed, no grading). Each such rollout is a task_results.jsonl row
    # with early_stopped=true, early_stop_phase=queued|in_flight and agent_error=EARLY_STOP_AGENT_ERROR (so
    # is_infra_abort drops it from every outcome count); a restarted worker skips those rows (done keys) and rebuilds
    # the tracker from the existing rows.
    early_stop = MixedStopTracker() if kw.get("early_stop_mixed") else None
    if early_stop is not None and (results_dir / "task_results.jsonl").is_file():
        for ln in (results_dir / "task_results.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                early_stop.record(json.loads(ln))

    async def _write_early_stopped(item: dict[str, Any], phase: str, seconds: float) -> dict[str, Any]:
        iid_ = str(item["instance_id"])
        rec_ = {
            "run_key": item_run_key(item), "instance_id": iid_, "rollout_idx": int(item.get("rollout_idx", 0)),
            "repo": public_tasks[iid_].repo, "resolved": False, "strict_clean_patch": False,
            "early_stopped": True, "early_stop_phase": phase,
            # agent_error makes v2.train.grpo_data.is_infra_abort drop the row everywhere: it is not an outcome
            "error": EARLY_STOP_AGENT_ERROR, "agent_error": EARLY_STOP_AGENT_ERROR,
            "tool_calls": 0, "total_llm_calls": 0, "phase1_seconds": round(seconds, 2),
            "duration_seconds": round(seconds, 2), "patch_chars": 0, "trace_path": None, "request_log_path": None,
            "grade": None,
        }
        if "pass_idx" in item:
            rec_["pass_idx"] = int(item["pass_idx"])
        if admission_file:
            rec_["t_end"] = time.time()
        async with file_lock:
            with (results_dir / "task_results.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec_, sort_keys=True) + "\n")
        return rec_

    async def _guarded_inner(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        # one rollout's unexpected exception (e.g. a Docker CLI timeout in grading) must not kill the worker and
        # every other in-flight rollout with it: it becomes that rollout's infra record
        try:
            return await _execute_one_inner(idx, item)
        except Exception as exc:
            return await _infra_record(item, f"worker crash: {type(exc).__name__}: {exc}"[:500])

    async def _execute_one(idx: int, item: dict[str, Any]) -> dict[str, Any] | None:
        if admission_file:
            if not await wait_admitted(Path(admission_file), int(item["unit"])):
                stopped["flag"] = True
                if stream:
                    await _task_done(unit_key(item))
                return None
        if not stream:
            return await _guarded_inner(idx, item)
        iid0 = str(item["instance_id"])
        u0 = unit_key(item)
        await ready[u0].wait()
        try:
            if early_stop is not None and early_stop.is_mixed(u0):
                return await _write_early_stopped(item, "queued", 0.0)
            if u0 in img_err:
                return await _infra_record(item, img_err[u0])
            return await _guarded_inner(idx, item)
        finally:
            await _task_done(u0)

    def _new_manager(iid_: str) -> Any:
        mgr = ContainerManager(
            ContainerConfig(
                image=batch_images[iid_]["agent_tag"],
                network_mode="none",
                mem_limit="4g",
                working_dir="/workspace",
                timeout_seconds=int(budgets["timeout_seconds"]),
                environment={
                    "TEST_TMPDIR": "/tmp",
                    "PIP_NO_INDEX": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                },
            )
        )
        _LIVE_MANAGERS.add(mgr)
        return mgr

    async def _execute_one_inner(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        manager: Any = None
        prewarm_s: float | None = None
        if prewarm_gate is not None and _control().get("prewarm", True):
            # Pre-warm while waiting for a rollout slot (at most `prewarm_sandboxes` at a time). Every rollout passes
            # this gate before the slot semaphore, so the slot waiters are pre-warmed ones and a freed slot goes to a
            # ready workspace. A failed pre-warm falls back to the normal in-slot setup (which records its error).
            await prewarm_gate.acquire()
            try:
                manager = _new_manager(str(item["instance_id"]))
                prewarm_s = await asyncio.to_thread(
                    _prewarm_sandbox, manager, snapshots_dir / f"{item['instance_id']}.tgz")
            except Exception:
                prewarm_s = None
            try:
                await _acquire_slot()
            finally:
                prewarm_gate.release()
        else:
            await _acquire_slot()
        held = {"slot": True}
        if stream:
            _unit_started(unit_key(item))

        def _free_slot() -> None:
            if held["slot"]:
                held["slot"] = False
                in_slot["n"] -= 1
                sem.release()
                if router is not None:
                    router.release(item_run_key(item))

        try:
            if early_stop is not None and early_stop.is_mixed(unit_key(item)):  # decided while waiting for a slot
                return await _write_early_stopped(item, "queued", 0.0)  # finally: slot freed, pre-warmed box removed
            iid = str(item["instance_id"])
            rollout_idx = int(item.get("rollout_idx", 0))
            run_key = item_run_key(item)
            base_task = public_tasks[iid]
            # Use run_key for distinct log & trace filenames across multi-rollout runs
            task_obj = Task(
                instance_id=run_key,
                repo=base_task.repo,
                base_commit=base_task.base_commit,
                problem_statement=base_task.problem_statement,
                hints_text=base_task.hints_text,
                created_at=base_task.created_at,
            )
            img_info = batch_images[iid]
            if manager is None:
                manager = _new_manager(iid)
            ctx = SwegemmaContext(
                docker_manager=manager,
                container_id="",
                problem_statement=task_obj.problem_statement,
                hints_text=task_obj.hints_text or "",
                task=task_obj,
                repo=task_obj.repo,
                graph_dir=cfg.graph_dir,
                embeddings_dir=cfg.embeddings_dir,
                budget=cfg.budget,
                harness=cfg.harness,
            )
            if prewarm_s is not None:
                # The session clock starts at context creation, before container start and setup; a pre-warmed
                # rollout did that work already, so its clock starts that many seconds in (same budget as before).
                ctx.task_start_time -= prewarm_s
                ctx.start_time = ctx.task_start_time

            t0 = time.monotonic()
            agent_patch = ""
            agent_error: str | None = None
            trace_path: str | None = None
            request_log = results_dir / "requests" / f"{run_key.replace('/', '__')}.jsonl"
            request_log.unlink(missing_ok=True)
            CURRENT_REQUEST_LOG.set(request_log)
            if router is not None:  # pin this rollout to one replica; each call asks the router (request_capture)
                CURRENT_ROUTE.set(functools.partial(router.base_for, run_key))
            cancelled = False
            try:
                agent_coro = agent_runner.run_agent_sandbox(
                    manager,
                    cfg,
                    task_obj,
                    snapshots_dir / f"{iid}.tgz",
                    task_index=idx + 1,
                    total_tasks=len(items),
                    context=ctx,
                )
                if early_stop is not None:
                    cancelled, out_ = await run_unless_mixed(agent_coro, early_stop.event(unit_key(item)))
                    if cancelled:
                        raise _ScreenCancelled()
                else:
                    out_ = await agent_coro
                agent_patch, agent_error, trace = out_
                trace_path = save_trace_artifact(trace, results_dir, run_key)
                (patches_dir / f"{run_key}.patch").write_text(
                    agent_patch or "", encoding="utf-8", newline="\n"
                )
            except _ScreenCancelled:
                pass
            except Exception as run_exc:
                agent_error = f"{type(run_exc).__name__}: {run_exc}"
            finally:
                try:
                    manager.cleanup_all()
                except Exception:
                    pass
                _LIVE_MANAGERS.discard(manager)
            if cancelled:  # early-stopped screening rollout: no grading
                _free_slot()
                return await _write_early_stopped(item, "in_flight", time.monotonic() - t0 + (prewarm_s or 0.0))

            phase1_sec = round(time.monotonic() - t0 + (prewarm_s or 0.0), 2)
            if free_slot_before_grading:  # grading needs no model: the next rollout may start meanwhile
                _free_slot()
            grade = await asyncio.to_thread(
                grade_rebench_patch,
                raw_patch=agent_patch or "",
                route=routes[iid],
                grader_tag=img_info["grader_tag"],
                grade_out_dir=results_dir / "grading" / run_key,
            )
            total_sec = round(time.monotonic() - t0 + (prewarm_s or 0.0), 2)

            rec = {
                "run_key": run_key,
                "instance_id": iid,
                "rollout_idx": rollout_idx,
                "repo": base_task.repo,
                "resolved": bool(grade["resolved"]),
                "strict_clean_patch": bool(grade.get("strict_clean_patch", False)),
                "error": grade.get("error") or agent_error,
                "agent_error": agent_error,
                "tool_calls": int(ctx.tool_calls_used),
                "total_llm_calls": int(ctx.llm_calls_used),
                "phase1_seconds": phase1_sec,
                "duration_seconds": total_sec,
                "patch_chars": len(agent_patch or ""),
                "trace_path": trace_path,
                "request_log_path": str(request_log) if request_log.is_file() else None,
                "grade": grade,
            }
            if "pass_idx" in item:
                rec["pass_idx"] = int(item["pass_idx"])
            if prewarm_s is not None:
                rec["prewarm_seconds"] = round(prewarm_s, 2)
            if admission_file:  # async RL: wall end, for server-outage attribution (async_rl.mark_outage)
                rec["t_end"] = time.time()
            async with file_lock:
                with (results_dir / "task_results.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    f.flush()
            if early_stop is not None:
                early_stop.record(rec)
            return rec
        finally:
            _free_slot()
            if manager is not None and manager in _LIVE_MANAGERS:  # failed before the agent ran
                try:
                    manager.cleanup_all()
                except Exception:
                    pass
                _LIVE_MANAGERS.discard(manager)

    with _task_native_swegemma_env():
        producer = asyncio.create_task(_producer()) if stream else None
        out_rows = await asyncio.gather(*(_execute_one(i, it) for i, it in enumerate(items)))
        if producer is not None:
            await producer
    return list(out_rows)


class _ScreenCancelled(Exception):
    """An early-stopped screening rollout (internal to _run_gpu_batch_items)."""


def _worker_batch_entry(payload_path: str) -> int:
    _raise_nofile_soft_limit()
    kw = json.loads(Path(payload_path).read_text(encoding="utf-8"))
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
    import signal

    def _on_term(signum: int, frame: Any) -> None:  # async_rl stops the worker with SIGTERM at the end of a run
        _cleanup_live_managers()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_term)
    asyncio.run(_run_gpu_batch_items(kw))
    return 0


# ---------------------------------------------------------------------------
# 5. Top-Level Rolling-Batch Dual-GPU Orchestrator
# ---------------------------------------------------------------------------
def build_mode_batch_items(
    mode: str,
    batch_tids: list[str],
    *,
    num_rollouts: int,
    secondary_num_rollouts: int | None = None,
    done_keys_0: set[str] | None = None,
    done_keys_1: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build GPU0 and GPU1 work item lists for `ab_paired`, `dual_replica_rollouts`, or `teacher_student_rollouts`."""
    dk0 = done_keys_0 or set()
    dk1 = done_keys_1 or set()
    if mode == "ab_paired":
        # One paired seed per rollout index on both arms (`--num-rollouts` used to be silently ignored here)
        n_ab = max(1, int(num_rollouts))
        items_0 = [
            {"instance_id": iid, "rollout_idx": r, "num_rollouts": n_ab}
            for iid in batch_tids
            for r in range(n_ab)
            if (iid if n_ab == 1 else f"{iid}__r{r}") not in dk0
        ]
        items_1 = [
            {"instance_id": iid, "rollout_idx": r, "num_rollouts": n_ab}
            for iid in batch_tids
            for r in range(n_ab)
            if (iid if n_ab == 1 else f"{iid}__r{r}") not in dk1
        ]
        return items_0, items_1
    if mode == "teacher_student_rollouts":
        n0 = max(1, int(num_rollouts))
        n1 = max(1, int(secondary_num_rollouts or num_rollouts))
        items_0 = [
            {"instance_id": iid, "rollout_idx": r, "num_rollouts": n0}
            for iid in batch_tids
            for r in range(n0)
            if (iid if n0 == 1 else f"{iid}__r{r}") not in dk0
        ]
        items_1 = [
            {"instance_id": iid, "rollout_idx": r, "num_rollouts": n1}
            for iid in batch_tids
            for r in range(n1)
            if (iid if n1 == 1 else f"{iid}__r{r}") not in dk1
        ]
        return items_0, items_1

    done_any = dk0 | dk1
    n = max(1, int(num_rollouts))
    all_items = [
        {"instance_id": iid, "rollout_idx": r, "num_rollouts": n}
        for iid in batch_tids
        for r in range(n)
    ]
    items_0 = [
        it
        for it in all_items[0::2]
        if (it["instance_id"] if it["num_rollouts"] == 1 else f"{it['instance_id']}__r{it['rollout_idx']}")
        not in done_any
    ]
    items_1 = [
        it
        for it in all_items[1::2]
        if (it["instance_id"] if it["num_rollouts"] == 1 else f"{it['instance_id']}__r{it['rollout_idx']}")
        not in done_any
    ]
    return items_0, items_1


def _summarize_rollout_rows(rows: list[dict[str, Any]], num_rollouts: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_task[str(r["instance_id"])].append(r)
    pass_at_g = sum(1 for rs in by_task.values() if any(x["resolved"] for x in rs))
    clean_resolved = sum(1 for x in rows if x["resolved"] and x["strict_clean_patch"])
    return {
        "num_rollouts_per_task": int(num_rollouts),
        "total_rollouts": len(rows),
        "total_resolved_rollouts": sum(1 for x in rows if x["resolved"]),
        "strict_clean_resolved_rollouts": clean_resolved,
        "tasks_with_at_least_1_resolved": pass_at_g,
        "pass_at_g_rate": round(pass_at_g / max(1, len(by_task)), 4),
        "per_task_solve_counts": {k: sum(1 for x in v if x["resolved"]) for k, v in sorted(by_task.items())},
        "failure_modes": summarize_failure_modes(rows),
    }


def append_missing_records(
    shard_dir: Path, expected_items: list[dict[str, Any]], routes_all: dict[str, dict[str, Any]], worker_rc: int
) -> int:
    """Check 5 (Trustworthy Grading): every expected rollout without a record gets a crash/timeout record (infra
    abort), so failures never disappear from task_results.jsonl. Returns the number of records appended."""
    if not expected_items:
        return 0
    from v2.eval.async_protocol import item_run_key

    res_file = shard_dir / "task_results.jsonl"
    seen_keys: set[str] = set()
    if res_file.is_file():
        for ln in res_file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                seen_keys.add(str(json.loads(ln).get("run_key", "")))
    missing_records: list[dict[str, Any]] = []
    for it in expected_items:
        iid = str(it["instance_id"])
        ridx = int(it.get("rollout_idx", 0))
        rkey = item_run_key(it)
        if rkey not in seen_keys:
            mrec = {
                "run_key": rkey,
                "instance_id": iid,
                "rollout_idx": ridx,
                "repo": str(routes_all[iid].get("repo", "")),
                "resolved": False,
                "strict_clean_patch": False,
                "error": f"Worker crash or hard timeout before completion (rc={worker_rc})",
                "agent_error": f"WorkerSubprocessExit({worker_rc})",
                "tool_calls": 0,
                "total_llm_calls": 0,
                "phase1_seconds": 0.0,
                "duration_seconds": 0.0,
                "patch_chars": 0,
                "trace_path": None,
                "grade": {
                    "resolved": False,
                    "strict_clean_patch": False,
                    "test_exit_code": -1,
                    "error": f"Worker crash or hard timeout (rc={worker_rc})",
                },
            }
            if "pass_idx" in it:
                mrec["pass_idx"] = int(it["pass_idx"])
            missing_records.append(mrec)
    if missing_records:
        with res_file.open("a", encoding="utf-8", newline="\n") as f:
            for mrec in missing_records:
                f.write(json.dumps(mrec, sort_keys=True) + "\n")
            f.flush()
    return len(missing_records)


def load_rebench_task_inputs(
    rebench_dir_arg: Path | str,
    tasks_path_arg: Path | str,
    *,
    max_tasks: int = 0,
    corpus_rows: Any = None,
) -> tuple[Path, Path, Path, dict[str, dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    """(rebench_dir, tasks_path, snapshots_dir, routes_all, task_ids, raw_rows_by_id) as the orchestrator uses them
    (shared with the async RL rollout driver, v2/train/async_rl.py)."""
    rebench_dir = Path(rebench_dir_arg).resolve()
    tasks_path = Path(tasks_path_arg).resolve()
    snapshots_dir = rebench_dir / "snapshots"
    routes_file = (
        rebench_dir / "evaluator_routes.json"
        if (rebench_dir / "evaluator_routes.json").is_file()
        else rebench_dir / "evaluator_routes36.json"
    )
    routes_all: dict[str, dict[str, Any]] = json.loads(routes_file.read_text(encoding="utf-8"))

    task_ids: list[str] = []
    raw_rows_by_id: dict[str, dict[str, Any]] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tdoc = json.loads(line)
            iid = str(tdoc["instance_id"])
            task_ids.append(iid)
            raw_rows_by_id[iid] = tdoc
    if max_tasks and max_tasks > 0:
        task_ids = task_ids[:max_tasks]

    raw_corpus_arg = corpus_rows
    corpus_paths: list[Path] = []
    if isinstance(raw_corpus_arg, Path):
        corpus_paths = [raw_corpus_arg]
    elif isinstance(raw_corpus_arg, (list, tuple)):
        corpus_paths = [Path(p) for p in raw_corpus_arg]
    elif raw_corpus_arg is None:
        default_corpus = Path("/path/to/gemma-v2/eval_data/swe_rebench_corpus/combined_all_waves_1_to_8.json")
        if default_corpus.is_file():
            corpus_paths = [default_corpus]
    for cp in corpus_paths:
        if cp.is_file():
            for crow in json.loads(cp.read_text(encoding="utf-8")):
                if isinstance(crow, dict) and crow.get("instance_id") in raw_rows_by_id:
                    raw_rows_by_id[str(crow["instance_id"])] = {**crow, **raw_rows_by_id[str(crow["instance_id"])]}
    return rebench_dir, tasks_path, snapshots_dir, routes_all, task_ids, raw_rows_by_id


def orchestrate_rebench_run(args: argparse.Namespace) -> dict[str, Any]:
    _raise_nofile_soft_limit()
    out_root = args.results_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rebench_dir, tasks_path, snapshots_dir, routes_all, task_ids, raw_rows_by_id = load_rebench_task_inputs(
        args.rebench_dir, args.tasks_path, max_tasks=int(args.max_tasks or 0), corpus_rows=getattr(args, "corpus_rows", None)
    )

    pri_sub = args.primary_submission.resolve()
    sec_sub = (args.secondary_submission or args.primary_submission).resolve()
    sec_model = args.secondary_model_path or args.primary_model_path
    sec_alias = args.secondary_alias or args.primary_alias
    sec_gpu_mem = float(getattr(args, "secondary_gpu_memory_utilization", None) or args.gpu_memory_utilization)
    sec_concurrency = int(getattr(args, "secondary_concurrency_per_gpu", None) or args.concurrency_per_gpu)
    sec_num_rollouts = int(getattr(args, "secondary_num_rollouts", None) or args.num_rollouts)

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
        gpu_memory_utilization=sec_gpu_mem,
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
    t_run0 = time.monotonic()

    try:
        log_mode = "a" if getattr(args, "resume", False) else "w"
        log0 = (logs_dir / "vllm_gpu0.log").open(log_mode, encoding="utf-8")
        log1 = (logs_dir / "vllm_gpu1.log").open(log_mode, encoding="utf-8")
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
        ready_0 = wait_for_vllm_health(args.base_port, proc_0, timeout_sec=args.startup_timeout)
        ready_1 = wait_for_vllm_health(args.base_port + 1, proc_1, timeout_sec=args.startup_timeout)

        if args.mode == "ab_paired":
            dir_0 = out_root / "anchor_gpu0"
            dir_1 = out_root / "candidate_gpu1"
        elif args.mode == "teacher_student_rollouts":
            dir_0 = out_root / "teacher_gpu0"
            dir_1 = out_root / "student_gpu1"
        else:
            dir_0 = out_root / "shard0_gpu0"
            dir_1 = out_root / "shard1_gpu1"
        dir_0.mkdir(parents=True, exist_ok=True)
        dir_1.mkdir(parents=True, exist_ok=True)

        done_keys_0: set[str] = set()
        done_keys_1: set[str] = set()
        if getattr(args, "resume", False):
            for s_dir, s_set in ((dir_0, done_keys_0), (dir_1, done_keys_1)):
                rf = s_dir / "task_results.jsonl"
                if rf.is_file():
                    for ln in rf.read_text(encoding="utf-8").splitlines():
                        if ln.strip():
                            s_set.add(str(json.loads(ln).get("run_key", "")))

        stream = bool(getattr(args, "stream", False))
        batch_size = len(task_ids) if stream else max(1, int(args.batch_size))
        batches = [task_ids[i : i + batch_size] for i in range(0, len(task_ids), batch_size)]

        for b_idx, batch_tids in enumerate(batches):
            items_0, items_1 = build_mode_batch_items(
                args.mode,
                batch_tids,
                num_rollouts=int(args.num_rollouts),
                secondary_num_rollouts=sec_num_rollouts,
                done_keys_0=done_keys_0,
                done_keys_1=done_keys_1,
            )

            needed_tids = sorted({str(it["instance_id"]) for it in (items_0 + items_1)})
            if not needed_tids:
                continue

            # Watchdog: verify both vLLM servers respond before starting each batch (auto-recovers hung APIServer)
            if items_0 and not _is_vllm_responsive(args.base_port, proc_0):
                terminate_process_group(proc_0)
                proc_0 = server_popen(
                    cmd_0,
                    env={**base_server_env, "CUDA_VISIBLE_DEVICES": "0"},
                    stdout=log0,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_for_vllm_health(args.base_port, proc_0, timeout_sec=args.startup_timeout)
            if items_1 and not _is_vllm_responsive(args.base_port + 1, proc_1):
                terminate_process_group(proc_1)
                proc_1 = server_popen(
                    cmd_1,
                    env={**base_server_env, "CUDA_VISIBLE_DEVICES": "1"},
                    stdout=log1,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_for_vllm_health(args.base_port + 1, proc_1, timeout_sec=args.startup_timeout)

            batch_routes = {iid: routes_all[iid] for iid in needed_tids}
            stream_extra: dict[str, Any] = {}
            if stream:
                if args.mode == "dual_replica_rollouts":
                    # Whole tasks per GPU, so each worker owns (and cleans up) its tasks' images
                    n_r = max(1, int(args.num_rollouts))
                    done_any = done_keys_0 | done_keys_1

                    def _task_items(ts: list[str]) -> list[dict[str, Any]]:
                        return [{"instance_id": i, "rollout_idx": r, "num_rollouts": n_r} for i in ts for r in range(n_r)
                                if (i if n_r == 1 else f"{i}__r{r}") not in done_any]

                    items_0, items_1 = _task_items(needed_tids[0::2]), _task_items(needed_tids[1::2])
                batch_imgs = {i: {"agent_tag": f"gemma-v2-rebench-agent:{_image_slug(i)}",
                                  "grader_tag": f"gemma-v2-rebench-grader:{_image_slug(i)}"} for i in needed_tids}
                stream_extra = {
                    "stream": True,
                    "build_ctx_root": str(out_root / "tmp_docker_contexts" / "stream"),
                    "raw_rows": {i: raw_rows_by_id.get(i) for i in needed_tids},
                    # ab_paired shares each task's images across both GPUs, so only the parent cleans up at the end
                    "cleanup_images": args.mode == "dual_replica_rollouts" and not args.keep_batch_images,
                    # screen-only early stop: stream dual_replica keeps each task's rollouts in one worker
                    "early_stop_mixed": bool(getattr(args, "early_stop_mixed", False))
                    and args.mode == "dual_replica_rollouts",
                }
            else:
                batch_imgs = prepare_batch_images(
                    needed_tids,
                    batch_routes,
                    out_root / "tmp_docker_contexts" / f"batch_{b_idx}",
                    raw_rows_by_id=raw_rows_by_id,
                )
                if b_idx + 1 < len(batches):
                    prefetch_task_images(batches[b_idx + 1], routes_all)
            try:

                p0_json = out_root / f"payload_b{b_idx}_gpu0.json"
                p1_json = out_root / f"payload_b{b_idx}_gpu1.json"
                p0_json.write_text(
                    json.dumps(
                        {
                            "tasks_path": str(tasks_path),
                            "snapshots_dir": str(snapshots_dir),
                            "results_dir": str(dir_0),
                            "submission_dir": str(pri_sub),
                            "api_base": f"http://127.0.0.1:{args.base_port}/v1",
                            "served_model_id": args.primary_model_path,
                            "declared_model_alias": args.primary_alias,
                            "concurrency": args.concurrency_per_gpu,
                            "routes": batch_routes,
                            "batch_images": {} if stream else batch_imgs,
                            "items": items_0,
                            **stream_extra,
                        }
                    ),
                    encoding="utf-8",
                )
                p1_json.write_text(
                    json.dumps(
                        {
                            "tasks_path": str(tasks_path),
                            "snapshots_dir": str(snapshots_dir),
                            "results_dir": str(dir_1),
                            "submission_dir": str(sec_sub),
                            "api_base": f"http://127.0.0.1:{args.base_port + 1}/v1",
                            "served_model_id": sec_model,
                            "declared_model_alias": sec_alias,
                            "concurrency": sec_concurrency,
                            "routes": batch_routes,
                            "batch_images": {} if stream else batch_imgs,
                            "items": items_1,
                            **stream_extra,
                        }
                    ),
                    encoding="utf-8",
                )

                w_log0 = (logs_dir / f"worker_b{b_idx}_gpu0.log").open("w", encoding="utf-8")
                w_log1 = (logs_dir / f"worker_b{b_idx}_gpu1.log").open("w", encoding="utf-8")
                w0 = subprocess.Popen(
                    [args.eval_python, "-m", "v2.eval.run_rebench_v2", "_worker_batch", str(p0_json)],
                    stdout=w_log0,
                    stderr=subprocess.STDOUT,
                ) if items_0 else None
                w1 = subprocess.Popen(
                    [args.eval_python, "-m", "v2.eval.run_rebench_v2", "_worker_batch", str(p1_json)],
                    stdout=w_log1,
                    stderr=subprocess.STDOUT,
                ) if items_1 else None

                fail_streak_0 = 0
                fail_streak_1 = 0
                while (w0 is not None and w0.poll() is None) or (w1 is not None and w1.poll() is None):
                    time.sleep(15.0)
                    if w0 is not None and w0.poll() is None:
                        # A busy server can answer slowly under 48+ agents; restarting it kills every in-flight
                        # request, so restart only when the process died or it stayed unresponsive ~5 min
                        if _is_vllm_responsive(args.base_port, proc_0, timeout_sec=60.0):
                            fail_streak_0 = 0
                        else:
                            fail_streak_0 += 1
                            if fail_streak_0 >= 4 or proc_0 is None or proc_0.poll() is not None:
                                print(f"[watchdog] restarting vLLM gpu0 (streak={fail_streak_0})", flush=True)
                                terminate_process_group(proc_0)
                                proc_0 = server_popen(
                                    cmd_0,
                                    env={**base_server_env, "CUDA_VISIBLE_DEVICES": "0"},
                                    stdout=log0,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                                wait_for_vllm_health(args.base_port, proc_0, timeout_sec=args.startup_timeout)
                                fail_streak_0 = 0
                    if w1 is not None and w1.poll() is None:
                        # A busy server can answer slowly under 48+ agents; restarting it kills every in-flight
                        # request, so restart only when the process died or it stayed unresponsive ~5 min
                        if _is_vllm_responsive(args.base_port + 1, proc_1, timeout_sec=60.0):
                            fail_streak_1 = 0
                        else:
                            fail_streak_1 += 1
                            if fail_streak_1 >= 4 or proc_1 is None or proc_1.poll() is not None:
                                print(f"[watchdog] restarting vLLM gpu1 (streak={fail_streak_1})", flush=True)
                                terminate_process_group(proc_1)
                                proc_1 = server_popen(
                                    cmd_1,
                                    env={**base_server_env, "CUDA_VISIBLE_DEVICES": "1"},
                                    stdout=log1,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                                wait_for_vllm_health(args.base_port + 1, proc_1, timeout_sec=args.startup_timeout)
                                fail_streak_1 = 0

                rc0 = w0.wait() if w0 is not None else 0
                rc1 = w1.wait() if w1 is not None else 0
                w_log0.close()
                w_log1.close()
                p0_json.unlink(missing_ok=True)
                p1_json.unlink(missing_ok=True)

                # Check 5 (Trustworthy Grading): crashes/timeouts NEVER disappear from task_results.jsonl!
                for worker_rc, expected_items, shard_dir in (
                    (rc0, items_0, dir_0),
                    (rc1, items_1, dir_1),
                ):
                    append_missing_records(shard_dir, expected_items, routes_all, worker_rc)
            finally:
                if not args.keep_batch_images:
                    cleanup_batch_images(batch_imgs)
                shutil.rmtree(out_root / "tmp_docker_contexts", ignore_errors=True)

        def _read_jsonl(p: Path) -> list[dict[str, Any]]:
            if not p.is_file():
                return []
            return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

        rows_0 = _read_jsonl(dir_0 / "task_results.jsonl")
        rows_1 = _read_jsonl(dir_1 / "task_results.jsonl")

        summary: dict[str, Any] = {
            "mode": args.mode,
            "total_unique_tasks": len(task_ids),
            "elapsed_wall_seconds": round(time.monotonic() - t_run0, 2),
            "vllm_ready_seconds": {"gpu0": ready_0, "gpu1": ready_1},
        }
        if args.mode == "ab_paired":
            res_0 = sum(1 for r in rows_0 if r["resolved"])
            res_1 = sum(1 for r in rows_1 if r["resolved"])
            summary["gpu0_anchor"] = {
                "resolved": res_0,
                "total": len(rows_0),
                "rate": round(res_0 / max(1, len(rows_0)), 4),
                "failure_modes": summarize_failure_modes(rows_0),
            }
            summary["gpu1_candidate"] = {
                "resolved": res_1,
                "total": len(rows_1),
                "rate": round(res_1 / max(1, len(rows_1)), 4),
                "failure_modes": summarize_failure_modes(rows_1),
            }
            if args.primary_alias == sec_alias and len(rows_0) == len(rows_1) and rows_0:
                # One paired "seed" per rollout index (each run must list every task exactly once)
                seeds = sorted({int(r.get("rollout_idx", 0)) for r in rows_0} & {int(r.get("rollout_idx", 0)) for r in rows_1})
                gate = evaluate_paired_promotion(
                    anchor_runs=[[r for r in rows_0 if int(r.get("rollout_idx", 0)) == k] for k in seeds],
                    candidate_runs=[[r for r in rows_1 if int(r.get("rollout_idx", 0)) == k] for k in seeds],
                    projected_120task_hours=5.0,
                    min_seeds=1,
                )
                summary["promotion_gate"] = gate.to_dict()
        elif args.mode == "teacher_student_rollouts":
            summary["teacher_gpu0"] = _summarize_rollout_rows(rows_0, int(args.num_rollouts))
            summary["student_gpu1"] = _summarize_rollout_rows(rows_1, sec_num_rollouts)
            t_solved = {
                str(r["instance_id"]) for r in rows_0 if r.get("resolved") and r.get("strict_clean_patch", True)
            }
            s_solved = {
                str(r["instance_id"]) for r in rows_1 if r.get("resolved") and r.get("strict_clean_patch", True)
            }
            summary["teacher_student_union"] = {
                "tasks_solved_by_teacher": len(t_solved),
                "tasks_solved_by_student": len(s_solved),
                "tasks_solved_by_either": len(t_solved | s_solved),
                "teacher_only_solved_count": len(t_solved - s_solved),
                "teacher_only_solved_instance_ids": sorted(t_solved - s_solved),
            }
        else:
            summary["rollout_collection"] = _summarize_rollout_rows(rows_0 + rows_1, int(args.num_rollouts))

        (out_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return summary
    finally:
        terminate_process_group(proc_0)
        terminate_process_group(proc_1)
        _run_cmd(["docker", "rmi", "-f", "gemma-v2-rebench-base-anchor:keep"], timeout=60, check=False)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "_worker_batch":
        return _worker_batch_entry(argv[1])

    parser = argparse.ArgumentParser(prog="python -m v2.eval.run_rebench_v2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mat = sub.add_parser("materialize")
    p_mat.add_argument(
        "--splits-json",
        type=Path,
        default=Path("/path/to/gemma-v2/splits/rebench_sealed_splits_v2.json"),
    )
    p_mat.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("/path/to/gemma-devagent-20261001/expansion-pilot-v1"),
    )
    p_mat.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/gemma-v2/eval_data/rebench36"),
    )

    p_run = sub.add_parser("run")
    p_run.add_argument(
        "--mode",
        choices=["ab_paired", "dual_replica_rollouts", "teacher_student_rollouts"],
        required=True,
    )
    p_run.add_argument("--rebench-dir", type=Path, default=Path("/path/to/gemma-v2/eval_data/rebench36"))
    p_run.add_argument("--tasks-path", type=Path, required=True)
    p_run.add_argument("--results-root", type=Path, required=True)
    p_run.add_argument("--primary-submission", type=Path, required=True)
    p_run.add_argument("--secondary-submission", type=Path, default=None)
    p_run.add_argument("--primary-model-path", required=True)
    p_run.add_argument("--secondary-model-path", default=None)
    p_run.add_argument("--primary-alias", required=True)
    p_run.add_argument("--secondary-alias", default=None)
    p_run.add_argument(
        "--vllm-python",
        default="/path/to/gemma-devagent-20261001/server-venv/bin/python",
    )
    p_run.add_argument("--eval-python", default=sys.executable)
    p_run.add_argument("--server-pythonpath", default="/path/to/gemma-v2/py_shim")
    p_run.add_argument("--hf-home", default="/path/to/gemma-v2/cache/hf")
    p_run.add_argument("--vllm-cache-root", default="/path/to/gemma-v2/cache/vllm")
    p_run.add_argument("--base-port", type=int, default=18910)
    p_run.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p_run.add_argument("--secondary-gpu-memory-utilization", type=float, default=None)
    p_run.add_argument("--concurrency-per-gpu", type=int, default=16)
    p_run.add_argument("--secondary-concurrency-per-gpu", type=int, default=None)
    p_run.add_argument("--batch-size", type=int, default=32)
    p_run.add_argument("--num-rollouts", type=int, default=4)
    p_run.add_argument("--secondary-num-rollouts", type=int, default=None)
    p_run.add_argument("--max-tasks", type=int, default=0)
    p_run.add_argument("--startup-timeout", type=float, default=900.0)
    p_run.add_argument("--keep-batch-images", action="store_true")
    p_run.add_argument("--stream", action="store_true",
                       help="One barrier-free pass: per-GPU workers prepare images just in time (no batch tails)")
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--early-stop-mixed", action="store_true",
                       help="SCREENING ONLY (stream dual_replica_rollouts): stop a task's remaining rollouts once it "
                            "has a resolved and an unresolved graded rollout (records carry early_stopped=true)")
    p_run.add_argument("--corpus-rows", type=Path, nargs="+", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "run" and getattr(args, "early_stop_mixed", False) and (
        args.mode != "dual_replica_rollouts" or not args.stream
    ):
        parser.error("--early-stop-mixed is for screens only: requires --mode dual_replica_rollouts --stream")
    if args.cmd == "materialize":
        rep = materialize_rebench36(
            splits_json=args.splits_json,
            pilot_root=args.pilot_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0

    rep = orchestrate_rebench_run(args)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
