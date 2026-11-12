"""Parallel Zero-Compromise 6-Gate Container Verifier & Dataset Materializer for `evalhost`.

Scales our verified task pool from 36 tasks -> 400-800+ tasks across hundreds of Python repositories
while enforcing EVERY SINGLE ONE of our 6 non-negotiable quality checks in isolated `--network none`
Docker containers:
  1. Clear requirements   — substantive prompt, concrete anchor, zero hidden APIs/flags/exception strings
  2. Reproducible bug     — 2x offline base fail (PYTHONHASHSEED=0,42), rc=1, 0 errors, editable /testbed
  3. Verified fixes       — 2x gold pass (PYTHONHASHSEED=0,42) + distinct AST-equivalent alt solution pass
  4. Strong tests         — F2P + P2P assertions + live AST/shortcut Mutation Kill Score K_mut >= 0.75
  5. Trustworthy grading  — exact swegemma 0.2.11 JUnit XML parity + cryptographic XML nonces + zero test edits
  6. Clean separation     — 5-way repo disjointness + GitHub repo_id/rename map + family dedup + zero prompt leaks
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any
import uuid

from v2.constants import canonicalize_repo
from v2.curation.quality_gates import (
    audit_task_quality,
    build_repo_disjoint_splits,
    parse_unified_diff,
)
from v2.curation.rebench_unpause import (
    generate_gold_patch_line_mutants,
    load_candidate_rows,
    synthesize_alternative_gold_patch,
)
from v2.eval.run_rebench_v2 import (
    CONDA_PATH_ENV,
    ISOLATED_GRADER_LIMITS,
    _copy_bytes_into_container,
    _decode_test_list,
    _junit_key,
    _parse_junit_bytes,
    _run_cmd,
    _safe_tar_members,
    _sha256_bytes,
)

_ALLOWED_IMAGE_RE = re.compile(
    r"^(?:docker\.io/)?(?:swerebench(?:v2)?|swebench|r2egym|xingyaoww)/[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?"
    r"(?:@sha256:[0-9a-f]{64})?$"  # digest-pinned refs (SWE-rebench V2.1 RL)
)


def load_harbor_july_sealed_repos(pilot_root: Path) -> set[str]:
    """Load all 65 Harbor July sealed repositories from either task list or repo list format."""
    sealed: set[str] = set()
    for rel in (
        "evaluator-only-source/harbor-july-metadata.json",
        "sealed-repositories-v1.json",
    ):
        p = pilot_root / rel
        if not p.is_file():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(doc, list):
            sealed.update(str(x) for x in doc)
        elif isinstance(doc, dict):
            if isinstance(doc.get("tasks"), list):
                for t in doc["tasks"]:
                    pkg = ((t.get("task_version") or {}).get("package") or {}).get("name", "")
                    if pkg and "-" in pkg:
                        sealed.add(pkg.rsplit("-", 1)[0].replace("__", "/"))
            for key in ("repositories", "sealed_repositories", "repos"):
                if isinstance(doc.get(key), list):
                    sealed.update(str(x) for x in doc[key])
    return {canonicalize_repo(r) for r in sealed if r}


def load_github_provenance_enrichment(
    pilot_root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return `(repo_license_map, repo_rename_map, repo_github_id_map)` from provenance JSONs."""
    repo_lic: dict[str, str] = {}
    repo_rename: dict[str, str] = {}
    repo_gh_id: dict[str, str] = {}

    prov_file = pilot_root / "connected-github-repository-provenance-v7.json"
    if prov_file.is_file():
        doc = json.loads(prov_file.read_text(encoding="utf-8"))
        for rec in doc.get("records", []) if isinstance(doc, dict) else []:
            if not isinstance(rec, dict):
                continue
            req = str(rec.get("requested_repo") or "").strip().lower()
            full = str(rec.get("full_name") or "").strip().lower()
            gh_id = str(rec.get("github_repository_id") or rec.get("id") or "").strip()
            spdx = str(rec.get("current_license_spdx") or "").strip()
            canon_target = canonicalize_repo(full or req)
            if req and full and req != full:
                repo_rename[canonicalize_repo(req)] = canon_target
            for k in (req, full, canon_target):
                if k:
                    if spdx:
                        repo_lic[k] = spdx
                    if gh_id:
                        repo_gh_id[k] = gh_id

    sel_file = pilot_root / "selection-manifest-v7.json"
    if sel_file.is_file():
        doc = json.loads(sel_file.read_text(encoding="utf-8"))
        for t in doc.get("tasks", []) if isinstance(doc, dict) else []:
            if isinstance(t, dict) and t.get("repo") and t.get("current_repository_license_spdx"):
                rk = canonicalize_repo(str(t["repo"]))
                repo_lic.setdefault(rk, str(t["current_repository_license_spdx"]))
    return repo_lic, repo_rename, repo_gh_id


def classify_license_bytes(data: bytes) -> str | None:
    text = data.decode("utf-8", errors="replace")
    if "Permission is hereby granted, free of charge" in text:
        return "MIT"
    if "Apache License" in text and "Version 2.0" in text:
        return "Apache-2.0"
    if "Redistribution and use in source and binary forms" in text:
        return "BSD-3-Clause"
    if "ISC License" in text or "Permission to use, copy, modify, and/or distribute this software" in text:
        return "ISC"
    if "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" in text.upper():
        return "PSF-2.0"
    if "Zero-Clause BSD" in text or "0BSD" in text:
        return "0BSD"
    if "This is free and unencumbered software released into the public domain" in text:
        return "Unlicense"
    return None


_FORBIDDEN_WORKDIRS = frozenset(
    {"", "/", "/workspace", "/root", "/tmp", "/usr", "/var", "/opt", "/etc", "/bin", "/sbin"}
)
_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _resolve_container_workdir(insp: dict[str, Any]) -> str:
    """Resolve and validate repo working directory across `swerebench` (`/testbed`) and `swerebenchv2` (`/<repo>`)."""
    raw_wd = str((insp.get("Config") or {}).get("WorkingDir") or "").strip()
    workdir = raw_wd.rstrip("/") or "/testbed"
    if not workdir.startswith("/") or workdir in _FORBIDDEN_WORKDIRS:
        raise RuntimeError(f"Unsafe or invalid container WorkingDir: {raw_wd!r}")
    return workdir


_PULL_LOCK = threading.Lock()
_PULL_STATE = {"enabled": False, "tokens": 0, "margin": 5, "checked": 0.0}


_HEAVY_SEM = threading.Semaphore(64)  # local image builds in flight; --serial-image-ops sets 1
_PULL_SEM = threading.Semaphore(64)  # registry pulls in flight; --max-concurrent-pulls N (default: 1 with
#                                      --serial-image-ops, else 64)
_LOW_IMPACT = {"max_docker_latency_sec": 0.0, "blkio_weight": 0}


def _wait_for_docker_daemon() -> None:
    """Block while `docker ps` takes longer than the threshold (daemon contention hurts live rollouts)."""
    thr = _LOW_IMPACT["max_docker_latency_sec"]
    if thr <= 0:
        return
    while True:
        t0 = time.monotonic()
        try:
            subprocess.run(["docker", "ps", "-q"], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() - t0 <= thr:
            return
        time.sleep(30)


def _docker_hub_pull_quota() -> int | None:
    """Remaining anonymous Docker Hub manifest pulls for this IP (a HEAD request does not consume one)."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull",
            timeout=20,
        ) as resp:
            token = json.loads(resp.read())["token"]
        req = urllib.request.Request(
            "https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest",
            method="HEAD",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            hdr = resp.headers.get("ratelimit-remaining") or resp.headers.get("x-ratelimit-remaining") or ""
        return int(hdr.split(";")[0]) if hdr else None
    except Exception:
        return None


def _acquire_pull_token() -> None:
    """Token bucket backed by the registry's own remaining-quota header (keeps a small safety margin)."""
    while True:
        with _PULL_LOCK:
            if _PULL_STATE["tokens"] > 0:
                _PULL_STATE["tokens"] -= 1
                return
            if time.monotonic() - _PULL_STATE["checked"] > 60:
                _PULL_STATE["checked"] = time.monotonic()
                rem = _docker_hub_pull_quota()
                if rem is not None:
                    _PULL_STATE["tokens"] = max(0, rem - _PULL_STATE["margin"])
                    if _PULL_STATE["tokens"] > 0:
                        _PULL_STATE["tokens"] -= 1
                        return
        time.sleep(20)


def _registry_pull(image: str) -> subprocess.CompletedProcess:
    """`docker pull`, optionally gated by the Docker Hub quota (anonymous limit: 100 manifest pulls/hour/IP).

    A rate-limit response is retried after a pause instead of dropping the candidate."""
    for attempt in range(4):
        if _PULL_STATE["enabled"] and image.split("/")[0] in ("swerebench", "swerebenchv2", "swebench", "xingyaoww", "r2egym", "docker.io"):
            _acquire_pull_token()
        _wait_for_docker_daemon()
        with _PULL_SEM:
            res = _run_cmd(["docker", "pull", "--quiet", image], timeout=900, check=False)
        err = res.stderr.decode("utf-8", errors="replace").lower()
        if res.returncode == 0 or not any(t in err for t in ("rate limit", "toomanyrequests", "429")):
            return res
        with _PULL_LOCK:
            _PULL_STATE["tokens"] = 0
        time.sleep(300 * (attempt + 1))
    return res


def _build_local_swerebenchv2_image(
    *,
    repo: str,
    base_commit: str,
    install_cfg: dict[str, Any],
    target_tag: str,
) -> bool:
    """Build a reproducible `swerebenchv2` image locally from `install_config` when cached base images exist."""
    if not _REPO_SLUG_RE.match(repo) or not _HEX_SHA_RE.match(base_commit):
        return False
    install_cmds = [
        c.strip()
        for c in (install_cfg.get("install") or [])
        if isinstance(c, str) and c.strip() and not c.strip().startswith("pytest") and "&& pytest " not in c
    ]
    if not install_cmds:
        return False

    base_name = str(install_cfg.get("base_image_name") or "python_base_310").strip()
    local_base = (
        "local-swerebenchv2/python_base_37:base"
        if "37" in base_name
        else "local-swerebenchv2/python_base_310:base"
    )
    if _run_cmd(["docker", "image", "inspect", local_base], timeout=15, check=False).returncode != 0:
        return False

    repo_name = repo.split("/", 1)[1]
    workdir = f"/{repo_name}"
    if workdir in _FORBIDDEN_WORKDIRS:
        return False

    clone_cmd = f"git clone -o origin https://github.com/{repo}.git {shlex.quote(workdir)}"
    joined_installs = "\n    ".join(f"( {cmd} ) || true;" for cmd in install_cmds)

    dockerfile = (
        f"FROM {local_base}\n"
        "USER root\n"
        "RUN --mount=type=cache,target=/tmp/pipcache <<'DOCKER_RUN_EOF'\n"
        "set -eux\n"
        "export PIP_CACHE_DIR=/tmp/pipcache PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_SRC=/opt/pip_src\n"
        "mkdir -p /opt/pip_src\n"
        f"rm -rf {shlex.quote(workdir)}\n"
        f"{clone_cmd}\n"
        f"cd {shlex.quote(workdir)}\n"
        "git config core.filemode false\n"
        f"(git checkout --force --detach {base_commit} || (git fetch origin {base_commit} && git checkout --force --detach {base_commit}))\n"
        f"git reset --hard {base_commit}\n"
        f"    {joined_installs}\n"
        "printf '\\n*.egg-info/\\nbuild/\\n.eggs/\\n' >> .git/info/exclude\n"
        f"git checkout --force --detach {base_commit}\n"
        f"git reset --hard {base_commit}\n"
        "git clean -fd\n"
        "for r in $(git remote); do git remote remove \"$r\"; done\n"
        "git for-each-ref --format='delete %(refname)' refs/heads refs/remotes refs/tags | git update-ref --stdin\n"
        "git reflog expire --expire=now --expire-unreachable=now --all\n"
        "git gc --prune=now --quiet\n"
        f"chmod -R a+rwX {shlex.quote(workdir)}\n"
        "DOCKER_RUN_EOF\n"
        f"WORKDIR {workdir}\n"
    )
    with tempfile.TemporaryDirectory(prefix="v2-local-sweb2-") as td:
        df_path = Path(td) / "Dockerfile"
        df_path.write_text(dockerfile, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        env["DOCKER_BUILDKIT"] = "1"
        _wait_for_docker_daemon()
        with _HEAVY_SEM:
            proc = subprocess.run(
                ["docker", "build", "--quiet", "-t", target_tag, str(td)],
                env=env,
                capture_output=True,
                timeout=600,
            )
        return proc.returncode == 0


def _build_local_swerebenchv1_image(
    *,
    repo: str,
    base_commit: str,
    install_cfg: dict[str, Any],
    requirements_txt: str,
    target_tag: str,
) -> bool:
    """Build a reproducible `swerebench` v1 image locally at `/testbed` from `requirements` + `install_config`."""
    if not _REPO_SLUG_RE.match(repo) or not _HEX_SHA_RE.match(base_commit):
        return False
    py_ver = str(install_cfg.get("python") or "3.10").strip()
    if py_ver.startswith(("2.", "3.4", "3.5")):
        return False
    local_base = (
        "local-swerebenchv2/python_base_37:base"
        if py_ver.startswith(("3.6", "3.7"))
        else "local-swerebenchv2/python_base_310:base"
    )
    if _run_cmd(["docker", "image", "inspect", local_base], timeout=15, check=False).returncode != 0:
        return False

    pre_cmds = install_cfg.get("pre_install") or []
    if isinstance(pre_cmds, str):
        pre_cmds = [pre_cmds]
    pre_cmds = [c.strip() for c in pre_cmds if isinstance(c, str) and c.strip()]

    install_cmd = str(install_cfg.get("install") or "pip install -e .").strip()
    pip_pkgs = [
        p.strip()
        for p in (install_cfg.get("pip_packages") or ["pytest"])
        if isinstance(p, str) and p.strip()
    ]
    if "pytest" not in pip_pkgs:
        pip_pkgs.insert(0, "pytest")

    pre_block = "\n    ".join(f"( {c} ) || true;" for c in pre_cmds)
    pip_extra = " ".join(shlex.quote(p) for p in pip_pkgs)
    workdir = "/testbed"

    dockerfile = (
        f"FROM {local_base}\n"
        "USER root\n"
        "COPY _rebench_reqs.txt /tmp/_rebench_reqs.txt\n"
        "RUN --mount=type=cache,target=/tmp/pipcache <<'DOCKER_RUN_EOF'\n"
        "set -eux\n"
        "export PIP_CACHE_DIR=/tmp/pipcache PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_SRC=/opt/pip_src\n"
        "mkdir -p /opt/pip_src\n"
        "rm -rf /testbed\n"
        f"git clone -o origin https://github.com/{repo}.git /testbed\n"
        "cd /testbed\n"
        "git config core.filemode false\n"
        f"(git checkout --force --detach {base_commit} || (git fetch origin {base_commit} && git checkout --force --detach {base_commit}))\n"
        f"git reset --hard {base_commit}\n"
        "if [ -s /tmp/_rebench_reqs.txt ]; then\n"
        "    pip install -r /tmp/_rebench_reqs.txt || (grep -Ev '^(#|$|conda|pywin32)' /tmp/_rebench_reqs.txt | xargs -r -n 15 pip install || true)\n"
        "fi\n"
        "rm -f /tmp/_rebench_reqs.txt\n"
        f"    {pre_block}\n"
        f"pip install {pip_extra} || true\n"
        f"( {install_cmd} ) || pip install --no-build-isolation -e . || true\n"
        "printf '\\n*.egg-info/\\nbuild/\\n.eggs/\\n' >> .git/info/exclude\n"
        f"git checkout --force --detach {base_commit}\n"
        f"git reset --hard {base_commit}\n"
        "git clean -fd\n"
        "for r in $(git remote); do git remote remove \"$r\"; done\n"
        "git for-each-ref --format='delete %(refname)' refs/heads refs/remotes refs/tags | git update-ref --stdin\n"
        "git reflog expire --expire=now --expire-unreachable=now --all\n"
        "git gc --prune=now --quiet\n"
        "chmod -R a+rwX /testbed\n"
        "DOCKER_RUN_EOF\n"
        f"WORKDIR {workdir}\n"
    )
    with tempfile.TemporaryDirectory(prefix="v2-local-sweb1-") as td:
        td_path = Path(td)
        (td_path / "_rebench_reqs.txt").write_text(requirements_txt + "\n" if requirements_txt else "", encoding="utf-8", newline="\n")
        (td_path / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        env["DOCKER_BUILDKIT"] = "1"
        _wait_for_docker_daemon()
        with _HEAVY_SEM:
            proc = subprocess.run(
                ["docker", "build", "--quiet", "-t", target_tag, str(td_path)],
                env=env,
                capture_output=True,
                timeout=600,
            )
        return proc.returncode == 0


def _run_pytest_in_container(
    cid: str,
    test_cmd: str,
    f2p: list[str],
    p2p: list[str],
    hash_seed: int,
    timeout_sec: int,
    workdir: str = "/testbed",
) -> tuple[dict[str, Any], bytes]:
    nonce_xml = f"/tmp/verify-{uuid.uuid4().hex}.xml"
    script = (
        "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH; "
        "export PYTHONPYCACHEPREFIX=/tmp/.cache/pycache XDG_CACHE_HOME=/tmp/.cache; "
        f"export PIP_NO_INDEX=1; export PYTHONHASHSEED={hash_seed}; "
        f"cd {shlex.quote(workdir)} && {test_cmd} --junitxml={nonce_xml}"
    )
    t0 = time.monotonic()
    res = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", script], timeout=timeout_sec, check=False)
    dur = round(time.monotonic() - t0, 2)
    xml_bytes = _run_cmd(
        ["docker", "exec", cid, "/bin/bash", "-c", f"cat {nonce_xml} && rm -f {nonce_xml}"],
        timeout=30,
        check=False,
    ).stdout
    if not xml_bytes.strip():
        return (
            {
                "returncode": int(res.returncode),
                "f2p_all_failed": False,
                "p2p_all_passed": False,
                "failed_count": 999,
                "error_count": 999,
                "duration_sec": dur,
                "status_signature_sha256": "",
                "parse_error": "missing_junit_xml",
            },
            b"",
        )
    try:
        parsed = _parse_junit_bytes(xml_bytes)
    except Exception as exc:
        return (
            {
                "returncode": int(res.returncode),
                "f2p_all_failed": False,
                "p2p_all_passed": False,
                "failed_count": 999,
                "error_count": 999,
                "duration_sec": dur,
                "status_signature_sha256": "",
                "parse_error": str(exc),
            },
            xml_bytes,
        )

    states = parsed["statuses"]
    f2p_states = [states.get(_junit_key(req), "MISSING") for req in f2p]
    p2p_states = [states.get(_junit_key(req), "MISSING") for req in p2p]

    f2p_all_failed = bool(f2p) and all(s == "FAILED" for s in f2p_states)
    f2p_all_passed = bool(f2p) and all(s == "PASSED" for s in f2p_states)
    p2p_all_passed = all(s == "PASSED" for s in p2p_states)

    sig_payload = json.dumps(
        {k: states[k] for k in sorted(states)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return (
        {
            "returncode": int(res.returncode),
            "f2p_all_failed": f2p_all_failed,
            "f2p_all_passed": f2p_all_passed,
            "p2p_all_passed": p2p_all_passed,
            "failed_count": int(parsed["counts"].get("FAILED", 0)),
            "error_count": int(parsed["counts"].get("ERROR", 0)),
            "collected_count": int(parsed["collected"]),
            "duration_sec": dur,
            "status_signature_sha256": hashlib.sha256(sig_payload).hexdigest(),
        },
        xml_bytes,
    )


def _probe_editable_testbed_import(cid: str, gold_files: list[str], workdir: str = "/testbed") -> dict[str, Any]:
    """Inspect touched module origins without executing packaging scripts or package side effects."""
    candidates: list[str] = []
    for rel in gold_files:
        if not rel.endswith(".py"):
            continue
        clean = rel.strip("/")
        if clean.startswith("src/"):
            clean = clean[4:]
        parts = [part for part in clean[:-3].split("/") if part and part != "__init__"]
        if parts and all(part.isidentifier() for part in parts) and parts[0] not in candidates:
            candidates.append(parts[0])
    if not candidates:
        return {"verified": True, "modules": [], "reason": "no_touched_python_module"}
    # Importing setup.py runs setuptools and can exit even when source resolution is correct.
    # find_spec on top-level names inspects the same resolution path without executing them.
    py_check = "\n".join([
        "import importlib.util, json, pathlib",
        f"names = {candidates!r}",
        f"root = pathlib.Path({workdir!r}).resolve()",
        "records = []",
        "for name in names:",
        "    spec = importlib.util.find_spec(name)",
        "    paths = ([spec.origin] if spec is not None and spec.origin else [])",
        "    if spec is not None and not spec.origin:",
        "        paths = list(spec.submodule_search_locations or [])",
        "    inside = bool(paths) and all(pathlib.Path(p).is_absolute() and str(pathlib.Path(p).resolve()).startswith(str(root).rstrip('/') + '/') for p in paths)",
        "    records.append(dict(module=name, origins=paths, inside_workspace=inside, found=spec is not None))",
        "print(json.dumps(dict(verified=all(r['inside_workspace'] for r in records), modules=records)))",
    ])
    res = _run_cmd([
        "docker", "exec", cid, "/bin/bash", "-c",
        "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH; "
        "export PYTHONPYCACHEPREFIX=/tmp/.cache/pycache; "
        f"cd {shlex.quote(workdir)} && "
        'PY_BIN="$(command -v python || command -v python3)" && '
        f'"$PY_BIN" -c {shlex.quote(py_check)}',
    ], timeout=30, check=False)
    if res.returncode != 0:
        return {"verified": False, "reason": "import_resolution_probe_error", "returncode": res.returncode,
                "stderr": res.stderr.decode("utf-8", errors="replace")[-2000:]}
    try:
        probe = json.loads(res.stdout.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        return {"verified": False, "reason": "import_resolution_probe_error", "error": str(exc)}
    probe["reason"] = "editable_source_verified" if probe["verified"] else (
        "missing_touched_module" if any(not item["found"] for item in probe["modules"])
        else "shadowed_site_packages_import"
    )
    return probe


def _verify_editable_testbed_import(cid: str, gold_files: list[str], workdir: str = "/testbed") -> bool:
    """Retain the existing boolean helper interface for callers outside this module."""
    return bool(_probe_editable_testbed_import(cid, gold_files, workdir)["verified"])


def _build_sanitized_tgz_from_tar_bytes(base_tar_bytes: bytes, dst_tgz: Path) -> str:
    """Build single-commit `.git`-squashed agent snapshot `.tgz` with zero history or test_patch."""
    dst_tgz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-snap-") as td:
        work = Path(td)
        with tarfile.open(fileobj=io.BytesIO(base_tar_bytes)) as archive:
            archive.extractall(work, members=_safe_tar_members(archive), filter="data")
        for git_cmd in (
            ["git", "init", "--quiet", str(work)],
            ["git", "-C", str(work), "config", "user.name", "Gemma Evaluation"],
            ["git", "-C", str(work), "config", "user.email", "evaluation@localhost"],
            ["git", "-C", str(work), "add", "--all"],
            ["git", "-C", str(work), "commit", "--quiet", "--allow-empty", "-m", "Pinned task base snapshot"],
        ):
            _run_cmd(git_cmd, timeout=120)
        tmp_tgz = dst_tgz.with_suffix(".tgz.tmp")
        with tmp_tgz.open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as out_tar:
                    for p in sorted(work.rglob("*")):
                        out_tar.add(p, arcname=p.relative_to(work), recursive=False)
        tmp_tgz.replace(dst_tgz)
    return _sha256_bytes(dst_tgz.read_bytes())


def _synthesize_in_container_ast_probes(
    cid: str,
    base_commit: str,
    gold_files: list[str],
    need_alt: bool,
    need_mutant: bool,
    workdir: str = "/testbed",
) -> tuple[str | None, str | None]:
    """Full-file AST fallback inside `workdir` for multi-line or deletion-only Python patches."""
    py_files = [f for f in gold_files if f.endswith(".py")]
    if not py_files or (not need_alt and not need_mutant):
        return None, None

    helper_py = r'''
import ast, json, subprocess, sys

base_commit = sys.argv[1]
py_files = json.loads(sys.argv[2])
need_alt = sys.argv[3] == "1"
need_mut = sys.argv[4] == "1"

def reset_and_gold():
    subprocess.run(["git", "reset", "--hard", base_commit], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "apply", "/tmp/gold.patch"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def changed_lines_for(path):
    out = subprocess.check_output(["git", "diff", "-U0", base_commit, "--", path], text=True)
    lines = set()
    import re
    for ln in out.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", ln)
        if m:
            start = int(m.group(1))
            cnt = int(m.group(2)) if m.group(2) is not None else 1
            for k in range(start, start + max(1, cnt) + 1):
                lines.add(k)
    return lines

alt_diff = None
mut_diff = None

for rel in py_files:
    reset_and_gold()
    try:
        src = open(rel, "r", encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        continue
    clines = changed_lines_for(rel)
    src_lines = src.splitlines(keepends=True)

    # Locate innermost statement (Return, If, Assign, Expr) overlapping changed lines (or inside changed FunctionDef)
    target_stmt = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Return, ast.Assign, ast.AnnAssign)) and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            span = set(range(node.lineno, node.end_lineno + 1))
            if span & clines:
                target_stmt = node
                break
    if target_stmt is None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if set(range(node.lineno, node.end_lineno + 1)) & clines:
                    for sub in ast.walk(node):
                        if isinstance(sub, (ast.If, ast.Return, ast.Assign)):
                            target_stmt = sub
                            break
                if target_stmt is not None:
                    break

    if target_stmt is None or not hasattr(ast, "unparse"):
        continue

    indent = src_lines[target_stmt.lineno - 1][: len(src_lines[target_stmt.lineno - 1]) - len(src_lines[target_stmt.lineno - 1].lstrip())]

    if need_alt and alt_diff is None:
        alt_lines = None
        if isinstance(target_stmt, ast.Return) and target_stmt.value is not None:
            expr_txt = ast.unparse(target_stmt.value)
            alt_lines = [f"{indent}_rebench_alt_res = ({expr_txt})\n", f"{indent}return _rebench_alt_res\n"]
        elif isinstance(target_stmt, ast.If):
            test_txt = ast.unparse(target_stmt.test)
            alt_lines = [f"{indent}if bool({test_txt}) is True:\n"] + src_lines[target_stmt.lineno : target_stmt.end_lineno]
        elif isinstance(target_stmt, ast.Assign) and len(target_stmt.targets) == 1:
            lhs_txt = ast.unparse(target_stmt.targets[0])
            rhs_txt = ast.unparse(target_stmt.value)
            alt_lines = [f"{indent}_rebench_alt_tmp = ({rhs_txt})\n", f"{indent}{lhs_txt} = _rebench_alt_tmp\n"]
        if alt_lines is not None:
            cand_src = "".join(src_lines[: target_stmt.lineno - 1] + alt_lines + src_lines[target_stmt.end_lineno :])
            try:
                compile(cand_src, rel, "exec")
                open(rel, "w", encoding="utf-8").write(cand_src)
                d = subprocess.check_output(["git", "diff", base_commit], text=True)
                if d.strip():
                    alt_diff = d
            except Exception:
                pass

    reset_and_gold()
    if need_mut and mut_diff is None:
        mut_lines = None
        if isinstance(target_stmt, ast.If):
            test_txt = ast.unparse(target_stmt.test)
            mut_lines = [f"{indent}if not ({test_txt}):\n"] + src_lines[target_stmt.lineno : target_stmt.end_lineno]
        elif isinstance(target_stmt, ast.Return):
            mut_lines = [f"{indent}return None\n"]
        elif isinstance(target_stmt, ast.Assign) and len(target_stmt.targets) == 1:
            lhs_txt = ast.unparse(target_stmt.targets[0])
            mut_lines = [f"{indent}{lhs_txt} = None\n"]
        if mut_lines is not None:
            cand_src = "".join(src_lines[: target_stmt.lineno - 1] + mut_lines + src_lines[target_stmt.end_lineno :])
            try:
                compile(cand_src, rel, "exec")
                open(rel, "w", encoding="utf-8").write(cand_src)
                d = subprocess.check_output(["git", "diff", base_commit], text=True)
                if d.strip():
                    mut_diff = d
            except Exception:
                pass

    if (not need_alt or alt_diff) and (not need_mut or mut_diff):
        break

reset_and_gold()
print(json.dumps({"alt_diff": alt_diff, "mut_diff": mut_diff}))
'''
    _copy_bytes_into_container(cid, helper_py.encode("utf-8"), "/tmp", "_ast_probe.py")
    cmd_str = (
        "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH; "
        "export PYTHONPYCACHEPREFIX=/tmp/.cache/pycache; "
        f"cd {shlex.quote(workdir)} && "
        'PY_BIN="$(command -v python || command -v python3)" && '
        f'"$PY_BIN" /tmp/_ast_probe.py '
        f"{shlex.quote(base_commit)} {shlex.quote(json.dumps(py_files))} "
        f"{'1' if need_alt else '0'} {'1' if need_mutant else '0'}"
    )
    res = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", cmd_str], timeout=45, check=False)
    if res.returncode != 0:
        return None, None
    try:
        doc = json.loads(res.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
        return doc.get("alt_diff"), doc.get("mut_diff")
    except Exception:
        return None, None


def verify_single_candidate_in_container(
    row: dict[str, Any],
    *,
    output_dir: Path,
    sealed_repos: set[str],
    repo_rename_map: dict[str, str],
    llm_judge_dir: Path | None = None,
    max_test_duration_sec: float = 120.0,
    min_mutation_kill_score: float = 0.75,
    max_mutants: int = 4,
    prune_image_after: bool = True,
    container_cpu_shares: int = 0,
    v1_image_policy: str = "local_first",
) -> dict[str, Any]:
    """Run full 6-Gate container verification + AST/LLM mutation + alt-solution probe for one task."""
    from swegemma.harness.verification import _validate_junit_xml

    iid = str(row["instance_id"])
    ctrl_dir = output_dir / "controls"
    snaps_dir = output_dir / "snapshots"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)
    ctrl_file = ctrl_dir / f"{iid}.json"

    if ctrl_file.is_file():
        cached = json.loads(ctrl_file.read_text(encoding="utf-8"))
        if cached.get("completed"):
            need_reprobe = False
            if llm_judge_dir is not None:
                jfile_chk = llm_judge_dir / "verdicts" / f"{iid}.json"
                if jfile_chk.is_file():
                    jdoc_chk = json.loads(jfile_chk.read_text(encoding="utf-8"))
                    if cached.get("reason") == "rejected_by_llm_dual_judge" and jdoc_chk.get("dual_judge_admitted", False):
                        need_reprobe = True
                    elif (
                        cached.get("admitted")
                        and jdoc_chk.get("dual_judge_admitted", False)
                        and isinstance(jdoc_chk.get("claude_verdict"), dict)
                        and not cached.get("claude_codex_patches_probed", False)
                    ):
                        need_reprobe = True
            if not need_reprobe:
                if cached.get("admitted"):
                    up_ref = str(cached.get("upstream_registry_digest") or "")
                    img_cand = str(row.get("docker_image") or row.get("image_name") or "").strip()
                    if (
                        up_ref
                        and _run_cmd(["docker", "image", "inspect", up_ref], timeout=15, check=False).returncode != 0
                    ):
                        icfg = row.get("install_config") or {}
                        if isinstance(icfg, str):
                            icfg = json.loads(icfg)
                        rebuilt = False
                        if "swerebenchv2/" in img_cand:
                            rebuilt = _build_local_swerebenchv2_image(
                                repo=str(row.get("repo") or "").strip(),
                                base_commit=str(row.get("base_commit") or "").strip(),
                                install_cfg=icfg,
                                target_tag=img_cand,
                            )
                        elif "swerebench/" in img_cand:
                            rebuilt = _build_local_swerebenchv1_image(
                                repo=str(row.get("repo") or "").strip(),
                                base_commit=str(row.get("base_commit") or "").strip(),
                                install_cfg=icfg,
                                requirements_txt=str(row.get("requirements") or "").strip(),
                                target_tag=img_cand,
                            )
                        if rebuilt:
                            cached["upstream_registry_digest"] = img_cand
                            ctrl_file.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return cached

    # Optional Codex Astra High / Claude Opus 5.5 High dual-judge verdict & adversarial patches
    llm_alt_patches: list[tuple[str, str]] = []
    llm_shortcut_patches: list[tuple[str, str]] = []
    both_llm_verdicts_present = False
    if llm_judge_dir is not None:
        jfile = llm_judge_dir / "verdicts" / f"{iid}.json"
        if jfile.is_file():
            jdoc = json.loads(jfile.read_text(encoding="utf-8"))
            if not jdoc.get("dual_judge_admitted", True):
                res = {
                    "instance_id": iid,
                    "completed": True,
                    "admitted": False,
                    "reason": "rejected_by_llm_dual_judge",
                }
                ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return res
            both_llm_verdicts_present = isinstance(jdoc.get("codex_verdict"), dict) and isinstance(
                jdoc.get("claude_verdict"), dict
            )
            for tag, vdict in (("codex", jdoc.get("codex_verdict")), ("claude", jdoc.get("claude_verdict"))):
                if isinstance(vdict, dict):
                    ap = str(vdict.get("alt_correct_patch") or "").strip()
                    sp = str(vdict.get("plausible_shortcut_patch") or "").strip()
                    if ap:
                        llm_alt_patches.append((f"{tag}_distinct_algorithm", ap))
                    if sp:
                        llm_shortcut_patches.append((f"m0_{tag}_plausible_shortcut", sp))

    image = str(row.get("docker_image") or row.get("image_name") or "").strip()
    if not _ALLOWED_IMAGE_RE.match(image):
        res = {"instance_id": iid, "completed": True, "admitted": False, "reason": f"disallowed_image:{image}"}
        ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return res

    install_cfg = row.get("install_config") or {}
    if isinstance(install_cfg, str):
        install_cfg = json.loads(install_cfg)
    test_cmd = str(install_cfg.get("test_cmd") or "").strip()
    if not test_cmd:
        res = {"instance_id": iid, "completed": True, "admitted": False, "reason": "missing_test_cmd"}
        ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return res

    f2p = _decode_test_list(row.get("FAIL_TO_PASS") or [])
    p2p = _decode_test_list(row.get("PASS_TO_PASS") or [])
    gold_patch = str(row.get("patch") or "")
    test_patch = str(row.get("test_patch") or "")
    base_commit = str(row.get("base_commit") or "").strip()
    gold_files = parse_unified_diff(gold_patch)["files"]

    llm_mutants: list[dict[str, Any]] = []
    for mid_tag, sp in llm_shortcut_patches:
        if sp.startswith("diff --git ") and sp != gold_patch.strip():
            llm_mutants.append(
                {
                    "mutant_id": mid_tag,
                    "kind": "llm_plausible_shortcut",
                    "original_line": "",
                    "mutated_line": "",
                    "mutated_patch": sp + ("\n" if not sp.endswith("\n") else ""),
                }
            )
    ast_mutants = generate_gold_patch_line_mutants(gold_patch, max_mutants=max_mutants)
    mutants = llm_mutants + ast_mutants

    alt_candidates: list[dict[str, str]] = []
    for akind, ap in llm_alt_patches:
        if ap.startswith("diff --git ") and ap != gold_patch.strip():
            alt_candidates.append(
                {
                    "kind": akind,
                    "alt_patch": ap + ("\n" if not ap.endswith("\n") else ""),
                }
            )
    diff_alt_spec = synthesize_alternative_gold_patch(gold_patch)
    if diff_alt_spec is not None:
        alt_candidates.append(
            {
                "kind": diff_alt_spec["kind"],
                "alt_patch": diff_alt_spec["alt_patch"],
            }
        )

    cid: str | None = None
    pulled_digest = ""
    task_admitted = False
    image_preexisted = True  # never remove an image this run did not create (shared daemon)
    created_images: list[str] = []  # derived images this run created (removed unless admitted)
    repair_info = None
    try:
        local_chk = _run_cmd(["docker", "image", "inspect", image], timeout=15, check=False)
        image_preexisted = local_chk.returncode == 0
        if local_chk.returncode != 0 and "@sha256:" in image:
            # digest-pinned upstream image (V2.1 RL): exactly that image or nothing; never a local rebuild
            pinned = _registry_pull(image)
            if pinned.returncode != 0:
                raise RuntimeError(f"pinned image pull failed: {pinned.stderr.decode('utf-8', errors='replace')[-300:]}")
            local_chk = _run_cmd(["docker", "image", "inspect", image], timeout=15, check=False)
        if local_chk.returncode != 0 and v1_image_policy == "pull_first" and "swerebench/" in image:
            # Official upstream SWE-rebench v1 image first (the environment the task was validated in);
            # the local BuildKit rebuild below stays the fallback when the upstream image is gone.
            pull_first = _registry_pull(image)
            if pull_first.returncode != 0:
                err_first = pull_first.stderr.decode("utf-8", errors="replace").lower()
                if any(tok in err_first for tok in ("rate limit", "toomanyrequests", "429")):
                    return {
                        "instance_id": iid,
                        "completed": False,
                        "admitted": False,
                        "reason": "transient_docker_hub_rate_limit",
                    }
            local_chk = _run_cmd(["docker", "image", "inspect", image], timeout=15, check=False)
        if local_chk.returncode != 0:
            built_local = False
            if "swerebenchv2/" in image:
                built_local = _build_local_swerebenchv2_image(
                    repo=str(row.get("repo") or "").strip(),
                    base_commit=base_commit,
                    install_cfg=install_cfg,
                    target_tag=image,
                )
            elif "swerebench/" in image:
                built_local = _build_local_swerebenchv1_image(
                    repo=str(row.get("repo") or "").strip(),
                    base_commit=base_commit,
                    install_cfg=install_cfg,
                    requirements_txt=str(row.get("requirements") or "").strip(),
                    target_tag=image,
                )
            if not built_local:
                pull_res = _registry_pull(image)
                if pull_res.returncode != 0:
                    err_txt = pull_res.stderr.decode("utf-8", errors="replace")
                    if any(tok in err_txt.lower() for tok in ("rate limit", "toomanyrequests", "429")):
                        return {
                            "instance_id": iid,
                            "completed": False,
                            "admitted": False,
                            "reason": "transient_docker_hub_rate_limit",
                        }
                    raise RuntimeError(f"docker pull failed ({pull_res.returncode}): {err_txt[-300:]}")
        insp = json.loads(_run_cmd(["docker", "image", "inspect", image], timeout=30).stdout)[0]
        if insp.get("Config", {}).get("Volumes"):
            raise RuntimeError("Unsafe image Volumes")
        digests = insp.get("RepoDigests") or []
        pulled_digest = str(digests[0]) if digests else image
        workdir = _resolve_container_workdir(insp)
        q_wd = shlex.quote(workdir)

        # Pin first base anchor so shared conda base layers stay cached across pulls
        chk_anchor = _run_cmd(
            ["docker", "image", "inspect", "gemma-v2-curation-anchor:keep"],
            timeout=15,
            check=False,
        )
        if chk_anchor.returncode != 0:
            _run_cmd(["docker", "tag", pulled_digest, "gemma-v2-curation-anchor:keep"], timeout=30, check=False)

        _wait_for_docker_daemon()
        cid = (
            _run_cmd(
                [
                    "docker",
                    "run",
                    "-d",
                    *ISOLATED_GRADER_LIMITS,
                    *(["--blkio-weight", str(_LOW_IMPACT["blkio_weight"])] if _LOW_IMPACT["blkio_weight"] else []),
                    *(["--cpu-shares", str(container_cpu_shares)] if container_cpu_shares > 0 else []),
                    "--entrypoint",
                    "/bin/bash",
                    pulled_digest,
                    "-c",
                    "sleep infinity",
                ],
                timeout=120,
            )
            .stdout.decode("utf-8")
            .strip()
        )

        # SWE-rebench V2.1 RL images ship environment repairs as uncommitted working-tree edits on top of base_commit
        # (e.g. pyproject.toml pytest settings); their test.sh never resets the tree. Keep that state exactly: commit
        # it once as a synthetic "environment repair" commit, snapshot the container into a derived image, and use
        # that commit as the task base everywhere (gates, snapshot, routes, grader reset).
        repair_info: dict[str, Any] | None = None
        if "@sha256:" in image and str(row.get("source_corpus") or "") == "swe_rebench_v2.1_rl":
            head0 = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git rev-parse HEAD"],
                             timeout=30).stdout.decode("utf-8").strip()
            dirty0 = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git status --porcelain"],
                              timeout=30).stdout.decode("utf-8", errors="replace")
            if head0 == base_commit and dirty0.strip():
                stat = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c",
                                 f"cd {q_wd} && git add -A && git -c user.name=pool-v2 -c user.email=pool-v2@localhost "
                                 f"commit -q -m 'SWE-rebench V2.1 environment repair (image working tree)' && "
                                 f"git show --stat --format= HEAD"], timeout=120)
                if stat.returncode != 0:
                    raise RuntimeError("could not commit the V2.1 environment repair")
                base_commit = _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git rev-parse HEAD"],
                                       timeout=30).stdout.decode("utf-8").strip()
                repaired_tag = f"pool-v21-repaired/{iid.lower().replace('__', '_')}:base"
                _run_cmd(["docker", "commit", cid, repaired_tag], timeout=600)
                created_images.append(repaired_tag)
                pulled_digest = repaired_tag
                repair_info = {"upstream_image": image, "repair_commit": base_commit, "derived_image": repaired_tag,
                               "status_porcelain": dirty0[-4000:],
                               "diffstat": stat.stdout.decode("utf-8", errors="replace")[-4000:]}

        # Reset any build-time .egg-info/.pyc dirtiness and verify HEAD == base_commit
        _run_cmd(
            [
                "docker",
                "exec",
                cid,
                "/bin/bash",
                "-c",
                f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd",
            ],
            timeout=60,
            check=False,
        )
        head_sha = (
            _run_cmd(["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git rev-parse HEAD"], timeout=30)
            .stdout.decode("utf-8")
            .strip()
        )
        dirty = _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git status --porcelain"], timeout=30
        ).stdout.strip()
        if head_sha != base_commit or dirty:
            raise RuntimeError(f"Dirty or mismatched container HEAD: {head_sha} vs {base_commit}")

        base_tar_bytes = _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git archive --format=tar {shlex.quote(base_commit)}"],
            timeout=120,
        ).stdout

        # Verify permissive license from root LICENSE/COPYING if row lacked SPDX
        detected_spdx = str(row.get("license_spdx") or "").strip() or None
        if not detected_spdx:
            with tarfile.open(fileobj=io.BytesIO(base_tar_bytes)) as arc:
                for member in arc:
                    if member.isfile() and "/" not in member.name and re.match(
                        r"(?i)^(license|copying|copyright)", member.name
                    ):
                        fobj = arc.extractfile(member)
                        if fobj is not None:
                            spdx_hit = classify_license_bytes(fobj.read())
                            if spdx_hit:
                                detected_spdx = spdx_hit
                                break
        if not detected_spdx:
            res = {"instance_id": iid, "completed": True, "admitted": False, "reason": "unverified_repo_license"}
            ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return res

        row_enriched = {**row, "license_spdx": detected_spdx}
        import_probe = _probe_editable_testbed_import(cid, gold_files, workdir=workdir)
        editable_ok = bool(import_probe["verified"])
        import_dir = output_dir / "import_probes"
        import_dir.mkdir(parents=True, exist_ok=True)
        (import_dir / f"{iid}.json").write_text(json.dumps(import_probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not editable_ok:
            res = {"instance_id": iid, "completed": import_probe["reason"] != "import_resolution_probe_error",
                   "admitted": False, "reason": import_probe["reason"], "import_probe": import_probe}
            ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return res

        _copy_bytes_into_container(cid, test_patch.encode("utf-8"), "/tmp", "test.patch")
        _copy_bytes_into_container(cid, gold_patch.encode("utf-8"), "/tmp", "gold.patch")

        timeout_int = max(45, int(max_test_duration_sec) + 25)

        # --- Check 2: Base Repeat 1 (seed=0) & Repeat 2 (seed=42) ---
        _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && git apply /tmp/test.patch"],
            timeout=60,
        )
        base_r1, _ = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
        if (
            base_r1["returncode"] != 1
            or not base_r1["f2p_all_failed"]
            or not base_r1["p2p_all_passed"]
            or base_r1["error_count"] > 0
            or base_r1["duration_sec"] > max_test_duration_sec
        ):
            res = {
                "instance_id": iid,
                "completed": True,
                "admitted": False,
                "reason": "failed_base_repeat_1",
                "base_repeat_1": base_r1,
            }
            ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return res

        base_r2, _ = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=42, timeout_sec=timeout_int, workdir=workdir)

        # --- Check 3 & 5: Gold Patch Repeat 1 (seed=0) & Repeat 2 (seed=42) + JUnit Parity ---
        _run_cmd(
            ["docker", "exec", cid, "/bin/bash", "-c", f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && git apply /tmp/gold.patch && git apply /tmp/test.patch"],
            timeout=60,
        )
        ref_r1, ref_xml1 = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
        ref_r2, ref_xml2 = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=42, timeout_sec=timeout_int, workdir=workdir)

        ok_xml1, _ = _validate_junit_xml(ref_xml1.decode("utf-8", errors="replace"), tuple(f2p), tuple(p2p), test_patch)
        ok_xml2, _ = _validate_junit_xml(ref_xml2.decode("utf-8", errors="replace"), tuple(f2p), tuple(p2p), test_patch)
        raw_junit_valid = bool(ok_xml1 and ok_xml2 and ref_r1.get("f2p_all_passed") and ref_r2.get("f2p_all_passed"))

        if ref_r1["returncode"] != 0 or ref_r2["returncode"] != 0 or not raw_junit_valid:
            res = {
                "instance_id": iid,
                "completed": True,
                "admitted": False,
                "reason": "failed_gold_or_swegemma_junit_parity",
                "ref_repeat_1": ref_r1,
                "swegemma_raw_junit_valid": raw_junit_valid,
            }
            ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return res

        # If diff-level AST probes were empty, synthesize full-file AST probes inside workdir
        if not alt_candidates or not mutants:
            ast_alt_diff, ast_mut_diff = _synthesize_in_container_ast_probes(
                cid,
                base_commit,
                gold_files,
                need_alt=not alt_candidates,
                need_mutant=not mutants,
                workdir=workdir,
            )
            if ast_alt_diff:
                alt_candidates.append({"kind": "container_fullfile_ast_equiv", "alt_patch": ast_alt_diff})
            if ast_mut_diff:
                mutants.append(
                    {
                        "mutant_id": "m_container_ast_mutant",
                        "kind": "container_fullfile_ast_mutant",
                        "original_line": "",
                        "mutated_line": "",
                        "mutated_patch": ast_mut_diff,
                    }
                )

        # --- Check 3b: Distinct Alternative Equivalent Solution Probe ---
        chosen_alt_kind = "none_applicable"
        alt_applied = False
        alt_passed = False
        for idx_a, acad in enumerate(alt_candidates):
            a_name = f"alt_{idx_a}.patch"
            _copy_bytes_into_container(cid, acad["alt_patch"].encode("utf-8"), "/tmp", a_name)
            app_a = _run_cmd(
                [
                    "docker",
                    "exec",
                    cid,
                    "/bin/bash",
                    "-c",
                    f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && (git apply /tmp/{a_name} || git apply -C1 /tmp/{a_name}) && git apply /tmp/test.patch",
                ],
                timeout=60,
                check=False,
            )
            if app_a.returncode == 0:
                alt_applied = True
                chosen_alt_kind = acad["kind"]
                alt_res, alt_xml = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
                alt_junit_ok, _ = _validate_junit_xml(alt_xml.decode("utf-8", errors="replace"), tuple(f2p), tuple(p2p), test_patch)
                alt_passed = bool(
                    alt_res["returncode"] == 0
                    and alt_res.get("f2p_all_passed")
                    and alt_res.get("p2p_all_passed")
                    and alt_res["failed_count"] == 0
                    and alt_res["error_count"] == 0
                    and alt_junit_ok
                )
                if alt_passed:
                    break

        if not alt_passed:
            ast_alt_diff2, _ = _synthesize_in_container_ast_probes(
                cid, base_commit, gold_files, need_alt=True, need_mutant=False, workdir=workdir
            )
            if ast_alt_diff2:
                _copy_bytes_into_container(cid, ast_alt_diff2.encode("utf-8"), "/tmp", "alt_fb.patch")
                app_fb = _run_cmd(
                    [
                        "docker",
                        "exec",
                        cid,
                        "/bin/bash",
                        "-c",
                        f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && git apply /tmp/alt_fb.patch && git apply /tmp/test.patch",
                    ],
                    timeout=60,
                    check=False,
                )
                if app_fb.returncode == 0:
                    alt_applied = True
                    chosen_alt_kind = "container_fullfile_ast_equiv"
                    alt_res, alt_xml = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
                    alt_junit_ok, _ = _validate_junit_xml(alt_xml.decode("utf-8", errors="replace"), tuple(f2p), tuple(p2p), test_patch)
                    alt_passed = bool(
                        alt_res["returncode"] == 0
                        and alt_res.get("f2p_all_passed")
                        and alt_res.get("p2p_all_passed")
                        and alt_res["failed_count"] == 0
                        and alt_res["error_count"] == 0
                        and alt_junit_ok
                    )

        if not alt_applied:
            res = {
                "instance_id": iid,
                "completed": True,
                "admitted": False,
                "reason": "could_not_synthesize_alternative_solution_probe",
            }
            ctrl_file.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return res

        # --- Check 4: Plausible Incorrect/Shortcut Mutants (K_mut >= 0.75) ---
        killed_mutants = 0
        tested_mutants = 0
        max_total_tested = max_mutants + len(llm_mutants)
        mutant_outcomes: list[dict[str, Any]] = []
        for idx_m, mut in enumerate(mutants):
            if tested_mutants >= max_total_tested:
                break
            m_name = f"mut_{idx_m}.patch"
            _copy_bytes_into_container(cid, mut["mutated_patch"].encode("utf-8"), "/tmp", m_name)
            app = _run_cmd(
                [
                    "docker",
                    "exec",
                    cid,
                    "/bin/bash",
                    "-c",
                    f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && (git apply /tmp/{m_name} || git apply -C1 /tmp/{m_name}) && git apply /tmp/test.patch",
                ],
                timeout=60,
                check=False,
            )
            if app.returncode != 0:
                continue
            tested_mutants += 1
            try:
                m_res, _ = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
                survived = (
                    m_res["returncode"] == 0
                    and m_res.get("f2p_all_passed", False)
                    and m_res.get("p2p_all_passed", False)
                    and m_res["failed_count"] == 0
                    and m_res["error_count"] == 0
                )
            except subprocess.TimeoutExpired:
                survived = False
            killed = not survived
            if killed:
                killed_mutants += 1
            mutant_outcomes.append({"mutant_id": mut["mutant_id"], "kind": mut["kind"], "killed": killed})

        if tested_mutants == 0:
            _, fb_mut_diff = _synthesize_in_container_ast_probes(
                cid, base_commit, gold_files, need_alt=False, need_mutant=True, workdir=workdir
            )
            if fb_mut_diff:
                _copy_bytes_into_container(cid, fb_mut_diff.encode("utf-8"), "/tmp", "mut_fb.patch")
                app_mfb = _run_cmd(
                    [
                        "docker",
                        "exec",
                        cid,
                        "/bin/bash",
                        "-c",
                        f"cd {q_wd} && git reset --hard {shlex.quote(base_commit)} && git clean -fd && git apply /tmp/mut_fb.patch && git apply /tmp/test.patch",
                    ],
                    timeout=60,
                    check=False,
                )
                if app_mfb.returncode == 0:
                    tested_mutants += 1
                    try:
                        m_res, _ = _run_pytest_in_container(cid, test_cmd, f2p, p2p, hash_seed=0, timeout_sec=timeout_int, workdir=workdir)
                        survived = (
                            m_res["returncode"] == 0
                            and m_res.get("f2p_all_passed", False)
                            and m_res.get("p2p_all_passed", False)
                            and m_res["failed_count"] == 0
                            and m_res["error_count"] == 0
                        )
                    except subprocess.TimeoutExpired:
                        survived = False
                    killed = not survived
                    if killed:
                        killed_mutants += 1
                    mutant_outcomes.append({"mutant_id": "m_fb_ast", "kind": "container_fullfile_ast_mutant", "killed": killed})

        mut_kill_score = (killed_mutants / tested_mutants) if tested_mutants > 0 else 0.0

        control_report = {
            "base_repeat_1": base_r1,
            "base_repeat_2": base_r2,
            "ref_repeat_1": ref_r1,
            "ref_repeat_2": ref_r2,
            "editable_workspace_verified": editable_ok,
            "swegemma_raw_junit_valid": raw_junit_valid,
            "swegemma_certified_identity_valid": raw_junit_valid,
            "alt_solution_kind": chosen_alt_kind,
            "alt_solution_passed": alt_passed,
            "mutation_tested_count": tested_mutants,
            "mutation_killed_count": killed_mutants,
            "mutation_kill_score": round(mut_kill_score, 4),
            "mutant_outcomes": mutant_outcomes,
        }

        audit = audit_task_quality(
            row_enriched,
            sealed_repos=sealed_repos,
            control_report=control_report,
            max_test_duration_sec=max_test_duration_sec,
            require_raw_swegemma_junit=True,
            min_mutation_kill_score=min_mutation_kill_score,
            require_container_mutation_and_alt=True,
            repo_rename_map=repo_rename_map,
        )

        snap_sha = ""
        if audit.admitted:
            dst_tgz = snaps_dir / f"{iid}.tgz"
            snap_sha = _build_sanitized_tgz_from_tar_bytes(base_tar_bytes, dst_tgz)

        task_admitted = bool(audit.admitted)
        final_doc = {
            "instance_id": iid,
            "repo": str(row.get("repo", "")),
            "canonical_repo": audit.canonical_repo,
            "license_spdx": detected_spdx,
            "completed": True,
            "admitted": audit.admitted,
            "claude_codex_patches_probed": both_llm_verdicts_present,
            "grading_tier": audit.grading_tier,
            "rejection_summary": list(audit.rejection_summary),
            "upstream_registry_digest": pulled_digest,
            "verified_base_commit": base_commit,
            "v21_environment_repair": repair_info,
            "workdir": workdir,
            "snapshot_sha256": snap_sha,
            "test_cmd": test_cmd,
            "control_report": control_report,
        }
        ctrl_file.write_text(json.dumps(final_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return final_doc
    except Exception as exc:
        err_doc = {
            "instance_id": iid,
            "completed": True,
            "admitted": False,
            "reason": f"container_verification_exception:{type(exc).__name__}:{str(exc)[:300]}",
        }
        ctrl_file.write_text(json.dumps(err_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return err_doc
    finally:
        if cid:
            try:
                _run_cmd(["docker", "rm", "-f", cid], timeout=180, check=False)
            except Exception:
                pass
        if not task_admitted and created_images:
            try:
                _run_cmd(["docker", "rmi", "-f", *created_images], timeout=180, check=False)
            except Exception:
                pass
        if prune_image_after and not task_admitted and not image_preexisted and (pulled_digest or image):
            refs = [r for r in (image, pulled_digest) if r]
            try:
                _run_cmd(["docker", "rmi", "-f", *refs], timeout=180, check=False)
            except Exception:
                pass


def run_parallel_curation(args: argparse.Namespace) -> dict[str, Any]:
    pilot_root = args.pilot_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_candidate_rows(args.rows.resolve())
    sealed_repos = load_harbor_july_sealed_repos(pilot_root)
    repo_lic, repo_rename, repo_gh_id = load_github_provenance_enrichment(pilot_root)

    # Enrich rows with SPDX & GitHub repo_id where known, then run instant static pre-screen (Gates 1, 2, 3, 5-static)
    static_survivors: list[dict[str, Any]] = []
    static_ledger: list[dict[str, Any]] = []
    static_rejected = 0
    for r in rows:
        rk = canonicalize_repo(str(r.get("repo") or ""))
        enriched = dict(r)
        if not enriched.get("license_spdx") and rk in repo_lic:
            enriched["license_spdx"] = repo_lic[rk]
        if rk in repo_gh_id:
            enriched["github_repo_id"] = repo_gh_id[rk]

        # Probe static gates with a synthetic dummy control report so Gates 1, 2, 3, 5-static reject fast
        dummy_ctrl = {
            "base_repeat_1": {"returncode": 1, "f2p_all_failed": True, "p2p_all_passed": True, "error_count": 0, "duration_sec": 1.0, "status_signature_sha256": "s"},
            "base_repeat_2": {"returncode": 1, "f2p_all_failed": True, "p2p_all_passed": True, "error_count": 0, "duration_sec": 1.0, "status_signature_sha256": "s"},
            "ref_repeat_1": {"returncode": 0, "failed_count": 0, "error_count": 0, "duration_sec": 1.0, "status_signature_sha256": "r"},
            "ref_repeat_2": {"returncode": 0, "failed_count": 0, "error_count": 0, "duration_sec": 1.0, "status_signature_sha256": "r"},
            "editable_workspace_verified": True,
            "swegemma_raw_junit_valid": True,
        }
        pre_row = enriched if enriched.get("license_spdx") else {**enriched, "license_spdx": "MIT"}
        pre_audit = audit_task_quality(
            pre_row,
            sealed_repos=sealed_repos,
            control_report=dummy_ctrl,
            repo_rename_map=repo_rename,
        )
        static_ledger.append(
            {
                "instance_id": str(r.get("instance_id")),
                "repo": str(r.get("repo") or ""),
                "static_prescreen_passed": bool(pre_audit.admitted),
                "rejection_summary": list(pre_audit.rejection_summary),
                "note": "static prescreen with a synthetic control report; not executable gate evidence",
            }
        )
        if pre_audit.admitted:
            static_survivors.append(enriched)
        else:
            static_rejected += 1
    with (out_dir / "static_verdicts.jsonl").open("w", encoding="utf-8") as fh:
        for rec in static_ledger:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    if args.require_llm_judge and args.llm_judge_dir is not None:
        jdir = args.llm_judge_dir.resolve() / "verdicts"
        llm_filtered: list[dict[str, Any]] = []
        for r in static_survivors:
            jf = jdir / f"{r['instance_id']}.json"
            if jf.is_file():
                jdoc = json.loads(jf.read_text(encoding="utf-8"))
                if jdoc.get("dual_judge_admitted", False) and (
                    not args.require_claude_judge or isinstance(jdoc.get("claude_verdict"), dict)
                ):
                    llm_filtered.append(r)
        static_survivors = llm_filtered

    if args.max_candidates > 0:
        static_survivors = static_survivors[: args.max_candidates]

    print(
        json.dumps(
            {
                "event": "static_prescreen_complete",
                "total_rows": len(rows),
                "sealed_harbor_july_repos": len(sealed_repos),
                "static_survivors": len(static_survivors),
                "static_rejected": static_rejected,
                "workers": args.workers,
            }
        ),
        flush=True,
    )

    admitted_count = 0
    done_count = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                verify_single_candidate_in_container,
                r,
                output_dir=out_dir,
                sealed_repos=sealed_repos,
                repo_rename_map=repo_rename,
                llm_judge_dir=args.llm_judge_dir.resolve() if args.llm_judge_dir else None,
                max_test_duration_sec=args.max_test_duration_sec,
                min_mutation_kill_score=args.min_mutation_kill_score,
                max_mutants=args.max_mutants,
                prune_image_after=not args.keep_images,
                container_cpu_shares=args.container_cpu_shares,
                v1_image_policy=args.v1_image_policy,
            ): str(r["instance_id"])
            for r in static_survivors
        }
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:
                res = {"instance_id": futs[fut], "admitted": False, "error": str(exc)}
            done_count += 1
            if res.get("admitted"):
                admitted_count += 1
            if args.builder_prune_every > 0 and done_count % args.builder_prune_every == 0:
                try:
                    _run_cmd(["docker", "builder", "prune", "-f", "--keep-storage", "120GB"], timeout=120, check=False)
                except Exception:
                    pass
            if done_count % 10 == 0 or done_count == len(static_survivors):
                print(
                    json.dumps(
                        {
                            "event": "container_progress",
                            "done": done_count,
                            "total": len(static_survivors),
                            "admitted_so_far": admitted_count,
                            "elapsed_sec": round(time.monotonic() - t0, 1),
                        }
                    ),
                    flush=True,
                )

    pin_train, pin_cv = _load_historical_split_pins(
        pin_manifests=getattr(args, "pin_manifests", None),
        pin_train_jsonls=getattr(args, "pin_train_jsonls", None),
        pin_cv_jsonls=getattr(args, "pin_cv_jsonls", None),
    )
    return finalize_verified_splits(
        rows=static_survivors,
        output_dir=out_dir,
        sealed_repos=sealed_repos,
        repo_rename_map=repo_rename,
        llm_judge_dir=args.llm_judge_dir.resolve() if args.llm_judge_dir else None,
        require_llm_judge=args.require_llm_judge,
        require_claude_judge=args.require_claude_judge,
        cv_repo_fraction=args.cv_repo_fraction,
        max_tasks_per_cv_repo=args.max_tasks_per_cv_repo,
        max_test_duration_sec=args.max_test_duration_sec,
        min_mutation_kill_score=args.min_mutation_kill_score,
        pinned_train_repos=pin_train,
        pinned_unseen_cv_repos=pin_cv,
    )


def _load_historical_split_pins(
    *,
    pin_manifests: list[Path] | None = None,
    pin_train_jsonls: list[Path] | None = None,
    pin_cv_jsonls: list[Path] | None = None,
) -> tuple[set[str], set[str]]:
    """Collect monotonic `(pinned_train_repos, pinned_unseen_cv_repos)` across prior rounds."""
    pin_train: set[str] = set()
    pin_cv: set[str] = set()
    for mp in pin_manifests or []:
        if not mp.is_file():
            continue
        doc = json.loads(mp.read_text(encoding="utf-8"))
        pin_train.update(canonicalize_repo(str(r)) for r in doc.get("train_repos", []) if r)
        pin_cv.update(canonicalize_repo(str(r)) for r in doc.get("unseen_cv_repos", []) if r)
    for tp in pin_train_jsonls or []:
        if not tp.is_file():
            continue
        for ln in tp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                pin_train.add(canonicalize_repo(str(json.loads(ln).get("repo", ""))))
    for cp in pin_cv_jsonls or []:
        if not cp.is_file():
            continue
        for ln in cp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                pin_cv.add(canonicalize_repo(str(json.loads(ln).get("repo", ""))))
    pin_train.discard("")
    pin_cv.discard("")
    pin_cv -= pin_train
    return pin_train, pin_cv


def finalize_verified_splits(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    sealed_repos: set[str],
    repo_rename_map: dict[str, str],
    llm_judge_dir: Path | None = None,
    require_llm_judge: bool = False,
    require_claude_judge: bool = False,
    cv_repo_fraction: float,
    max_tasks_per_cv_repo: int,
    max_test_duration_sec: float,
    min_mutation_kill_score: float,
    pinned_train_repos: set[str] | None = None,
    pinned_unseen_cv_repos: set[str] | None = None,
) -> dict[str, Any]:
    ctrl_dir = output_dir / "controls"
    controls_by_id: dict[str, dict[str, Any]] = {}
    meta_by_id: dict[str, dict[str, Any]] = {}
    for p in sorted(ctrl_dir.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        iid = str(doc.get("instance_id") or p.stem)
        if llm_judge_dir is not None:
            jfile = llm_judge_dir / "verdicts" / f"{iid}.json"
            if jfile.is_file():
                jdoc = json.loads(jfile.read_text(encoding="utf-8"))
                if not jdoc.get("dual_judge_admitted", True):
                    continue
                if require_claude_judge and not isinstance(jdoc.get("claude_verdict"), dict):
                    continue
            elif require_llm_judge:
                continue
        meta_by_id[iid] = doc
        if isinstance(doc.get("control_report"), dict):
            controls_by_id[iid] = doc["control_report"]

    enriched_rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for r in rows:
        iid = str(r["instance_id"])
        rcopy = dict(r)
        if iid in meta_by_id and meta_by_id[iid].get("license_spdx"):
            rcopy["license_spdx"] = meta_by_id[iid]["license_spdx"]
        enriched_rows.append(rcopy)
        rows_by_id[iid] = rcopy

    splits = build_repo_disjoint_splits(
        enriched_rows,
        controls_by_id,
        sealed_repos=sealed_repos,
        cv_repo_fraction=cv_repo_fraction,
        max_tasks_per_cv_repo=max_tasks_per_cv_repo,
        max_test_duration_sec=max_test_duration_sec,
        min_mutation_kill_score=min_mutation_kill_score,
        require_container_mutation_and_alt=True,
        repo_rename_map=repo_rename_map,
        pinned_train_repos=pinned_train_repos or set(),
        pinned_unseen_cv_repos=pinned_unseen_cv_repos or set(),
    )

    train_ids: list[str] = splits["train_instance_ids"]
    cv_ids: list[str] = splits["unseen_cv_instance_ids"]
    all_ids: list[str] = sorted(set(train_ids) | set(cv_ids))

    public_rows: dict[str, dict[str, Any]] = {}
    evaluator_routes: dict[str, dict[str, Any]] = {}
    for iid in all_ids:
        row = rows_by_id[iid]
        meta = meta_by_id[iid]
        public_rows[iid] = {
            "instance_id": iid,
            "repo": str(row.get("repo", "")),
            "base_commit": str(meta.get("verified_base_commit") or row.get("base_commit", "")),
            "problem_statement": str(row.get("problem_statement", "")),
            "hints_text": str(row.get("hints_text", "") or ""),
            "created_at": str(row.get("created_at", "") or ""),
        }
        evaluator_routes[iid] = {
            "instance_id": iid,
            "repo": str(row.get("repo", "")),
            "split": "train_pool" if iid in set(train_ids) else "unseen_repo_cv",
            "base_commit": str(meta.get("verified_base_commit") or row.get("base_commit", "")),
            "upstream_registry_digest": str(meta["upstream_registry_digest"]),
            "workdir": str(meta.get("workdir") or "/testbed"),
            "snapshot_sha256": str(meta["snapshot_sha256"]),
            "test_cmd": str(meta["test_cmd"]),
            "test_patch": str(row["test_patch"]),
            "FAIL_TO_PASS": _decode_test_list(row["FAIL_TO_PASS"]),
            "PASS_TO_PASS": _decode_test_list(row["PASS_TO_PASS"]),
        }

    for fname, id_list in (
        ("train.jsonl", train_ids),
        ("unseen_cv.jsonl", cv_ids),
        ("all_verified.jsonl", all_ids),
    ):
        (output_dir / fname).write_text(
            "".join(json.dumps(public_rows[i], ensure_ascii=False) + "\n" for i in id_list),
            encoding="utf-8",
            newline="\n",
        )

    routes_path = output_dir / "evaluator_routes.json"
    routes_path.write_text(
        json.dumps(evaluator_routes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(routes_path, 0o600)

    splits_compact = {k: v for k, v in splits.items() if k != "audits"}
    (output_dir / "sealed_splits_manifest.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(splits_compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return splits_compact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.curation.verify_rebench_containers")
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("/path/to/gemma-devagent-20261001/expansion-pilot-v1/evaluator-only-source/rows.json"),
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("/path/to/gemma-devagent-20261001/expansion-pilot-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/gemma-v2/eval_data/rebench_v3"),
    )
    parser.add_argument("--llm-judge-dir", type=Path, default=None)
    parser.add_argument("--require-llm-judge", action="store_true")
    parser.add_argument("--require-claude-judge", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--cv-repo-fraction", type=float, default=0.18)
    parser.add_argument("--max-tasks-per-cv-repo", type=int, default=4)
    parser.add_argument("--max-test-duration-sec", type=float, default=120.0)
    parser.add_argument("--min-mutation-kill-score", type=float, default=0.75)
    parser.add_argument("--max-mutants", type=int, default=4)
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument(
        "--builder-prune-every",
        type=int,
        default=250,
        help="Run `docker builder prune --keep-storage 120GB` every N candidates; 0 disables (shared daemons).",
    )
    parser.add_argument(
        "--container-cpu-shares",
        type=int,
        default=0,
        help="Optional docker --cpu-shares for curation containers (lower = yields CPU to other workloads).",
    )
    parser.add_argument(
        "--v1-image-policy",
        choices=("local_first", "pull_first"),
        default="local_first",
        help="swerebench/ (v1) images: rebuild locally first (default) or pull the official upstream image first.",
    )
    parser.add_argument(
        "--docker-hub-quota-gate",
        action="store_true",
        help="Gate registry pulls on Docker Hub's remaining anonymous quota; rate-limited pulls are retried later.",
    )
    parser.add_argument("--serial-image-ops", action="store_true", help="At most one local image build at a time "
                        "(and one pull, unless --max-concurrent-pulls says otherwise).")
    parser.add_argument("--max-concurrent-pulls", type=int, default=0, help="Registry pulls in flight (0 = default).")
    parser.add_argument(
        "--pause-if-docker-latency-sec",
        type=float,
        default=0.0,
        help="Before pulls/builds/container starts, wait while `docker ps` takes longer than this (0 = off).",
    )
    parser.add_argument("--container-blkio-weight", type=int, default=0, help="docker --blkio-weight (10-1000; 0 = off).")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--pin-manifests", type=Path, nargs="*", default=None)
    parser.add_argument("--pin-train-jsonls", type=Path, nargs="*", default=None)
    parser.add_argument("--pin-cv-jsonls", type=Path, nargs="*", default=None)
    args = parser.parse_args(argv)

    if args.finalize_only:
        pilot_root = args.pilot_root.resolve()
        rows = load_candidate_rows(args.rows.resolve())
        sealed_repos = load_harbor_july_sealed_repos(pilot_root)
        _, repo_rename, _ = load_github_provenance_enrichment(pilot_root)
        pin_train, pin_cv = _load_historical_split_pins(
            pin_manifests=args.pin_manifests,
            pin_train_jsonls=args.pin_train_jsonls,
            pin_cv_jsonls=args.pin_cv_jsonls,
        )
        rep = finalize_verified_splits(
            rows=rows,
            output_dir=args.output_dir.resolve(),
            sealed_repos=sealed_repos,
            repo_rename_map=repo_rename,
            llm_judge_dir=args.llm_judge_dir.resolve() if args.llm_judge_dir else None,
            require_llm_judge=args.require_llm_judge,
            require_claude_judge=args.require_claude_judge,
            cv_repo_fraction=args.cv_repo_fraction,
            max_tasks_per_cv_repo=args.max_tasks_per_cv_repo,
            max_test_duration_sec=args.max_test_duration_sec,
            min_mutation_kill_score=args.min_mutation_kill_score,
            pinned_train_repos=pin_train,
            pinned_unseen_cv_repos=pin_cv,
        )
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0

    if args.docker_hub_quota_gate:
        _PULL_STATE["enabled"] = True
    global _HEAVY_SEM, _PULL_SEM
    if args.serial_image_ops:
        _HEAVY_SEM = threading.Semaphore(1)
    n_pulls = args.max_concurrent_pulls or (1 if args.serial_image_ops else 64)
    _PULL_SEM = threading.Semaphore(max(1, n_pulls))
    _LOW_IMPACT["max_docker_latency_sec"] = args.pause_if_docker_latency_sec
    _LOW_IMPACT["blkio_weight"] = args.container_blkio_weight
    rep = run_parallel_curation(args)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
