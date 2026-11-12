"""Per-GPU launch lock + retry for vLLM startups on a shared machine (stdlib only).

Author: Claude Opus 5.5 (async RL track), 2026-11-05.

Problem: vLLM's startup memory profiling measures free GPU memory before and after a profiling run and attributes any
change to itself; another process allocating or freeing memory on the SAME GPU during that window aborts the start
("Error in memory profiling ... other processes ... release GPU memory", or "No available memory for the cache
blocks"). Seen three times on evalhost on 05-Nov (screen restart, async-rl x2).

Protocol (documented in COORDINATION.md): every process that starts a vLLM engine, or makes a large (> ~1 GB) GPU
allocation burst at startup (model load), takes the lock of each GPU it uses for that window:
  - lock file  <LOCK_DIR>/gpu<N>.lock, LOCK_DIR = $V2_GPU_LOCK_DIR or /path/to/gemma-v2/tii-dev/.gpu_locks;
  - an OS advisory lock (fcntl.flock): released automatically if the holder dies; holder info in gpu<N>.lock.info;
  - GPU indices = physical ids from CUDA_VISIBLE_DEVICES (with CUDA_DEVICE_ORDER=PCI_BUS_ID, as all our scripts set).
Hold it from process start until the server is healthy (or the model is loaded), then release; steady-state running
and serving do not hold it. Locks of different GPUs are independent.

Usage:
  with gpu_launch_lock([0], owner="engine C3"):           # offline LLM(...) / trainer model load
      llm = LLM(**kw)
  proc = server_popen(cmd, env=env, ...)                   # orchestrators: lock taken at Popen (opt-in, see below)
  wait_for_vllm_health(port, proc)                         # releases it (run_dual_gpu_ab.wait_for_vllm_health)
  start_with_retry(start, wait, log_path)                  # async_rl: lock + retry on the profiling failure

`server_popen` is used by run_rebench_v2 / run_dual_gpu_ab and only takes the lock when V2_GPU_LAUNCH_LOCK=1 (their
default behaviour is unchanged).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterator, Sequence

DEFAULT_LOCK_DIR = "/path/to/gemma-v2/tii-dev/.gpu_locks"
PROFILING_FAILURES = (
    "Error in memory profiling",
    "No available memory for the cache blocks",
    "release GPU memory while vLLM is profiling",
)


def lock_dir() -> Path:
    return Path(os.environ.get("V2_GPU_LOCK_DIR") or DEFAULT_LOCK_DIR)


def gpus_from_env(env: dict[str, str] | None = None) -> list[int]:
    """Physical GPU indices named by CUDA_VISIBLE_DEVICES (empty/unset -> [] = no lock needed)."""
    v = (env if env is not None else os.environ).get("CUDA_VISIBLE_DEVICES", "")
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out


class GpuLaunchLock:
    """Exclusive flock on gpu<N>.lock for each GPU (acquired in ascending order: no deadlock between holders)."""

    def __init__(self, gpus: Sequence[int], *, owner: str = "", timeout_s: float = 3600.0, poll_s: float = 1.0) -> None:
        self.gpus = sorted(set(int(g) for g in gpus))
        self.owner, self.timeout_s, self.poll_s = owner, float(timeout_s), float(poll_s)
        self.fds: list[int] = []
        self.waited_s = 0.0

    def acquire(self) -> "GpuLaunchLock":
        d = lock_dir()
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            for g in self.gpus:
                path = d / f"gpu{g}.lock"
                fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - t0 > self.timeout_s:
                            os.close(fd)
                            raise TimeoutError(f"GPU launch lock {path} busy for {self.timeout_s:.0f}s "
                                               f"(holder: {_read_info(path)})")
                        time.sleep(self.poll_s)
                self.fds.append(fd)
                info = {"pid": os.getpid(), "owner": self.owner, "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                Path(str(path) + ".info").write_text(json.dumps(info), encoding="utf-8")
        except BaseException:
            self.release()
            raise
        self.waited_s = time.monotonic() - t0
        return self

    def release(self) -> None:
        while self.fds:
            fd = self.fds.pop()
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self) -> "GpuLaunchLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _read_info(path: Path) -> str:
    try:
        return Path(str(path) + ".info").read_text(encoding="utf-8")
    except OSError:
        return "unknown"


@contextlib.contextmanager
def gpu_launch_lock(gpus: Sequence[int] | None = None, *, owner: str = "", timeout_s: float = 3600.0) -> Iterator[GpuLaunchLock]:
    lk = GpuLaunchLock(gpus_from_env() if gpus is None else gpus, owner=owner, timeout_s=timeout_s)
    with lk:
        yield lk


def server_popen(cmd: list[str], *, env: dict[str, str] | None = None, **kw: Any) -> subprocess.Popen[Any]:
    """subprocess.Popen for a vLLM server. With V2_GPU_LAUNCH_LOCK=1 the server's GPU lock is taken first and attached
    to the process; `release_launch_lock(proc)` (called by wait_for_vllm_health) releases it."""
    lk = None
    if os.environ.get("V2_GPU_LAUNCH_LOCK") == "1":
        lk = GpuLaunchLock(gpus_from_env(env), owner=f"vllm server {' '.join(cmd[:4])}").acquire()
    try:
        proc = subprocess.Popen(cmd, env=env, **kw)
    except BaseException:
        if lk is not None:
            lk.release()
        raise
    proc._v2_launch_lock = lk  # type: ignore[attr-defined]
    return proc


def release_launch_lock(proc: Any) -> None:
    lk = getattr(proc, "_v2_launch_lock", None)
    if lk is not None:
        lk.release()
        proc._v2_launch_lock = None


def is_profiling_failure(log_text: str) -> bool:
    return any(m in log_text for m in PROFILING_FAILURES)


def start_with_retry(start: Callable[[], Any], wait: Callable[[Any], float], log_path: Path, *, gpus: Sequence[int],
                     owner: str, retries: int = 3, backoff_s: float = 30.0) -> tuple[Any, float, int]:
    """Start (start() -> proc) and wait (wait(proc) -> seconds) under the GPU launch lock; if the start dies with the
    vLLM memory-profiling failure (read from the new part of log_path), retry after a backoff. Returns
    (proc, startup seconds, attempts)."""
    for attempt in range(1, retries + 2):
        off = log_path.stat().st_size if log_path.is_file() else 0
        with gpu_launch_lock(gpus, owner=owner):
            proc = start()
            try:
                return proc, wait(proc), attempt
            except RuntimeError:
                tail = ""
                if log_path.is_file():
                    with log_path.open("rb") as f:
                        f.seek(off)
                        tail = f.read().decode("utf-8", errors="replace")
                if attempt > retries or not is_profiling_failure(tail):
                    raise
        time.sleep(backoff_s)
    raise AssertionError("unreachable")
