"""Engine hang dump for the TII aligned vLLM engine (plugin P7, on by default with TII_ALIGNED=1; TII_HANG_WATCH=0 off).

Author: Claude Opus 5.5 (engine track), 2026-11-06, after the wave-2 engine hang (12:27Z: no step for 330 s, GPU busy
~40 s then idle, no traceback).

What it records, without CUDA_LAUNCH_BLOCKING and without any device synchronisation:
  * every model step (GPUModelRunner.execute_model / sample_tokens, AsyncGPUModelRunnerOutput.get_output): enter/exit
    time, token / request counts, CUDA-graph or eager;
  * an op ring: each TII op (seg7 attention, cascade groups, LoRA shrink, fused norms, log_softmax, decoder layer)
    appends (time, name, shape info) and, when the stream is NOT being captured, a CUDA event recorded right after
    the launch. At dump time event.query() says which ops the GPU has finished: the first unfinished op (or the
    first op after the last finished one) names the kernel the GPU is stuck in; all finished + a blocked Python
    stack means a host-side block;
  * SIGUSR1 -> faulthandler dump of every Python thread (kill -USR1 <EngineCore pid>);
  * a watchdog thread: when a step has been in flight for TII_HANG_S seconds (default 45), it writes ONE dump
    (in-flight steps, the op ring with event status, all Python thread stacks) to
    $TII_HANG_DIR (default $TII_RECEIPT_DIR or /tmp)/engine-hang-<pid>-<time>.txt and logs the path.
Numerics: none (host-side bookkeeping only; events are recorded only outside graph capture).
"""

from __future__ import annotations

import collections
import faulthandler
import os
import signal
import sys
import threading
import time

_RING: collections.deque = collections.deque(maxlen=int(os.environ.get("TII_HANG_RING", "600")))
_INFLIGHT: dict[int, tuple[str, float, str]] = {}
_LOCK = threading.Lock()
_SEQ = [0]
_STATE = {"dumped_for": None, "installed": False, "dir": None, "steps": 0}


def _dir() -> str:
    d = os.environ.get("TII_HANG_DIR") or os.environ.get("TII_RECEIPT_DIR") or "/tmp"
    os.makedirs(d, exist_ok=True)
    return d


def trace(name: str, info: str = "") -> None:
    """Record one TII op launch (call right AFTER launching the kernel(s) of the op)."""
    if not _STATE["installed"]:
        return
    ev = None
    try:
        import torch

        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            ev = torch.cuda.Event()
            ev.record()
    except Exception:  # noqa: BLE001 - bookkeeping must never break the engine
        ev = None
    _RING.append((time.time(), name, info, ev))


def _enter(name: str, info: str) -> int:
    with _LOCK:
        _SEQ[0] += 1
        k = _SEQ[0]
        _INFLIGHT[k] = (name, time.time(), info)
    return k


def _exit(k: int) -> None:
    with _LOCK:
        _INFLIGHT.pop(k, None)
        _STATE["steps"] += 1


def _wrap(cls, meth: str, info_fn) -> None:
    orig = getattr(cls, meth, None)
    if orig is None or getattr(orig, "_tii_hang", False):
        return

    def w(self, *a, **k):
        try:
            info = info_fn(self, a, k)
        except Exception:  # noqa: BLE001
            info = ""
        key = _enter(f"{cls.__name__}.{meth}", info)
        if not _STATE.get(f"seen_{meth}"):
            _STATE[f"seen_{meth}"] = True
            print(f"[tii-hang-watch] pid {os.getpid()}: first {cls.__name__}.{meth} call is watched", file=sys.stderr,
                  flush=True)
        trace(f"step:{meth}:begin", info)
        try:
            return orig(self, *a, **k)
        finally:
            trace(f"step:{meth}:end", "")
            _exit(key)

    w._tii_hang = True
    setattr(cls, meth, w)


def _sched_info(self, a, k) -> str:
    so = a[0] if a else k.get("scheduler_output")
    try:
        return f"tokens={so.total_num_scheduled_tokens} reqs={len(so.num_scheduled_tokens)}"
    except Exception:  # noqa: BLE001
        return ""


def dump(reason: str) -> str:
    path = os.path.join(_dir(), f"engine-hang-{os.getpid()}-{int(time.time())}.txt")
    now = time.time()
    with open(path, "w") as fh:
        fh.write(f"TII engine hang dump: {reason}\npid {os.getpid()} time {time.strftime('%FT%TZ', time.gmtime(now))}"
                 f" steps_done {_STATE['steps']}\n\n== in-flight steps\n")
        with _LOCK:
            for k, (n, t0, info) in sorted(_INFLIGHT.items()):
                fh.write(f"  #{k} {n} for {now - t0:.1f} s  {info}\n")
        fh.write("\n== op ring (oldest first): time-ago, gpu status, op, info\n")
        first_pending = None
        for i, (t, n, info, ev) in enumerate(list(_RING)):
            if ev is None:
                st = "graph/no-event"
            else:
                try:
                    st = "done" if ev.query() else "PENDING"
                except Exception as e:  # noqa: BLE001
                    st = f"query-error {e!r}"
            if st == "PENDING" and first_pending is None:
                first_pending = (i, n, info)
            fh.write(f"  -{now - t:8.2f}s {st:14s} {n} {info}\n")
        fh.write(f"\n== first op the GPU has not finished: {first_pending}\n")
        fh.write("\n== python stacks (all threads)\n")
        fh.flush()
        faulthandler.dump_traceback(file=fh, all_threads=True)
    print(f"[tii-hang-watch] wrote {path} ({reason})", file=sys.stderr, flush=True)
    return path


def _watch(limit: float) -> None:
    while True:
        time.sleep(max(1.0, limit / 9))
        now = time.time()
        with _LOCK:
            old = [(k, n, now - t0) for k, (n, t0, _i) in _INFLIGHT.items() if now - t0 > limit]
        if old:
            k = min(x[0] for x in old)
            if _STATE["dumped_for"] != k:
                _STATE["dumped_for"] = k
                try:
                    dump(f"step #{k} {old[0][1]} in flight for {old[0][2]:.0f} s (> TII_HANG_S={limit:g})")
                except Exception as e:  # noqa: BLE001
                    print(f"[tii-hang-watch] dump failed: {e!r}", file=sys.stderr, flush=True)


def install() -> dict:
    if _STATE["installed"] and _STATE.get("pid") == os.getpid():
        return {"dir": _STATE["dir"]}
    # (re)install: also in a process forked from an installed one (threads and signal handlers do not survive fork)
    _STATE["pid"] = os.getpid()
    _RING.clear()
    with _LOCK:
        _INFLIGHT.clear()
    from vllm.v1.worker import gpu_model_runner as gmr

    _STATE["dir"] = _dir()
    fh = open(os.path.join(_STATE["dir"], f"engine-stacks-{os.getpid()}.txt"), "a")
    _STATE["sigfile"] = fh
    try:
        faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True, chain=False)
    except (ValueError, RuntimeError):  # not the main thread / no signal support: watchdog only
        pass
    _wrap(gmr.GPUModelRunner, "execute_model", _sched_info)
    _wrap(gmr.GPUModelRunner, "sample_tokens", lambda s, a, k: "")
    _wrap(gmr.AsyncGPUModelRunnerOutput, "get_output", lambda s, a, k: "")
    limit = float(os.environ.get("TII_HANG_S", "45"))
    threading.Thread(target=_watch, args=(limit,), name="tii-hang-watch", daemon=True).start()
    if not _STATE.get("atfork"):
        os.register_at_fork(after_in_child=_reinstall_in_child)
        _STATE["atfork"] = True
    _STATE["installed"] = True
    print(f"[tii-hang-watch] installed in pid {os.getpid()} (TII_HANG_S={limit:g}, dir {_STATE['dir']})",
          file=sys.stderr, flush=True)
    return {"dir": _STATE["dir"], "hang_s": limit}


def _reinstall_in_child() -> None:
    try:
        install()
    except Exception as e:  # noqa: BLE001
        print(f"[tii-hang-watch] reinstall after fork failed: {e!r}", file=sys.stderr, flush=True)
