"""Rollout <-> learner protocol for asynchronous RL (stdlib only; imported by rollout workers and by the learner).

Author: Claude Opus 5.5 (async RL track), 2026-11-05.

Work units. The wave's task list is streamed in repeated passes; one *unit* = one task in one pass = one group of
G rollouts. Units are numbered in admission order, `unit = pass_idx * n_tasks + task_pos`, and every rollout's
`run_key` encodes task, pass and rollout index (`<iid>__p<pass>__r<ridx>`), so groups never mix passes. Items
without `pass_idx` keep the synchronous run_key (`<iid>` or `<iid>__r<ridx>`) byte for byte.

Admission (bounded staleness, AReaL/PipelineRL style). The learner writes a small JSON file
`{"limit": L, "stop": false, ...}`; a rollout worker may start unit u only while u < L. With learner version k
(= optimizer steps taken), lag cap D, groups per step S, `refunded` = groups the learner dropped (zero variance,
all-infra, stale, missing version) and `extra` over-provisioned groups:

    L = refunded + (k + D + 1) * S + extra

Groups consumed in admission order are then trained at learner version <= (their admission version) + D, because a
version only advances after S usable groups. A group admitted at version v samples every call with a version >= v
(the policy pointer only moves forward), so every token's lag k - v_token <= D. Groups that finish later than their
slot (tails; over-provisioned groups) can exceed D; the learner drops them at train time and refunds their slot.
`stop: true` ends admission (termination).

Behaviour versions. Each model call reads the policy pointer once and is served entirely by that adapter
(`policy-v<N>`), so versions switch only at call (turn) boundaries; the call's request record keeps the name.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

_RUN_KEY_PASS = re.compile(r"^(?P<iid>.+)__p(?P<pass>\d+)__r(?P<ridx>\d+)$")
_VERSION = re.compile(r"^policy-v(\d+)$")


def run_key(instance_id: str, rollout_idx: int, num_rollouts: int, pass_idx: int | None = None) -> str:
    """Synchronous form (pass_idx None) is unchanged: `<iid>` for one rollout, else `<iid>__r<ridx>`."""
    if pass_idx is None:
        return instance_id if int(num_rollouts) == 1 else f"{instance_id}__r{int(rollout_idx)}"
    return f"{instance_id}__p{int(pass_idx)}__r{int(rollout_idx)}"


def item_run_key(item: dict[str, Any]) -> str:
    p = item.get("pass_idx")
    return run_key(str(item["instance_id"]), int(item.get("rollout_idx", 0)), int(item.get("num_rollouts", 1)),
                   None if p is None else int(p))


def parse_run_key(key: str) -> tuple[str, int, int] | None:
    """(instance_id, pass_idx, rollout_idx) for a pass-encoded key, else None."""
    m = _RUN_KEY_PASS.match(key)
    return (m.group("iid"), int(m.group("pass")), int(m.group("ridx"))) if m else None


def unit_key(item: dict[str, Any]) -> str:
    """Group/unit identity: the task id, plus the pass when the item carries one."""
    p = item.get("pass_idx")
    return str(item["instance_id"]) if p is None else f"{item['instance_id']}__p{int(p)}"


def build_pass_items(task_ids: list[str], *, passes: int, group_size: int, first_pass: int = 0) -> list[dict[str, Any]]:
    """Pass-major work list: all G rollouts of task 0 pass p, task 1 pass p, ..., then pass p+1."""
    n = len(task_ids)
    if n == 0 or passes <= 0 or group_size <= 0:
        return []
    if len(set(task_ids)) != n:
        raise ValueError("duplicate task ids in the wave")
    out = []
    for p in range(first_pass, first_pass + passes):
        for t, iid in enumerate(task_ids):
            for r in range(group_size):
                out.append({"instance_id": iid, "rollout_idx": r, "num_rollouts": group_size, "pass_idx": p,
                            "unit": p * n + t})
    return out


def parse_version(name: Any) -> int | None:
    m = _VERSION.match(str(name or ""))
    return int(m.group(1)) if m else None


def version_name(v: int) -> str:
    return f"policy-v{int(v)}"


def admission_limit(*, version: int, max_lag: int, groups_per_step: int, refunded: int = 0, extra: int = 0) -> int:
    return int(refunded) + (int(version) + int(max_lag) + 1) * int(groups_per_step) + int(extra)


def write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".adm")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def write_admission(path: Path, *, limit: int, stop: bool = False, **info: Any) -> None:
    write_json_atomic(Path(path), {"limit": int(limit), "stop": bool(stop), "t": time.time(), **info})


def read_admission(path: Path) -> tuple[int, bool]:
    """(limit, stop); a missing or half-written file admits nothing (limit 0) and does not stop."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return int(doc.get("limit", 0)), bool(doc.get("stop", False))
    except (OSError, ValueError):
        return 0, False


_ADMISSION_CACHE: dict[str, tuple[float, tuple[int, bool]]] = {}


def read_admission_cached(path: Path, max_age_s: float) -> tuple[int, bool]:
    """read_admission, shared by all coroutines of a worker (at most one file read per `max_age_s`)."""
    key = str(path)
    now = time.monotonic()
    hit = _ADMISSION_CACHE.get(key)
    if hit is not None and now - hit[0] < max_age_s:
        return hit[1]
    val = read_admission(path)
    _ADMISSION_CACHE[key] = (now, val)
    return val


_ADMISSION_DOC_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def read_admission_doc_cached(path: Path, max_age_s: float) -> dict[str, Any]:
    """The whole admission document (optional run-time controls such as `slot_cap` / `prewarm` written by the
    learner's PipelineGuard); {} when missing or half-written. At most one file read per `max_age_s`."""
    key = str(path)
    now = time.monotonic()
    hit = _ADMISSION_DOC_CACHE.get(key)
    if hit is not None and now - hit[0] < max_age_s:
        return hit[1]
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        doc = doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        doc = hit[1] if hit is not None else {}
    _ADMISSION_DOC_CACHE[key] = (now, doc)
    return doc


async def wait_admitted(path: Path, unit: int, *, poll_s: float = 0.25) -> bool:
    """Block until `unit` is admitted (True) or the learner stops admission (False)."""
    while True:
        limit, stop = read_admission_cached(path, poll_s / 2)
        if stop:
            return False
        if int(unit) < limit:
            return True
        await asyncio.sleep(poll_s)


# ------------------------------------------------------------------------------------------- replica routing
# Several rollout servers (replicas of the SAME engine closure, e.g. one on the rollout GPU and one next to the learner)
# serve one worker. The learner writes a routing file
#   {"servers": [{"base": "http://127.0.0.1:18990", "slots": 48, "open": true}, ...], "policy": "migrate", "t": ...}
# and every rollout is pinned to one server (prefix-cache locality across its turns). New rollouts go to the open
# server with the lowest active/slots; a call whose server is closed (the colocated replica during a learner step)
# either moves the rollout to an open server ("migrate", one prefix-cache miss) or waits until it reopens ("wait").
# Each call still reads the policy pointer once, and every replica has every version loaded before the pointer flips
# (PolicySync loads on all servers first), so routing never changes which version samples a call.
def write_routing(path: Path, servers: list[dict[str, Any]], *, policy: str = "migrate", **info: Any) -> None:
    if policy not in ("migrate", "wait"):
        raise ValueError(f"unknown routing policy {policy!r}")
    write_json_atomic(Path(path), {"servers": [dict(s) for s in servers], "policy": policy, "t": time.time(), **info})


def read_routing(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("servers") else None


class Router:
    """Per-worker sticky rollout -> server assignment driven by the routing file (single asyncio loop; no locks)."""

    def __init__(self, path: Path | None = None, *, servers: list[dict[str, Any]] | None = None,
                 poll_s: float = 0.25) -> None:
        self.path = Path(path) if path else None
        self.poll_s = float(poll_s)
        self._static = servers
        self._doc: dict[str, Any] | None = None
        self._t = -1e9
        self.assigned: dict[str, int] = {}
        self.active: dict[int, int] = {}
        self.migrations = 0
        self.waits = 0
        self.assign_log: dict[int, int] = {}

    def doc(self) -> dict[str, Any]:
        now = time.monotonic()
        if self.path is not None and now - self._t >= self.poll_s / 2:
            d = read_routing(self.path)
            if d is not None:
                self._doc = d
            self._t = now
        if self._doc is None:
            if self._static is None:
                raise RuntimeError(f"no routing document at {self.path}")
            self._doc = {"servers": self._static, "policy": "migrate"}
        return self._doc

    def _best(self, servers: list[dict[str, Any]], exclude: int | None = None) -> int | None:
        best, key = None, None
        for i, s in enumerate(servers):
            if not s.get("open", True) or i == exclude:
                continue
            k = ((self.active.get(i, 0) + 1) / max(1e-9, float(s.get("slots", 1))), i)
            if key is None or k < key:
                best, key = i, k
        return best

    def assign(self, rk: str) -> int:
        """Pin a new rollout to the open server with the lowest load (active / slots); all closed -> server 0."""
        if rk in self.assigned:
            return self.assigned[rk]
        servers = self.doc()["servers"]
        i = self._best(servers)
        i = 0 if i is None else i
        self.assigned[rk] = i
        self.active[i] = self.active.get(i, 0) + 1
        self.assign_log[i] = self.assign_log.get(i, 0) + 1
        return i

    def release(self, rk: str) -> None:
        i = self.assigned.pop(rk, None)
        if i is not None:
            self.active[i] = max(0, self.active.get(i, 0) - 1)

    async def base_for(self, rk: str) -> str:
        """Server base URL for the next call of rollout `rk` (assigning it on first use)."""
        i = self.assign(rk)
        while True:
            d = self.doc()
            servers = d["servers"]
            if i < len(servers) and servers[i].get("open", True):
                return str(servers[i]["base"])
            if d.get("policy", "migrate") == "migrate":
                j = self._best(servers, exclude=i)
                if j is not None:
                    self.active[i] = max(0, self.active.get(i, 0) - 1)
                    self.active[j] = self.active.get(j, 0) + 1
                    self.assigned[rk] = i = j
                    self.migrations += 1
                    return str(servers[j]["base"])
            self.waits += 1
            await asyncio.sleep(self.poll_s)

    def stats(self) -> dict[str, Any]:
        return {"active": dict(self.active), "assigned_total": dict(self.assign_log), "migrations": self.migrations,
                "wait_polls": self.waits}
