"""Confirmation eval of one served adapter on a rebench-format task set, against an already-running vLLM server.

Author: Claude Fable 5 (keef)
Date: 10-November-2026
PURPOSE: golden_ab's external arm is fixed to golden_v2. This runs the same `_worker_batch` worker on any rebench-format
    directory (tasks.jsonl, evaluator_routes.json, snapshots), optionally restricted to the split that v2.steer.states
    never used for fitting directions (--split heldout), with private image tags (REBENCH_IMAGE_NS) so it can run next
    to other arms and next to a training run that uses the same tasks.
  Writes <out>/payload.json, <out>/shard0_gpu0/* (worker output), <out>/confirm.json (per-task resolved + summary).
  Usage: python -m v2.steer.confirm run --data DIR --served NAME --api-base URL --harness DIR --out DIR
             [--split heldout|train|all] [--limit N] [--concurrency 8] [--rollouts 1]
         python -m v2.steer.confirm compare --anchor OUT_A --candidate OUT_B
SRP/DRY check: Pass - payload fields mirror golden_ab.external_arm_jobs; the worker is run_rebench_v2._worker_batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from v2.constants import E4B_MODEL_ID


def records(out: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rec: dict[tuple[str, int], dict[str, Any]] = {}
    for f in sorted((out / "shard0_gpu0").rglob("task_results.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rec[(r["instance_id"], int(r.get("rollout_idx", 0)))] = r
    return rec


def summarize(out: Path) -> dict[str, Any]:
    rec = records(out)
    n = len(rec)
    res = sum(bool(r.get("resolved")) for r in rec.values())
    summ = {"n": n, "resolved": res, "rate": res / max(n, 1),
            "status": dict(sorted(__import__("collections").Counter(str(r.get("status")) for r in rec.values()).items()))}
    (out / "confirm.json").write_text(json.dumps({"summary": summ, "resolved": {f"{i}|{k}": bool(r.get("resolved"))
                                                                            for (i, k), r in sorted(rec.items())}}, indent=1))
    return summ


def run(a: argparse.Namespace) -> int:
    from v2.eval.run_rebench_v2 import load_rebench_task_inputs
    from v2.steer.states import split_of

    _, tasks_path, snapshots_dir, routes_all, task_ids, raw_rows = load_rebench_task_inputs(a.data, a.data / "tasks.jsonl")
    ids = [i for i in task_ids if a.split == "all" or split_of(i) == a.split]
    if a.limit:
        ids = ids[: a.limit]
    out = a.out.resolve()
    if (out / "payload.json").exists():
        print(f"exists: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks_path": str(tasks_path), "snapshots_dir": str(snapshots_dir), "results_dir": str(out / "shard0_gpu0"),
        "submission_dir": str(a.harness.resolve()), "api_base": a.api_base.rstrip("/"), "served_model_id": a.served,
        "declared_model_alias": a.alias, "concurrency": a.concurrency,
        "routes": {i: routes_all[i] for i in ids}, "batch_images": {},
        "items": [{"instance_id": i, "rollout_idx": r, "num_rollouts": a.rollouts} for i in ids for r in range(a.rollouts)],
        "stream": True, "build_ctx_root": str(out / "tmp_docker_contexts"), "raw_rows": {i: raw_rows.get(i) for i in ids},
        "cleanup_images": True,
    }
    (out / "payload.json").write_text(json.dumps(payload))
    env = dict(os.environ, REBENCH_IMAGE_NS=a.image_ns or f"cf-{out.name}")
    print(f"{len(ids)} tasks x {a.rollouts}, image ns {env['REBENCH_IMAGE_NS']}", flush=True)
    rc = subprocess.run([sys.executable, "-m", "v2.eval.run_rebench_v2", "_worker_batch", str(out / "payload.json")], env=env).returncode
    print(json.dumps(summarize(out)))
    return rc


def compare(a: argparse.Namespace) -> int:
    ra, rc = records(a.anchor), records(a.candidate)
    keys = sorted(set(ra) & set(rc))
    b = sum(bool(ra[k].get("resolved")) and not rc[k].get("resolved") for k in keys)   # anchor only
    c = sum(bool(rc[k].get("resolved")) and not ra[k].get("resolved") for k in keys)   # candidate only
    n = len(keys)
    pa = sum(bool(ra[k].get("resolved")) for k in keys) / max(n, 1)
    pc = sum(bool(rc[k].get("resolved")) for k in keys) / max(n, 1)
    m = b + c
    # exact two-sided McNemar (binomial, p = 0.5)
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m) if m else 1.0
    se = math.sqrt(max(b + c - (c - b) ** 2 / max(n, 1), 0.0)) / max(n, 1)
    print(json.dumps({"n": n, "anchor": pa, "candidate": pc, "delta": pc - pa, "se": se, "anchor_only": b,
                      "candidate_only": c, "mcnemar_p": p, "only_anchor_n": len(ra) - n, "only_candidate_n": len(rc) - n}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.confirm")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--served", required=True)
    r.add_argument("--api-base", required=True)
    r.add_argument("--harness", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", choices=("heldout", "train", "all"), default="heldout")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--rollouts", type=int, default=1)
    r.add_argument("--alias", default=E4B_MODEL_ID)
    r.add_argument("--image-ns", default="")
    c = sub.add_parser("compare")
    c.add_argument("--anchor", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    a = p.parse_args(argv)
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
