"""Screening run of one served edit on a subset of the golden tasks (same harness, same workers as golden_ab).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: A full golden run costs ~3 GPU-hours per 8 arms, and 95 of the 170 tasks were never solved in 56 earlier
  runs of any arm, so they carry no signal for ranking edits. This runs the unchanged golden_ab external-server arm
  on the listed task ids only (the plan comes from `golden_ab --dry-run`; the comp worker's task_ids and the rebench
  payload are filtered). It is for ranking candidates; every claim in the paper is re-measured on all 170 tasks.
  Usage: python -m v2.steer.screen --port 18000 --out results/steer_eval/s1_<arm> --arm <served name> --tasks ids.txt
SRP/DRY check: Pass - v2.eval.golden_ab builds the jobs; this only narrows their task lists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

R = Path("/path/to/gemma-v2")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.screen")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--conc", type=int, default=4)
    p.add_argument("--harness", type=Path, default=R / "harnesses" / "steer_ext_w5")
    a = p.parse_args(argv)
    keep = set(a.tasks.read_text().split())
    if a.out.exists():
        print(f"exists: {a.out}", file=sys.stderr)
        return 2
    subprocess.run([sys.executable, "-m", "v2.eval.golden_ab", "--golden", "v2", "--arm", str(a.harness),
                    "--external-api-base", f"http://127.0.0.1:{a.port}/v1", "--served-adapter", a.arm, "--out",
                    str(a.out), "--seeds", "1", "--concurrency-per-gpu", str(a.conc), "--keep-images", "--dry-run"],
                   check=True)
    from v2.eval.run_rebench_v2 import append_missing_records

    plan = json.loads((a.out / "external_plan.json").read_text())
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    n = 0
    for job in plan["jobs"]:
        cmd, payload = job["cmd"], None
        if "--internal-worker-json" in cmd:
            cfg = json.loads(cmd[-1])
            ids = [json.loads(l)["instance_id"] for l in open(cfg["tasks_path"])]
            cfg["task_ids"] = [i for i in ids if i in keep]
            n += len(cfg["task_ids"])
            cmd = cmd[:-1] + [json.dumps(cfg)]
        else:
            path = Path(cmd[-1])
            payload = json.loads(path.read_text())
            payload["items"] = [it for it in payload["items"] if it["instance_id"] in keep]
            for k in ("routes", "raw_rows"):
                payload[k] = {i: v for i, v in payload[k].items() if i in keep}
            n += len(payload["items"])
            path.write_text(json.dumps(payload))
        print(f"== {job['name']}: {n} tasks so far", flush=True)
        with (a.out / "logs" / f"{job['name']}.log").open("w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False).returncode
        if payload is not None:
            append_missing_records(Path(payload["results_dir"]), payload["items"], payload["routes"], rc)
        if rc != 0:
            print(f"{job['name']} failed rc={rc}", file=sys.stderr)
            return rc
    (a.out / "screen.json").write_text(json.dumps({"tasks": sorted(keep), "n_run": n, "arm": a.arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
