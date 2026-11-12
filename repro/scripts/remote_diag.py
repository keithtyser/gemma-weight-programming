"""Diagnosis, replay and held-out (run 1) extracts on the evaluation host (CPU, read-only).

Prints one JSON object. Usage (from the project root on the host): python3 - < remote_diag.py > remote_diag.json
"""
import collections, datetime, json, math, os, sys
from pathlib import Path

R = Path("results")
S = R / "steer"
out = {}


def mh(rows, key):
    st = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        st[r["instance_id"]][(0 if r[key] else 2) + (0 if r["resolved"] else 1)] += 1
    num = den = o = e = v = 0.0
    k = 0
    for a, b, c, d in st.values():
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        k += 1
        num += a * d / n
        den += b * c / n
        o += a
        e += (a + b) * (a + c) / n
        v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    return (num / den if den else None), ((o - e) / math.sqrt(v) if v else 0.0), k


for name in ["tracefeat_base", "tracefeat_basefam"]:
    rows = [json.loads(l) for l in (S / name / "episodes.jsonl").open()]
    arms = sorted({r["arm"] for r in rows})
    d = {"episodes": len(rows), "arms": arms, "tasks": len({r["instance_id"] for r in rows}),
         "solved": sum(r["resolved"] for r in rows), "features": {}}
    for key in [k for k, v in rows[0].items() if isinstance(v, bool) and k != "resolved"]:
        on = [r for r in rows if r[key]]
        off = [r for r in rows if not r[key]]
        orr, z, k = mh(rows, key)
        d["features"][key] = {"n_on": len(on), "solved_on": sum(r["resolved"] for r in on), "n_off": len(off),
                              "solved_off": sum(r["resolved"] for r in off), "mh_or": orr, "z": z, "strata": k}
    out[name] = d

for f in ["replay_r2/summary.json", "replay_b6_neg/summary.json", "replay_b5_sub/summary.json", "replay_b1/summary.json",
          "mine/stats.json", "states_suc/states_summary.json", "states_ef/states_summary.json"]:
    p = S / f
    if p.exists():
        j = json.loads(p.read_text())
        if isinstance(j, dict):
            j = {k.split("/")[-1]: v for k, v in j.items()}
        out[f] = j

# held-out run 1: kept rows (request log written before the cutoff) plus rerun rows
cut = datetime.datetime.fromisoformat("2026-11-12T01:26:00+00:00").timestamp()


def records(o):
    rec = {}
    for f in sorted((o / "shard0_gpu0").rglob("task_results.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rec[(r["instance_id"], int(r.get("rollout_idx", 0)))] = r
    return rec


v1 = []
for arm in ["base", "base2", "bnm25_s1", "bneg_t0.125", "bneg_t0.25", "bneg_t0.5"]:
    o = (R / "steer_confirm" / f"heldout_v1_{arm}").resolve()
    keep = {}
    for k, r in records(o).items():
        p = r.get("request_log_path")
        if p and os.path.exists(p) and os.path.getmtime(p) < cut:
            keep[k] = r
    new = records(o.with_name(o.name + "_p2"))
    assert not set(keep) & set(new)
    for (i, k), r in sorted(keep.items()):
        v1.append([arm, i, k, int(bool(r.get("resolved"))), 1])
    for (i, k), r in sorted(new.items()):
        v1.append([arm, i, k, int(bool(r.get("resolved"))), 0])
out["heldout_v1_rows"] = v1  # arm, task, rollout, resolved, kept(1)/rerun(0)
json.dump(out, sys.stdout)
