"""Pre-registered analysis of held-out v2 (results/steer_confirm/heldout_v2_prereg.md). Pure python.

    python heldout2_analyze.py --tags g1 [g0] --out heldout_v2_analysis.json
Per task: solved rollouts (0..3) per arm; differences are within a replicate (one server) and summed over replicates.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

C = Path("/path/to/gemma-v2/results/steer_confirm")
ARMS = ["base", "ac_xp_pre", "ac_xp_const_ctrl", "am_damped_dual_explore_concise"]
PAIRS = [("ac_xp_pre", "base", "primary"), ("ac_xp_pre", "ac_xp_const_ctrl", "secondary"),
         ("am_damped_dual_explore_concise", "base", "exploratory"), ("ac_xp_const_ctrl", "base", "control vs unedited"),
         ("ac_xp_pre", "am_damped_dual_explore_concise", "exploratory")]


def load(arm, tag):
    rows = {}
    for f in (C / f"heldout_v2_{arm}_{tag}").rglob("task_results.jsonl"):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[(r["instance_id"], r.get("rollout_idx", 0))] = (bool(r.get("resolved")), r.get("tool_calls") or 0)
    return rows


def t_sf2(t, df):
    """Two-sided p of Student t by numeric integration of the density (Simpson)."""
    t = abs(t)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    n = 20000
    h = t / n
    s = f(0) + f(t) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    return max(0.0, 1 - 2 * s * h / 3)


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = {(arm, tag): load(arm, tag) for arm in ARMS for tag in a.tags}
    rep = {"tags": a.tags, "arms": {}, "pairs": []}
    for (arm, tag), rows in data.items():
        rep["arms"][f"{arm}_{tag}"] = {"rollouts": len(rows), "solved": sum(v[0] for v in rows.values()),
                                       "tasks": len({k[0] for k in rows}),
                                       "mean_tool_calls": round(sum(v[1] for v in rows.values()) / max(1, len(rows)), 2)}
    for x, y, role in PAIRS:
        diff = defaultdict(float)
        nroll = 0
        for tag in a.tags:
            rx, ry = data[(x, tag)], data[(y, tag)]
            keys = set(rx) & set(ry)                       # rollouts both arms finished
            nroll += len(keys)
            for k in keys:
                diff[k[0]] += rx[k][0] - ry[k][0]
        d = list(diff.values())
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
        t = m / (sd / math.sqrt(n)) if sd else 0.0
        better, worse = sum(v > 0 for v in d), sum(v < 0 for v in d)
        rep["pairs"].append({"x": x, "y": y, "role": role, "tasks": n, "paired_rollouts": nroll,
                             "sum_diff_rollouts": sum(d), "tasks_equivalent": round(sum(d) / (3 * len(a.tags)), 2),
                             "better": better, "worse": worse, "t": round(t, 3), "p_t": round(t_sf2(t, n - 1), 4),
                             "p_sign": round(sign_p(better, worse), 4)})
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for k, v in rep["arms"].items():
        print(k, v)
    for r in rep["pairs"]:
        print(r)


if __name__ == "__main__":
    main()
