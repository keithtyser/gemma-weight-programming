"""Score arena edits against the unedited arm of the same batch, paired per task and pooled over batches.

For every arm, each batch that holds both the arm and `base` gives one solved-count difference and one per-task
difference. Per-task differences are summed over batches; the sign test and the paired t test are over tasks.
`--versus ARM=OTHER` adds the same comparison between two arms (an edit and its control) over their shared batches.

Usage: python arena_score.py --root results/steer_eval --batches b21 b22 ... [--versus ac_xp_pre=ac_xp_const_ctrl] [--out f.json]
"""
import argparse
import json
import math
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for f in d.rglob("task_results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["instance_id"]] = (bool(r["resolved"]), r.get("tool_calls"))
    return out


def sign_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def t_p(xs: list) -> tuple:
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    if v == 0:
        return 0.0, 1.0
    t = m / math.sqrt(v / n)
    # two-sided p from the normal distribution (n = 170 tasks, so t is close to normal)
    return t, math.erfc(abs(t) / math.sqrt(2))


def compare(runs: dict, a: str, b: str, batches: list) -> dict | None:
    shared = [x for x in batches if a in runs[x] and b in runs[x]]
    if not shared:
        return None
    per = {}
    counts = []
    for x in shared:
        ra, rb = runs[x][a], runs[x][b]
        assert set(ra) == set(rb), (x, a, b)
        counts.append((x, sum(v[0] for v in ra.values()), sum(v[0] for v in rb.values())))
        for t in ra:
            per[t] = per.get(t, 0) + int(ra[t][0]) - int(rb[t][0])
    better = sum(1 for v in per.values() if v > 0)
    worse = sum(1 for v in per.values() if v < 0)
    t, p = t_p([v / len(shared) for v in per.values()])
    tc = lambda arm: sum(v[1] or 0 for x in shared for v in runs[x][arm].values()) / sum(len(runs[x][arm]) for x in shared)
    return {"arm": a, "versus": b, "batches": [c[0] for c in counts], "solved": [c[1] for c in counts],
            "solved_versus": [c[2] for c in counts], "mean_delta": sum(c[1] - c[2] for c in counts) / len(counts),
            "tasks_better": better, "tasks_worse": worse, "sign_p": round(sign_p(better, better + worse), 4),
            "t": round(t, 3), "t_p": round(p, 4), "tool_calls": round(tc(a), 2), "tool_calls_versus": round(tc(b), 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--batches", nargs="+", required=True)
    p.add_argument("--versus", nargs="*", default=[])
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    runs = {}
    for x in a.batches:
        runs[x] = {d.name[len(x) + 1:]: load(d) for d in sorted(a.root.glob(f"{x}_*")) if d.is_dir()}
        runs[x] = {k: v for k, v in runs[x].items() if len(v) == 170}
    arms = sorted({k for r in runs.values() for k in r if k != "base"})
    rows = [compare(runs, arm, "base", a.batches) for arm in arms]
    rows += [compare(runs, *pair.split("="), a.batches) for pair in a.versus]
    rows = [r for r in rows if r]
    base = [(x, sum(v[0] for v in runs[x]["base"].values())) for x in a.batches if "base" in runs[x]]
    print("base", base)
    for r in rows:
        print(f"{r['arm']:42s} vs {r['versus']:38s} n={len(r['batches'])} {r['solved']} vs {r['solved_versus']} "
              f"mean {r['mean_delta']:+.2f} better/worse {r['tasks_better']}/{r['tasks_worse']} sign p {r['sign_p']} "
              f"t p {r['t_p']} calls {r['tool_calls']}/{r['tool_calls_versus']}")
    if a.out:
        a.out.write_text(json.dumps({"base": base, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
