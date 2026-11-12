"""Cost of an episode per arm, paired with the unedited arm of the same batch. Pure python.

    python3 cost_per_arm.py --root results/steer_eval --batches b21 ... --out cost_per_arm.json
Per batch and arm: solved, mean tool calls, mean LLM calls, mean duration (s), and the same per solved task.
Paired (same task, same batch) differences vs base are averaged per batch; the test uses batches as the unit.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

KEYS = ["tool_calls", "total_llm_calls", "duration_seconds"]


def load(d):
    rows = {}
    for f in d.rglob("task_results.jsonl"):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[r["instance_id"]] = r
    return rows


def t_p(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    if sd == 0:
        return None
    t, df = abs(m / (sd / math.sqrt(n))), n - 1
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    k = 20000
    h = t / k
    s = f(0) + f(t) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, k))
    return round(max(0.0, 1 - 2 * s * h / 3), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--batches", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    per = defaultdict(list)          # arm -> list of per-batch dicts
    for b in a.batches:
        base = load(root / f"{b}_base")
        for d in sorted(root.glob(f"{b}_*")):
            arm = d.name[len(b) + 1:]
            rows = load(d)
            keys = sorted(set(rows) & set(base))
            rec = {"batch": b, "tasks": len(keys), "solved": sum(bool(rows[k].get("resolved")) for k in keys),
                   "base_solved": sum(bool(base[k].get("resolved")) for k in keys)}
            for key in KEYS:
                x = [rows[k].get(key) or 0 for k in keys]
                y = [base[k].get(key) or 0 for k in keys]
                rec[key] = round(sum(x) / len(keys), 2)
                rec[key + "_base"] = round(sum(y) / len(keys), 2)
                rec[key + "_diff"] = round((sum(x) - sum(y)) / len(keys), 3)
                rec[key + "_per_solved"] = round(sum(x) / max(1, rec["solved"]), 1)
                rec[key + "_per_solved_base"] = round(sum(y) / max(1, rec["base_solved"]), 1)
            per[arm].append(rec)
    out = {}
    for arm, recs in per.items():
        if arm == "base":
            continue
        s = {"batches": [r["batch"] for r in recs], "solved": [r["solved"] for r in recs],
             "base_solved": [r["base_solved"] for r in recs]}
        for key in KEYS:
            d = [r[key + "_diff"] for r in recs]
            s[key] = {"arm_mean": round(sum(r[key] for r in recs) / len(recs), 2),
                      "base_mean": round(sum(r[key + "_base"] for r in recs) / len(recs), 2),
                      "diff_mean": round(sum(d) / len(d), 3), "p_batch_t": t_p(d),
                      "per_solved_arm": round(sum(r[key + "_per_solved"] for r in recs) / len(recs), 1),
                      "per_solved_base": round(sum(r[key + "_per_solved_base"] for r in recs) / len(recs), 1)}
        out[arm] = s
    Path(a.out).write_text(json.dumps({"summary": out, "per_batch": per}, indent=1))
    for arm, s in sorted(out.items(), key=lambda kv: -len(kv[1]["batches"])):
        print(arm, len(s["batches"]), "solved", sum(s["solved"]), "vs", sum(s["base_solved"]),
              "| tools", s["tool_calls"]["arm_mean"], s["tool_calls"]["base_mean"], s["tool_calls"]["p_batch_t"],
              "| dur", s["duration_seconds"]["arm_mean"], s["duration_seconds"]["base_mean"], s["duration_seconds"]["p_batch_t"],
              "| dur/solved", s["duration_seconds"]["per_solved_arm"], s["duration_seconds"]["per_solved_base"])


if __name__ == "__main__":
    main()
