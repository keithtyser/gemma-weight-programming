"""Mean, SD (n-1) and SE of per-batch differences in solved tasks (edit minus unedited model of the same batch).
    python scripts/batch_sd.py src/arena_score_b21_b28.json      Pure python; prints the structure it finds if keys differ."""
import json, math, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
def stat(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return f"n {n} mean {m:.2f} sd {sd:.2f} se {sd / math.sqrt(n):.2f}"
print("unedited", stat([b[1] for b in d["base"]]))
for r in d["rows"]:
    print(r["arm"], "vs", r["versus"], "| arm", stat(r["solved"]), "| diff", stat([a - b for a, b in zip(r["solved"], r["solved_versus"])]))
