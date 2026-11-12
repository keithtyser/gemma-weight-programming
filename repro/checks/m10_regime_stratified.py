"""M10: Test 1 outage, regime-stratified paired comparison (read-only).

Input: src/heldout_v1_per_rollout.tsv  (arm, task, rollout k, solved 0/1, kept 0/1)
  kept = 1: rollout finished before the first outage (pre-outage regime)
  kept = 0: rollout rerun after the outage (post-outage regime)
Arms compared (the three primary arms run side by side): base, bneg_t0.25, bnm25_s1.
Run from repro/:  python checks/m10_regime_stratified.py
"""
import collections, math

ROWS = collections.defaultdict(dict)  # (task,k) -> arm -> (solved, kept)
for line in open("src/heldout_v1_per_rollout.tsv", encoding="utf-8"):
    arm, task, k, res, kept = line.rstrip("\n").split("\t")
    ROWS[(task, int(k))][arm] = (int(res), int(kept))

ARMS = ["base", "bneg_t0.25", "bnm25_s1", "base2", "bneg_t0.125", "bneg_t0.5"]


def binom_two_sided(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def betainc(a, b, x):
    # regularized incomplete beta via continued fraction (Numerical Recipes)
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
            de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * cf(a, b, x) / a
    return 1 - math.exp(lbt) * cf(b, a, 1 - x) / b


def paired_t(diffs):
    n = len(diffs)
    if n < 2: return float("nan")
    m = sum(diffs) / n
    v = sum((d - m) ** 2 for d in diffs) / (n - 1)
    if v == 0: return 1.0 if m == 0 else 0.0
    t = m / math.sqrt(v / n)
    return betainc((n - 1) / 2, 0.5, (n - 1) / (n - 1 + t * t))


print("== Per-arm regime counts and solve rates (kept = finished before outage)")
for a in ARMS:
    kept = [v[a] for v in ROWS.values() if v[a][1] == 1]
    rer = [v[a] for v in ROWS.values() if v[a][1] == 0]
    print(f"{a:12s} kept {len(kept)} solved {sum(s for s, _ in kept)} ({100*sum(s for s,_ in kept)/len(kept):.1f}%)"
          f" | rerun {len(rer)} solved {sum(s for s, _ in rer)} ({100*sum(s for s,_ in rer)/len(rer):.1f}%)")

def pooled(arms, kept_flag):
    xs = [v[a] for v in ROWS.values() for a in arms if v[a][1] == kept_flag]
    return sum(s for s, _ in xs), len(xs)

for name, arms in [("primary three arms", ["base", "bneg_t0.25", "bnm25_s1"]),
                   ("other three arms", ["base2", "bneg_t0.125", "bneg_t0.5"])]:
    sk, nk = pooled(arms, 1); sr, nr = pooled(arms, 0)
    print(f"pooled {name}: kept {sk}/{nk} = {100*sk/nk:.1f}%, rerun {sr}/{nr} = {100*sr/nr:.1f}%")

sk, nk = pooled(["base", "bneg_t0.25", "bnm25_s1"], 1); sr, nr = pooled(["base", "bneg_t0.25", "bnm25_s1"], 0)
gap = sk / nk - sr / nr
print("\n== Exposure penalty implied by unequal pre-outage completion (primary arms' pooled regime gap)")
for a in ["bneg_t0.25", "bnm25_s1"]:
    extra = sum(1 for v in ROWS.values() if v["base"][1] == 1) - sum(1 for v in ROWS.values() if v[a][1] == 1)
    print(f"{a}: {extra} more rerun rollouts than base x gap {gap:.3f} = {extra*gap:.2f} rollouts = {extra*gap/3:.2f} tasks-equivalent")


def compare(x, y, label):
    print(f"\n== {label}: {x} minus {y}")
    for regime, flag in [("all three primary arms pre-outage (kept)", 1), ("all three primary arms rerun", 0), ("either (pooled strata)", None)]:
        items = [(t, k) for (t, k), v in ROWS.items()
                 if (flag is None and len({v[a][1] for a in ("base", "bneg_t0.25", "bnm25_s1")}) == 1)
                 or (flag is not None and all(v[a][1] == flag for a in ("base", "bneg_t0.25", "bnm25_s1")))]
        sx = sum(ROWS[i][x][0] for i in items); sy = sum(ROWS[i][y][0] for i in items)
        b = sum(1 for i in items if ROWS[i][x][0] == 1 and ROWS[i][y][0] == 0)
        c = sum(1 for i in items if ROWS[i][x][0] == 0 and ROWS[i][y][0] == 1)
        per_task = collections.defaultdict(int)
        for (t, k) in items:
            per_task[t] += ROWS[(t, k)][x][0] - ROWS[(t, k)][y][0]
        diffs = list(per_task.values())
        print(f"  {regime}: {len(items)} rollout slots over {len(per_task)} tasks; solved {sx} vs {sy};"
              f" diff {sx - sy:+d} rollouts = {(sx - sy)/3:+.2f} tasks-equivalent;"
              f" discordant {b} vs {c}, exact McNemar p = {binom_two_sided(b, c):.3f};"
              f" paired t over tasks p = {paired_t(diffs):.3f}")


compare("bneg_t0.25", "base", "Primary edit vs unedited")
compare("bneg_t0.25", "bnm25_s1", "Primary edit vs control")
compare("bnm25_s1", "base", "Control vs unedited")
