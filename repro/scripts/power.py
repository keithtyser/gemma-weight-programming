"""Power of the pre-registered held-out design (130 tasks x 3 rollouts, paired by task). Pure python.

    python scripts/power.py src/heldout_v2_per_task.tsv

Input: one row per task, solved rollouts (0..3) per arm. For each registered pair the per-task difference d_i is
computed, with mean m and sample s.d. s (n - 1 denominator). Normal approximation to the paired t test:
    minimum detectable mean difference at two-sided alpha and power 1 - b:  (z_{1-alpha/2} + z_{1-b}) * s / sqrt(n)
    power to detect a true mean difference D:  Phi(D * sqrt(n) / s - z_{1-alpha/2})
Differences are reported in solved rollouts per task, in tasks-equivalent (sum over tasks / 3) and in solve-rate points.
"""
import math
import sys

Z975, Z80 = 1.959964, 0.841621
PAIRS = [("ac_xp_pre", "base"), ("ac_xp_pre", "ac_xp_const_ctrl"), ("am_damped_dual_explore_concise", "base"),
         ("ac_xp_const_ctrl", "base")]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main(path, rollouts=3):
    rows = [ln.rstrip("\n").split("\t") for ln in open(path, encoding="utf-8") if ln.strip()]
    head, rows = rows[0], rows[1:]
    col = {a: [int(r[head.index(a)]) for r in rows] for a in head[1:]}
    n = len(rows)
    print(f"tasks {n}; solved rollouts per arm: " + ", ".join(f"{a} {sum(v)}" for a, v in col.items()))
    for a in head[1:]:
        k = [sum(1 for v in col[a] if v == j) for j in range(rollouts + 1)]
        print(f"  {a}: tasks solved 0/1/2/3 times = {k}; solved at least once {n - k[0]}")
    for x, y in PAIRS:
        d = [p - q for p, q in zip(col[x], col[y])]
        m = sum(d) / n
        s = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
        se = s / math.sqrt(n)
        mde = (Z975 + Z80) * se
        print(f"{x} - {y}: mean {m:.4f} rollouts/task, sd {s:.4f}, se {se:.4f}, t {m / se:.3f}; "
              f"95% CI tasks-eq [{(m - Z975 * se) * n / rollouts:.1f}, {(m + Z975 * se) * n / rollouts:.1f}]; "
              f"MDE(80%) {mde:.4f} rollouts/task = {mde * n / rollouts:.1f} tasks-eq = {100 * mde / rollouts:.1f} points; "
              f"power at +3 tasks-eq {phi(3 * rollouts / n / se - Z975):.2f}, at +5 {phi(5 * rollouts / n / se - Z975):.2f}, "
              f"at +10 {phi(10 * rollouts / n / se - Z975):.2f}; "
              f"tasks needed for 80% power at the observed sd for +5 tasks-eq per 130: "
              f"{math.ceil(((Z975 + Z80) * s / (5 * rollouts / n)) ** 2)}")


if __name__ == "__main__":
    main(sys.argv[1])
