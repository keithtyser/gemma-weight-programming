"""Extra statistics for the key paired comparisons. Pure python, fixed seed.

    python scripts/stats.py src/arena_per_task_b21_b28.tsv src/heldout_v2_per_rollout.tsv

For each comparison (arm x versus arm y, runs paired within a batch / within a rollout index on one server):
  * 95% confidence interval of the difference: paired percentile bootstrap over TASKS (10,000 resamples, seed 0).
    A resample draws tasks with replacement and keeps all of a task's runs together.
  * exact McNemar test on the paired binary outcomes (one pair per task and batch, or per task and rollout):
    b = pairs solved by x only, c = pairs solved by y only, two-sided exact binomial p on (b, c).
    Pairs of the same task in different batches are not independent, so with more than one batch per task
    the task-level tests (sign, t, bootstrap) are the ones to rely on.
  * Cohen's h = 2 asin(sqrt(p_x)) - 2 asin(sqrt(p_y)) on the pooled solve rates.
"""
import math, random, sys
from collections import defaultdict

B, SEED = 10000, 0


def binom2(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def compare(x, y, scale, label):
    """x, y: {(task, rep): 0/1}. scale converts a summed difference into the reported unit."""
    keys = sorted(set(x) & set(y))
    per = defaultdict(int)
    b = c = 0
    for k in keys:
        per[k[0]] += x[k] - y[k]
        b += x[k] > y[k]
        c += x[k] < y[k]
    d = [per[t] for t in sorted(per)]
    n = len(d)
    rng = random.Random(SEED)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) * scale for _ in range(B))
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    px, py = sum(x[k] for k in keys) / len(keys), sum(y[k] for k in keys) / len(keys)
    h = 2 * math.asin(math.sqrt(px)) - 2 * math.asin(math.sqrt(py))
    print(f"{label}: pairs {len(keys)} tasks {n} diff {sum(d) * scale:+.2f} boot95 [{lo:+.2f}, {hi:+.2f}] "
          f"McNemar b={b} c={c} p={binom2(b, c):.4f} rate {px * 100:.1f}% vs {py * 100:.1f}% h={h:+.3f}")


def main():
    arena, held = sys.argv[1], sys.argv[2]
    runs = defaultdict(dict)                                   # (batch, arm) -> {task: 0/1}
    for ln in open(arena, encoding="utf-8"):
        bt, arm, task, r = ln.rstrip("\n").split("\t")
        runs[(bt, arm)][task] = int(r)
    batches = sorted({k[0] for k in runs})
    arms = sorted({k[1] for k in runs} - {"base"})
    print("# development tasks (170), unit = solved tasks per batch")
    pairs = [(a, "base") for a in arms] + [("ac_xp_pre", "ac_xp_const_ctrl"), ("ac_xp_early", "ac_xp_const_ctrl"),
                                           ("ag_explore_then_rotate", "ag_think_start_explore_ctrl"),
                                           ("am_damped_dual_explore_concise", "am_damped_dual_explore_concise_ctrl")]
    for a, o in pairs:
        shared = [bt for bt in batches if (bt, a) in runs and (bt, o) in runs]
        x = {(t, bt): v for bt in shared for t, v in runs[(bt, a)].items()}
        y = {(t, bt): v for bt in shared for t, v in runs[(bt, o)].items()}
        compare(x, y, 1 / len(shared), f"{a} - {o} [{','.join(shared)}]")
    # repeats only (first batch of the arm dropped)
    shared = [bt for bt in batches if (bt, "ac_xp_pre") in runs][1:]
    x = {(t, bt): v for bt in shared for t, v in runs[(bt, "ac_xp_pre")].items()}
    y = {(t, bt): v for bt in shared for t, v in runs[(bt, "base")].items()}
    compare(x, y, 1 / len(shared), f"ac_xp_pre - base, repeats only [{','.join(shared)}]")

    print("# held-out tasks (130 x 3 rollouts), unit = tasks-equivalent (solved rollouts / 3)")
    hv = defaultdict(dict)
    for ln in open(held, encoding="utf-8"):
        arm, task, ri, r = ln.rstrip("\n").split("\t")
        hv[arm][(task, ri)] = int(r)
    for a, o in [("ac_xp_pre", "base"), ("ac_xp_pre", "ac_xp_const_ctrl"), ("am_damped_dual_explore_concise", "base"),
                 ("ac_xp_const_ctrl", "base")]:
        compare(hv[a], hv[o], 1 / 3, f"{a} - {o}")


if __name__ == "__main__":
    main()
