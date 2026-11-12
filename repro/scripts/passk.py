"""pass@k from saved per-task results. Pure python, fixed seed. Writes src/passk.json.

Estimator (Chen et al. 2021, unbiased): per task 1 - C(n-c, k)/C(n, k), n = runs of the arm on the task, c = solved
runs; pass@k of an arm = mean over tasks. Paired comparison: per-task difference of pass@k between two arms on the same
tasks; percentile bootstrap over tasks, 10,000 resamples, random.Random(0) per comparison (same convention as
review_stats.py). Decoding is greedy: runs differ only through non-repeatable serving, so pass@k measures run-to-run
variability, not sampling diversity.

Inputs: src/dev_runs_b3_b16.tsv, src/arena_per_task_b21_b28.tsv, src/heldout_v1_per_rollout.tsv (kept + rerun rows,
identical to src/remote_diag.json heldout_v1_rows), src/heldout_v2_per_rollout.tsv.
"""
import collections, json, math, random

B, SEED = 10000, 0


def r(x, d=4):
    return round(x, d)


def pk(n, c, k):
    if k > n:
        raise ValueError((n, c, k))
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def counts(runs):
    """runs: list of {task: 0/1}. Returns {task: (n, c)}; every run must cover the same tasks."""
    tasks = set(runs[0])
    assert all(set(x) == tasks for x in runs), "runs cover different task sets"
    return {t: (len(runs), sum(x[t] for x in runs)) for t in tasks}


def arm_passk(nc, ks):
    T = len(nc)
    out = {}
    for k in ks:
        s = sum(pk(n, c, k) for n, c in nc.values())
        out[f"pass@{k}"] = {"rate": r(s / T), "tasks_eq": r(s, 2)}
    return out


def boot(d):
    n = len(d)
    rng = random.Random(SEED)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) for _ in range(B))
    return [r(bs[int(.025 * B)], 2), r(bs[int(.975 * B) - 1], 2)]


def compare(x, y, ks):
    """x, y: {task: (n, c)} on the same tasks. Difference x - y in tasks-equivalent and points, bootstrap CI."""
    assert set(x) == set(y)
    tasks = sorted(x)
    out = {}
    for k in ks:
        d = [pk(*x[t], k) - pk(*y[t], k) for t in tasks]
        ci = boot(d)
        out[f"pass@{k}"] = {"x": r(sum(pk(*x[t], k) for t in tasks) / len(tasks)), "y": r(sum(pk(*y[t], k) for t in tasks) / len(tasks)),
                            "diff_tasks_eq": r(sum(d), 2), "diff_points": r(100 * sum(d) / len(tasks), 2),
                            "boot95_tasks_eq": ci, "boot95_points": [r(100 * c / len(tasks), 2) for c in ci],
                            "tasks_better": sum(v > 1e-12 for v in d), "tasks_worse": sum(v < -1e-12 for v in d)}
    return out


OUT = {"method": __doc__.strip(), "bootstrap": {"resamples": B, "seed": SEED, "unit": "task", "type": "paired percentile"}}

# ------------------------------------------------------------------ load
DEV = collections.defaultdict(dict)
for l in open("src/dev_runs_b3_b16.tsv", encoding="utf-8"):
    if "\t" not in l:
        continue
    a, j = l.split("\t", 1)
    d = json.loads(j)
    if d.get("instance_id"):
        DEV[a][d["instance_id"]] = int(bool(d.get("resolved")))
AR = collections.defaultdict(dict)
for l in open("src/arena_per_task_b21_b28.tsv", encoding="utf-8"):
    b, a, t, v = l.rstrip("\n").split("\t")
    AR[(b, a)][t] = int(v)

# ------------------------------------------------------------------ 1. unedited, 170 dev tasks
dev_b = [f"b{i}" for i in range(3, 16)]
ar_b = [f"b{i}" for i in range(21, 29)]
base13 = [DEV[f"{b}_base"] for b in dev_b]
base21 = base13 + [AR[(b, "base")] for b in ar_b]
assert all(len(x) == 170 for x in base21)
U = {}
for nm, runs, ks in (("all_21_runs", base21, [1, 3, 5, 10, 21]), ("b3_b15_13_runs", base13, [1, 3, 5, 10, 13]),
                     ("b21_b28_8_runs", base21[13:], [1, 3, 5, 8])):
    nc = counts(runs)
    U[nm] = {"runs": len(runs), "solved_per_run": [sum(x.values()) for x in runs], "passk": arm_passk(nc, ks),
             "tasks_solved_at_least_once": sum(c > 0 for _, c in nc.values()),
             "tasks_solved_every_run": sum(c == n for n, c in nc.values()), "tasks": len(nc)}
OUT["unedited_dev"] = U

# ------------------------------------------------------------------ 2. arena repeated edits vs unedited, same batches
ARENA = {"ac_xp_pre": "Claude before-first-edit gate", "ac_xp_early": "Claude early-calls gate",
         "ag_explore_then_rotate": "GPT gate + rotation", "am_damped_dual_explore_concise": "Gemini explore + concise"}
A2 = {}
for a, lab in ARENA.items():
    bs = [b for b in ar_b if (b, a) in AR]
    x = counts([AR[(b, a)] for b in bs]); y = counts([AR[(b, "base")] for b in bs])
    n = len(bs)
    ks = sorted({1, 3, min(6, n), n})
    A2[a] = {"label": lab, "batches": bs, "runs": n, "edit": arm_passk(x, ks), "unedited": arm_passk(y, ks),
             "edit_tasks_any": sum(c > 0 for _, c in x.values()), "unedited_tasks_any": sum(c > 0 for _, c in y.values()),
             "edit_minus_unedited": compare(x, y, ks)}
OUT["arena_vs_unedited"] = A2

# ------------------------------------------------------------------ 3. always-on explore-more (0.125-0.5) vs unedited, b3-b15
E = ["bneg_t0.125", "bneg_t0.25", "bneg_t0.5"]
C = ["bnegmatch_t0.5", "bnm25_s0", "bnm25_s1", "bnm25_s2"]
A3 = {}
for a in E:
    bs = [b for b in dev_b if f"{b}_{a}" in DEV]
    x = counts([DEV[f"{b}_{a}"] for b in bs]); y = counts([DEV[f"{b}_base"] for b in bs])
    n = len(bs)
    ks = list(range(1, n + 1))
    A3[a] = {"batches": bs, "runs": n, "edit": arm_passk(x, ks), "unedited": arm_passk(y, ks),
             "edit_tasks_any": sum(c > 0 for _, c in x.values()), "unedited_tasks_any": sum(c > 0 for _, c in y.values()),
             "edit_minus_unedited": compare(x, y, ks)}
# pooled: every run of the three strengths vs the unedited run of each of those batches (unequal run counts: k <= min)
eb = [b for b in dev_b if any(f"{b}_{a}" in DEV for a in E)]
er = [f"{b}_{a}" for b in eb for a in E if f"{b}_{a}" in DEV]
x = counts([DEV[a] for a in er]); y = counts([DEV[f"{b}_base"] for b in eb])
ks = sorted({1, 3, 5, 10, len(eb)})
A3["pooled_0125_05"] = {"edit_runs": er, "unedited_batches": eb, "edit_n": len(er), "unedited_n": len(eb),
                        "edit": arm_passk(x, ks + [len(er)]), "unedited": arm_passk(y, ks),
                        "edit_tasks_any": sum(c > 0 for _, c in x.values()), "unedited_tasks_any": sum(c > 0 for _, c in y.values()),
                        "edit_minus_unedited": compare(x, y, ks),
                        "note": "edit pass@k uses all edit runs (n = edit_n); unedited uses one run per batch (n = unedited_n); compare only k <= unedited_n"}
# pooled edit vs pooled matched controls on the batches that have controls
cb = [b for b in eb if any(f"{b}_{c}" in DEV for c in C)]
er_c = [f"{b}_{a}" for b in cb for a in E if f"{b}_{a}" in DEV]
cr = [f"{b}_{c}" for b in cb for c in C if f"{b}_{c}" in DEV]
xe = counts([DEV[a] for a in er_c]); xc = counts([DEV[a] for a in cr]); yb = counts([DEV[f"{b}_base"] for b in cb])
kk = min(len(er_c), len(cr))
ks = sorted({1, 3, 5, len(cb), kk})
A3["pooled_vs_controls"] = {"batches": cb, "edit_runs": er_c, "control_runs": cr, "edit": arm_passk(xe, ks), "control": arm_passk(xc, ks),
                            "unedited_same_batches": arm_passk(yb, [k for k in ks if k <= len(cb)]),
                            "edit_minus_control": compare(xe, xc, ks),
                            "control_minus_unedited": compare(xc, yb, [k for k in ks if k <= len(cb)])}
OUT["always_on_vs_unedited_b3_b15"] = A3

# ------------------------------------------------------------------ 4. held-out tests (130 tasks x 3 rollouts per arm)
def load_ho(path, five):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in open(path, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        arm, task, k, res = f[0], f[1], int(f[2]), int(f[3])
        d[arm][task][k] = res
    out = {}
    for arm, tasks in d.items():
        assert len(tasks) == 130 and all(sorted(v) == [0, 1, 2] for v in tasks.values()), arm
        out[arm] = {t: (3, sum(v.values())) for t, v in tasks.items()}
    return out


H1 = load_ho("src/heldout_v1_per_rollout.tsv", True)
H2 = load_ho("src/heldout_v2_per_rollout.tsv", False)
ks = [1, 2, 3]
SERV = {"A": ["base", "bneg_t0.25", "bnm25_s1"], "B": ["base2", "bneg_t0.125", "bneg_t0.5"]}
t1 = {"arms": {a: dict(server=s, **arm_passk(H1[a], ks), tasks_any=sum(c > 0 for _, c in H1[a].values()), solved_rollouts=sum(c for _, c in H1[a].values()))
               for s, arms in SERV.items() for a in arms},
      "comparisons": {"A: explore 0.25 - unedited": compare(H1["bneg_t0.25"], H1["base"], ks),
                      "A: explore 0.25 - control": compare(H1["bneg_t0.25"], H1["bnm25_s1"], ks),
                      "A: control - unedited": compare(H1["bnm25_s1"], H1["base"], ks),
                      "B: explore 0.125 - unedited2": compare(H1["bneg_t0.125"], H1["base2"], ks),
                      "B: explore 0.5 - unedited2": compare(H1["bneg_t0.5"], H1["base2"], ks),
                      "unedited2 (B) - unedited (A)": compare(H1["base2"], H1["base"], ks)},
      "rows": "kept + rerun (primary analysis of the outage rule)"}
t2 = {"arms": {a: dict(**arm_passk(H2[a], ks), tasks_any=sum(c > 0 for _, c in H2[a].values()), solved_rollouts=sum(c for _, c in H2[a].values()))
               for a in ["base", "ac_xp_pre", "ac_xp_const_ctrl", "am_damped_dual_explore_concise"]},
      "comparisons": {"gate - unedited": compare(H2["ac_xp_pre"], H2["base"], ks),
                      "gate - control": compare(H2["ac_xp_pre"], H2["ac_xp_const_ctrl"], ks),
                      "gemini - unedited": compare(H2["am_damped_dual_explore_concise"], H2["base"], ks),
                      "control - unedited": compare(H2["ac_xp_const_ctrl"], H2["base"], ks)}}
OUT["heldout"] = {"test1": t1, "test2": t2}

json.dump(OUT, open("src/passk.json", "w", encoding="utf-8"), indent=1)


def show(o, pre=""):
    for k, v in o.items():
        if isinstance(v, dict):
            if "rate" in v:
                print(f"{pre}{k}: {100 * v['rate']:.2f}% ({v['tasks_eq']})")
            elif "diff_tasks_eq" in v:
                print(f"{pre}{k}: x {100 * v['x']:.2f}% y {100 * v['y']:.2f}% diff {v['diff_tasks_eq']} tasks ({v['diff_points']} pts) CI {v['boot95_tasks_eq']} better/worse {v['tasks_better']}/{v['tasks_worse']}")
            else:
                print(pre + k); show(v, pre + "  ")
        elif not isinstance(v, list) or len(v) < 25:
            print(f"{pre}{k}: {v}")


show({k: v for k, v in OUT.items() if k not in ("method",)})
