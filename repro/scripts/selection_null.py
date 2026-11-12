"""Selection null for the largest screening difference among the 12 edit arms (b21-b23), and a
within-batch permutation test of the gate's mean over the five replication batches (b24-b28).

    python -I scripts/selection_null.py      (from repro/)  ->  src/selection_null.json
Pure python. Inputs: src/arena_per_task_b21_b28.tsv, src/dev_runs_b3_b16.tsv.
"""
import collections, json, math, random, statistics as st

ROOT = ""
NPERM = 10000
SCREEN = ["b21", "b22", "b23"]
REPL = ["b24", "b25", "b26", "b27", "b28"]
GATE = "ac_xp_pre"

runs = collections.defaultdict(dict)
for ln in open(ROOT + "src/arena_per_task_b21_b28.tsv", encoding="utf-8"):
    bt, arm, task, v = ln.rstrip("\n").split("\t")
    runs[(bt, arm)][task] = int(v)
S = {k: sum(v.values()) for k, v in runs.items()}
tasks = {b: sorted(runs[(b, "base")]) for b in SCREEN + REPL}
for (b, a), d in runs.items():
    assert sorted(d) == tasks[b], (b, a)

edits = [(b, a) for b in SCREEN for (bb, a) in runs if bb == b and a != "base" and not a.endswith("_ctrl")]
ctrls = [(b, a) for b in SCREEN for (bb, a) in runs if bb == b and a.endswith("_ctrl")]
edits.sort(); ctrls.sort()
assert len(edits) == 12 and len(ctrls) == 3
obs_diff = {f"{b}/{a}": S[(b, a)] - S[(b, "base")] for b, a in edits}
obs_ctrl = {f"{b}/{a}": S[(b, a)] - S[(b, "base")] for b, a in ctrls}
obs_max = max(obs_diff.values())
assert obs_max == 9


def summ(v, obs):
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(math.ceil(p * n)) - 1)]
    return {"n": n, "mean": round(st.mean(v), 3), "median": q(0.5), "p95": q(0.95), "p99": q(0.99),
            "observed": obs, "p_ge_observed": round(sum(x >= obs for x in v) / n, 4),
            "p_ge_observed_plus1": round((sum(x >= obs for x in v) + 1) / (n + 1), 4)}


OUT = {"inputs": ["src/arena_per_task_b21_b28.tsv", "src/dev_runs_b3_b16.tsv"], "n_perm": NPERM, "seed": 0,
       "screening_edit_diffs": obs_diff, "screening_control_diffs": obs_ctrl, "observed_max": obs_max}

# ---------------------------------------------------------------- (b1) paired swap per task, per edit arm (spec)
rng = random.Random(0)
dvec = {(b, a): [runs[(b, a)][t] - runs[(b, "base")][t] for t in tasks[b] if runs[(b, a)][t] != runs[(b, "base")][t]] for b, a in edits}
mx = []
for _ in range(NPERM):
    m = -10 ** 9
    for k in edits:
        s = sum(x if rng.random() < 0.5 else -x for x in dvec[k])
        m = max(m, s)
    mx.append(m)
OUT["b1_paired_swap_max12"] = summ(mx, obs_max)
OUT["b1_paired_swap_max12"]["note"] = ("per task, independently for each edit arm, swap (edit, unedited) labels with prob 1/2; "
                                       "statistic = max over 12 of (edit - unedited) solved; discordant tasks per arm: "
                                       + json.dumps({f"{b}/{a}": len(dvec[(b, a)]) for b, a in edits}))

# ---------------------------------------------------------------- (b2) joint relabelling of all six arms per task
rng = random.Random(0)
arms_b = {b: ["base"] + sorted(a for (bb, a) in runs if bb == b and a != "base") for b in SCREEN}
mx = []
for _ in range(NPERM):
    m = -10 ** 9
    for b in SCREEN:
        A = arms_b[b]
        tot = [0] * len(A)
        for t in tasks[b]:
            vals = [runs[(b, a)][t] for a in A]
            if min(vals) == max(vals):
                for i in range(len(A)):
                    tot[i] += vals[0]
                continue
            rng.shuffle(vals)
            for i in range(len(A)):
                tot[i] += vals[i]
        for i, a in enumerate(A):
            if a != "base" and not a.endswith("_ctrl"):
                m = max(m, tot[i] - tot[0])
    mx.append(m)
OUT["b2_joint_permutation_max12"] = summ(mx, obs_max)
OUT["b2_joint_permutation_max12"]["note"] = ("per task, permute the six arm labels of the batch (4 edits, control, unedited) jointly; "
                                             "statistic = max over the 12 edits of (edit - unedited); keeps the shared unedited arm")

# ---------------------------------------------------------------- (a) arms with no intended effect
R = collections.defaultdict(dict)
for l in open(ROOT + "src/dev_runs_b3_b16.tsv", encoding="utf-8"):
    if "\t" not in l:
        continue
    a, j = l.split("\t", 1)
    d = json.loads(j)
    if d.get("instance_id"):
        R[a][d["instance_id"]] = int(bool(d.get("resolved")))
T = {a: sum(v.values()) for a, v in R.items()}
unedited = [T[f"b{i}_base"] for i in range(3, 16)] + [S[(b, "base")] for b in SCREEN + REPL]
assert len(unedited) == 21 and min(unedited) == 21 and max(unedited) == 30
sd_run = st.stdev(unedited)
C = ["bnegmatch_t0.5", "bnm25_s0", "bnm25_s1", "bnm25_s2"]
dev_ctrl = {f"b{i}_{c}": T[f"b{i}_{c}"] - T[f"b{i}_base"] for i in range(3, 16) for c in C if f"b{i}_{c}" in T}
arena_ctrl = {f"{b}/{a}": S[(b, a)] - S[(b, "base")] for (b, a) in runs if a.endswith("_ctrl")}
ctrl_pool = list(dev_ctrl.values()) + list(arena_ctrl.values())
OUT["a_inputs"] = {"unedited_runs_21": unedited, "unedited_mean": round(st.mean(unedited), 2), "unedited_sd": round(sd_run, 3),
                   "control_minus_unedited_dev": dev_ctrl, "control_minus_unedited_arena": arena_ctrl,
                   "control_diff_pool_n": len(ctrl_pool), "control_diff_pool_mean": round(st.mean(ctrl_pool), 3),
                   "control_diff_pool_sd": round(st.stdev(ctrl_pool), 3),
                   "implied_diff_sd_from_unedited": round(sd_run * math.sqrt(2), 3)}

NSIM = 100000
# (a1) Gaussian runs: each arm and the unedited arm are independent runs with SD = SD of the 21 unedited runs
rng = random.Random(0)
mx = []
for _ in range(NSIM):
    m = -1e9
    for b in range(3):
        z0 = rng.gauss(0, sd_run)
        for k in range(4):
            m = max(m, rng.gauss(0, sd_run) - z0)
    mx.append(round(m))
OUT["a1_gaussian_runs_max12"] = summ(mx, obs_max)
OUT["a1_gaussian_runs_max12"]["note"] = f"{NSIM} sims; run total ~ N(mu, {sd_run:.3f}^2), shared unedited draw per batch, 3 batches x 4 arms; rounded to integer"
# (a2) resample whole run totals from the 21 unedited runs
rng = random.Random(0)
mx = []
for _ in range(NSIM):
    m = -1e9
    for b in range(3):
        z0 = rng.choice(unedited)
        for k in range(4):
            m = max(m, rng.choice(unedited) - z0)
    mx.append(m)
OUT["a2_resample_unedited_runs_max12"] = summ(mx, obs_max)
OUT["a2_resample_unedited_runs_max12"]["note"] = f"{NSIM} sims; every arm and the unedited arm drawn with replacement from the 21 unedited run totals"
# (a3) resample 12 control-minus-unedited differences
rng = random.Random(0)
mx = [max(rng.choice(ctrl_pool) for _ in range(12)) for _ in range(NSIM)]
OUT["a3_resample_control_diffs_max12"] = summ(mx, obs_max)
OUT["a3_resample_control_diffs_max12"]["note"] = (f"{NSIM} sims; 12 draws with replacement from the {len(ctrl_pool)} matched-control minus unedited "
                                                  "differences (14 dev runs b3-b15, 11 arena runs b21-b28); ignores the shared unedited arm")

# ---------------------------------------------------------------- replication mean of the gate, b24-b28
rep = {b: S[(b, GATE)] - S[(b, "base")] for b in REPL}
obs_mean = st.mean(rep.values())
assert abs(obs_mean - 4.2) < 1e-9
rng = random.Random(0)
dv = {b: [runs[(b, GATE)][t] - runs[(b, "base")][t] for t in tasks[b] if runs[(b, GATE)][t] != runs[(b, "base")][t]] for b in REPL}
ms = []
for _ in range(NPERM):
    ms.append(sum(sum(x if rng.random() < 0.5 else -x for x in dv[b]) for b in REPL) / len(REPL))
v = sorted(ms)
OUT["replication_gate_mean"] = {"per_batch": rep, "observed_mean": obs_mean, "discordant_tasks": {b: len(dv[b]) for b in REPL},
                                "gained": sum(x > 0 for b in REPL for x in dv[b]), "lost": sum(x < 0 for b in REPL for x in dv[b]),
                                "null_median": v[NPERM // 2], "null_p95": v[int(math.ceil(0.95 * NPERM)) - 1],
                                "p_one_sided_ge": round(sum(x >= obs_mean - 1e-9 for x in ms) / NPERM, 4),
                                "p_one_sided_ge_plus1": round((sum(x >= obs_mean - 1e-9 for x in ms) + 1) / (NPERM + 1), 4),
                                "p_two_sided_abs": round(sum(abs(x) >= obs_mean - 1e-9 for x in ms) / NPERM, 4),
                                "note": "per task within each batch, swap (gate, unedited) labels with prob 1/2; statistic = mean over b24-b28 of batch difference"}
# exact binomial (sign) on the pooled discordant tasks, for reference
g, l_ = OUT["replication_gate_mean"]["gained"], OUT["replication_gate_mean"]["lost"]
n = g + l_
OUT["replication_gate_mean"]["exact_one_sided_sign_p"] = round(sum(math.comb(n, k) for k in range(g, n + 1)) / 2 ** n, 4)

json.dump(OUT, open(ROOT + "src/selection_null.json", "w", encoding="utf-8"), indent=1)
for k in ["b1_paired_swap_max12", "b2_joint_permutation_max12", "a1_gaussian_runs_max12", "a2_resample_unedited_runs_max12", "a3_resample_control_diffs_max12"]:
    print(k, {x: OUT[k][x] for x in ["median", "p95", "p99", "mean", "p_ge_observed"]})
print("a_inputs", {k: OUT["a_inputs"][k] for k in ["unedited_mean", "unedited_sd", "control_diff_pool_n", "control_diff_pool_mean", "control_diff_pool_sd", "implied_diff_sd_from_unedited"]})
print("replication", {k: v for k, v in OUT["replication_gate_mean"].items() if k != "note"})
print("edits", obs_diff, "ctrls", obs_ctrl)
