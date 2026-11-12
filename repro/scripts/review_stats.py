"""Statistics for the revision pass. Pure python, fixed seed. Writes src/review_stats.json.

    python scripts/review_stats.py     (run from the paper directory)

Inputs (all in src/): arena_per_task_b21_b28.tsv, dev_runs_b3_b16.tsv, heldout_v2_per_rollout.tsv, remote_diag.json.
Unit of analysis: the BATCH (all arms of a batch share one server and one unedited run). For every comparison we
give the per-batch differences, their mean, SD, SE, the 95% Student-t confidence interval, the two-sided paired t
test over batches (exact Student distribution) and the exact two-sided sign test over batches. Task-level tests
(paired t over the 170 tasks on differences summed over batches; exact McNemar on task-batch pairs; paired
percentile bootstrap over tasks, 10,000 resamples, seed 0) are secondary: runs of one task in different batches
are not independent replicates of the edit effect.
"""
import collections, json, math, random, statistics as st

B, SEED = 10000, 0
OUT = {}


def _betacf(a, b, x):
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, alpha=0.05):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_p(mid, df) > alpha: lo = mid
        else: hi = mid
    return hi


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def r(x, k=3):
    return None if x is None else round(x, k)


def paired(d):
    """d: list of paired differences. Mean, SD, SE, 95% t interval, t test, sign test."""
    n = len(d)
    o = {"n": n, "diffs": [r(x, 2) for x in d], "mean": r(st.mean(d))}
    w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    o.update(up=w, down=l, sign_p=r(sign_p(w, l), 4), min_sign_p=r(2 / 2 ** n if n else None, 4))
    if n >= 2:
        sd = st.stdev(d); se = sd / math.sqrt(n)
        o.update(sd=r(sd), se=r(se))
        if se > 0:
            t = st.mean(d) / se; c = t_crit(n - 1)
            o.update(t=r(t), df=n - 1, p=r(t_p(t, n - 1), 4), ci95=[r(st.mean(d) - c * se, 2), r(st.mean(d) + c * se, 2)])
    return o


def task_level(x, y, scale=1.0):
    """x, y: {(task, rep): 0/1}. Secondary task-level statistics."""
    keys = sorted(set(x) & set(y))
    per = collections.defaultdict(int)
    b = c = 0
    for k in keys:
        per[k[0]] += x[k] - y[k]
        b += x[k] > y[k]; c += x[k] < y[k]
    d = [per[t] for t in sorted(per)]
    n = len(d)
    rng = random.Random(SEED)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) * scale for _ in range(B))
    px, py = sum(x[k] for k in keys) / len(keys), sum(y[k] for k in keys) / len(keys)
    sd = st.stdev(d); t = st.mean(d) / (sd / math.sqrt(n)) if sd else 0.0
    w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
    return {"pairs": len(keys), "tasks": n, "diff": r(sum(d) * scale, 2), "boot95": [r(boots[int(.025 * B)], 2), r(boots[int(.975 * B) - 1], 2)],
            "task_t_p": r(t_p(t, n - 1), 4), "tasks_better": w, "tasks_worse": l, "task_sign_p": r(sign_p(w, l), 4),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": r(sign_p(b, c), 4), "rate_x": r(px, 4), "rate_y": r(py, 4),
            "cohen_h": r(2 * math.asin(math.sqrt(px)) - 2 * math.asin(math.sqrt(py)))}


def holm(ps):
    """ps: {name: p}. Holm-Bonferroni adjusted p-values."""
    order = sorted(ps, key=ps.get)
    m, run, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = r(run, 4)
    return adj


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [r(c - h), r(c + h)]


# ---------------------------------------------------------------- three-author benchmark (batches 21-28)
runs = collections.defaultdict(dict)
for ln in open("src/arena_per_task_b21_b28.tsv", encoding="utf-8"):
    bt, arm, task, v = ln.rstrip("\n").split("\t")
    runs[(bt, arm)][task] = int(v)
batches = sorted({k[0] for k in runs})
S = {k: sum(v.values()) for k, v in runs.items()}
CTRL = {"ac_xp_pre": "ac_xp_const_ctrl", "ac_xp_early": "ac_xp_const_ctrl", "ag_explore_then_rotate": "ag_think_start_explore_ctrl",
        "am_damped_dual_explore_concise": "am_damped_dual_explore_concise_ctrl"}
arms = sorted({k[1] for k in runs} - {"base"})
edits = [a for a in arms if not a.endswith("_ctrl")]


def cmp(a, o, bs=None):
    shared = [bt for bt in batches if (bt, a) in runs and (bt, o) in runs and (bs is None or bt in bs)]
    if not shared:
        return None
    d = paired([S[(bt, a)] - S[(bt, o)] for bt in shared])
    d["batches"] = shared
    d["solved_x"] = [S[(bt, a)] for bt in shared]; d["solved_y"] = [S[(bt, o)] for bt in shared]
    x = {(t, bt): v for bt in shared for t, v in runs[(bt, a)].items()}
    y = {(t, bt): v for bt in shared for t, v in runs[(bt, o)].items()}
    d["task_level"] = task_level(x, y, 1.0 / len(shared))
    return d


A = {"unedited_per_batch": {bt: S[(bt, "base")] for bt in batches}}
ub = list(A["unedited_per_batch"].values())
A["unedited_mean_sd"] = [r(st.mean(ub)), r(st.stdev(ub))]
A["vs_unedited"] = {a: cmp(a, "base") for a in arms}
A["edit_vs_control"] = {a: cmp(a, c) for a, c in CTRL.items()}
A["control_vs_unedited"] = {c: cmp(c, "base") for c in sorted(set(CTRL.values()))}
rep = {}
for a, c in CTRL.items():
    own = [bt for bt in batches if (bt, a) in runs]
    screen, later = own[0], own[1:]
    rep[a] = {"screening_batch": screen, "screening_diff_vs_unedited": S[(screen, a)] - S[(screen, "base")],
              "screening_diff_vs_control": (S[(screen, a)] - S[(screen, c)]) if (screen, c) in runs else None,
              "replication_vs_unedited": cmp(a, "base", later), "replication_vs_control": cmp(a, c, later),
              "replication_control_vs_unedited": cmp(c, "base", later)}
A["screening_vs_replication"] = rep
# Holm over the 12 screened edits: (i) task-level paired t over all batches of each edit, (ii) exact McNemar in the
# screening batch alone (the only test that exists for all twelve on equal footing).
p_all = {a: A["vs_unedited"][a]["task_level"]["task_t_p"] for a in edits}
first = {a: [bt for bt in batches if (bt, a) in runs][0] for a in edits}
p_scr = {a: cmp(a, "base", [first[a]])["task_level"]["mcnemar_p"] for a in edits}
A["holm"] = {"task_t_all_batches": {"raw": p_all, "holm": holm(p_all)}, "mcnemar_screening_batch": {"raw": p_scr, "holm": holm(p_scr)}}
# among the four repeated edits only, replication batches, batch-level t
p_rep = {a: rep[a]["replication_vs_unedited"].get("p") for a in CTRL}
A["holm"]["batch_t_replication_four_edits"] = {"raw": p_rep, "holm": holm(p_rep)}
OUT["arena"] = A

# calibration (prediction: point, lo, hi) from the authors' prediction files; outcome = first-run difference
PRED = {"ac_xp_const": (1.5, -3, 6), "ac_xp_pre": (1.0, -4, 6), "ac_xp_think": (0.5, -5, 6), "ac_xp_early": (0.0, -6, 5), "ac_xp_const_ctrl": (-0.5, -5, 3),
        "ag_think_start_explore": (2.0, -5, 9), "ag_explore_then_rotate": (1.5, -7, 10), "ag_error_recheck": (0.5, -6, 7), "ag_decision_rotation": (0.5, -6, 7),
        "ag_think_start_explore_ctrl": (0.0, -6, 6), "am_damped_dual_explore_concise": (3.8, -1.5, 9.0), "am_oproj_onesided_commit_damper": (3.0, -3.0, 9.0),
        "am_whitened_cond_rushed_read": (2.6, -3.0, 8.5), "am_hybrid_explore_verify_guard": (2.3, -3.5, 8.0), "am_damped_dual_explore_concise_ctrl": (-0.2, -5.0, 4.5)}
try:
    chk = {}
    for au in ("claude", "gpt", "gemini"):
        j = json.load(open(f"src/{au}/PROPOSALS.json", encoding="utf-8"))
        chk[au] = json.dumps(j)[:0]  # presence check only; values are compared by scripts/check_pred.py if needed
except Exception as e:  # noqa
    chk = {"error": str(e)}
cal = {}
for a, (pt, lo, hi) in PRED.items():
    bt = [b for b in batches if (b, a) in runs][0]
    out = S[(bt, a)] - S[(bt, "base")]
    cal[a] = {"author": {"ac": "Claude", "ag": "GPT", "am": "Gemini"}[a[:2]], "point": pt, "lo": lo, "hi": hi, "width": hi - lo, "batch": bt,
              "outcome": out, "inside": lo <= out <= hi, "abs_error": abs(out - pt), "is_control": a.endswith("_ctrl")}
summ = {}
for au in ("Claude", "GPT", "Gemini"):
    v = [c for c in cal.values() if c["author"] == au]
    summ[au] = {"n": len(v), "inside": sum(c["inside"] for c in v), "mean_width": r(st.mean(c["width"] for c in v), 2),
                "mean_abs_error": r(st.mean(c["abs_error"] for c in v), 2), "mean_signed_error": r(st.mean(c["outcome"] - c["point"] for c in v), 2)}
ed = [c for c in cal.values() if not c["is_control"]]
OUT["calibration"] = {"rows": cal, "per_author": summ, "all_inside": sum(c["inside"] for c in cal.values()), "edits_inside": sum(c["inside"] for c in ed),
                      "n_edits": len(ed), "wilson95_12_of_15": wilson(sum(c["inside"] for c in cal.values()), 15)}

# ---------------------------------------------------------------- always-on explore-more (batches 3-16)
R = collections.defaultdict(dict)
for l in open("src/dev_runs_b3_b16.tsv", encoding="utf-8"):
    if "\t" not in l:
        continue
    a, j = l.split("\t", 1)
    d = json.loads(j)
    if d.get("instance_id"):
        R[a][d["instance_id"]] = int(bool(d.get("resolved")))
T = {a: sum(v.values()) for a, v in R.items()}
E = ["bneg_t0.125", "bneg_t0.25", "bneg_t0.5"]
C = ["bnegmatch_t0.5", "bnm25_s0", "bnm25_s1", "bnm25_s2"]
bl = [f"b{i}" for i in range(3, 16)]
sched = {b: {a: T[f"{b}_{a}"] for a in ["base"] + E + ["bneg_t1.0"] + C if f"{b}_{a}" in T} for b in bl}
D = {"schedule": sched, "unedited_runs": [T[f"{b}_base"] for b in bl]}
D["unedited_mean_sd_range"] = [r(st.mean(D["unedited_runs"]), 2), r(st.stdev(D["unedited_runs"]), 2), min(D["unedited_runs"]), max(D["unedited_runs"])]


def dev_cmp(xs, ys):
    """xs, ys: lists of run names paired one to one (same batch)."""
    x = {(t, i): v for i, a in enumerate(xs) for t, v in R[a].items()}
    y = {(t, i): v for i, a in enumerate(ys) for t, v in R[a].items()}
    g = sum(1 for k in x if k in y and x[k] > y[k]); l = sum(1 for k in x if k in y and x[k] < y[k])
    o = paired([T[a] - T[b] for a, b in zip(xs, ys)])
    o.update(runs=xs, gained=g, lost=l, task_level=task_level(x, y, 1.0 / len(xs)), solved=[T[a] for a in xs], mean_solved=r(st.mean(T[a] for a in xs), 2))
    return o


per = {}
for a in E + ["bneg_t1.0"]:
    bs = [b for b in bl if f"{b}_{a}" in T]
    per[a] = dev_cmp([f"{b}_{a}" for b in bs], [f"{b}_base" for b in bs])
D["per_strength_vs_unedited"] = per
crun = [f"{b}_{c}" for b in bl for c in C if f"{b}_{c}" in T]
D["control_runs_vs_unedited"] = dev_cmp(crun, [x.split("_")[0] + "_base" for x in crun])
D["control_runs_per_adapter"] = {c: [b for b in bl if f"{b}_{c}" in T] for c in C}
erun = [f"{b}_{a}" for b in bl for a in E if f"{b}_{a}" in T]
D["edit_runs_0125_05"] = {"n": len(erun), "mean_solved": r(st.mean(T[x] for x in erun), 2)}
# pooled: mean over the edit arms of a batch
eb = [b for b in bl if any(f"{b}_{a}" in T for a in E)]
me = {b: st.mean(T[f"{b}_{a}"] for a in E if f"{b}_{a}" in T) for b in eb}
mc = {b: st.mean(T[f"{b}_{c}"] for c in C if f"{b}_{c}" in T) for b in eb if any(f"{b}_{c}" in T for c in C)}
D["pooled_edit_vs_unedited"] = dict(paired([me[b] - T[f"{b}_base"] for b in eb]), batches=eb)
D["pooled_edit_vs_control"] = dict(paired([me[b] - mc[b] for b in mc]), batches=list(mc))
D["pooled_control_vs_unedited"] = dict(paired([mc[b] - T[f"{b}_base"] for b in mc]), batches=list(mc))
# strength 0.25 against the controls of its batch (0.25 is the strength the first held-out test registered)
b25 = [b for b in mc if f"{b}_bneg_t0.25" in T]
D["s025_vs_control"] = dict(paired([T[f"{b}_bneg_t0.25"] - mc[b] for b in b25]), batches=b25)
# cohorts: every split of the 12 edit batches into first k / rest
coh = {}
for k in (6, 7):
    for nm, bs in (("first", eb[:k]), ("rest", eb[k:])):
        coh[f"{nm}_{k}" if nm == "first" else f"after_first_{k}"] = dict(
            paired([me[b] - T[f"{b}_base"] for b in bs]), batches=bs, mean_edit=r(st.mean(me[b] for b in bs), 2),
            mean_unedited=r(st.mean(T[f"{b}_base"] for b in bs), 2),
            mean_edit_runs=r(st.mean(T[f"{b}_{a}"] for b in bs for a in E if f"{b}_{a}" in T), 2))
# also with batch 3 (no edit arm) counted among the unedited runs, as an earlier draft did
coh["earlier_draft_first7_b3_b9"] = {"mean_edit_runs": r(st.mean(T[f"{b}_{a}"] for b in bl[:7] for a in E if f"{b}_{a}" in T), 2),
                                     "mean_unedited": r(st.mean(T[f"{b}_base"] for b in bl[:7]), 2)}
coh["earlier_draft_last6_b10_b15"] = {"mean_edit_runs": r(st.mean(T[f"{b}_{a}"] for b in bl[7:] for a in E if f"{b}_{a}" in T), 2),
                                      "mean_unedited": r(st.mean(T[f"{b}_base"] for b in bl[7:]), 2)}
D["cohorts"] = coh
# never-solved tasks over all complete runs in the file
full = [a for a in R if len(R[a]) == 170]
ever = set(t for a in full for t, v in R[a].items() if v)
D["complete_runs_in_file"] = len(full); D["tasks_ever_solved"] = len(ever)
# rule edits (batches 14, 15) and rank-16 batch (16)
rule = {}
for b in ("b14", "b15"):
    for a in sorted(T):
        if a.startswith(b + "_hp") and "match" not in a:
            rule[a] = {"solved": T[a], "unedited": T[f"{b}_base"], "task": task_level({(t, 0): v for t, v in R[a].items()}, {(t, 0): v for t, v in R[f"{b}_base"].items()})}
D["rule_const_neuron_runs"] = rule
D["rule_const_mean"] = r(st.mean(v["solved"] for v in rule.values()), 2)
D["rule_const_controls"] = {a: T[a] for a in T if a in ("b14_hpematch_t3", "b15_hpxmatch_t3")}
b16 = [a for a in R if a.startswith("b16_")]
common = set.intersection(*[set(R[a]) for a in b16])
D["rank16_batch"] = {"common_tasks": len(common), "finished": {a: len(R[a]) for a in b16}, "solved_on_common": {a: sum(R[a][t] for t in common) for a in b16}}
D["prompt_rule_batch13"] = {a: T[a] for a in T if a.startswith("b13_")}
OUT["explore_more_dev"] = D

# ---------------------------------------------------------------- held-out tests
dg = json.load(open("src/remote_diag.json", encoding="utf-8"))


def heldout(rows_by_arm, x, y):
    tasks = sorted({t for (t, k) in rows_by_arm[x]} & {t for (t, k) in rows_by_arm[y]})
    def tm(a):
        d = collections.defaultdict(list)
        for (t, k), v in rows_by_arm[a].items():
            d[t].append(v)
        return {t: st.mean(v) for t, v in d.items()}
    mx, my = tm(x), tm(y)
    d = [mx[t] - my[t] for t in tasks]
    n = len(d); sd = st.stdev(d); se = sd / math.sqrt(n); t = st.mean(d) / se; c = t_crit(n - 1)
    w = [v for v in d if v > 0]; l = [v for v in d if v < 0]
    mde = (c + t_crit(n - 1, 0.40)) * se           # 80% power: t quantile 0.80 one-sided = two-sided alpha 0.40
    return {"tasks": n, "diff_tasks_eq": r(sum(d), 2), "diff_points": r(100 * st.mean(d), 2), "t": r(t), "p": r(t_p(t, n - 1), 4),
            "ci95_tasks_eq": [r((st.mean(d) - c * se) * n, 2), r((st.mean(d) + c * se) * n, 2)],
            "ci95_points": [r(100 * (st.mean(d) - c * se), 2), r(100 * (st.mean(d) + c * se), 2)],
            "mde80_tasks_eq": r(mde * n, 2), "mde80_points": r(100 * mde, 2),
            "tasks_better": len(w), "tasks_worse": len(l), "sign_p": r(sign_p(len(w), len(l)), 4),
            "sum_gain_better_tasks_eq": r(sum(w), 2), "sum_loss_worse_tasks_eq": r(-sum(l), 2),
            "mean_gain_when_better_rollouts": r(3 * st.mean(w), 2) if w else None, "mean_loss_when_worse_rollouts": r(-3 * st.mean(l), 2) if l else None}


v1 = collections.defaultdict(dict); kept = collections.Counter(); tot = collections.Counter(); ks = collections.Counter(); rs = collections.Counter()
for arm, task, k, res, kp in dg["heldout_v1_rows"]:
    v1[arm][(task, k)] = res; tot[arm] += 1; kept[arm] += kp
    (ks if kp else rs)[arm] += res
H1 = {"arms": {a: {"rollouts": tot[a], "solved": sum(v1[a].values()), "rate": r(sum(v1[a].values()) / tot[a], 4), "kept": kept[a], "rerun": tot[a] - kept[a],
                   "rerun_fraction": r(1 - kept[a] / tot[a], 4), "kept_solved": ks[a], "rerun_solved": rs[a],
                   "rate_wilson95": wilson(sum(v1[a].values()), tot[a])} for a in v1}}
H1["rerun_fraction_all"] = r(1 - sum(kept.values()) / sum(tot.values()), 4)
H1["server_A_rerun"] = [sum(rs[a] for a in ("base", "bnm25_s1", "bneg_t0.25")), sum(tot[a] - kept[a] for a in ("base", "bnm25_s1", "bneg_t0.25"))]
H1["server_B_rerun"] = [sum(rs[a] for a in ("base2", "bneg_t0.125", "bneg_t0.5")), sum(tot[a] - kept[a] for a in ("base2", "bneg_t0.125", "bneg_t0.5"))]
H1["primary_vs_unedited"] = heldout(v1, "bneg_t0.25", "base")
H1["primary_vs_control"] = heldout(v1, "bneg_t0.25", "bnm25_s1")
H1["control_vs_unedited"] = heldout(v1, "bnm25_s1", "base")
H1["serverB_0125_vs_unedited2"] = heldout(v1, "bneg_t0.125", "base2")
H1["serverB_05_vs_unedited2"] = heldout(v1, "bneg_t0.5", "base2")
H1["unedited2_vs_unedited"] = heldout(v1, "base2", "base")
OUT["heldout_test1"] = H1
v2 = collections.defaultdict(dict)
for ln in open("src/heldout_v2_per_rollout.tsv", encoding="utf-8"):
    arm, task, k, res = ln.rstrip("\n").split("\t")
    v2[arm][(task, int(k))] = int(res)
H2 = {"arms": {a: {"rollouts": len(v), "solved": sum(v.values()), "rate": r(sum(v.values()) / len(v), 4), "rate_wilson95": wilson(sum(v.values()), len(v))} for a, v in v2.items()}}
H2["gate_vs_unedited"] = heldout(v2, "ac_xp_pre", "base")
H2["gate_vs_control"] = heldout(v2, "ac_xp_pre", "ac_xp_const_ctrl")
H2["gemini_vs_unedited"] = heldout(v2, "am_damped_dual_explore_concise", "base")
H2["control_vs_unedited"] = heldout(v2, "ac_xp_const_ctrl", "base")
OUT["heldout_test2"] = H2

# ---------------------------------------------------------------- replay (binomial Wilson intervals) and diagnosis
rp = {}
sub = dg["replay_b5_sub/summary.json"]
rp["submit_sooner"] = {"n": sub["n"], "arms": {a: {"rate": v["all"]["submit"], "k": round(v["all"]["submit"] * v["all"]["n"]), "wilson95": wilson(round(v["all"]["submit"] * v["all"]["n"]), v["all"]["n"]),
                                                  "think": v["all"]["think"], "rate_class1": v["1"]["submit"], "rate_class0": v["0"]["submit"], "n_class": v["1"]["n"]}
                                              for a, v in sub.items() if isinstance(v, dict)}}
neg = dg["replay_b6_neg/summary.json"]
rp["explore_more"] = {"n": neg["n"], "arms": {a: {"first_edit_rate": v["all"]["edit"], "k": round(v["all"]["edit"] * v["all"]["n"]), "wilson95": wilson(round(v["all"]["edit"] * v["all"]["n"]), v["all"]["n"]),
                                                 "edit_rate_class1": v["1"]["edit"], "edit_rate_class0": v["0"]["edit"], "think": v["all"]["think"]}
                                             for a, v in neg.items() if isinstance(v, dict)}}
r2 = dg["replay_r2/summary.json"]
rp["think_briefly"] = {"n": r2["n"], "arms": {a: {"think": v["all"]["think"], "edit_rate": v["all"]["edit"]} for a, v in r2.items() if isinstance(v, dict)}}
OUT["replay"] = rp
dz = {}
for nm in ("tracefeat_base", "tracefeat_basefam"):
    t = dg[nm]
    dz[nm] = {"episodes": t["episodes"], "arms": t["arms"], "tasks": t["tasks"], "solved": t["solved"],
              "features": {k: {kk: (r(vv) if isinstance(vv, float) else vv) for kk, vv in t["features"][k].items()}
                           for k in ("any_edit_failed", "read_before_first_edit", "long_completion", "first_edit_failed", "run_after_last_ok_edit", "no_edit_attempt")}}
OUT["diagnosis"] = dz
OUT["probe_counts"] = {"states_suc": dg["states_suc/states_summary.json"], "states_ef": dg["states_ef/states_summary.json"]["counts"]}

json.dump(OUT, open("src/review_stats.json", "w", encoding="utf-8"), indent=1)
print("written src/review_stats.json")
