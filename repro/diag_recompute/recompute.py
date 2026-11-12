"""Independent recompute of the paper's diagnosis tallies from saved base-arm episode traces.

Input : raw/b{3..15}_base/{comp_s0/shard_0,rebench/shard0_gpu0}/{task_results.jsonl,traces/trace_*.json}
        (read-only copy of /path/to/gemma-v2/results/steer_eval on evalhost)
Output: result.json (and printed summary). Run: python -I recompute.py <diag_recompute_dir>

Definitions (chosen to match diag.py / diag2.py, deviations flagged as sensitivity):
  * episode = one (batch, task) trace; solved = task_results 'resolved'.
  * tool call = each entry of a step's tool_calls list, in step order.
  * edit_file failure (primary) = the tool observation's content JSON has status == "error".
    Sensitivity 'strict': also counts edit_file calls whose observation has no status but an
    'error' key (harness rejected the call, e.g. missing old_string) or no observation at all.
  * first file edit = first edit_file or write_file call (regardless of success);
    n_pre = number of tool calls before it; if the episode never edits, n_pre = all its calls.
  * MH odds ratio stratified by task (instance_id), bin vs all other episodes of that task;
    95% CI from the Robins-Breslow-Greenland variance of log OR_MH.
"""
import collections, glob, json, math, os, random, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
RAW = os.path.join(BASE, "raw")
BATCHES = range(3, 16)
SUBDIRS = ("comp_s0/shard_0", "rebench/shard0_gpu0")


def edit_status(step, n_calls):
    """Return (primary_failed, strict_failed) for an edit_file call in this step."""
    obs = step.get("observation")
    if not isinstance(obs, dict) or obs.get("content") is None:
        return False, True  # no observation at all
    try:
        c = json.loads(obs["content"])
    except Exception:
        return False, True
    if not isinstance(c, dict):
        return False, True
    if "status" in c:
        st = c["status"] == "error"
        if n_calls > 1:  # shared observation: attribute to the edit only if it is an edit error
            st = st and c.get("error_type") == "FileEditError"
        return st, st
    return False, "error" in c


def load():
    eps = []
    for b in BATCHES:
        for sd in SUBDIRS:
            d = os.path.join(RAW, f"b{b}_base", sd)
            res = {}
            for line in open(os.path.join(d, "task_results.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                res[r["instance_id"]] = bool(r.get("resolved"))
            seen = set()
            for f in glob.glob(os.path.join(d, "traces", "trace_*.json")):
                iid = os.path.basename(f)[len("trace_"):-len(".json")]
                assert iid in res, (b, sd, iid)
                seen.add(iid)
                calls = []  # (name, primary_fail, strict_fail)
                for s in json.load(open(f, encoding="utf-8"))["steps"]:
                    tcs = s.get("tool_calls") or []
                    for tc in tcs:
                        name = tc.get("function_name")
                        if name == "edit_file":
                            p, st = edit_status(s, len(tcs))
                        else:
                            p = st = False
                        calls.append((name, p, st))
                eps.append({"batch": b, "task": iid.lower(), "solved": res[iid], "calls": calls})
            assert seen == set(res), (b, sd, "trace/result mismatch")
    return eps


def mh(strata):
    """strata: list of (a,b,c,d) with a=exposed&solved, b=exposed&unsolved, c=unexposed&solved, d=unexposed&unsolved.
    Returns OR_MH, RBG 95% CI, CMH chi2 (no continuity correction), R, S."""
    R = S = PR = PSQR = QS = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        r, s = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R += r; S += s; PR += p * r; PSQR += p * s + q * r; QS += q * s
        m1, n1 = a + c, a + b
        sa += a; ea += n1 * m1 / n
        va += n1 * (n - n1) * m1 * (n - m1) / (n * n * (n - 1))
    if R == 0 or S == 0:
        return {"or": None, "lo": None, "hi": None, "R": R, "S": S}
    orr = R / S
    var = PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    chi2 = (sa - ea) ** 2 / va if va > 0 else None
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 is not None else None
    return {"or": orr, "lo": math.exp(math.log(orr) - 1.96 * se), "hi": math.exp(math.log(orr) + 1.96 * se),
            "se_log": se, "R": R, "S": S, "cmh_chi2": chi2, "cmh_p": p}


def strata_for(eps, expo):
    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    for e in eps:
        x = expo(e)
        k = (0 if e["solved"] else 1) if x else (2 if e["solved"] else 3)
        by[e["task"]][k] += 1
    return list(by.values())


def wilson(k, n):
    if n == 0:
        return None, None
    z = 1.96; p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def task_boot(eps, stat, B=2000, seed=0):
    """Percentile CI of stat(eps) resampling tasks (clusters) with replacement."""
    by = collections.defaultdict(list)
    for e in eps:
        by[e["task"]].append(e)
    keys = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = [e for k in (rng.choice(keys) for _ in keys) for e in by[k]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    eps = load()
    out = {"n_episodes": len(eps), "n_solved": sum(e["solved"] for e in eps),
           "n_tasks": len({e["task"] for e in eps}), "n_batches": len({e["batch"] for e in eps})}
    tools = collections.Counter(c[0] for e in eps for c in e["calls"])
    out["tool_counts"] = dict(tools.most_common())

    # ---- 1. edit_file failures
    for mode, idx in (("primary", 1), ("strict", 2)):
        tot = sum(1 for e in eps for c in e["calls"] if c[0] == "edit_file")
        fail = sum(1 for e in eps for c in e["calls"] if c[0] == "edit_file" and c[idx])

        def share(sample, idx=idx):
            t = sum(1 for e in sample for c in e["calls"] if c[0] == "edit_file")
            return sum(1 for e in sample for c in e["calls"] if c[0] == "edit_file" and c[idx]) / t if t else None

        anyf = lambda e, idx=idx: any(c[0] == "edit_file" and c[idx] for c in e["calls"])
        w = [e for e in eps if anyf(e)]
        wo = [e for e in eps if not anyf(e)]
        kw, kwo = sum(e["solved"] for e in w), sum(e["solved"] for e in wo)
        out[f"edit_{mode}"] = {
            "edit_calls": tot, "failed": fail, "share_failed": fail / tot,
            "share_failed_ci_taskboot": task_boot(eps, share),
            "eps_with_failed_edit": len(w), "solved_with": kw, "rate_with": kw / len(w), "rate_with_wilson": wilson(kw, len(w)),
            "eps_without": len(wo), "solved_without": kwo, "rate_without": kwo / len(wo), "rate_without_wilson": wilson(kwo, len(wo)),
            "mh_any_failed_edit": mh(strata_for(eps, anyf)),
        }

    # ---- 2. calls before first file edit
    def npre(e):
        names = [c[0] for c in e["calls"]]
        return next((i for i, n in enumerate(names) if n in ("edit_file", "write_file")), len(names))

    out["mean_calls_before_first_edit"] = sum(npre(e) for e in eps) / len(eps)
    out["never_edited"] = sum(1 for e in eps if not any(c[0] in ("edit_file", "write_file") for c in e["calls"]))
    bins = {}
    for lo, hi, lab in ((0, 2, "0-2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 10 ** 9, "8+")):
        inb = lambda e, lo=lo, hi=hi: lo <= npre(e) <= hi
        w = [e for e in eps if inb(e)]
        wo = [e for e in eps if not inb(e)]
        kw, kwo = sum(e["solved"] for e in w), sum(e["solved"] for e in wo)
        st = strata_for(eps, inb)
        bins[lab] = {"n": len(w), "share": len(w) / len(eps), "solved_in": kw, "rate_in": kw / len(w),
                     "rate_in_wilson": wilson(kw, len(w)), "solved_out": kwo, "rate_out": kwo / len(wo),
                     "rate_out_wilson": wilson(kwo, len(wo)),
                     "informative_tasks": sum(1 for a, b, c, d in st if (a + b) and (c + d) and (a + c) and (b + d)),
                     "mh": mh(st)}
    out["n_pre_bins"] = bins

    # ---- 3. per-task solve counts
    per = collections.defaultdict(int)
    runs = collections.defaultdict(int)
    for e in eps:
        per[e["task"]] += e["solved"]
        runs[e["task"]] += 1
    assert set(runs.values()) == {13}
    out["tasks_solved_at_least_once"] = sum(1 for v in per.values() if v > 0)
    out["tasks_solved_every_run"] = sum(1 for v in per.values() if v == 13)
    out["solve_count_histogram"] = dict(sorted(collections.Counter(per.values()).items()))
    json.dump(out, open(os.path.join(BASE, "result.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
