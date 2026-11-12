"""DR fix M12 / m38 / m39 / m40: round-4 and round-5 recount from raw task_results.jsonl (read-only).

Run:  ssh evalhost python3 - < r4_limit_calib.py      (reads /path/to/gemma-v2/results/steer_eval, writes nothing)

Per batch: tasks finished by every arm ("common"), solved on common tasks per arm, delta vs the batch's unedited arm,
won/lost, and session-limit rows (error contains "session timeout") counted (a) over all rows of the arm and
(b) over the common tasks. Table 15 (tab:round4b) limit-row counts are used to decide which definition the paper used.
Then: calibration of the written round-4 predictions (round4_predictions.md, interval on the mean gain over the two
decision-rule batches) and round-5 predictions (round5_predictions.md, interval for 170 tasks; b59 stopped early).
"""
import glob, json, os
from math import comb

ROOT = "/path/to/gemma-v2/results/steer_eval"


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*", "*", "task_results.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[r["instance_id"]] = (bool(r.get("resolved")), "session timeout" in str(r.get("error") or ""))
    return out


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


res = {}
for b in ["b35", "b36", "b43", "b44", "b45", "b46", "b47", "b48", "b49", "b50", "b59"]:
    arms = {}
    for d in sorted(glob.glob(os.path.join(ROOT, b + "_*"))):
        a = os.path.basename(d)[len(b) + 1:]
        a = a.replace("ac3_", "").replace("ac2_", "").replace("ac5_", "")
        arms[a] = load(d)
    if "base" not in arms:
        print(b, "no base")
        continue
    common = set.intersection(*[set(v) for v in arms.values()])
    base = arms["base"]
    nb = sum(base[t][0] for t in common)
    print(f"\n{b}: common {len(common)}; rows per arm {min(map(len, arms.values()))}-{max(map(len, arms.values()))}; "
          f"base solved {nb}; base limit rows all/common {sum(x[1] for x in base.values())}/{sum(base[t][1] for t in common)}")
    res[b] = {}
    for a, v in arms.items():
        if a == "base":
            continue
        s = sum(v[t][0] for t in common)
        w = sum(v[t][0] and not base[t][0] for t in common)
        l = sum(base[t][0] and not v[t][0] for t in common)
        lim_all, lim_c = sum(x[1] for x in v.values()), sum(v[t][1] for t in common)
        res[b][a] = s - nb
        print(f"  {a:14s} rows {len(v):3d} solved {s:3d} d {s - nb:+3d} won/lost {w}/{l} p {sign_p(w, l):.3f}  "
              f"limit all {lim_all:3d} common {lim_c:3d}")

# ---- round-4 calibration: decision-rule batch pairs (Amendment 3) ----
pairs = {"first": ("b45", "b47"), "second": ("b36", "b48"), "third": ("b43", "b49"), "fourth": ("b44", "b50")}
fam = {"u33": "first", "u36": "first", "u63": "first", "u33_L5": "first", "u33_c6": "first",
       "late30": "second", "late60": "second", "pu": "second", "u33_ctrl": "second", "pre_early": "second",
       "pe_lo": "third", "pe_hi": "third", "pe_L5": "third", "pe_c6": "third", "pe_ctrl": "third",
       "pe_L14": "fourth", "pe_L20": "fourth", "pe_L22": "fourth", "pre15": "fourth", "pre35": "fourth"}
mean = {a: (res[pairs[f][0]][a] + res[pairs[f][1]][a]) / 2 for a, f in fam.items()}
# intervals as written in round4_predictions.md ("Predictions" section); relative ones resolved against the measured
# mean of the 3-layer version ("within 2 of their 3-layer versions")
pred = {"u33": (3, 8), "u36": (3, 8), "u33_c6": (3, 8), "pu": (3, 8), "late30": (0, 4), "late60": (0, 4),
        "u63": (0, 3), "pe_lo": (1, 4), "pe_hi": (-2, 3), "pe_c6": (1, 5), "pre15": (0, 3), "pre35": (0, 4),
        "pe_L14": (0, 3), "pe_L20": (1, 5), "pe_L22": (-1, 3), "pre_early": (1, 5), "u33_ctrl": (-2, 2),
        "pe_ctrl": (-2, 2)}
pred["u33_L5"] = (mean["u33"] - 2, mean["u33"] + 2)
# 3-layer version of pe_L5 = pre_early (build_r3.py: pe_L5 = layers 16-24, [pre(0.25), early(0.3)])
pred["pe_L5"] = (mean["pre_early"] - 2, mean["pre_early"] + 2)
print("\nRound 4: interval vs measured two-batch mean")
hits, widths, misses = 0, [], []
for a in fam:
    lo, hi = pred[a]
    m = mean[a]
    ok = lo <= m <= hi
    hits += ok
    widths.append(hi - lo)
    if not ok:
        misses.append((a, m, lo, hi))
    print(f"  {a:10s} pred [{lo:+.1f}, {hi:+.1f}] measured {m:+.1f} ({res[pairs[fam[a]][0]][a]:+d}, "
          f"{res[pairs[fam[a]][1]][a]:+d}) {'hit' if ok else 'MISS'}")
print(f"  hits {hits}/{len(fam)}; mean width {sum(widths) / len(widths):.2f}; "
      f"misses above {sum(m > hi for _, m, lo, hi in misses)}, below {sum(m < lo for _, m, lo, hi in misses)}")
# also against each single batch (40 edit-batches)
sh = sum(pred[a][0] <= res[b][a] <= pred[a][1] for a in fam for b in pairs[fam[a]])
print(f"  single-batch hits {sh}/40")

# ---- round 5 (b59, stopped early; intervals were written for 170 tasks) ----
p5 = {"late60": (0, 5), "late90": (0, 5), "late60_c6": (0, 5), "late100_c6": (-2, 6), "late60_x3": (-3, 5),
      "late60_x5": (-8, 4), "late60_ctrl": (-2, 2)}
print("\nRound 5 (b59, common tasks only; intervals were for 170 tasks)")
h5 = 0
for a, (lo, hi) in p5.items():
    m = res["b59"][a]
    h5 += lo <= m <= hi
    print(f"  {a:12s} pred [{lo:+d}, {hi:+d}] measured {m:+d} {'hit' if lo <= m <= hi else 'MISS'}")
print(f"  hits {h5}/{len(p5)}")
