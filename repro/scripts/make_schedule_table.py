"""Write the batch-by-arm schedule tables (appendix) from the saved per-task results.

Usage: python make_schedule_table.py ../src/dev_runs_b3_b16.tsv ../src/arena_per_task_b21_b28.tsv ../app_schedule.tex
Every cell is the number of solved tasks of 170 for that arm in that batch.
"""
import collections, json, sys

dev, arena, out = sys.argv[1:4]
runs = collections.defaultdict(dict)
for line in open(dev, encoding="utf-8"):
    arm, j = line.rstrip("\n").split("\t", 1)
    d = json.loads(j)
    runs[arm][d["instance_id"]] = bool(d["resolved"])


def tex(s):
    return s.replace("_", r"\_")


L = [r"\begin{table}[h]",
     r"\caption{Schedule of development batches 3--15: every arm of every batch with its solved tasks of 170 (arm identifiers as in the released results). \texttt{base}: unedited model; \texttt{bneg\_t$s$}: always-on explore-more at strength $s$; \texttt{bnegmatch\_t0.5} and \texttt{bnm25\_s0}--\texttt{s2}: its matched random controls; \texttt{w5}: wave-5 RL adapter; \texttt{sub}, \texttt{bth}, \texttt{bsuc}, \texttt{vfyD}/\texttt{bvfyD}, \texttt{bedit}: submit-sooner, think-briefly, success, verify and edit-sooner edits; \texttt{p*}: prompt rules; \texttt{hp*}: constant-neuron rule edits; a name containing \texttt{match} is a control. \texttt{bnegW} writes explore-more at five layers (10--26); \texttt{becond}, \texttt{econd}, \texttt{bse}, \texttt{bns}, \texttt{bn50\_th50} and \texttt{bcombo} are exploratory conditional or combined variants that the text does not analyse.}",
     r"\label{tab:schedule}", r"\centering", r"\scriptsize",
     r"\begin{tabular}{rp{0.92\linewidth}}", r"\toprule", r"Batch & Arm: solved \\", r"\midrule"]
for b in range(3, 16):
    arms = sorted(a for a in runs if a.startswith(f"b{b}_"))
    cells = []
    for a in arms:
        assert len(runs[a]) == 170, (a, len(runs[a]))
        cells.append(r"\texttt{%s}: %d" % (tex(a.split("_", 1)[1]), sum(runs[a].values())))
    L.append(f"{b} & " + "; ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

ar = collections.defaultdict(lambda: collections.defaultdict(int))
n = collections.defaultdict(lambda: collections.defaultdict(int))
for line in open(arena, encoding="utf-8"):
    b, arm, _, r = line.rstrip("\n").split("\t")
    ar[arm][b] += int(r)
    n[arm][b] += 1
names = [("base", "Unedited model"), ("ac_xp_const", "Claude: always-on, 0.125"), ("ac_xp_pre", "Claude: before-first-edit gate"),
         ("ac_xp_think", "Claude: generated-positions gate"), ("ac_xp_early", "Claude: early-calls gate"),
         ("ac_xp_const_ctrl", "Claude: random control"), ("ag_think_start_explore", "GPT: decision-token gate"),
         ("ag_explore_then_rotate", "GPT: decision-token gate + rotation"), ("ag_error_recheck", "GPT: re-check after error"),
         ("ag_decision_rotation", "GPT: rank-two rotation")]
known = {k for k, _ in names}
rest = sorted(a for a in ar if a not in known)
bs = [f"b{i}" for i in range(21, 29)]
L += [r"\begin{table}[h]",
      r"\caption{Schedule of the three-author benchmark: solved tasks of 170 per arm and batch; an empty cell means the arm was not in that batch. Batches 21--23 are the screening batches, 24--28 the replication batches. Rows below the first ten use the arm identifiers of the released results (\texttt{ag\_*}: GPT; \texttt{am\_*}: Gemini; \texttt{ctrl}: random control).}",
      r"\label{tab:arenaschedule}", r"\centering", r"\footnotesize",
      r"\begin{tabular}{l" + "r" * 8 + "}", r"\toprule",
      "Arm & " + " & ".join(str(i) for i in range(21, 29)) + r" \\", r"\midrule"]
for k, lab in names + [(a, r"\texttt{%s}" % tex(a)) for a in rest]:
    if k not in ar:
        continue
    for b in ar[k]:
        assert n[k][b] == 170, (k, b, n[k][b])
    L.append(lab + " & " + " & ".join(str(ar[k][b]) if b in ar[k] else "" for b in bs) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(out, "w", encoding="utf-8").write("\n".join(L))
print(sorted(ar))
