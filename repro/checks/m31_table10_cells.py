"""m31/M6: values for the blank cells of Table 10 (tab:stats) and screening-only Holm, read from src/review_stats.json
(produced by scripts/review_stats.py from src/arena_per_task_b21_b28.tsv). Run from repro/."""
import json
A = json.load(open("src/review_stats.json"))["arena"]
def row(name, v):
    tl = v["task_level"]
    print(f"{name}: n={v['n']} mean={v['mean']} se={v.get('se')} t_p={v.get('p')} sign_p={v['sign_p']} | task_t_p={tl['task_t_p']} "
          f"mcnemar={tl['mcnemar_b']}/{tl['mcnemar_c']} p={tl['mcnemar_p']} boot95={tl['boot95']} h={tl['cohen_h']} tasks better/worse={tl['tasks_better']}/{tl['tasks_worse']}")
row("gate replication vs unedited", A["screening_vs_replication"]["ac_xp_pre"]["replication_vs_unedited"])
for k, v in A["control_vs_unedited"].items(): row("control-unedited " + k, v)
for k, v in A["edit_vs_control"].items(): row("edit-control " + k, v)
h = A["holm"]["mcnemar_screening_batch"]
print("screening-batch McNemar raw:", sorted(h["raw"].items(), key=lambda x: x[1]))
print("screening-batch McNemar Holm:", h["holm"])
