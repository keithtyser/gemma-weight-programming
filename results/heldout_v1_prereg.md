# Held-out confirmation of the explore-more edit: plan written before the run

Written 11 November 2026, before any agent run on these tasks with any edit.

## Why

The strength of the explore-more edit was chosen on the 170 evaluation tasks that the paper reports. This run fixes
everything in advance and measures the edit on tasks that played no part in building or tuning it.

## Tasks

130 SWE-rebench-format tasks from 78 repositories (`results/heldout_v1_ids.txt`), selected by rule from a 300-task set
prepared for an unrelated training run:

- not one of the 170 evaluation tasks, and not from any repository in the evaluation set;
- never present in the logged rollouts that the directions were mined from (1,626 task ids excluded).

No task was dropped for any other reason and none will be dropped after the run except for infrastructure aborts,
which are reported.

## Arms (all on the unedited INT4 base model, same harness as the paper, greedy)

| Arm | Edit |
|---|---|
| `base` | none |
| `bneg_t0.25` | explore-more, strength 0.25 (**primary**) |
| `bnm25_s1` | norm-matched random control for strength 0.25 |
| `bneg_t0.125`, `bneg_t0.5` | explore-more at the two other strengths in the paper (secondary) |

Three runs per arm and task (greedy runs of this stack are not repeatable, so runs are replicates).
The adapters are the ones already built for the paper; nothing is rebuilt or retuned.

## Primary outcome and test

Per task, the mean solve rate over the three runs. Primary comparison: `bneg_t0.25` minus `base`, and `bneg_t0.25`
minus `bnm25_s1`; paired over tasks; two-sided sign test over tasks whose means differ and the paired t statistic.
Secondary: the same for strengths 0.125 and 0.5, and control minus base.

## Prediction

On the evaluation set the edit solved about 3 more of 170 tasks than the unedited model and about 3.6 more than its
controls (roughly +12% relative). The prediction is the same direction and relative size here: the edit's solve rate
exceeds both the unedited model's and the control's, by 1 to 4 percentage points; control minus base within ±2 points.
The run counts as a confirmation only if the primary edit beats both comparators; any other outcome is reported as a
failure to confirm.
