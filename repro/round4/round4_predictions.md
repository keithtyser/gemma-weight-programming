# Round 4 (wide screen) — predictions and decision rule, written before launch

Written 12 Nov 2026, about 12:05 PM ET, before any round-4 episode ran. Round 3 (b33, b34) was still running and
no stack was ahead of unedited at that point.

## Setup

- Eight servers on a rented 8-GPU AMD node (vLLM ROCm build, a different serving stack from rounds 1-3), one per GPU.
  Sandboxes run on evalhost as before. Each server carries its own `base` arm; an edit is only ever compared with the
  `base` served by the same server in the same batch.
- 20 edits, each run in two batches on two different servers (b35-b38, replicated as b39-b42), 170 dev tasks,
  1 rollout, concurrency 4 per arm.

| batches | arms (plus `base`) |
|---|---|
| b35, b39 | ac3_u33, ac3_u36, ac3_u63, ac3_u33_L5, ac3_u33_c6 |
| b36, b40 | ac3_late30, ac3_late60, ac3_pu, ac3_u33_ctrl, ac2_pre_early |
| b37, b41 | ac3_pe_lo, ac3_pe_hi, ac3_pe_L5, ac3_pe_c6, ac3_pe_ctrl |
| b38, b42 | ac3_pe_L14, ac3_pe_L20, ac3_pe_L22, ac3_pre15, ac3_pre35 |

## Diagnosis behind the new edits

In 13 unedited runs (2,210 episodes), solving against the number of calls before the first file edit is an inverted
U: 0-2 calls odds ratio 0.76, 3-4 calls 1.42, 5-7 calls 1.72, 8+ calls 0.12. Earlier edits only pushed "explore
more before editing". The `late` neuron pushes towards the first edit once 8+ calls have passed without one; `u*`
edits combine the early explore push with the late commit push.

## Predictions (tasks solved minus same-server unedited, mean of two batches)

- `u33`, `u36`, `u33_c6`, `pu`: +3 to +8. These are the edits I expect to be best; `u36` and `pu` first.
- `late30`, `late60` alone: 0 to +4 (fixes only the 8+ call tail, about 15% of episodes).
- `u63`: 0 to +3 (0.6 early dose was harmful in round 2).
- `u33_L5`, `pe_L5`: within 2 of their 3-layer versions.
- `pe_lo`: +1 to +4; `pe_hi`: -2 to +3; `pe_c6`: +1 to +5; `pre15`: 0 to +3; `pre35`: 0 to +4.
- Depth variants: `pe_L14` 0 to +3, `pe_L20` +1 to +5, `pe_L22` -1 to +3.
- `ac2_pre_early` (round-2 best, +5 in one batch): +1 to +5.
- Controls `u33_ctrl`, `pe_ctrl`: -2 to +2.
- Behaviour: every `u*` edit lowers the share of episodes with 8+ calls before the first edit relative to
  same-server base; every `pe*` / `pre*` edit raises median calls before the first edit; controls change neither.

## Decision rule (fixed now)

An edit goes to a held-out confirmation only if all three hold:

1. it beats same-server unedited by at least 8 tasks in the mean of its two batches;
2. it beats same-server unedited by at least 5 tasks in each batch;
3. its mean gain exceeds the mean gain of its family's control (`u33_ctrl` for `u*`, `late*`, `pu`; `pe_ctrl` for
   the rest) by at least 5 tasks.

If several pass, the one with the largest mean gain goes; ties go to the lower dose. If none pass, the screen is
reported as a negative result with all 20 rows. Twenty edits are screened, so any dev-set p-value is reported with
a Holm correction over 20 and the held-out run is the only confirmatory test.

## Amendment (12 Nov 2026, 1:05 PM ET; written with every round-4 batch under 15% complete)

The plan above was not executed as written. Deviations, all forced by sandbox capacity on evalhost and none chosen
after looking at an edit's score:

- b35-b42 were launched together at 12:17-12:20 PM. With round 3 still running this put about 240 sandboxes on
  evalhost, whose disk could not keep up (load 190, 77% IO wait, episode times 3x normal). b37-b42 were stopped at
  about 12:38 PM; their partial directories are ignored.
- b35 (first arm set) and b36 (second) continued at concurrency 4. The third and fourth arm sets were relaunched as
  b43 and b44 at concurrency 6. There are no replicate batches: each edit has one batch.
- Round 3 (b33, b34) was stopped at 1:03 PM at 52% / 42% complete to free sandbox capacity. Its stacks were not
  ahead of unedited or of the control on the tasks done by all arms (b33: 9 unedited, 10 control, 8-9 stacks of 53;
  b34: 7, 9, 6-8 of 45). It is reported as stopped for capacity, not as a completed test.
- Episodes run during the overload took longer in wall-clock time. This affects all arms of a batch alike, since
  arms of a batch run interleaved, but differs between batches.

Decision rule with one batch per edit: rules 1 and 2 collapse to "beats same-server unedited by at least 8 tasks in
its batch"; rule 3 is unchanged, with the family control taken from its own batch's gain over its own unedited arm.
An edit that passes is first replicated in one more batch (same server arm set, on evalhost or the rented node) and
must gain at least 5 there before any held-out run. If a batch is incomplete when the rented node's lease ends,
it is scored on the tasks completed by every arm of that batch and labelled incomplete.

## Amendment 2 (12 Nov 2026, 1:51 PM ET; written before any episode of b47-b50)

- Replicate batches b47-b50 run entirely on the AMD node: sandboxes start in a second docker daemon on the node
  (image store on its NVMe volume), and the harness runs there from a copy of tii-steer, eval_data/golden_v2 and
  eval-venv (comp image id checked against the golden_v2 manifest). evalhost runs none of their containers.
- Servers and arms are the replicate sets of the table above: b47 = GPU 4 (the b35/b39 arms), b48 = GPU 5 (the
  b36/b40 arms), b49 = GPU 6 (the b37/b41 arms), b50 = GPU 7 (the b38/b42 arms). Each has its own `base`.
- Concurrency is 8 per arm (not 4), so the batches can finish inside the lease. An edit is still compared only with
  the `base` of its own batch, which runs at the same concurrency on the same server and host.
- Predictions and the decision rule are unchanged. For the rule, the two batches of an edit are its first batch
  (b35, b36, b43 or b44) and its node batch (b47-b50); b45 and b46, if they finish, are reported as third batches.
- These batches differ from the first ones in sandbox host and concurrency. That is a batch-level difference shared
  by every arm of a batch; it is noted wherever the two batches are averaged.

## Amendment 3 (12 Nov 2026, 2:12 PM ET; written with b47-b50 under 10% complete and unscored)

Two sessions edited this file. An amendment written at 1:40 PM was overwritten by the 1:51 PM one above; its
content is restored here, and the two versions of the decision rule are reconciled.

- b45 (first arm set) and b46 (second arm set) were started at 1:20 PM on evalhost's own two GPUs (sandboxes on evalhost,
  concurrency 4), each with its own unedited arm.
- b35 (first arm set, node-served, sandboxes on evalhost) was stopped at 1:38 PM at 27% complete for sandbox capacity:
  about 11% of rows in the node-served batches with evalhost sandboxes were ending at the 7.5-minute session limit
  against about 4% in earlier batches, and b35 could not finish before the node lease ends. At the stop its arms
  were within +2/-1 of unedited on 41 common tasks. b35 is reported as incomplete.
- Session-limit rows are counted per arm in every table.
- Decision rule, final form. Since every edit now has two batches again, the original three-part rule applies
  (mean gain at least 8, at least 5 in each batch, mean gain at least 5 above the family control's mean gain). The
  two batches of an edit are: first arm set b45 and b47 (b35 being incomplete); second arm set b36 and b48, with
  b46 reported as a third batch; third arm set b43 and b49; fourth arm set b44 and b50. This replaces the
  one-batch rule of the first amendment. A batch that is incomplete at scoring time is scored on the tasks
  completed by every arm of that batch and labelled incomplete.

## Amendment 4 (12 Nov 2026, 2:25 PM ET; written before any episode of b51-b58 was scored)

- Eight further batches b51-b58 are being started by the second session on a rented 8-GPU CUDA box (the usual
  serving stack and flags), with sandboxes and harness on the AMD node: b51/b55 first arm set, b52/b56 second,
  b53/b57 third, b54/b58 fourth; each with its own unedited arm; 1 rollout, concurrency 8 per arm.
- The decision rule and its batch pairs (Amendment 3) are unchanged. b51-b58 are extra replicate batches. They
  enter only a secondary pooled analysis: for each edit, the paired difference from the same-batch unedited arm
  summed over every batch of that edit (complete or labelled incomplete), with a task-level sign test and a Holm
  correction over the 20 edits. The pooled analysis cannot by itself send an edit to held-out; it can be reported.
- If the AMD node lease ends while these batches run, unfinished ones are scored on common tasks and labelled
  incomplete, or restarted under a new batch name with sandboxes on evalhost; a restarted batch is a new batch.

## Amendment 5 (12 Nov 2026, 2:32 PM ET)

- b46 (second arm set, third batch, not part of the decision rule) is stopped at about 2:33 PM at roughly 35%
  complete to free one evalhost GPU and 24 sandboxes for round 5 (wsp_arena/round5_predictions.md). At 2:27 PM, on the
  57 tasks done by all its arms: unedited 10, ac2_pre_early 10, ac3_late30 7, ac3_late60 10, ac3_pu 7,
  ac3_u33_ctrl 12. It is reported as incomplete with those numbers (rescored at the stop).
- The decision rule and its batch pairs are unchanged.

## Amendment 6 (12 Nov 2026, 4:07 PM ET)

- b36 (second arm set, node-served, sandboxes on evalhost) was stopped at 4:06 PM at 156 of 170 tasks complete in every
  arm, to keep evalhost's sandbox load down for round 6 (201 containers, 34% IO wait at 4:05 PM). Scored at the stop
  (wsp_arena/b36_at_stop.txt): unedited 22, ac2_pre_early 25, ac3_late30 24, ac3_late60 28, ac3_pu 25,
  ac3_u33_ctrl 29. It is reported as incomplete with those numbers.
- b43, b44 and b45 finished all 170 tasks. The decision rule and its batch pairs are unchanged.
