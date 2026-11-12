# Round 6 (dev replicate of one edit) — hypothesis and test, written before launch

Written 12 Nov 2026, 3:34 PM ET, before any episode of the batches named here.

## Why this edit

Selected after looking at round-4 data. The interim pooled round-4 analysis at 3:33 PM (all round-4 batches, partial)
had `ac3_pe_lo` ahead of same-batch unedited by 19 net tasks over 508 paired episodes (30 won, 11 lost, sign test
p = .004, Holm over 20 edits .082), with its matched random-direction control `ac3_pe_ctrl` at -2. No edit met the
round-4 decision rule and nothing passed the Holm correction. Round 6 tests this one edit on new episodes only.

## Design

- Arms: `base` (all-zero adapter), `ac3_pe_lo`, `ac3_pe_ctrl`. Adapters are the round-4 files, unchanged.
- Batches b60, b61 (evalhost GPU 0) and b62, b63 (evalhost GPU 1): 170 dev tasks, 1 rollout each, concurrency 7 per arm,
  sandboxes on evalhost. Each batch has its own `base`; comparisons are within batch.
- The second session may add batches b64-b71 with the same three arms on the rented CUDA box. Any of b60-b71 that
  have run by scoring time are included; a batch incomplete at scoring time is scored on the tasks completed by all
  three of its arms and labelled incomplete.
- Round 5 (b59) is stopped at 3:34 PM to free the GPU and sandboxes; it is reported as incomplete with its
  scores at the stop (wsp_arena/b59_at_stop.txt).

## Test (fixed now)

- Primary: task-level paired comparison of `ac3_pe_lo` with same-batch `base`, pooled over the round-6 batches only
  (no round-4 episodes), exact two-sided sign test on discordant pairs, alpha .05. One hypothesis, no correction.
- Secondary: the same test for `ac3_pe_ctrl` against `base`, and for `ac3_pe_lo` against `ac3_pe_ctrl`.
- Session-limit rows are counted per arm.

## Predictions

- `ac3_pe_lo`: +2 to +7 tasks per complete batch over same-batch unedited.
- `ac3_pe_ctrl`: -3 to +2 per batch.

## What can be claimed

- If the primary test rejects with `ac3_pe_lo` ahead: the edit's dev gain replicated in independent episodes. These
  are the same 170 dev tasks the edit was selected on, so this is not evidence of generalisation to new tasks; no
  held-out run fits before the deadline.
- If it does not reject: reported as a failed replication, with the round-4 pooled number described as exploratory.

## Amendment 1 (12 Nov 2026, 3:55 PM ET; written before any episode of b72)

- One further batch, b72, runs on the second rented CUDA box (GPU 1, usual serving stack and flags), sandboxes on
  evalhost, 170 dev tasks, 1 rollout, concurrency 6 per arm: `base`, `ac3_pe_L20`, `ac3_pe_L5` (round-4 adapters
  unchanged, checksums verified after copy).
- Reason: in the round-4 pooled table the four explore-first variants were all positive (pe_lo +19, pe_L20 +17,
  pe_L5 +12, pe_c6 +9) while their control was -2. b72 gives the next two an independent batch.
- b72 does not enter the round-6 primary test. Secondary, exploratory: each of the two edits against b72's own
  unedited arm, task-level exact sign test, Holm over 2. Prediction: each +1 to +6; a gain of 10 or more in b72
  would be outside the prediction and would still need a replicate before any claim.
- Same scoring cutoff (5:40 PM ET); scored on common tasks and labelled incomplete if unfinished.

## Amendment 2 (4:10 PM ET, 12 Nov 2026; written before any episode of b66-b71 or b73-b80)

- More batches join the primary test, all with the same three arms (`base`, `ac3_pe_lo`, `ac3_pe_ctrl`), 170 dev
  tasks, 1 rollout, own unedited arm each, sandboxes on the AMD node:
  - b66-b71 on the CUDA box as planned, but at concurrency 12 per arm (not 4), so they can finish before the cutoff;
  - b73-b76 served by AMD-node GPUs 4-7 (vLLM ROCm build, the stack of b47-b50), concurrency 5 per arm;
  - b77-b80 served by AMD-node GPUs 0-3, same settings, if started.
  (b72 is the exploratory batch of Amendment 1 and stays outside the primary test.)
- Reason: power. The decision to add them was taken on capacity becoming free, with round 6 at this interim state
  (ac3_pe_lo minus unedited on common tasks): b60 and b61 about 60 tasks each, b62-b65 under 40 each; no batch
  differs from unedited by more than 2 tasks in either direction. The exact interim scores are in
  wsp_arena/r6_interim_amend2.txt.
- The primary test is unchanged: pooled task-level paired comparison over every round-6 batch (b60-b71, b73-b80)
  that has episodes at the 5:40 PM cutoff, exact two-sided sign test, alpha .05. Serving stack (CUDA, ROCm) and
  concurrency differ between batches but are shared by the arms of each batch; a per-stack breakdown is reported.
- The AMD node can be reclaimed at any time; batches cut off are scored on common tasks and labelled incomplete.

## Amendment 3 (12 Nov 2026, 4:42 PM ET)

- b66-b71 (CUDA, 5090 box, concurrency 12) were stopped by the second session at 4:40 PM ET after 14-19 minutes of
  running, on Keith's instruction to use those six servers for a broader exploratory screen (round 7,
  wsp_arena/round7_predictions.md). They were stopped without their rows being read by either session.
- Under the primary-test definition above (every round-6 batch with episodes at the cutoff), their partial rows stay
  in the pooled test, scored on the tasks completed by every arm of the batch and labelled incomplete. The pooled
  result is also reported with b66-b71 excluded, as a sensitivity check.
- Still running for round 6: b60-b63 (evalhost), b64, b65 (CUDA), b73-b80 (ROCm). Cutoff, tests and alpha unchanged.
- Deviation to record for b72 (Amendment 1): it was launched at concurrency 5 per arm, not 6 as written there.
