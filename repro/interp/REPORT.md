# Behavioural control scorecard (sub-agent analysis, 12 Nov 2026, about 4:40 PM ET)

Status: numbers come from one sub-agent pass over the episode files; spot-check in progress by the main session.
Scripts and raw outputs: `repro/interp/` (behav.py -> behav_r45.jsonl, behav_r1.jsonl; prelog.py -> prelog.jsonl;
corr.py -> corr_out.txt; r1_out.txt; l14.py -> l14_out.txt; score_snapshot.txt), copies in
`evalhost:/path/to/gemma-v2/wsp_arena/interp_scratch/`.

Metric: "calls before first edit" = tool calls before the first edit_file/write_file (all calls if the episode never
edits), on tasks finished by every arm of a batch, against the same batch's unedited arm. "8+ share" = share of
episodes with 8 or more such calls. Hit rule: pooled shift has the predicted sign and no batch reverses it; ties are
misses.

## 1. What the author LLM sees

It does not read weight values. Inputs: (a) behaviour statistics over 2,210 unedited episodes; (b) activations stored
at labelled decision points ("the next call makes the first edit" vs not). It forms a class-mean-difference
direction, fits a ridge-regression gate, and its code writes both into one nearly silent MLP neuron per layer, with
the write size solved so the recomputed shift on stored activations equals a target number of class gaps. Raw
weights enter only as inputs to that code (dequantised MLP matrices and post-MLP norm scale, to pick a quiet neuron
and undo the norm; `claude/code/build.py` lines 101-118). No author used a logit-lens / unembedding readout.

## 2. Scorecard

### Arena round 1 (b21-b28), mean calls before first edit, pooled base -> edited

| edit (author) | predicted | measured | verdict |
|---|---|---|---|
| xp_const (Claude) | 4.8 -> 5.2 (5.0-5.6) | 4.59 -> 5.18 | hit |
| xp_pre (Claude) | -> 6.0 (5.5-6.8) | 4.75 -> 6.71, up in 6/6 batches | hit (8+ share 16.0% -> 31.4% vs 23% predicted: overshoot) |
| xp_think (Claude) | -> 5.6 (5.1-6.5) | 4.59 -> 5.15 | hit |
| xp_early (Claude) | 8+ share stays below 22% | 16.0% -> 29.8% | miss |
| xp_const_ctrl (Claude) | within 0.4 of base | 4.72 -> 5.16 | control not flat |
| think_start_explore (GPT) | 4.80 -> 5.3 | 4.95 -> 4.91 | miss |
| explore_then_rotate (GPT) | -> 5.5 | 4.81 -> 4.92 (3 up / 3 down) | miss |
| decision_rotation (GPT) | -> 5.0 | 4.95 -> 4.81 | miss |
| think_start_explore_ctrl (GPT) | flat | 4.81 -> 4.73 | flat |
| error_recheck (GPT) | error->submit share 10.9% -> 8.5% | not measured | not checkable |
| damped_dual_explore_concise (Gemini) | 4.7 -> 5.7 | 4.76 -> 5.09, 3/3 up | right direction, a third of predicted size |
| oproj_onesided_commit_damper (Gemini) | -> 5.8 | 4.61 -> 4.59 | miss |
| whitened_cond_rushed_read (Gemini) | -> 5.5 | 4.61 -> 4.93 | right direction, 40% of size |
| hybrid_explore_verify_guard (Gemini) | -> 5.3 | 4.61 -> 5.24 | hit |
| damped..._ctrl (Gemini) | flat | 4.76 -> 4.62 | flat |

### Round 4 (Claude), pooled over each edit's 4-5 batches

| edit | predicted | 8+ share base -> edited | mean calls base -> edited | verdict |
|---|---|---|---|---|
| u33 | 8+ share down | .151 -> .252 | 4.57 -> 5.97 | miss (up in 5/5 batches) |
| u36 | 8+ share down | .151 -> .255 | -> 6.00 | miss (5/5) |
| u63 | 8+ share down | .151 -> .473 | -> 7.94 | miss (5/5) |
| u33_L5 | 8+ share down | .151 -> .258 | -> 5.91 | miss (5/5) |
| u33_c6 | 8+ share down | .151 -> .283 | -> 6.11 | miss (5/5) |
| pe_lo | median up | .172 -> .242 | 4.99 -> 6.02 | hit (weak: median up in 2, tied in 2) |
| pe_hi | median up | .172 -> .445 | -> 9.35 | hit (4/4) |
| pe_L5 | median up | .172 -> .435 | -> 8.39 | hit (4/4) |
| pe_c6 | median up | .172 -> .328 | -> 7.31 | hit (4/4) |
| pe_L20 | median up | .168 -> .313 | 4.71 -> 6.82 | hit (4/4) |
| pe_L22 | median up | .168 -> .331 | -> 7.82 | hit (3 up, 1 tied) |
| pre15 | median up | .168 -> .246 | -> 5.66 | miss on the stated median (tied 4/4; mean up 4/4) |
| pre35 | median up | .168 -> .369 | -> 8.20 | hit (4/4) |
| pe_L14 | median up | .168 -> .003 | -> 1.97 | miss (collapse) |
| u33_ctrl | flat | .159 -> .220 | 4.75 -> 5.42 | not flat (8+ share up 5/5; median tied) |
| pe_ctrl | flat | .172 -> .222 | 4.99 -> 5.29 | not flat (8+ share up 4/4; median tied 3/4) |

### Round 5 (b59; one batch stopped at 52 tasks, underpowered)

Non-controls lower the 8+ share (base .096): 3 of 6 (late60 .058, late60_x3 .038, late100_c6 .019; not late90 .115,
late60_x5 .135, late60_c6 .096). "More with dose": miss.

### Counts

- Direction right: 18 / 30 edits (sign test p = 0.18 one-sided). Arena non-control 7/10 (Claude 3/3, GPT 1/3,
  Gemini 3/4); round-4 u* 0/5; round-4 pe*/pre* by mean 8/9; round 5 3/6.
- Single-direction "explore first" MLP pushes (xp_const, xp_pre, xp_think, nine pe*/pre*): 11 / 12, one-sided
  p = 0.003. Edit-by-batch comparisons excluding pe_L14: mean calls up in 46 / 46 (not independent; descriptive).
- Composite early+late edits (u*): 0 / 5, wrong in 25 / 25 batches.
- Strict hits (magnitude where an interval was given; controls counted): 18 / 39, 1 not checkable.
- Controls flat: 3 / 6.

### Dose-response (change vs same-batch base, pooled)

| pair | change | ordered |
|---|---|---|
| pre15 -> pre35 | mean calls +0.95 -> +3.49 | yes |
| pe_lo -> pe_hi | +1.03 -> +4.36 | yes |
| u33 -> u63 (early dose) | +1.39 -> +3.37 | yes |
| late30 -> late60 | 8+ share -0.013 -> -0.045 | yes |
| u33 -> u36 (late dose) | 8+ share +0.101 -> +0.104 | no |
| round-5 ladders (n = 52) | | no (2 of 2) |

Pre-serving write size vs measured shift within pe/pre: Spearman +0.55 (mean calls), +0.60 (8+ share), n = 8.

## 3. Specificity

Explore edits lengthen whole episodes: total calls per episode pe_hi 15.15 vs base 11.14 (b43); u63 14.85 vs 11.68
(b45); pe_lo 12.15 vs 11.14; pe_ctrl 11.67 vs 11.14. Session-limit rows pooled (r4_pool_1555.txt): pe_lo 31 vs 28,
pe_L5 39 vs 28, pe_ctrl 59 vs 28, pre35 65 vs 49. Malformed tool-call rate does not rise for the real explore edits
(b44 base .083; pe_lo .073; pe_L20 .064; u63 .065; pe_ctrl .069).

pe_L14 (b44): 4.21 calls per episode vs 11.32; median 1 call before first edit; 42% of agent steps had no tool call
(base 15%); 24% of calls malformed (base 8%); 156 / 170 episodes still ended in submit_patch. Visible before serving
in `claude/work/r3_pe_L14.log`: both layer-14 neurons hit the bisection ceiling (c = 1048.576; every other layer
needed c <= 0.32); relative change norm 0.49 at layer 14 vs 0.02-0.04 at layers 16 and 18; the recomputed shift had
mixed signs across position groups (-0.34 at edit@0; +0.40 at edit@32; +0.38 bg@think; +0.25 bg@prompt), whereas in
pe_lo every group is -0.15 to -0.19. pe_L22 also hit the ceiling at layer 26 (rel 0.40, signs consistent) and was the
worst non-collapsed edit.

## 4. Human-readable chains

pe_lo: contrast "next call makes the first edit" vs not -> class-mean difference at layers 18/20/22 -> ridge gate
(held-out AUC 0.99 / 0.95) -> quiet neurons 3359/4218 (L18), 2972/3970 (L20), 4698/6259 (L22), recomputed shift
-0.15 to -0.25 class gaps, uniform across position groups, residual change 1.6-5.5% -> predicted median calls up ->
measured mean 4.99 -> 6.02, 8+ share .172 -> .242, up in 4/4 batches.

u33: early neuron -0.43 to -0.54 gaps (calls <= 4) plus late neuron +0.34 to +0.42 gaps (calls >= 8) -> predicted 8+
share down -> measured .151 -> .252, up in 5/5. The late neuron alone (late60) lowered the 8+ share by only 0.045.
Inference (not tested): shifts measured on base-model states do not compose once the edited rollout changes the
states the late gate sees.

Proposed table "contrast -> write -> predicted -> measured": rows pe_lo, pre35, u33, pe_L14, pe_ctrl; columns
layers/neurons, gate AUC, recomputed shift with sign-consistency marker, ceiling flag, predicted behaviour, measured
change in mean calls and 8+ share with batch counts.

## 5. Claim

Supported: direction-level control of one agent behaviour (exploration before the first edit) by single-direction
closed-form edits, stated in advance (11/12, p = 0.003; dose ordered in 4/5 pairs); the build log carried a warning
sign before the one collapse.

Not supported: "correct every time" (18/30); composite edits (0/5); magnitude across constructions; specificity;
clean controls (3/6 flat); that moving behaviour changes solve rate (rank correlation about 0 across 16 edits); that
the LLM reads weights.

Prior-work lines: automated interpretability (Bills et al. 2023; MAIA, Shaham et al. 2024); activation steering /
representation engineering (Turner et al. 2023; Zou et al. 2023; Rimsky et al. 2024; Li et al. 2023; Arditi et al.
2024); closed-form editing (ROME, MEMIT); weight-space programming. Verify each citation before use.

## 6. Solve-rate predictions (secondary)

Arena round 1: 15/15 inside 80% intervals that were 9-17 tasks wide. Round 4: 11/20 (a constant -2..+3 range would
hit 13/20); predicted vs realised rank correlation +0.21 (n = 19).

## 7. Draft text

Kaggle (about 115 words):
Can one LLM steer another's behaviour by writing its weights? The author never reads weight values. It reads agent
traces and stored activations, finds the direction that separates "about to make the first edit" from "still
exploring", and compiles a neuron that pushes along it. For single-direction pushes, the predicted change held in 11
of 12 edits and in all 46 batch comparisons, and larger doses moved behaviour more (4 of 5 pairs). Composite edits
that pushed early and pulled late went the opposite way in 5 of 5. Overall, 18 of 30 pre-stated directions held. The
one collapse was flagged by the author's own build log before serving: the write-size solver hit its ceiling.

Full paper (about 340 words):
Behavioural control. The author's inputs are behavioural statistics over 2,210 unedited episodes and activations
stored at labelled decision points, not weight values. It forms a class-mean-difference direction ("the next call
makes the first edit"), fits a ridge gate (held-out AUC 0.95-0.99), and writes a quiet MLP neuron per layer. The
write size is solved so that the recomputed shift on stored activations equals a target. Base weights enter only as
inputs to this computation. We scored every pre-stated behavioural prediction against the same-batch unedited arm.
The direction held in 18 of 30 edits (sign test p = 0.18). The record splits by construction. Single-direction
"explore first" pushes raised calls before the first edit in 11 of 12 edits (p = 0.003) and in all 46 edit-batch
comparisons, and a larger dose moved behaviour more in 4 of 5 pairs. Edits combining an early explore push with a
late commit push were predicted to cut episodes with 8+ calls before the first edit. All five raised that share
instead, in 25 of 25 batches. Shifts measured on base-model states do not compose once the rollout leaves those
states. Of six random-direction controls, three stayed flat. The other three left the median unchanged but raised
the 8+ share by 0.05-0.06. One control doubled session-limit endings. The pushes are not specific: total calls per
episode also rose. The one collapse (pe_L14) made the model submit after a median of one call, with 42% of turns
lacking a tool call and 24% of calls malformed. It was visible before serving: at layer 14 the solver hit its
ceiling and the shift's sign differed across position groups. Solve-rate predictions were weaker (round 4: 11 of 20,
against 13 for a constant no-effect range), and behavioural shift did not track solve change (rank correlation about
0). The supportable claim is direction-level behavioural control for simple edits, not outcome prediction.

Note on the full-paper draft: "the other three ... raised the 8+ share by 0.05-0.06" is measured for the two round-4
controls only; xp_const_ctrl's non-flatness is +0.45 mean calls. Reword before use.

## 8. Not done

Unembedding token projection; GPT error_recheck metric; Gemini secondary predictions; round-2 prediction file not
located; round 6 states no behavioural prediction.
