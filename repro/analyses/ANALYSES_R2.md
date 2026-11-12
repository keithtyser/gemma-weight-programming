# Round-2 analyses (2026-11-12): selection null and predicted vs realised shift

Saved data only; no jobs run. Scripts: `scripts/selection_null.py` → `src/selection_null.json`; `scripts/shift_check.py <copy dir>` → `src/shift_check.json` (build reports read from evalhost `/path/to/gemma-v2/wsp_arena/{claude,gpt,gemini}/edits/*/`, read only, not modified).

## 1. Selection null for the largest of twelve screening differences

Data: `src/arena_per_task_b21_b28.tsv` (170 tasks per arm per batch, same 170 tasks in every batch and in all 21 unedited runs, b3–b15 from `src/dev_runs_b3_b16.tsv`).

Observed screening differences (edit − unedited, b21–b23): b21 −1, −2, +4, −1; b22 +3, +6, **+9** (`ac_xp_pre`), +1; b23 +7, +6, +4, +3. Controls: −1, +1, +8.

| Null for max over 12 edits of (edit − unedited) | median | 95th pct | 99th pct | P(max ≥ 9) |
|---|---|---|---|---|
| (b1) paired swap per task, independently per edit arm (10,000 perms, seed 0) | 7 | 11 | 13 | 0.22 |
| (b2) joint permutation of all six arm labels per task within batch (10,000, seed 0) | 6 | 11 | 13 | 0.18 |
| (a1) Gaussian runs, SD 2.66 = SD of the 21 unedited runs, shared unedited arm per batch (100,000 sims) | 5 | 10 | 12 | 0.11 |
| (a2) resample run totals from the 21 unedited runs | 5 | 8 | 9 | 0.025 (bounded: range 21–30 caps max at 9) |
| (a3) resample 12 of the 25 control − unedited differences | 6 | 8 | 8 | 0 (bounded: pool max is +8) |

(a2) and (a3) cannot exceed their pools' extremes and are not used for p-values.

Replication of the selected gate, b24–b28: +2, +3, +9, +5, +2, mean +4.2; 75 discordant tasks, 48 gained, 27 lost. Within-batch per-task paired swap (10,000, seed 0): null median −0.2, 95th percentile +3.0, one-sided p = 0.0093 (two-sided 0.018; exact sign test on pooled discordant tasks 0.010). The batch-level t-test in `src/review_stats.json` gives p = 0.033; the permutation treats tasks as the unit and so ignores batch-level variation (server/batch effects), which the t-test includes.

Assumptions: under the null each task's outcomes in an edit arm and the unedited arm of the same batch are exchangeable (sharp null, conditional on the observed per-task outcomes); runs are independent; (a1) assumes edited and unedited runs have the run-to-run SD of the unedited model. The permutation nulls condition on the observed discordance, which includes any real effects, so they are slightly wide if some edits work.

Paper text: *Under a within-batch permutation null (per task, the labels of each edit arm and the unedited arm swapped at random; 10,000 permutations), the largest of the twelve screening differences has median +7 and 95th percentile +11, so the screening winner's +9 is what selection alone produces (p = 0.22; 0.18 under a joint permutation of all arms, 0.11 under a Gaussian model with the unedited run-to-run SD of 2.7).* *The evidence for the gate therefore rests on its five replication batches, where it was fixed in advance: +4.2 per batch, 48 tasks gained and 27 lost, task-level permutation p = 0.009 (batch-level t-test p = 0.033).*

## 2. Predicted (Eq. 3) vs realised shift along the direction, in class gaps

Prediction: GPT and Gemini set β from Eq. 3 for a target (uncapped in every layer), so the prediction is the target; Gemini's η is the projection of w_post ⊙ û on d_l (recomputed from its β and η at layers 18/20/22: 0.140 each). Claude's builder sized the write by bisection on the exact shift, so its Eq. 3 prediction is computed from its recorded β_on and K_l = s_l√n η_l / gap_l taken from Table tab:params (same d_l, gaps and AUCs; K = 4.37, 4.16, 1.64). Realised: the builders' exact recompute on held-out positions (ON positions for gated edits; bf16 for GPT and Gemini, fp32 for Claude).

| Author / edit | Layer | Kind | β | Predicted | Realised | Ratio |
|---|---|---|---|---|---|---|
| Claude xp_const | 18/20/22 | always-on | .028/.030/.076 | .124/.124/.124 | .125/.125/.125 | 1.01/1.01/1.01 |
| Claude xp_pre | 18/20/22 | gated | .059/.062/.165 | .258/.258/.266 | .250 each | 0.97/0.97/0.94 |
| Claude xp_think | 18/20/22 | gated | .059/.061/.154 | .259/.254/.250 | .250 each | 0.97/0.99/1.00 |
| Claude xp_early | 18/20/22 | gated | .097/.104/.298 | .423/.430/.467 | .300 each | 0.71/0.70/0.64 |
| GPT think_start_explore | 18/20/22 | gated | .080/.084/.219 | .35 | .343/.343/.321 | 0.98/0.98/0.92 |
| GPT error_recheck | 18/20/22 | gated | .059/.056/.057 | .25 | .211/.210/.236 | 0.84/0.84/0.94 |
| Gemini damped_dual (explore) | 18/20/22 | always-on | .018/.021/.017 | .14 | .134/.155/.159 | 0.96/1.11/1.14 |
| Gemini damped_dual (concise) | 24/26/28 | always-on | .019/.028/.025 | .12 | .125/.131/.133 | 1.04/1.09/1.11 |
| Gemini hybrid (explore) | 18/20/22 | always-on | .017/.020/.016 | .13 | .124/.143/.148 | 0.96/1.10/1.14 |
| Gemini hybrid (guard) | 23/24/28 | gated | .012/.017/.023 | .10 | .169/.186/.214 | 1.69/1.86/2.14 |
| Gemini whitened_cond | 17/18/20/22 | gated | .026/.025/.028/.022 | .18 | .177/.165/.183/.200 | 0.98/0.92/1.02/1.11 |

Ratio realised/predicted, 34 edit×layer rows: mean 1.05, range 0.64–2.14. Always-on (12): mean 1.06, range 0.96–1.14. Gated (22): mean 1.05, range 0.64–2.14. Gated GPT/Gemini rows divided by the gate's held-out firing relative to its reference (13 rows; firing 0.87–0.99, Gemini guard 1.71–1.99): mean 1.01, range 0.92–1.17.

Departures: Gemini's guard fires 1.7–2.0 times its reference level on held-out positives, so it overshoots by that factor (per unit firing 0.99–1.08). Claude's gated β_on is averaged over training and held-out ON positions while the shift is held-out only, so ratios below 1 (xp_early 0.64–0.71) partly reflect the gate firing less on held-out ON positions (xp_early: only 66 % of held-out ON positions fire above 1 at layer 18; mean held-out ON activation 2.23 against 4 at full firing) and are not a pure Eq. 3 error.

Not tabulated (no Eq. 3 prediction or no data): GPT's rotations (linearised sizing, angle at its 0.15 cap, class gap not recorded; realised −0.012/−0.101/−0.119), Gemini's attention edit (realised −0.11 to −0.16), controls (prediction 0; realised |shift| ≤ 0.019). **The first builder's always-on edits of Table tab:params (bneg_t*) have no realised-shift record in `src/` or under the permitted evalhost paths; their predicted-vs-realised comparison is not available here.** Claude's xp_const is the same edit (same d_l, neurons 3359/2972 at layers 18/20) at t = 0.125 and gives ratio 1.01.

Paper text: *Across the 12 always-on edit×layer pairs built by Claude and Gemini, the shift recomputed on held-out positions is 0.96–1.14 times the Eq. (3) prediction (mean 1.06).* *For gated edits the ratio tracks how strongly the gate fires on held-out positions: per unit of firing it is 0.92–1.17 (13 pairs, GPT and Gemini), while the raw ratio ranges from 0.64 (Claude's early-calls gate) to 2.14 (a Gemini guard that fires twice its reference level).*
