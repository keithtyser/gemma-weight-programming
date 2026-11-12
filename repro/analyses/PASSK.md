# pass@k from saved per-task results

Script: `scripts/passk.py` (pure Python) → `src/passk.json`. Run from `repro/`: `python scripts/passk.py`.

## Method

- Estimator (Chen et al. 2021, unbiased): for each task, pass@k = 1 − C(n−c, k) / C(n, k), where n = runs of that arm on
  the task and c = solved runs; an arm's pass@k is the mean over tasks. "Tasks-eq" = pass@k × number of tasks.
- Paired comparison: per-task difference of pass@k between two arms on the same tasks (same batches for dev/arena,
  same server for held-out). 95% CI = percentile bootstrap over tasks, 10,000 resamples, `random.Random(0)` per
  comparison (the convention of `scripts/review_stats.py`). Better/worse = tasks where the per-task pass@k is higher/lower.
- **Decoding is greedy.** Runs of an arm differ only through non-repeatable serving (batching, kernel order), so pass@k
  here measures run-to-run variability of a fixed model, not diversity from sampling.
- Inputs: `src/dev_runs_b3_b16.tsv` (b3–b15; b16 is partial and unused), `src/arena_per_task_b21_b28.tsv`,
  `src/heldout_v2_per_rollout.tsv`, and `src/heldout_v1_per_rollout.tsv` (new: kept + rerun rows of Test 1, read on evalhost
  with `v2.steer.confirm_resume.merged/kept` and the registered cutoff 2026-11-12T01:26:00Z; identical, row for row, to
  `heldout_v1_rows` in `src/remote_diag.json`; columns arm, task, rollout, solved, kept).
- Every unedited dev run covers the same 170 tasks (asserted). Held-out: 130 tasks × rollouts {0,1,2} for every arm (asserted).
- At k = 1 the difference equals the difference in mean solved per run, so pass@1 reproduces the paper's +5.0 (gate,
  arena) and the held-out pass@1 CI reproduces Appendix B (−6.0 to +11.3).

## 1. Unedited model, 170 dev tasks

| Runs | pass@1 | pass@3 | pass@5 | pass@10 | pass@n | Solved ≥ once | Solved every run |
|---|---|---|---|---|---|---|---|
| All 21 complete runs (b3–b15, b21–b28) | 15.2% (25.9) | 22.1% (37.6) | 25.2% (42.8) | 29.0% (49.4) | 32.9% (56), n = 21 | 56 | 9 |
| b3–b15, 13 runs | 15.5% (26.3) | 22.1% (37.6) | 25.0% (42.5) | 28.8% (48.9) | 30.0% (51), n = 13 | 51 | 9 |
| b21–b28, 8 runs | 14.9% (25.3) | 21.9% (37.1) | 25.3% (43.0) | — | 28.2% (48), n = 8 | 48 | 12 |

Per-run solved (21 runs): 28 24 22 29 22 29 27 26 25 27 30 28 25 | 29 24 21 27 28 22 25 26. Tasks-eq in parentheses.

## 2. Repeated arena edits vs unedited, same batches (170 tasks)

| Edit (batches) | k | Edit | Unedited | Δ tasks-eq | Δ points | 95% CI (tasks-eq) | Better / worse |
|---|---|---|---|---|---|---|---|
| Claude before-first-edit gate `ac_xp_pre` (b22, b24–b28) | 1 | 17.8% | 14.9% | +5.00 | +2.94 | +1.33 to +8.83 | 22 / 12 |
| | 3 | 24.1% | 21.6% | +4.25 | +2.50 | −0.85 to +9.75 | 20 / 10 |
| | 6 | 28.2% (48) | 25.9% (44) | +4.00 | +2.35 | −4.0 to +12.0 | 10 / 6 |
| Claude early-calls gate `ac_xp_early` (b22, b24–b28) | 1 | 17.6% | 14.9% | +4.67 | +2.75 | +0.33 to +9.33 | 27 / 15 |
| | 3 | 25.0% | 21.6% | +5.85 | +3.44 | −0.85 to +12.65 | 24 / 12 |
| | 6 | 29.4% (50) | 25.9% (44) | +6.00 | +3.53 | −4.0 to +16.0 | 15 / 9 |
| GPT gate + rotation `ag_explore_then_rotate` (b21, b24–b28) | 1 | 16.8% | 15.4% | +2.33 | +1.37 | −1.0 to +5.83 | 25 / 17 |
| | 3 | 24.5% | 23.2% | +2.30 | +1.35 | −3.0 to +7.5 | 23 / 13 |
| | 6 | 29.4% (50) | 28.2% (48) | +2.00 | +1.18 | −6.0 to +10.0 | 10 / 8 |
| Gemini explore + concise `am_damped_dual_explore_concise` (b23, b25, b27) | 1 | 17.8% | 14.5% | +5.67 | +3.33 | +0.67 to +11.0 | 19 / 10 |
| | 3 = n | 25.3% (43) | 19.4% (33) | +10.00 | +5.88 | +2.0 to +18.0 | 13 / 3 |

## 3. Always-on explore-more vs unedited, b3–b15 (170 tasks)

Per strength, the edit's runs vs the unedited run of the same batches (equal n; k up to n).

| Strength (batches, n) | k | Edit | Unedited | Δ tasks-eq | 95% CI | Better / worse |
|---|---|---|---|---|---|---|
| 0.125 (b6, b7, b10, b12; 4) | 1 | 17.9% | 15.3% | +4.50 | +0.25 to +8.75 | 22 / 13 |
| | 2 | 22.5% | 19.6% | +4.83 | −0.67 to +10.5 | 18 / 9 |
| | 3 | 24.6% | 21.9% | +4.50 | −2.25 to +11.5 | 14 / 8 |
| | 4 = n | 25.9% (44) | 23.5% (40) | +4.00 | −4.0 to +12.0 | 10 / 6 |
| 0.25 (b5–b7, b10–b12, b15; 7) | 1 | 17.1% | 14.8% | +4.00 | 0.0 to +8.0 | 32 / 19 |
| | 3 | 24.3% | 21.7% | +4.46 | −2.0 to +11.09 | 28 / 16 |
| | 5 | 27.6% | 24.8% | +4.71 | −3.67 to +13.38 | 20 / 14 |
| | 7 = n | 30.0% (51) | 27.1% (46) | +5.00 | −5.0 to +16.0 | 17 / 12 |
| 0.5 (b4–b11, b13, b14; 10) | 1 | 17.0% | 15.4% | +2.70 | −0.7 to +6.1 | 28 / 17 |
| | 3 | 25.2% | 22.2% | +5.08 | −0.06 to +10.24 | 26 / 15 |
| | 5 | 29.2% | 25.4% | +6.41 | +0.1 to +12.73 | 24 / 14 |
| | 10 = n | 34.1% (58) | 30.0% (51) | +7.00 | −1.0 to +15.0 | 13 / 6 |
| Pooled 0.125–0.5 (21 edit runs, 12 batches; unedited n = 12) | 1 | 17.2% | 15.4% | +3.12 | +0.25 to +6.08 | 40 / 23 |
| | 3 | 25.0% | 22.2% | +4.79 | +0.11 to +9.51 | 37 / 21 |
| | 5 | 28.4% | 25.2% | +5.50 | −0.31 to +11.21 | 36 / 20 |
| | 10 | 33.0% | 29.1% | +6.70 | −0.92 to +14.4 | 26 / 18 |
| | 12 | 34.2% | 30.0% (51) | +7.09 | −0.92 to +15.37 | 19 / 21 |

Pooled edit runs reach 64 tasks solved at least once over 21 runs (pass@21 = 37.6%); not comparable to the unedited
arm's 51 over 12 runs, which is why the comparison stops at k = 12.

Against the matched random controls (7 batches with controls: b5–b7, b10–b13; 16 edit runs vs 14 control runs):

| k | Edit | Control | Δ tasks-eq | 95% CI | Better / worse |
|---|---|---|---|---|---|
| 1 | 17.5% | 15.0% | +4.12 | +1.61 to +6.77 | 39 / 18 |
| 3 | 25.2% | 22.1% | +5.34 | +1.33 to +9.46 | 35 / 16 |
| 5 | 28.6% | 24.9% | +6.34 | +1.30 to +11.56 | 34 / 16 |
| 7 | 30.8% | 26.4% | +7.51 | +1.71 to +13.55 | 29 / 14 |
| 14 | 35.1% (59.7) | 28.8% (49) | +10.69 | +3.10 to +18.57 | 15 / 9 |

Controls − unedited (same 7 batches, unedited n = 7): pass@1 −0.29 (CI −2.86 to +2.21), pass@3 −0.57 (−4.37 to +3.10),
pass@7 −2.16 (−8.16 to +3.53).

## 4. Held-out tests (130 tasks × 3 rollouts per arm; kept + rerun rows for Test 1)

| Test, server | Arm | pass@1 | pass@2 | pass@3 (= solved ≥ once) |
|---|---|---|---|---|
| 1, A | Unedited | 34.6% (45.0) | 46.2% | 53.8% (70) |
| 1, A | Explore-more 0.25 (primary) | 34.9% (45.3) | 47.2% | 54.6% (71) |
| 1, A | Matched random control | 35.6% (46.3) | 46.2% | 52.3% (68) |
| 1, B | Unedited (second copy) | 43.6% (56.7) | 57.2% | 66.2% (86) |
| 1, B | Explore-more 0.125 | 41.0% (53.3) | 56.9% | 64.6% (84) |
| 1, B | Explore-more 0.5 | 42.6% (55.3) | 60.3% | 69.2% (90) |
| 2 | Unedited | 41.3% (53.7) | 56.7% | 63.1% (82) |
| 2 | Before-first-edit gate (primary) | 43.6% (56.7) | 59.7% | 68.5% (89) |
| 2 | Claude's random control | 40.8% (53.0) | 55.1% | 61.5% (80) |
| 2 | Gemini explore + concise | 41.8% (54.3) | 56.7% | 63.8% (83) |

Paired differences (tasks-eq of 130; bootstrap 95% CI; better / worse tasks):

| Comparison | pass@1 | pass@3 |
|---|---|---|
| T2 gate − unedited | +3.00 (−6.0 to +11.33); 45 / 36 | +7.00 (−6.0 to +20.0); 27 / 20 |
| T2 gate − control | +3.67 (−4.67 to +11.67); 42 / 34 | +9.00 (−3.0 to +21.0); 24 / 15 |
| T2 Gemini − unedited | +0.67 (−6.67 to +8.33); 37 / 35 | +1.00 (−10.0 to +12.0); 16 / 15 |
| T2 control − unedited | −0.67 (−8.67 to +7.67); 37 / 39 | −2.00 (−14.0 to +10.0); 18 / 20 |
| T1-A explore 0.25 − unedited | +0.33 (−6.33 to +7.33); 24 / 30 | +1.00 (−9.0 to +11.0); 14 / 13 |
| T1-A explore 0.25 − control | −1.00 (−7.67 to +6.0); 23 / 27 | +3.00 (−7.0 to +13.0); 14 / 11 |
| T1-A control − unedited | +1.33 (−5.0 to +7.67); 26 / 25 | −2.00 (−12.0 to +8.0); 13 / 15 |
| T1-B explore 0.125 − unedited2 | −3.33 (−11.67 to +5.33); 33 / 44 | −2.00 (−14.0 to +10.0); 18 / 20 |
| T1-B explore 0.5 − unedited2 | −1.33 (−10.0 to +7.0); 40 / 40 | +4.00 (−8.0 to +16.0); 22 / 18 |
| T1 unedited B − unedited A (same weights) | +11.67 (+3.67 to +20.0); 46 / 23 | +16.00 (+3.0 to +29.0); 30 / 14 |

pass@2 for every comparison is in `src/passk.json`.
