# Diagnosis tallies: independent recompute (base runs b3–b15)

Data: 13 base runs (b3–b15) × 170 dev tasks = 2,210 episodes, 342 solved. Read-only copy of
`results/steer_eval/b{3..15}_base/{comp_s0/shard_0, rebench/shard0_gpu0}/` with the
task_results.jsonl and traces (106 + 64 tasks per batch, disjoint sets) is in `raw/`.
Script: `recompute.py` (own code). Output: `result.json`. `theirs/diag.py` is the authors' script
copied for a cross-check; only its ROOT path and Windows path handling were changed.

## Definitions
- A tool call is each entry in a step's `tool_calls`.
- An edit_file call fails when its observation JSON has `status == "error"` (the
  primary definition, matching diag.py). The sensitivity variant "strict" also counts
  165 calls the harness rejected before running them (missing `old_string` etc., with no
  `status` field) and 1 call with no observation.
- The first file edit is the first edit_file or write_file call. n_pre is the number of
  calls before it, or all calls if the episode never edits (6 episodes).
- The MH OR compares episodes in the bin with all other episodes of the same task,
  stratified by task. The 95% CI uses the Robins–Breslow–Greenland variance. The p-value is
  from a CMH chi-square test with no continuity correction.

## Results

| Quantity | Paper | Recomputed (95% CI) |
|---|---|---|
| edit_file calls | 4,916 | 4,916 |
| failed / share | 35% | 1,727 / 35.1% (task-bootstrap 31.3–38.6%) |
| solve rate, no failed edit | 21% (318/1,540) | 20.7% (318/1,538; Wilson 18.7–22.8%) |
| solve rate, ≥1 failed edit | 4% (24/670) | 3.6% (24/672; Wilson 2.4–5.3%) |
| MH OR, any failed edit | 0.42 | 0.42 (0.23–0.74), p = 0.001 |
| n_pre 0–2: share; solve in/out; MH OR | 0.76 | 27.1%; 28.3% / 10.7%; 0.76 (0.39–1.48), p = 0.39 |
| n_pre 3–4 | 1.42 | 34.1%; 14.3% / 16.1%; 1.42 (0.86–2.35), p = 0.16 |
| n_pre 5–7 | 1.72 | 21.9%; 11.3% / 16.6%; 1.72 (0.97–3.06), p = 0.051 |
| n_pre 8+ | 0.12 | 16.9%; 2.7% / 18.1%; 0.12 (0.04–0.38), p < 0.001 |
| tasks solved ≥1 of 13 runs | 51 | 51 |
| tasks solved in all 13 runs | 9 | 9 |

Strict sensitivity, which also counts rejected calls as failures: 1,893/4,916 failed (38.5%),
34/724 vs 308/1,486 solved, MH OR 0.44 (0.25–0.79).

## Mismatches
1. **670/1,540 vs 672/1,538.** The paper's split comes from diag.py's "call after first failed
   edit" buckets. Those buckets drop 2 failed-edit episodes into "no failed edit", and both were unsolved:
   - (a) tobymao__sqlglot-4004 in b4. Its only failed edit is the last call of the trajectory.
     diag.py `continue`s before recording the first-failure kind.
   - (b) One step issued two calls at once (edit_file + read_file). diag.py sets `err=None`
     for multi-call steps, but the shared observation is a FileEditError.

   diag2.py (any_edit_failed) misses only (b), so it gives 671. The rates do not change at
   the reported precision (21% / 4%), and the MH OR 0.42 is the same.
2. **1,726 vs 1,727 failed calls.** The difference is the same multi-call step (b). The 35% is unchanged.
3. **Bin ORs** match to 2 decimals. Only the 8+ bin's CI excludes 1. The CIs for 0–2 and 3–4
   include 1, and 5–7 is borderline (0.97–3.06). The raw and within-task directions disagree for
   0–2 (raw solve rate 28% vs 11%, but within-task OR 0.76) and for 5–7 (raw 11% vs 17%, OR 1.72).
   Easy tasks get edited early, so the raw rates are confounded by task difficulty.
4. Caveat: few tasks inform the MH estimates. Only 19–36 tasks have both bin members and
   non-members and mixed outcomes, because 119/170 tasks are never solved.
