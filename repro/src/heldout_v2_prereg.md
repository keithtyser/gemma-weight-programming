# Held-out confirmation v2: gated explore-more edit from the arena (written before any v2 rollout)

Written 12 Nov 2026, 7:30 AM ET, before launch. Nothing below may change after the first rollout starts.

- Tasks: the 130 tasks of `eval_data/steer_heldout_v1` (never used for diagnosis, direction fitting, strength
  selection or arena authoring; they were run once before, for the v1 confirmation of the ungated edit).
- Settings: `v2.steer.confirm run --split all --rollouts 3 --concurrency 10`, harness `steer_ext_w5`, greedy.
- Arms, all served by ONE vLLM server per replicate: `base` (zero adapter), `ac_xp_pre` (primary edit),
  `ac_xp_const_ctrl` (its author's control), `am_damped_dual_explore_concise` (exploratory, second author).
- Replicates: g1 on evalhost GPU1 now; g0 on evalhost GPU0 with the same arms when that GPU frees, if time before the
  deadline allows. Arms are compared only with arms on the same server; per-task differences are summed over replicates.
- Primary test: `ac_xp_pre` minus `base`, per task = solved rollouts of the edit minus solved rollouts of base
  (summed over rollouts and replicates); two-sided paired t over the 130 tasks, alpha 0.05. Reported with the
  sign test over tasks and the difference in tasks-equivalent (solved rollouts / 3 per replicate).
- Secondary: `ac_xp_pre` minus `ac_xp_const_ctrl`, same test. Exploratory: `am_damped_dual_explore_concise` minus `base`.
- Confirmed = primary p < 0.05 with a positive difference. Anything else is reported as not confirmed.
- If a server fails mid-run, the outcome-blind rule of `v2.steer.confirm_resume` applies and all arms of that
  replicate are rerun on one server.
