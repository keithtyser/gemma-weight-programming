# Weight-Space Programming Notes (`gemini`)

## What We Examined
We audited all 23 evaluation batches (`b0..b20`, `s1`, `s2`), ATIF traces across 13 unedited base runs (2,210 episodes), the exact vLLM-aligned Gemma 4 E4B equations (`aligned_forward.py`), and persistent activation datasets (`acts_based/acts_base_rl.pt`, `acts_bases/acts_base.pt`) linked to full tool-call histories (`states_w5d`, `states_w5s`).

Four findings shaped our edits:

1. **Dose-response & why `colscale` hurt:** Multi-neuron rescaling (`cg_*` in `b19–b20`) lost `-3` to `-8` tasks/batch from polysemantic collateral damage. Conversely, mild anti-premature-edit steering (`bneg_t0.125`, `0.375` gaps) won `4/4` batches (`+4.50/batch`, lifting the 42 unreliable tasks from `17.31` to `20.75` and `ok_edited` from `94.5%` to `96.5%`), and stacked concise thinking (`bn50_th50`) won `2/2` batches (`+4.00/batch`, cutting `old_string` quotes from `351` to `239` chars), though overdosing `bneg` at `t=0.5` lowered `ok_edited` to `90.3%`.
2. **`post_feedforward_layernorm` outlier blow-up:** Exact coordinate inversion `uh = unit(dh / w_post)` in `neuron.py` amplifies near-zero `w_post` outliers (`0.0015` at L26), collapsing write efficiency `36x` (`0.045`), saturating `beta = 0.60`, and shrinking natural MLP writes by `14.25%`. Tikhonov preconditioning `uh = unit(dh * w_post / (w_post^2 + 0.10 * median(w_post^2)))` preserves `cos > 0.993` while keeping `beta <= 0.028` and natural MLP shrinkage `<= 0.04%`.
3. **Exact `self_attn.o_proj` recovery:** Intersecting `xin_i / w_pre_ff_i` with the `post_attention_layernorm` sphere `||(s*v - h_{i-1}) / w_post_attn_i||^2 = 2560` recovered exact normalized `o_proj` outputs across all 41 layers, showing `edit@0` and `th@0` enter primarily via `self_attn.o_proj` (peaking at global `full_attention` anchor L23: heldout `aw_auc = 0.819`, gap `43.37`).
4. **Zero-leak conditional ridge gates:** Prior `cond` neurons (`bnegc_t1.0`) leaked `68.2%` background firing by omitting `bg@*` states when fitting gates. Including `bg@*` as negative ridge targets yields `0.0000` heldout background firing.

## Why We Chose These Edits
1. **`damped_dual_explore_concise` (+ `_ctrl`):** Pairs sweet-spot `-edit@0` (L18,20,22) and `-th@0` (L24,26,28) with Tikhonov preconditioning for cleaner exploration and shorter quotes without MLP distortion.
2. **`oproj_onesided_commit_damper`:** Pure `self_attn.o_proj` rank-1 attenuation orthogonalized against explore and background subspaces (`20x` one-sided selectivity, zero MLP changes).
3. **`whitened_cond_rushed_read`:** Zero-leak gate firing `6.7x` stronger on rushed edits (`<=4` calls) than informed edits (`>=6` calls), writing `read_file - edit_file`.
4. **`hybrid_explore_verify_guard`:** Combines damped pre-edit explore with a cross-dataset whitened gate discouraging unverified `submit_patch` (`85.1%` of base submissions).
