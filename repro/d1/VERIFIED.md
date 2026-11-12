# D1 verification (independent recompute, 2026-11-12)

Sources: `d1_fidelity_results.json` (= `d1_fidelity_summary.txt`; embedded `meta` is identical to `d1_fidelity_meta.json`), `d1_regen_results.json`, scripts `d1_fidelity.py`, `d1_prep.py`, `d1_regen.py`, `d1_summarize.py`, `d1_regen_summarize.py`. All numbers below recomputed from the JSON with `python -I`. Prefix `F.` = `d1_fidelity_results.json:stats.<pair>`, `R.` = `d1_regen_results.json:explore_more_0.25`. Pairs: `em` = `explore_more_0.25` (real `bneg_t0.25`, control `bnm25_s1`, both const mode); `gate` = `before_first_edit_gate` (real `ac_xp_pre`, gated; control `ac_xp_const_ctrl`, const mode).

## What is measured (from the scripts)

- Edits: each adapter changes exactly one neuron j (gate row, up row, down column) at layers 18, 20, 22 and nothing else (asserted in `edit_spec`).
- Arms: base = zero adapter; A = adapter as served (rank padded to 32); N = zero adapter + hook adding `gelu_tanh(g.x)(u.x) v` (fp32, cast to bf16) to the MLP output, g,u,v = the edited neuron's new rows/column; the overwritten neuron j is NOT removed in N (A replaces j, N adds a new neuron next to the old j); V = constant vector `m d_unit` added to the layer output at every position (gated edit: scaled by a(x)/a_ref); C/CN/CV = same for the control.
- Part 1 engine: AlignedGemma4 (vLLM-numerics twin, INT4 marlin, bf16), teacher-forced on logged prompt + logged completion of the edit@0 decision call.
- "cos": cosine between residual-stream deltas (arm layer output minus base layer output) at capture layers 18, 20, 22, 24, 28, 32, 36, 41; `cos_pooled` pools all positions of prompt+completion (mean 11,763 tokens) per point, then averages over points. `cos_last` = the decision position (last prompt token) only.
- R² = 1 − ‖ΔA−ΔN‖²/‖ΔA‖² (uncentred, relative to A). Norm ratio = ‖ΔA‖/‖ΔN‖ pooled.
- KL: KL(p‖q) of fp32 next-token distributions (softcap(h·Eᵀ), not the engine lm-head) averaged over the rows that predict the completion tokens (positions P−1..L−2, mean ≈ 868 rows/point), then averaged over points. `first` = decision position only.
- CIs: normal 95% over points (n=76 or 57).
- A "point" = one held-out edit@0 decision call (states_w5d, split heldout), label-balanced seeded order.

## Verified numbers

| Quantity | em | gate | JSON key |
|---|---|---|---|
| n points (label 1) | 76 (38) | 76 (38) | `n_points`, `n_label1` |
| cos(ΔA,ΔN) L18 | 0.9918 [0.9917, 0.9919]; per-point 0.9908–0.9926 | 0.9944; 0.9905–0.9960 | `F.resid["A|N"]["18"].cos_pooled` |
| cos(ΔA,ΔN) L22 | 0.9922; 0.9902–0.9933 | 0.9958; 0.9924–0.9972 | `…["22"].cos_pooled` |
| cos(ΔA,ΔN) L41 | 0.9724; 0.9693–0.9756 | 0.9839; 0.9775–0.9874 | `…["41"].cos_pooled` |
| cos(ΔA,ΔN) range of means, all 8 layers | 0.972–0.992 | 0.984–0.996 | `F.resid["A|N"][*].cos_pooled.mean` |
| cos(ΔA,ΔN) at decision position, L18 | 0.990 [0.989, 0.991]; min 0.966 | **0.883 [0.831, 0.936]; min 0.044** | `…["18"].cos_last` |
| norm ratio ΔA/ΔN, L41 (range all layers) | 1.0007 (1.0002–1.0039) | 1.0000 (0.9996–1.0027) | `…norm_ratio_pooled` |
| R² A vs N, L18 / L41 (range all layers) | 0.9837 / 0.9448 (0.945–0.985) | 0.9889 / 0.9678 (0.968–0.992) | `…r2` |
| cos(ΔA,ΔV) L18 / L22 / L41 | 0.9494 / 0.8035 / 0.9464 | 0.9551 / **0.7315** / 0.9454 | `F.resid["A|V"][L].cos_pooled` |
| cos(ΔA,ΔC) L41 (all layers) | 0.0563 (−0.256 at L22 … 0.077) | 0.0180 (−0.068 … 0.185) | `F.resid["A|C"][L].cos_pooled` |
| cos(ΔC,ΔCN) L18–41 | 0.954–0.992 | 0.964–0.992 | `F.resid["C|CN"]` |
| cos(ΔC,ΔCV) L18 / L20 | 0.961 / 0.963 | **0.121 / 0.422** (nr 6.98 / 1.96) | `F.resid["C|CV"]` |
| proj on d / gap, L18, all positions, A / N / V | 0.236 / 0.236 / 0.250 | 0.249 / 0.249 / 0.265 | `F.proj[arm]["18"].all` |
| KL(base‖A) | 0.01124 [0.01047, 0.01201]; per-point 0.0044–0.0211 | 0.01271 [0.01147, 0.01395]; 0.0016–0.0253 | `F.kl["base|A"].mean` |
| KL(N‖A) | 1.14e-4 [1.08, 1.21]e-4; 0.5–1.7e-4 | 1.16e-4 [1.10, 1.22]e-4; 0.5–1.7e-4 | `F.kl["N|A"].mean` |
| KL(V‖A) | 3.90e-4 | 5.07e-4 | `F.kl["V|A"].mean` |
| KL(CN‖C) (control hook vs control adapter) | 1.12e-4 | 1.14e-4 | `F.kl["CN|C"].mean` |
| KL(base‖C) | 0.00291 | 0.00562 | `F.kl["base|C"].mean` |
| ratio KL(base‖A)/KL(N‖A) | 98.5 | 109.6 | derived |
| KL at decision position: base‖A / N‖A | 3.4e-5 / 1.0e-6 | 3.4e-5 / 7.9e-7 | `F.kl[*].first` |
| top-1 agreement N vs A over completion rows | 0.9959 (min 0.985) | 0.9958 (min 0.984) | `F.top1["N|A"]` |
| top-1 agreement base vs A | 0.9585 | 0.9576 | `F.top1["base|A"]` |

Part 2 (HF transformers, dense bf16 from the INT4 groups, greedy, batched arms; em pair only):

| Quantity | Value | JSON key |
|---|---|---|
| n points (label 1), s/point | 57 (28), 188.3 s | `R.n_points`, `R.n_label1`, `R.sec_per_point` |
| fake-quant-off check | max |Δlogit| 0.0, both shards | `R.meta[*].fake_quant_off_check` |
| edit_file share base / A / N / V / C / CV | 0.632 / 0.614 / 0.544 / 0.632 / 0.614 / 0.684 | `R.readouts[arm].all.edit` |
| same tool N\|A, V\|A, base\|A, C\|A | 0.860, 0.877, 0.877, 0.860 | `R.paired[..].same_tool` |
| edit-share diff N−A | −0.070 [−0.153, 0.013] | `R.paired["N|A"].edit_rate_diff` |
| edit-share diff V−A | 0.018 [−0.042, 0.078] | `R.paired["V|A"].edit_rate_diff` |
| edit-share diff A−base | −0.018 [−0.109, 0.074] | `R.paired["A|base"].edit_rate_diff` |
| edit-share diff N−V (not in RESULT) | **−0.088 [−0.162, −0.014]** | `R.paired["N|V"].edit_rate_diff` |
| think-length diff C−base, CV−C (not in RESULT) | −83 [−150, −16]; +132 [+20, +244] tokens | `R.paired["C|base"/"CV|C"].think_diff` |
| full token sequence identical to A: N / V / base / C | 5/57 / 5/57 / 0 / 0 | `R.identical_tokens_to_A` |
| mean common prefix with A (tokens): N / V / base / C | 182.8 / 147.7 / 34.5 / 37.2 | `R.common_prefix_with_A_mean` |
| HF decision-position KL base‖A / N‖A | 0.0065 [−0.0034, 0.0164] / <5e-5 (rounded 0.0) | `R.kl_first` |
| HF vs vLLM replay same tool: base / A / C | 0.842 / 0.860 / 0.860 | `R.vllm[..].same_tool` |
| edit share HF vs vLLM, base arm | 0.632 vs 0.719 | `R.vllm["base~base"]` |

## Discrepancies and issues in RESULT.md

1. Rounding: gate cos(A−base, V−base) at L22 is 0.7315 → 0.731, RESULT says 0.732. All other table numbers match to the shown precision.
2. Omitted: for the gated edit, at the decision position itself the A vs N agreement is much weaker at the write layers: cos_last 0.883 [0.831, 0.936] at L18 (per-point min 0.044), 0.949 at L20 (min 0.253), 0.975 at L22 (min 0.592); nr_last up to 13.5. RESULT reports only position-pooled cosines, which hide this. The output-level agreement at that position is still tight (KL(N‖A) first 7.9e-7 vs KL(base‖A) 3.4e-5), so it is most likely positions where the new neuron is nearly off and A−N is dominated by the removed old neuron (see 4); this is an inference, not verified.
3. "CIs ≤ ±0.003 for cos" holds for the pooled cosines in the table (max half-width 0.0025), not for decision-position cosines (gate L18 ±0.053).
4. "~1.6% A-vs-N residual is unattributed": 1−R² is 1.6% only at em L18; across layers it is 1.5–5.5% (em) and 0.8–3.2% (gate), growing with depth. The scripts make one source explicit: A replaces neuron j while N keeps old neuron j and adds the new one, so ΔA−ΔN includes −(old neuron j's output). The comparison is therefore "adapter vs add-new-neuron hook", not "adapter vs exact replacement". No numerics noise floor (e.g. A vs A in a second run) was measured.
5. "next-token distribution is about 100 times closer": ratios are 98.5 (em) and 109.6 (gate) — fine, but the edit itself is small (KL(base‖A) 0.011–0.013 nats/token, top-1 agreement base vs A 0.958), so "100×" is relative to a small effect. KL is over completion-predicting positions only (~868 rows/point), which RESULT does not state; cosines are over all ~11.8k positions.
6. "matched control": the em control is matched in predicted shift (0.249–0.260 vs 0.250 of gap). The gate control is not: it is const-mode (real is gated) and its predicted shift is 0.021–0.049 of gap vs 0.257–0.267 for the real edit (`meta.specs.*.pred_shift_over_gap`), so KL(base‖C)=0.0056 is not a magnitude-matched floor. Gate control vector hook is poor at L18 (cos 0.12) and also at L20 (cos 0.42), RESULT mentions only L18.
7. Regeneration "cannot separate the arms": correct for A vs N, A vs V, A vs base (all CIs include 0, half-widths 0.06–0.09). But not "no significant differences" overall: N−V edit share −0.088 [−0.162, −0.014] and two think-length differences exclude 0 (uncorrected, 12 paired comparisons × 2 metrics). RESULT also omits token-level evidence favouring fidelity: N and V reproduce A's full output on 5/57 points (base 0/57) and share 4–5× longer prefixes with A than base or C do.
8. HF decision-position KL(base‖A) = 0.0065 (CI includes negatives, i.e. heavy-tailed) vs 3.4e-5 in the vLLM twin at the same position. Not discussed; likely engine numerics plus a few outlier points, unverified. HF-vs-vLLM base edit share differs by 0.09 (0.632 vs 0.719), larger than any A-vs-hook difference.
9. Unverifiable from the files here: "320 planned points" (only the `d1_prep.py` docstring; points.jsonl absent and prep filters by length); "57 = longest 27 + shortest 30 of the first 104" (rows, shard membership and `--limit` are not in the results JSON; meta only shows shards 0/2 and 1/2); thinking budget 2560 and max tokens 3072 (script defaults, not recorded in meta). "β taken from edit reports" matches the code.

## Deviations from the planned design

- Part 1: 76 of 320 planned points (time limit `--minutes`); two edit pairs only; one engine (vLLM-numerics twin), with lm-head logits recomputed in fp32 rather than by the engine kernel.
- N hook does not remove the overwritten neuron j (design choice in code; differs from the adapter by j's old contribution).
- Part 2: 57 points instead of 100; length-tiered shards (longest and shortest prompts of the limited set), so mid-length prompts are excluded and the sample is not the label-balanced prefix used in Part 1; explore-more pair only (gate-pair regeneration not run); HF dense-bf16 engine, not vLLM; arms share a batch; the `--a2` batch-row determinism check was not run (no A2 arm in `R.meta.arms`).
- Gate pair control is not magnitude-matched and differs in mode (const vs gated).

## Paper text drafts

### (a) Full paper, LaTeX

```latex
\paragraph{Compiled edit versus activation hook (D1).}
We test whether a compiled adapter implements the activation intervention it encodes.
Two edits each rewrite one MLP neuron (gate row, up row, down column) at layers 18, 20 and 22 of Gemma~4 E4B: a constant ``explore-more'' write and a gated write before the first file edit.
We compare the adapter as served (A) with an inference-time hook that adds the new neuron's output $\mathrm{gelu}(g^\top x)(u^\top x)\,v$ to the MLP output (N), and with constant activation addition along the same residual direction (V).
All arms share one forward pass with vLLM numerics (INT4 weights, bf16), teacher-forced on 76 held-out decision calls (mean 11.8k tokens).
Relative to the unedited model, the residual-stream changes of A and N have cosine 0.972--0.992 (explore-more) and 0.984--0.996 (gated) across eight layers from 18 to 41, with norm ratios within 0.4\% of 1.
Over completion tokens, $\mathrm{KL}(N\,\|\,A)$ is $1.1\times10^{-4}$ and $1.2\times10^{-4}$ nats, 99 and 110 times smaller than $\mathrm{KL}(\text{base}\,\|\,A)$ (0.011 and 0.013).
Vector addition matches less closely (cosine 0.73--0.80 at layer 22; $\mathrm{KL}(V\,\|\,A)=3.9$--$5.1\times10^{-4}$).
```

Limitations sentence:

```latex
D1 covers two edits on 76 of 320 planned decision points in a single engine; the hook adds the new neuron without removing the overwritten one, which leaves a 1--6\% residual ($1-R^2$) between adapter and hook unattributed; greedy regeneration was run for one edit on 57 points only, with confidence intervals ($\pm$0.06--0.09 on edit rate) too wide to separate even the adapter from the unedited model.
```

### (b) Short Markdown version

The compiled adapter reproduces the neuron hook it was built from: on 76 held-out decision points its residual-stream change has cosine 0.97–1.00 with the hook's, and the next-token KL between adapter and hook (≈1×10⁻⁴) is about 100× smaller than the edit's own effect (0.011–0.013). Greedy regeneration (57 points, one edit) was too small to tell any arms apart, including adapter vs base.
