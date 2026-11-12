# D1: compiled weight edit vs inference-time activation addition

**Setup.** Held-out `edit@0` decision points (states3), same prompts for every arm. Arms: base; A = compiled adapter as served (rank-padded LoRA); N = hook adding the intended neuron write `gelu(g·x)(u·x)·v` to the MLP output at L18/20/22; V = hook adding the constant (or gate-scaled) vector `m·d` to the layer output; C/CN/CV = same for the matched control.
Part 1 (fidelity): vLLM-numerics twin `AlignedGemma4` (INT4 marlin, bf16), teacher-forced on the logged completion, 76 points, all positions. Part 2 (regeneration): HF transformers, checkpoint decompressed to dense bf16, greedy, thinking budget 2560, 57 points, explore-more only.

**Part 1, mean over 76 points (CIs ≤ ±0.003 for cos)**

| | explore-more 0.25 | before-first-edit gate |
|---|---|---|
| cos(A−base, N−base) L18 / L22 / L41 | 0.992 / 0.992 / 0.972 | 0.994 / 0.996 / 0.984 |
| norm ratio A/N, L41 | 1.001 | 1.000 |
| R² A vs N, L18 / L41 | 0.984 / 0.945 | 0.989 / 0.968 |
| cos(A−base, V−base) L18 / L22 / L41 | 0.949 / 0.804 / 0.946 | 0.955 / 0.732 / 0.945 |
| cos(A−base, C−base) L41 (floor) | 0.056 | 0.018 |
| projection on d / gap at L18, A / N / V | 0.236 / 0.236 / 0.250 | 0.249 / 0.249 / 0.265 |
| KL base‖A | 0.0112 | 0.0127 |
| KL N‖A / V‖A | 0.00011 / 0.00039 | 0.00012 / 0.00051 |
| KL base‖C | 0.0029 | 0.0056 |

**Part 2, explore-more, n=57**: edit share base 0.632, A 0.614, N 0.544, V 0.632, C 0.614, CV 0.684. Same tool: N|A 0.860, V|A 0.877, base|A 0.877, C|A 0.860. Edit-share difference N−A −0.070 [−0.153, 0.013], V−A 0.018 [−0.042, 0.078], A−base −0.018 [−0.109, 0.074]. HF vs vLLM replay, same arm: same tool 0.84–0.86.

**Reading.** The compiled adapter reproduces the exact hooked write: its next-token distribution is about 100 times closer to the hook's than to base. Compilation loss is negligible; a plain vector addition is a looser match (cos 0.73–0.80 at L22). Regeneration cannot separate the arms: agreement between hook and adapter equals agreement between base and adapter.

**Caveats.** Part 1 is not HF and stopped at 76 of 320 points to free the GPU. Part 2 has 57 points, not 100 (188 s/point); the 57 are the longest 27 and shortest 30 prompts of the first 104. Greedy sequences diverge chaotically, and batch composition changes outputs. β taken from edit reports; hook adds in bf16; R² uncentred; old neuron not removed in N; the ~1.6% A-vs-N residual is unattributed. Gate control's vector hook is a poor match at L18 (cos 0.12). Gate regeneration not run.
