# Programming a Coding Agent's Weights by Hand

**Subtitle:** An LLM diagnoses a Gemma 4 E4B coding agent's failure, writes a weight edit without training, predicts its effect, and verifies it.

## Abstract

Weight-space programming lets a large language model (LLM) write a smaller model's weights without gradients. So far it has been tested only on single-turn benchmarks. We turn it into a working loop for a multi-step coding agent. An LLM author reads logged episodes of a frozen Gemma 4 E4B agent, finds where it fails, and writes a small closed-form edit to a few MLP neurons. The edit can act only in a chosen agent state, and it ships as a standard LoRA adapter. Before any test, the author writes down what the edit will do. We then check each edit against the unedited model and a random edit of the same size.

The edits give precise, dose-dependent control of agent decisions. A "submit sooner" edit raises the submit rate from 15% to 41%, while its random control leaves it unchanged. On development tasks, the best edit solved 30.3 tasks per run against 25.3 for the unedited model and led in every batch. A pre-registered test on held-out tasks and a pre-registered replication both gave gains in the same direction. We also measure how much a greedy agent varies from run to run, which sets the bar for any claimed gain. We compare Claude, GPT and Gemini as authors, and we release all edits, predictions, code and per-task results.

## 1. Introduction

The Gemma 4 Developer Agent competition (Markowitz et al., 2026) asks for a Gemma 4 (Gemma Team, 2026) agent that fixes real bugs on consumer hardware. Post-training costs GPU time and gives little control over what changes.

A frontier LLM can write the weights of a small model without gradients (Takehi et al., 2026). Activation-derived directions compiled into MLP weights cut overthinking in Qwen3.8-27B against a norm-matched random control (Bennington et al., 2026). Both evaluate on single-turn benchmarks.

An agent is harder. An edit must change a decision that recurs over up to 34 tool calls, and the outcome depends on all of them. Measurement is harder too. With greedy decoding, the unedited model solved 21 to 30 of the same 170 tasks across 21 runs.

**Contributions.**

1. To our knowledge, the first LLM-authored weight edits for a multi-step coding agent: closed-form, state-gated neurons served as LoRA adapters (Hu et al., 2021) (§2.2).
2. Dose-responsive decision control against matched random controls (§2.4).
3. A solve-rate gain and its statistical status (§2.5, §2.6).
4. Failure diagnosis and evaluation lessons for small agents (§2.3, §2.8).
5. A reusable resource: the edit builder, an episode/decision dataset with labelled activations, and three authors' scored prediction files (§2.5).

## 2. Description of research

### 2.1 Setting

**Model and agent.** Gemma 4 E4B has 42 layers and residual width n = 2560. We serve its INT4 checkpoint with vLLM. The agent calls six tools with greedy decoding, at most 34 tool calls and 7.5 minutes per episode. A task is solved when hidden tests pass. We edit the released model (the *unedited model*) and, in three experiments, our own small reinforcement-learning (RL) adapter.

**Tasks.** 170 *development tasks* (106 competition-format, 64 SWE-rebench; Badertdinov et al., 2025) and 130 *held-out tasks* from 78 repositories disjoint from them. Diagnosis used development episodes, so development gains are in-sample. We split logged decisions into training and held-out sets by episode.

**Unit of comparison.** An *arm* is one model variant (unedited, an edit or a control) run on every task. A *batch* is one run of every arm, the unedited model included, on all tasks at once. Greedy runs are not repeatable, so we compare arms within a batch (mean per-batch difference, 95% t CI, paired t test).

### 2.2 Method

**Locating a direction.** A binary label on logged decisions defines a direction. For "explore more before editing" the label is whether the next tool call makes the episode's first edit. At each layer, d is the difference of class means of the layer output over 400 training decisions per class; its norm is the *class gap*. We keep d only if it separates 160 held-out decisions per class: AUC 0.77, 0.81 and 0.83 at layers 18, 20 and 22, the layers we write.

**The hand-written neuron.** The competition scores a checkpoint served by a stock engine, so the edit must live in the weights. An MLP neuron computes a = gelu(gate · x̂) · (up · x̂), where x̂ is the normalised input; its down column writes a to the output, which is then RMS-normalised. We fit a ridge read-out q with q · x̂ ≈ 1 at every position and, in each layer, overwrite the quietest neuron:

- gate = G·q with G = 4, and up = q, so the neuron fires at a constant level;
- down = ± β · ρ · √n · û / gelu(G), where û is d mapped back through the output normalisation weight, ρ is the mean RMS of the natural MLP output, and the author chooses the sign.

If the natural output is orthogonal to û, the layer output moves along d by

Δ(β) = s · β / √(1 + β²) · √n · η,

with s the layer's output scalar and η a factor fixed by the normalisation weight. An edit of *strength* t sets β per layer so that Δ equals t class gaps. On held-out positions, the realised shift is 0.96 to 1.14 times this prediction (mean 1.06) for 12 always-on edit–layer pairs. A *gated* edit adds an agent-state read-out to the gate row, so the neuron fires mainly in that state. We size gated edits by bisection on the measured shift. A neuron edit in three layers changes nine tensors and is stored exactly (up to bf16 rounding) as a LoRA adapter.

**The compiled edit reproduces the intended intervention.** On 76 held-out decisions, for two edits, the adapter's residual-stream change has cosine 0.97 to 0.996 with that of a hook adding the same neuron write. The adapter–hook next-token KL (0.0001) is about 100 times smaller than the edit's own effect (0.011 to 0.013).

**Matched random control:** same neurons, gate, up row and norm; a random direction orthogonal to d.

**The loop** (Figure 1).

*Figure 1. The loop. Steps 1–4 and 6 run on a CPU.*

The author is an LLM with shell access to weights, stored activations and logged episodes. It (1) diagnoses a behaviour separating a task's solved and unsolved episodes, (2) locates its direction, (3) writes the edit and its control, (4) predicts the behaviour and solve-rate changes (80% interval) before any evaluation, (5) verifies by replaying held-out decisions per arm and running full episodes in one batch, and (6) scores the prediction. Steps 1–4 need no GPU.

### 2.3 Where the agent fails

- **Inconsistent, not incapable.** Over 13 unedited runs, a single run solves 22–30 tasks. 51 different tasks are solved at least once, and 9 in every run.
- **Failed edits go with lost episodes.** In those runs (2,210 episodes), 35% of 4,916 `edit_file` calls fail. Episodes with no failed edit solve 21%; those with at least one solve 4% (within-task odds ratio 0.42, 95% CI 0.23–0.74).
- **Exploring too long is lost.** Episodes with 8 or more calls before the first edit (17% of episodes) solve far less than the same task's other episodes (Mantel–Haenszel odds ratio 0.12, 95% CI 0.04–0.38; also 0.12 among episodes that made an edit). For 0–2 calls the odds ratio is below 1 and for 3–7 calls above it; all three intervals include 1 (0.76, 1.42, 1.72).

*Figure 2. Odds of solving by calls before the first edit (95% CI): only 8+ calls is clearly worse.*

### 2.4 Edits control decisions

*Figure 3. Held-out replay: movement toward each edit's target by strength (explore-first 0.125–0.5 class gaps, others 0.25–1); orange squares are matched controls.*

In replay (Figure 3), edits move their decisions; matched controls do not:

- **Submit sooner** (written on the RL adapter): the submit rate over 100 decisions rises from 0.15 to 0.20, 0.29 and 0.41; the control gives 0.15. In full episodes, mean tool calls fall from 20.8 to 14.2 at strength 0.5 and 8.8 at 1.0; the 0.5-matched control gives 19.8.
- **Think briefly**: mean thinking tokens over 320 decisions fall from 659 to 399; the control gives 649.
- **Explore more**: the next call's first-edit rate falls from 0.66 to 0.49; four control adapters give 0.66, 0.62, 0.64 and 0.66. The weakest strength is not separated from the lowest control.

**Across all rounds.** Against same-batch unedited runs, single-direction explore pushes moved calls before the first edit as predicted in 11 of 12 edits (sign test p = 0.003); composite early-push, late-pull edits went the wrong way in 5 of 5. Overall, 18 of 30 stated directions held. The author never reads weight values, and the one collapse (layer 14) was visible in its build report before serving.


### 2.5 Solve rate on development tasks

**Always-on explore-more.** In batches 3–15 (Figure 4), we pool strengths 0.125–0.5 per batch. The edit exceeds the unedited model by 2.7 tasks over 12 batches (95% CI +0.3 to +5.0, p = 0.03). It exceeds its matched controls by 3.6 over 7 batches (p < 0.001). Strength 1.0 overshoots (18 against 24 in its batch). The gain is +4.3 in the first six batches (p = 0.07) and +1.1 in the last six (p = 0.27).

*Figure 4. Solved development tasks per run: always-on explore-more, unedited model, matched controls.*

**Three LLM authors, one brief.** Claude Fable 5.2, GPT Astra 6 and Gemini 4 Carbon each received the edit-failure and always-on findings, stored activations and tools in a fresh session (90 CPU-only minutes, no evaluation access). Each delivered four edits, a random control and a prediction file. Each author's five adapters ran in one *screening batch*. We repeated four edits in *replication batches*. All three, given a brief that already favoured low-dose explore-more, gated or weakened it: Claude on a "before the first edit" read-out, GPT on a decision's first token, and Gemini by pairing it with concise thinking.

*Figure 5. Each repeated edit and each author's random control, per batch; Claude's gate is ahead in all six.*

| Edit (author) | Edit − unedited, all batches | Replication only | Edit − author's control |
|---|---|---|---|
| Before-first-edit gate (Claude) | +5.0 [1.6, 8.4], p = .014 (6) | +4.2, p = .033 (5) | +3.75, p = .11 (4) |
| Early-calls gate (Claude) | +4.7 [1.2, 8.2], p = .019 (6) | +4.4, p = .054 (5) | +3.5, p = .035 (4) |
| Gate + rotation (GPT) | +2.3 [−1.3, 5.9], p = .16 (6) | +2.0, p = .30 (5) | +2.0, p = .27 (4) |
| Explore + concise (Gemini) | +5.7 [−0.1, 11.4], p = .051 (3) | +5.0, p = .24 (2) | +2.7, p = .32 (3) |

*Mean per-batch difference in solved tasks of 170; 95% CI; uncorrected paired t test; number of batches in parentheses.*

Statistical status of Claude's before-first-edit gate:

- Selection: its screening difference (+9) was the largest of twelve (within-batch permutation null, p = 0.22).
- On the screening batches, Holm–Bonferroni over twelve edits raises its p from 0.020 to 0.24. No edit survives correction.
- Its author's control was ungated, not orthogonalised and sized to Claude's always-on edit at 0.125 gaps.

**Pass@k.** The unedited model has pass@1 15.2% and pass@10 29.0% over 21 runs. The gate gains +5.0 tasks at pass@1 (CI +1.3 to +8.8) and +4.0 at pass@6 (CI −4 to +12). Its +9 screening gain beat the author's 80% upper bound (+6). Overall 12 of 15 outcomes fell inside the authors' intervals, as a fixed ±6 interval would.


### 2.6 Two pre-registered held-out tests

We wrote each plan first, then ran three rollouts per task and arm on the 130 held-out tasks. We report differences in *tasks-equivalent* (solved rollouts / 3).

**Test 1: always-on explore-more, strength 0.25.** +0.33 tasks-equivalent over the unedited model (95% CI −6.6 to +7.3, p = 0.92; −0.33 when stratified by outage regime) and −1.0 against its matched control (p = 0.78). *Not confirmed.* After an outage, two side-by-side copies of the unedited model differed by 11.7 tasks-equivalent (p = 0.006).

**Test 2: before-first-edit gate.** We served four arms together, with Claude's ungated control. +3.0 over the unedited model (95% CI −5.7 to +11.7, p = 0.50) and +3.7 over its control (p = 0.39). *Not confirmed*; the sign matches development.

Each test had 80% power for gains of 10 to 12 tasks-equivalent, about three times the development gain. Both intervals contain zero and the development gain.

### 2.7 What did not work

Control of a decision was not enough to solve more tasks (one run per arm unless stated):

- On the RL adapter, submit-sooner solved 33, 26 and 27 against 36 and test-before-submitting 17 against 30; think-briefly solved 28, 28 and 25 against 29.
- Gradient-scored one-shot weight deletion (after Lu et al., 2026) scored below random deletions and no deletion (8–9 against 10–14 of 64).
- **Naive stacking collides.** All builders overwrite the quietest neurons, so summed stacks shared neurons and stopped submitting. Rebuilt on separate neurons, stacks led by at most 3 tasks, against +2 for their control.
- Our RL adapter (same batches, not served together) solved 32, 30 and 36 against 28, 24 and 22 unedited.

No direction or rule moved a frequent failure: edits that quote outdated file text. Several round-4 explore pushes hit the time limit more often.

**Later rounds, all with predictions and rules written before launch.** *Round 4* targeted the 8+-call loss. It tested 18 Claude edits ("commit late" neurons and variants) and two random controls. No edit met its pre-registered pass rule; the best averaged +5.0 tasks over two batches. In an interim pooled analysis, the best (+22 net over 678 task pairs, p = 0.005) misses Holm over 20 (0.087). Writing at layer 14 broke the model (1 of 170). *Round 5* (one exploratory batch, 52 tasks): stronger commit-late pushes solved fewer tasks (8, 7, 6 at 1×, 3×, 5×; unedited 10, control 11). *Round 6* replicated that best edit, pre-registered, in 20 new batches: +26 net tasks over 1,930 task pairs (105 won, 79 lost; p = 0.065, just short of α = 0.05), and +3 to +6 in each complete batch, as predicted. Its random control fell 36 below unedited.

*Figure 6. Round 4: each edit's two batches against the unedited model of the same batch; the best edits average +5 tasks.*

### 2.8 Generality, lessons and limitations

**What carries beyond this competition.** The loop needs open weights, labelled logged decisions and an engine serving LoRA adapters; no step is Gemma- or software-specific. It worked on two checkpoints (released, RL-tuned) and for three LLM authors. It is cheap. Authors finished 14 to 45 minutes into 90 CPU-only minutes, and an edit builds on a CPU in seconds to minutes. One arm on 170 tasks takes about 23 GPU-minutes (68 GPU-hours in all). The edit needs a quiet neuron; without Gemma's MLP output normalisation, as in most open models, its shift is exactly linear in the write size.

**Practical use.** Edits are post-deployment dials needing no retraining. Think-briefly cut thinking tokens per call by 10% and 20% (659 → 594, 524) while solving 28 and 28 against 29; submit-sooner cut tool calls per episode from 20.8 to 8.8, at a cost (26–27 solved against 36). Edits also tune a trained adapter. Small weight changes are cheap diversity for best-of-N with a verifier: edited, control or RL runs solved 28 of the 108 tasks never solved in 42 unedited runs; random controls add similar reach to edits (0.52 against 0.59 per run).

**Lessons** (for any change to a small agent). One batch cannot rank edits; compare only within a batch; match control gate, norm and orthogonality; drop the selecting batch; pre-register.

**Limitations.** We test one model and one harness with greedy decoding. The failure diagnosis is observational. Gates are not exclusive: the before-first-edit gate also shifts later prompt positions (−0.19 to −0.21 class gaps).

## 3. Related works

**LLM-authored weights.** Takehi et al. (2026) coined weight-space programming; we adopt the norm-matched control of Bennington et al. (2026). Mehta et al. (2026) have an LLM agent write small interpretable transformers and brain encoding models from scratch; we edit a pretrained agent sparsely.

**Activation steering.** Activation addition (Turner et al., 2023), contrastive activation addition (Panickssery et al., 2023), conditional steering (Lee et al., 2024; the hook-based counterpart of our gate) and TACT for coding agents (Sui et al., 2026) all need inference hooks.

**Direct weight editing.** ROME (Meng et al., 2022) and refusal-direction ablation (Arditi et al., 2024) replace or remove what weights write in closed form; our neuron adds a new write.

**Reproducibility.** Code, edit build reports, the authors' brief, prediction files, pre-registration texts and per-task results are released under Apache 2.0 at github.com/keithtyser/gemma-weight-programming.

## References

- Arditi et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.
- Badertdinov et al. (2025). SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. arXiv:2505.20411.
- Bennington, Kaelar, Lee et al. (2026). Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B. Working paper. concept.country/research/weight-space-programming
- Gemma Team (2026). Gemma 4 Technical Report. arXiv:2607.02770.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Lee et al. (2024). Programming Refusal with Conditional Activation Steering. arXiv:2409.05907.
- Lu et al. (2026). The Surprising Effectiveness of Deleting Weights in LLM Post-Training. arXiv:2611.10866.
- Markowitz et al. (2026). Google - The Gemma 4 Developer Agent Paper Track. Kaggle.
- Mehta, Singh, Xu et al. (2026). Interpretable Weight-Space Programming Builds Transparent Transformers and Brain Encoding Models. Preprint. github.com/csinva/weight-space-programming-autoresearch
- Meng et al. (2022). Locating and Editing Factual Associations in GPT. arXiv:2202.05262.
- Panickssery et al. (2023). Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681.
- Sui et al. (2026). TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering. arXiv:2605.05980.
- Takehi, Teoh, Zhang, Khattab (2026). Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights. Blog post. rikiyat.github.io/blog/weight-space-programming
- Turner et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
