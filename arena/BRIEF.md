# Weight-space programming arena: brief for the author model

You are one of three LLM authors (the others get this same brief and the same budget). Your job: **improve the
task solve rate of a small coding-agent LLM by writing closed-form edits to its weights. No training, no gradient
steps, no optimisation against task outcomes.** You diagnose a failure from logged data, locate where the model
represents the relevant decision, write the edit by hand (code that computes it from weights and stored
activations), and predict its effect before anyone measures it. The edits are evaluated for you afterwards.

## Target model and task
- Gemma 4 E4B (int4 QAT checkpoint), 42 layers, residual width 2560, MLP 2560 -> 10240.
  Base weights: `/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct/model.safetensors`
  (dequantise with `v2.steer.fold.dequant_int4_bf16`; see `v2/steer/colscale.py` for a complete worked example).
- MLP of a layer: `a_j = gelu_tanh(gate_j . xh) * (up_j . xh)`, `y = sum_j down[:, j] a_j`,
  `h_out = layer_scalar * (x + w_post * y / rms(y))`. The MLP output is RMS-normalised, so only its direction counts.
- The model runs as a tool-calling agent (read_file, search, list, edit_file, run tests, submit) on 170 real
  repository bug-fix tasks, greedy decoding, with thinking. A task is solved if the hidden tests pass.
- Unedited model: about 26 of 170 solved per run (13 runs, mean 26.3, range 22-30; run-to-run noise is large).

## Delivery format (hard constraint)
An edit is a PEFT LoRA adapter with the config of the zero adapter
`/path/to/gemma-v2/results/steer/edits/zero/` (rank 16, alpha 32) on `self_attn.q_proj`, `self_attn.o_proj`,
`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` of every language-model layer. You may change the rank r to any
value up to 32 (one r for the whole adapter) as long as lora_alpha = 2 r, so delta W = 2 * B @ A. So every edit is
a rank <= 32 additive change per module. Anything expressible that way is allowed
(new neurons, rescaled neurons, direction ablations, attention changes, combinations).

## What you have (all on host `evalhost`, reach it with `ssh evalhost '<command>'`)
- Tool library (read-only, do not modify): `/path/to/gemma-v2/tii-steer/v2/steer/*.py`. Every file starts with a
  docstring saying what it does and how to call it. Most useful: `colscale.py` (rescale existing neurons),
  `neuron.py` (add hand-written neurons), `linmap.py` (compile a prompt rule into linear neurons), `fold.py`
  (direction ablation folded into LoRA), `scan_unembed.py`, `neuron_trace.py`, `directions.py`, `aucscan.py`,
  `probescan.py`, `states*.py` (how labelled positions were chosen), `editfail.py`, `diverge.py`, `endfeat.py`,
  `tracefeat.py`, `behav.py`, `batchstats.py`.
  Run as: `cd /path/to/gemma-v2/tii-steer && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=. /path/to/gemma-v2/tii-venv/bin/python -m v2.steer.<tool> ...`
- Stored activations of the unedited model at labelled decision positions (residual `h`, MLP input `xin`, `yrms`,
  `meta` with tag / label / split): `/path/to/gemma-v2/results/steer/acts_based/acts_base.pt`.
  Other labelled state sets and direction files: `/path/to/gemma-v2/results/steer/{states*,dirs,acts*,mine_base,diverge_base}`.
- Logged agent runs (traces and outcomes) of the unedited model and of every earlier edit:
  `/path/to/gemma-v2/results/steer_eval/<batch>_<arm>/`. Note each run dir holds TWO result sets with different
  schemas (`comp_s0/shard_0/task_results.jsonl`, about 106 tasks, and rebench rows elsewhere, about 64 tasks);
  key on `resolved`, not on patch-size fields.
- Earlier edits (adapters + reports): `/path/to/gemma-v2/results/steer/edits/`.

## What is already known (lab notebook; solved counts are out of 170, one run each)
Failure picture of the unedited model: solved ~25; quick confident wrong patch ~116; timeout ~6; no patch ~5.
- 32% of edit_file calls fail (stale or inexact quoted text). Episodes with zero failed edits solve 21%, with one
  or more 4%. The Nth edit of an episode fails 14 / 39 / 49 / 60%.
- Greedy runs of the same task diverge early (inside thinking, tool call 0-3). 119 tasks never solved, 9 always,
  42 sometimes: making the sometimes-tasks reliable is worth up to about +25.
- Episode endings: tests ran and passed locally 64% (solve rate .24), never edited 17% (.05), tests errored 16% (.12).
Edits tried so far:
- **Explore more before the first edit** (`bneg`: ablate / push against the edit-vs-keep-reading direction at the
  decision position). The only edit with a repeatable gain: 21 runs mean 29.3 vs unedited 26.3 vs matched random
  controls 25.6; +2.7 per batch over unedited (n=12, p~.03), +3.6 over control (7/7). The gain shrank in later
  batches. Reversed sign (edit sooner) hurts badly (12 vs 28).
- Think briefly: 28/28/25 vs 29. Success direction (AUC .61): 30/25/25 vs 27. Verify more before submit: 24.
  Submit sooner (on an RL checkpoint): no gain.
- Prompt rules ("read the file again before editing", etc.) in the system prompt: 25 / 29 / 30 vs 30. The same
  rules compiled into the weights as one constant neuron (R2 .19-.25) or a rank-16 linear map (R2 .41-.53): no gain.
- Decision-gated neuron rescaling (`colscale.py`, `cg_*` edits; neurons in layers 8-24 that fire at the
  edit decision and write along the decision direction): being measured now, no result yet.
Not yet tried: an edit that makes the model re-read a file after a failed edit_file; an edit targeted at the
42 unreliable tasks; attention (q_proj / o_proj) edits of any kind; stacks of two edits; anything you think of.

## Your budget and deliverables
- Wall time: 90 minutes. CPU only: never use a GPU, never start a model server, never launch an evaluation.
- Write ONLY under `/path/to/gemma-v2/wsp_arena/<AUTHOR>/` (your author name is given in the prompt). Do not read
  the other authors' directories. Do not modify or delete anything else on the host. Do not kill or signal any
  process you did not start. Do not read `~/.docker`, `~/.kaggle`, `~/.trainingjobs`, `/etc/fleet`, or any key file.
  Keep memory under 40 GB and threads at 8.
- Deliver in your directory:
  1. `edits/<name>/` : exactly **4 edits** aimed at raising the solve rate, plus **1 control** named
     `<name>_ctrl` for the edit you believe in most (same construction and size, but aimed at a random or
     irrelevant direction / neuron set, so it tests whether your targeting matters). Each must pass
     `/path/to/gemma-v2/tii-venv/bin/python /path/to/gemma-v2/wsp_arena/check_adapter.py edits/<name>`.
  2. `code/` : the scripts that build each edit from the base weights and stored data (reproducible, one command each).
  3. `PROPOSALS.json` : a list of 5 objects with keys `name`, `rank` (1 = most confident; control gets 0),
     `diagnosis` (the failure you target and the evidence, with numbers you measured),
     `mechanism` (what the edit changes, which layers / neurons / directions, how they were chosen),
     `predicted_behaviour` (a measurable change in agent behaviour, with a number, e.g. "tool calls before the
     first edit rise from 6 to 8"), `predicted_solved_delta` (your point estimate of solved tasks out of 170
     relative to the unedited model, and an 80% interval), `build_command`.
  4. `NOTES.md` : under 400 words, what you looked at and why you chose these edits.
- You are scored on (a) solved tasks of your edits against the unedited model run in the same batch and against
  your control, and (b) how well your predictions matched what was measured. Honest predictions matter: a
  confident wrong prediction scores worse than a calibrated one.
When your five adapters pass the check and the three files are written, reply with a one-paragraph summary and stop.
