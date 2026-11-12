# Programming a Coding Agent's Weights by Hand

Code and data to reproduce the results of the paper *Programming a Coding Agent's Weights by Hand*
(`paper/paper.pdf`, LaTeX source in `paper/`; the Kaggle Writeup text is `paper/kaggle_writeup.md`).

An LLM diagnoses the failures of a Gemma 4 E4B software agent, writes closed-form weight edits with no training,
predicts their effect, and checks each edit against the unedited model and norm-matched random controls. Every edit
is a standard PEFT LoRA adapter over the INT4 base model.

## Requirements

- Python 3.10+. All statistics, tables and the diagnosis recompute are pure Python (standard library only).
- Figures: `matplotlib`, `numpy`; the Kaggle cards also need `Pillow`.
- Paper: a LaTeX distribution with `pdflatex` and `bibtex` (the ICLR style files are included).
- Building edits: `torch`, `safetensors` and the Gemma 4 E4B INT4 checkpoint (CPU is enough).
- Running agent evaluations: a CUDA GPU, `vllm` (serving the edit as a LoRA module), Docker and the task images
  pinned in `v2/experiments/splits/golden_v2_manifest.json`.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Final LaTeX source, `paper.pdf`, `figures/` (figure files and the scripts that draw them), `kaggle_writeup.md`. |
| `v2/steer/` | Edit construction (`neuron.py`, `linmap.py`, `fold.py`), decision mining and directions (`mine.py`, `states*.py`, `directions.py`, `aucscan.py`), activation collection (`collect.py`), held-out replay (`replay.py`, `behav.py`, `probe.py`), forensics, and the exact build/serve/batch scripts (`scripts/`). |
| `v2/eval/`, `v2/tii/`, `v2/curation/`, `v2/harnesses/`, `v2/train/` | Evaluation runner, exact-INT4 Gemma 4 forward used by `collect.py`, task-image verification and the modules they import. |
| `harnesses/steer_ext_w5/` | The agent harness used for every run (system prompt, tools, greedy decoding, budgets). |
| `arena/` | Rounds 1-3 with three LLM authors: author brief, build and scoring scripts, held-out v2 pre-registration and analysis. |
| `analysis/` | Batch-level and task-paired tests on `results/runs.tsv`. |
| `results/` | `runs.tsv` (one row per run and task), the build report of every edit (`edit_reports/`), held-out v1 task ids and pre-registration. |
| `repro/src/` | Per-task and per-rollout result tables and the saved outputs every statistic is computed from. |
| `repro/scripts/`, `repro/checks/` | Statistics, tables and number checks behind the paper. |
| `repro/diag_recompute/` | Independent recompute of the failure diagnosis from the saved unedited-model traces (`raw/`). |
| `repro/d1/` | Compiled adapter against activation hooks (teacher-forced and regenerated decisions). |
| `repro/interp/`, `repro/novel/`, `repro/analyses/` | Behaviour/score correlations, novel-task solves and pass@k, selection null, predicted-vs-realised shift, compute accounting. |
| `repro/round4/` | Pre-registered predictions and decision rules for rounds 4-7 with their dated amendments. |

Run names are `<batch>_<arm>`: `base` is the unedited model, `bneg_t<s>` the explore-more edit at strength `s`,
`bnm25_s*` and names containing `match` or `ctrl` are matched random controls.

## Reproducing the results

All commands below run from the repository root unless a `cd` is shown; outputs are compared against the saved
files next to them.

### Diagnosis tallies (Section 3, odds ratios by calls before the first edit and failed edits)

```
cd repro
python -I diag_recompute/recompute.py diag_recompute       # -> diag_recompute/result.json (see RESULT.md)
python -I diag_recompute/theirs/diag.py                     # the authoring model's own tally, for comparison
python -I checks/m4_8plus_edited_only.py diag_recompute
```

### Replay dose-response (held-out decisions, Figure 2 and the dose table)

```
cd repro
python -I scripts/shift_check.py <dir with the arena edits' build reports>   # -> src/shift_check.json
python -I checks/m2_eta.py                                                    # write-size saturation, reads results/edit_reports
cd d1 && python d1_summarize.py <rows.jsonl> d1_fidelity_meta.json out.json  # adapter vs hooks; saved: d1_fidelity_results.json
```

The replay rows themselves are extracted from the logged agent calls by `v2/steer/replay.py` and
`repro/scripts/remote_diag.py` (saved output: `repro/src/remote_diag.json`); `repro/d1/vllm_ref/` holds the
regenerated decisions for the base model, the edit and its control.

### Solve-rate statistics (development batches 3-16, Figure 3)

```
python analysis/batch_level.py results/runs.tsv
python analysis/paired_sign.py results/runs.tsv b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15
cd repro
python -I scripts/consistency.py src/dev_runs_b3_b16.tsv
python -I scripts/passk.py                                  # -> src/passk.json
python -I scripts/make_schedule_table.py src/dev_runs_b3_b16.tsv src/arena_per_task_b21_b28.tsv /tmp/app_schedule.tex
```

### Benchmark statistics (arena batches 21-28, Figure 4, statistics and cost tables)

```
cd repro
python -I scripts/review_stats.py                           # -> src/review_stats.json
python -I scripts/stats.py src/arena_per_task_b21_b28.tsv src/heldout_v2_per_rollout.tsv
python -I scripts/batch_sd.py src/arena_score_b21_b28.json
python -I scripts/selection_null.py                         # -> src/selection_null.json
python -I checks/exp_checks.py
python -I checks/m31_table10_cells.py
python -I checks/m34_table11_cells.py
```

### Held-out tests

```
cd repro
python -I scripts/power.py src/heldout_v2_per_task.tsv      # power of the pre-registered design
python -I checks/m10_regime_stratified.py                   # held-out v1, before/after the server outage
```

Pre-registrations: `results/heldout_v1_prereg.md`, `arena/heldout_v2_prereg.md`; the registered analysis is
`arena/heldout2_analyze.py` (saved output `arena/heldout_v2_analysis.json`).

### Round 4 and later rounds

Predictions were written before evaluation and are in `repro/round4/`. Scores against the same batch's unedited
arm, per batch and pooled with Holm correction:

```
python repro/scripts/r4_score.py <results root> <batches...>
python repro/scripts/r4_pool.py <results root> --json r4.json <batches...>
```

These two read the per-run result directories written by the evaluation runner; `repro/checks/r4_limit_calib.out`
holds the saved session-limit calibration.

### Figures

```
cd paper/figures
python make_figures_v2.py      # fig1-fig4, fig6 (PDF/PNG/SVG) used by the LaTeX paper
python make_fig5.py            # fig5_invu
python make_figures_web.py     # web_fig*.png for the Kaggle Writeup
python make_cards.py           # card_fig*.png
```

### Paper

```
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## Building an edit and running an evaluation

`v2/steer/neuron.py` and `v2/steer/linmap.py` need only PyTorch, safetensors and the base checkpoint; they run on CPU
and write an adapter directory (`--control` builds the norm-matched random control). The commands behind every edit
are in `v2/steer/scripts/build_r*.sh`, `mk_hint_edits.sh`, `mk_lin_edits.sh`, `arena/build_r2.py` and
`repro/scripts/build_r3.py`, `build_r5.py`. One evaluation run is

```
python -m v2.eval.golden_ab --golden v2 --arm harnesses/steer_ext_w5 \
  --external-api-base http://127.0.0.1:<port>/v1 --served-adapter <edit> --out <dir> --seeds 1 --keep-images
```

against a local vLLM server that serves the edit as a LoRA module (serve commands in `v2/steer/scripts/`).
Paths such as `/path/to/gemma-v2` in the scripts stand for the project root and must be set for your machine.

## Not included

- The task data: the competition tasks come from the Kaggle Gemma 4 Developer Agent public data and the SWE-rebench
  tasks from their public images; the manifest pins both by hash.
- The edit adapters themselves (they are rebuilt from the scripts; every build report is in `results/edit_reports/`).
- The full per-run trace directories; per-task outcomes of every run are in `results/runs.tsv` and `repro/src/`.
- The RL trainer and the RL-trained reference adapter.

## Licence

Apache 2.0.
