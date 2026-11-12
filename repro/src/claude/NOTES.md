# Notes (claude)

All five adapters rewrite one quiet MLP neuron in each of layers 18, 20, 22 (gate, up, down only; q_proj and o_proj are zero). No gradient steps or fitting to task outcomes: only ridge readouts of stored activations and a class-mean difference. CPU, about 40 s each.

## What I measured (code/diag.py, code/diag2.py, logs in work/)

- 13 unedited runs, 2210 episodes, 15.5% solved. 35.1% of edit_file calls fail.
- Calls before the first edit versus solving, within task: 0-2 calls OR 0.76, 3-4 OR 1.42, 5-7 OR 1.72, 8+ OR 0.12.
- Re-reading after a failed edit is not a lever: the next edit still fails 69% of the time with a re-read and 82% without, and episodes with any failed edit rarely solve (1.6-5.8%) whatever the next call is.
- Earlier always-on "explore more" pushes: 0.125 gap solved 17.9%, 0.25 17.1%, 0.5 17.0%, 1.0 about 10.6%. Larger doses cut edit failures but push episodes into the 8+ bucket.

## The edits

- `xp_const`: always-on, 0.125 gap per layer. The best-evidenced dose, rebuilt.
- `xp_pre`: 0.25 gap, only before the first edit.
- `xp_think`: 0.25 gap, only at generated positions.
- `xp_early`: about 0.5 gap at calls <= 4, zero from call 8.
- `xp_const_ctrl`: xp_const with a random output direction, matched on mean residual change.

Each edit's `report.json` holds gate quality and the shift recomputed from the saved bf16 tensors. Build commands are in PROPOSALS.json (`xp_const` before `xp_const_ctrl`).

## Caveats

- The odds ratios are correlational: hard tasks may cause long exploration.
- The gain for the 0.125 dose came mostly from four early batches; later batches showed about zero. Hence small point estimates and wide intervals.
- `xp_pre` and `xp_early` still fire at post-edit prompt positions (shift about -0.2 and -0.1 gaps), which were not in the fit.
- `xp_early` changes the residual by up to 14% of its norm at early calls; it is the most likely to hurt.
- The control is matched on mean residual change, not on output-column norm (a random column of equal norm changed the residual 4 times less).
- I dropped a planned "re-read after failed edit" edit: the `rr` labels are read-versus-run before the first edit, and no activations exist for post-failure states.
- No attention edits: no stored attention data to derive one from.
