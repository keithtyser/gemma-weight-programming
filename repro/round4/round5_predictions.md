# Round 5 (commit-late dose and timing ladder) — predictions, written before launch

Written 12 Nov 2026, 2:32 PM ET, before any round-5 episode ran. Exploratory, one batch (b59), author `claude`.

## Why

Unedited episodes with 8 or more calls before the first file edit solve 2.7% of the time (17% of episodes; 13 runs,
observational). The `late` neuron pushes towards the first edit once many calls have passed without one. In round 4
so far `ac3_late60` is +5 on 73 common tasks in b36 (its random-direction family control +4) and 0 on 57 in b46.
Round 5 asks whether a stronger or earlier push does more, and adds the control matched to `late60` itself.

## Arms (b59, evalhost GPU 0, sandboxes on evalhost, 170 dev tasks, 1 rollout, concurrency 5 per arm)

| arm | construction (layers 18, 20, 22) |
|---|---|
| base | unedited |
| ac3_late60 | round-4 edit, repeated as the anchor |
| ac5_late90 | class-gap target 0.9 (solved write sizes 0.31 / 0.30 / 1.28) |
| ac5_late60_x3 | write sizes 3x those of late60 (0.61 / 0.56 / 1.79) |
| ac5_late60_x5 | write sizes 5x those of late60 (1.01 / 0.94 / 2.98) |
| ac5_late60_c6 | target 0.6, push from call 6 (gate: on at 6+, off at 3 or fewer) |
| ac5_late100_c6 | target 1.0, push from call 6 |
| ac5_late60_ctrl | random write direction, same mean residual change as late60 |

Build note: targets 1.2 and 2.0 and a 5-layer variant (16-24) were built first and discarded before serving: the
class-gap shift saturates at layer 22 (and at 16, 24), so the solver ran to its cap (write size about 1000).
`late60_x3` and `late60_x5` replace them with write sizes fixed as multiples of the solved `late60` sizes.

## Predictions (tasks solved minus same-batch unedited, 170 tasks)

- ac3_late60: 0 to +5. ac5_late90: 0 to +5. ac5_late60_c6: 0 to +5. ac5_late100_c6: -2 to +6.
- ac5_late60_x3: -3 to +5. ac5_late60_x5: -8 to +4 (most likely harmful: premature edits).
- ac5_late60_ctrl: -2 to +2.
- Behaviour: every non-control arm lowers the share of episodes with 8+ calls before the first edit relative to
  base, more with dose; the c6 arms also lower the 5-7 share; the control changes neither.
- Ceiling reasoning: fully repairing the 8+ bucket is worth about 3-4 tasks of 170 if the association were causal,
  so a gain of 8 or more would mean the edit helps beyond that bucket.

## Rule (fixed now)

An arm is called a pass only if it gains at least 8 over same-batch unedited and at least 5 more than
ac5_late60_ctrl gains. A pass is replicated in a second batch before any held-out run. Otherwise round 5 is
reported as a dose-response result with all rows, on the tasks completed by every arm if the batch is incomplete
at the paper deadline. Eight arms in one exploratory batch: any p-value is reported uncorrected and labelled so.
