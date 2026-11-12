# Round 7 (breadth: 30 long shots, one batch each) — written before any episode, 12 Nov 2026, 4:38 PM ET

Author: Claude (second session). Keith asked for breadth over depth: many untested edits, one batch each, looking
for a gain of 10 or more tasks. No new directions: every edit is a dose-scaled copy or a collision-free stack
(`wsp_arena/stack2.py`, scale multiplies the write only) of edits that already exist. Built by `build_r7.sh`;
all 30 pass `check_adapter.py`.

Why these: in round 4 the lowest dose of the gate was the best arm and the highest dose was flat, the same gate on
layers 20-24 was second, and the gate's matched control was flat. So: lower doses, the two best gates together on
different layers, the gate plus the "commit late" neuron, and the gate plus another author's edit.

| batch (5090 GPU) | arms (each batch also has `base`) |
|---|---|
| b81 (0) | s7_lo50, s7_lo75, s7_lo125 (pe_lo at 0.5x, 0.75x, 1.25x), s7_pre15_60 (pre15 at 0.6x), s7_ctrl50 (pe_ctrl at 0.5x) |
| b82 (1) | s7_L20_50, s7_L20_75 (pe_L20 at 0.5x, 0.75x), s7_L5_50 (pe_L5 at 0.5x), s7_c6_60 (pe_c6 at 0.6x), s7_L5_50_L20_50 |
| b83 (3) | s7_lo_L20, s7_lo50_L20_50, s7_lo_L20_50, s7_lo75_L20_75 (pe_lo + pe_L20 at the stated scales), s7_ctrl_stack (pe_ctrl + u33_ctrl 0.5x) |
| b84 (4) | s7_lo_late50, s7_lo_late (pe_lo + late30 at 0.5x, 1x), s7_L20_late50, s7_lo_u63_50, s7_u63_50 |
| b85 (5) | s7_lo_D50, s7_lo_D (pe_lo + Gemini's explore+concise at 0.5x, 1x), s7_L20_D, s7_lo_L20_D50, s7_D50 |
| b86 (7) | s7_lo_H, s7_lo_W (pe_lo + Gemini's two other edits), s7_H50, s7_W50, s7_u36_50 |

Setup: 170 dev tasks, 1 rollout, concurrency 8 per arm (48 per server), servers on the 8x5090 box (CUDA stack),
sandboxes on the AMD node, each edit compared only with the `base` of its own batch. To free the six servers,
round-6 batches b66-b71 are stopped unread (their rows are not looked at before the stop).

Predictions (solved minus same-batch unedited, one batch): every arm between -4 and +6; the two controls between
-3 and +3. I expect no arm at +10. One batch of the unedited model has ranged over 21-30 solved, so a single +10
is about the size of that spread.

What counts (fixed now): this round is exploratory and selects nothing by itself. An arm at +10 or more in its one
batch is reported as "+N in one unreplicated batch" with the count of arms tested (30). If time allows, any arm at
+10 or more is rerun in a second batch on a different server before 6 PM, and only an arm at +10 or more in the
mean of two batches with at least +6 in each is described as a candidate gain; even that is a dev-set result with
no held-out test. Session-limit rows are reported per arm.
