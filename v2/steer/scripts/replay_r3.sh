#!/bin/bash
# Replay screen of the round-3 edits at held-out decision points right after a failed edit (rf@0)
# and at edit_file calls (eok@0). Reference = unedited base.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
B=/path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct
AD=$B,brf_c0.5,brf_c1.0,brf_c2.0,brfconst_t0.5,brfmatch_c1.0,beok_t0.5,beok_t1.0,bneg_t0.25
for t in rf eok; do
  $PY -m v2.steer.replay --states /dev/shm/ef/states_ef --out /dev/shm/ef/replay_r3_$t --url http://127.0.0.1:18000/v1 --adapters $AD --tag $t@0 --workers 16
done
echo REPLAY_R3_DONE
