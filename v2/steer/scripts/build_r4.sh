#!/bin/bash
# Round 4: within-task success direction (v2.steer.states7), always on. Fit on pool tasks only.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=/dev/shm/e4; Z=/dev/shm/ef/edits/zero; A=/dev/shm/suc/acts/acts_base.pt
mkdir -p $E
b() { n=$1; shift; $PY -m v2.steer.neuron --base $R/models/gemma-4-e4b-it-qat-w4a16-ct --acts $A --out $E/$n "$@" > $E/$n.log 2>&1 && echo "built $n" || echo "FAILED $n"; }
SU="--tags suc@0 --layers 24,26,28 --beta 0.6 --sign 1 --mode const"
for t in 0.5 1.0 2.0; do b bsuc_t$t --adapter $Z $SU --target $t; done
b bsucmatch_t1.0 --adapter $Z $SU --target 1.0 --control match --seed 0
echo BUILD_R4_DONE
