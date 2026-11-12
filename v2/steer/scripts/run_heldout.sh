#!/bin/bash
# Pre-registered held-out confirmation (results/heldout_v1_prereg.md): one confirm run per arm, three rollouts per task.
# Usage: run_heldout.sh PORT OUTNAME:SERVED ...   (PORT = local end of the tunnel to the serving GPU)
P=$1; shift
cd /path/to/gemma-v2/tii-steer
R=/path/to/gemma-v2/results/steer_confirm
mkdir -p $R/logs
for x in "$@"; do
  o=${x%%:*}; s=${x##*:}
  [ -e $R/heldout_v1_$o ] && { echo "exists $o"; continue; }
  nohup /path/to/gemma-v2/eval-venv/bin/python -m v2.steer.confirm run --data /path/to/gemma-v2/eval_data/steer_heldout_v1 --split all \
    --served $s --api-base http://127.0.0.1:$P/v1 --harness /path/to/gemma-v2/harnesses/steer_ext_w5 \
    --out $R/heldout_v1_$o --rollouts 3 --concurrency 10 > $R/logs/heldout_v1_$o.log 2>&1 &
  echo "started $o served=$s port=$P pid=$!"
  sleep 20
done
