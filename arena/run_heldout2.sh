#!/bin/bash
# Held-out confirmation v2 (results/steer_confirm/heldout_v2_prereg.md): start one vLLM server with all arms on one
# evalhost GPU, then one confirm run per arm (130 tasks, three rollouts).
# Usage: run_heldout2.sh GPU PORT TAG arm [arm ...]     ("base" = the zero adapter; outputs heldout_v2_<arm>_<TAG>)
. /path/to/gemma-v2/env.sh
G=$1; P=$2; T=$3; shift 3
E=$S/edits
C=/path/to/gemma-v2/results/steer_confirm
L=""; for a in "$@"; do p=$E/$a; [ $a = base ] && p=$E/zero; [ -d $p ] || { echo "missing $p"; exit 1; }; L="$L $a=$p"; done
curl -sf http://127.0.0.1:$P/v1/models > /dev/null && { echo "port $P already serving"; exit 1; }
m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
[ $m -lt 3000 ] || { echo "GPU $G busy ($m MiB)"; exit 1; }
cd $R; nohup ./serve_w.sh $G $P 0.90 $L > $C/logs/serve_${P}_v2_$T.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:$P/v1/models > /dev/null && break; sleep 5; done
ids=$(curl -sf http://127.0.0.1:$P/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " ")
echo "served: $ids"
for a in "$@"; do echo "$ids" | grep -q "\"$a\"" || { echo "arm $a not served"; exit 1; }; done
cd /path/to/gemma-v2/tii-steer
for a in "$@"; do
  o=$C/heldout_v2_${a}_$T
  [ -e $o ] && { echo "exists $o"; continue; }
  nohup /path/to/gemma-v2/eval-venv/bin/python -m v2.steer.confirm run --data /path/to/gemma-v2/eval_data/steer_heldout_v1 --split all \
    --served $a --api-base http://127.0.0.1:$P/v1 --harness /path/to/gemma-v2/harnesses/steer_ext_w5 \
    --out $o --rollouts 3 --concurrency 10 > $C/logs/heldout_v2_${a}_$T.log 2>&1 < /dev/null &
  echo "started $a pid=$!"
  sleep 20
done
echo HELDOUT2_${T}_STARTED
