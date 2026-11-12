#!/bin/bash
# As run_steer_batch.sh but against a chosen local port. Usage: run_steer_batch_p.sh PORT BATCH SEEDS CONC name [name ...]
R=/path/to/gemma-v2; P=$R/tii-venv/bin/python; PORT=$1; B=$2; N=$3; C=$4; shift 4
cd $R/tii-steer
for a in "$@"; do
  O=$R/results/steer_eval/${B}_$a
  if [ -e $O ]; then echo "exists: $O"; continue; fi
  mkdir -p $R/results/steer_eval/logs
  nohup $P -m v2.eval.golden_ab --golden v2 --arm $R/harnesses/steer_ext_w5 --external-api-base http://127.0.0.1:$PORT/v1 \
    --served-adapter $a --out $O --seeds $N --concurrency-per-gpu $C --keep-images > $R/results/steer_eval/logs/${B}_$a.log 2>&1 < /dev/null &
  echo "started $a"
done
