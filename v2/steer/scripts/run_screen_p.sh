#!/bin/bash
# Subset screening batch against a served port. Usage: run_screen_p.sh PORT BATCH CONC TASKS name [name ...]
R=/path/to/gemma-v2; P=$R/tii-venv/bin/python; PORT=$1; B=$2; C=$3; T=$4; shift 4
cd $R/tii-steer
mkdir -p $R/results/steer_eval/logs
for a in "$@"; do
  nohup $P -m v2.steer.screen --port $PORT --arm $a --out $R/results/steer_eval/${B}_$a --tasks $T --conc $C \
    > $R/results/steer_eval/logs/${B}_$a.log 2>&1 < /dev/null &
  echo "started $a"
done
