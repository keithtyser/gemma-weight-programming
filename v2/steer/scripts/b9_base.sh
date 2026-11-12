#!/bin/bash
# Batch 9 (base model): within-task success direction at three strengths with a matched control, "quote less text"
# with a matched control, explore-more 0.5 again. GPU1 / port 18002. Swaps the idle batch-7 server.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
while [ $(ps aux | grep -c "[g]olden_ab.*steer_eval/b7_") -gt 0 ]; do sleep 30; done
for p in $(ps aux | grep "[a]pi_server.*--port 18002" | awk '{print $2}'); do kill $p; done
for i in $(seq 90); do m=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
E=$S/edits
A="base bneg_t0.5 bsuc_t0.5 bsuc_t1.0 bsuc_t2.0 bsucmatch_t1.0 bse_t1.0 bsematch_t1.0"
L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
cd $R; nohup ./serve_w.sh 1 18002 0.90 $L > results/steer_eval/logs/serve_18002_b9.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:18002/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:18002/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
./run_steer_batch_p.sh 18002 b9 1 4 $A
echo B9_STARTED
