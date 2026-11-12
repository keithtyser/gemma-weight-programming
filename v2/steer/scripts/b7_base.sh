#!/bin/bash
# Batch 7 (base model): second replicate of batch 6 arms on GPU1 / port 18002, started once batch 5 has finished there.
. /path/to/gemma-v2/env.sh
cd $R
while [ $(ps aux | grep "[g]olden_ab" | grep -c "b5_") -gt 0 ]; do sleep 60; done
echo "B5_DONE $(date -u +%H:%M)"
kill 3017524; for i in $(seq 60); do m=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
E=$S/edits
A="base bneg_t0.125 bneg_t0.25 bneg_t0.5 bnm25_s0 bnm25_s1 bnm25_s2 bnegmatch_t0.5"
L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
nohup ./serve_w.sh 1 18002 0.90 $L > results/steer_eval/logs/serve_18002_b7.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:18002/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:18002/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
./run_steer_batch_p.sh 18002 b7 1 4 $A
echo B7_STARTED
