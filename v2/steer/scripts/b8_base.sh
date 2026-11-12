#!/bin/bash
# Batch 8 (base model): think-briefly dose (0.25 / 0.5 / 1.0) with a matched control, explore-more 0.5 again, and
# two stacks of the two edits. GPU0 / port 18001. Waits for batch 6 to finish, then swaps the server.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
while [ $(ps aux | grep -c "[g]olden_ab.*steer_eval/b6_") -gt 0 ]; do sleep 30; done
for p in $(ps aux | grep "[a]pi_server.*--port 18001" | awk '{print $2}'); do kill $p; done
for i in $(seq 90); do m=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
E=$S/edits
[ -f $E/bn50_th50/adapter_config.json ] || $PY -m v2.steer.neuron --acts $S/acts_based/acts_base_rl.pt --mode const --adapter $E/bneg_t0.5 \
  --tags th@0 --layers 24,26,28 --beta 0.6 --sign -1 --target 0.5 --out $E/bn50_th50 > $E/bn50_th50.log 2>&1 || exit 1
A="base bneg_t0.5 bth_t0.25 bth_t0.5 bth_t1.0 bthmatch_t0.5 bn50_th50 bcombo_n25_th50"
L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
cd $R; nohup ./serve_w.sh 0 18001 0.90 $L > results/steer_eval/logs/serve_18001_b8.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:18001/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:18001/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
./run_steer_batch_p.sh 18001 b8 1 4 $A
echo B8_STARTED
