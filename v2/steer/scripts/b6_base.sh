#!/bin/bash
# Batch 6 (base model): replication of explore-more with three seeded matched controls and a dose point. GPU0 / port 18001.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
kill 3017523; for i in $(seq 60); do m=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
E=$S/edits
mk() { [ -e $E/$1 ] || CUDA_VISIBLE_DEVICES=0 $PY -m v2.steer.neuron --adapter $E/zero --acts $S/acts_based/acts_base.pt --tags edit@0 --layers 18,20,22 --beta 0.6 --mode const --sign -1 "${@:2}" --out $E/$1 || exit 1; }
mk bneg_t0.125 --target 0.125
mk bnm25_s0 --control match --target 0.25 --seed 0
mk bnm25_s1 --control match --target 0.25 --seed 1
mk bnm25_s2 --control match --target 0.25 --seed 2
A="base bneg_t0.125 bneg_t0.25 bneg_t0.5 bnm25_s0 bnm25_s1 bnm25_s2 bnegmatch_t0.5"
L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
cd $R; nohup ./serve_w.sh 0 18001 0.90 $L > results/steer_eval/logs/serve_18001_b6.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:18001/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:18001/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
./run_steer_batch_p.sh 18001 b6 1 4 $A
echo B6_STARTED
