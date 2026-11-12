#!/bin/bash
# Batches 14 (GPU0 / port 18001, after batch 11) and 15 (GPU1 / port 18002, after batch 12), base model:
# prompt rules compiled into one constant neuron at layer 18 (v2.steer.states8 contrast; mk_hint_edits.sh).
# hpe = edit-hygiene rule, hpx = explore-before-editing rule, hpb = both; tN = N times the mean shift the rule
# causes when it is in the system prompt; *match = matched random control.
. /path/to/gemma-v2/env.sh
E=$S/edits
go() { G=$1; P=$2; B=$3; shift 3; A="$*"
  for p in $(ps aux | grep "[a]pi_server.*--port $P" | awk '{print $2}'); do kill $p; done
  for i in $(seq 90); do m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
  L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
  cd $R; nohup ./serve_w.sh $G $P 0.90 $L > results/steer_eval/logs/serve_${P}_$B.log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -sf http://127.0.0.1:$P/v1/models > /dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$P/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
  ./run_steer_batch_p.sh $P $B 1 4 $A; echo "${B}_STARTED $(TZ=America/New_York date +%H:%M)"; }
after() { until [ $(ps aux | grep "[g]olden_ab" | grep -c "steer_eval/$1_") -eq 0 ]; do sleep 60; done; }
(after b11; go 0 18001 b14 base bneg_t0.5 hpe18_t1 hpe18_t3 hpe18_t6 hpematch_t3 hpx18_t3 hpb18_t3) &
(after b12; go 1 18002 b15 base bneg_t0.25 hpx18_t1 hpx18_t3 hpx18_t6 hpxmatch_t3 hpe18_t3 hpb18_t6) &
wait
