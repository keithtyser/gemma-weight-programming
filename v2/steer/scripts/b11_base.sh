#!/bin/bash
# Batches 11 (GPU0 / port 18001) and 12 (GPU1 / port 18002), base model: more replicates of explore-more at three
# strengths with random matched controls, the success-direction edit again, and stacks of explore-more (layers
# 18,20,22) on the success-direction edit (layers 24,26,28).
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=$S/edits; AD=$S/acts_based/acts_base.pt
st() { n=$1; t=$2; [ -f $E/$n/adapter_config.json ] && { echo "have $n"; return; }
  $PY -m v2.steer.neuron --base $R/models/gemma-4-e4b-it-qat-w4a16-ct --adapter $E/bsuc_t0.5 --acts $AD --tags edit@0 \
    --layers 18,20,22 --beta 0.6 --sign -1 --mode const --target $t --out $E/$n > $E/$n.log 2>&1 && echo "built $n" || exit 1; }
st bns_n25_s50 0.25
st bns_n50_s50 0.5
go() { G=$1; P=$2; B=$3; shift 3; A="$*"
  for p in $(ps aux | grep "[a]pi_server.*--port $P" | awk '{print $2}'); do kill $p; done
  for i in $(seq 90); do m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 3000 ] && break; sleep 2; done
  L=""; for a in $A; do p=$E/$a; [ $a = base ] && p=$E/zero; L="$L $a=$p"; done
  cd $R; nohup ./serve_w.sh $G $P 0.90 $L > results/steer_eval/logs/serve_${P}_$B.log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -sf http://127.0.0.1:$P/v1/models > /dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$P/v1/models | grep -o "\"id\": *\"[^\"]*\"" | tr "\n" " "; echo
  ./run_steer_batch_p.sh $P $B 1 4 $A; echo ${B}_STARTED; }
go 0 18001 b11 base bneg_t0.5 bneg_t0.25 bsuc_t0.5 bns_n25_s50 bns_n50_s50 bnm25_s1 bnegmatch_t0.5
go 1 18002 b12 base bneg_t0.125 bneg_t0.25 bn50_th50 bns_n25_s50 becond_t1.0 bnm25_s2 bsucmatch_t1.0
