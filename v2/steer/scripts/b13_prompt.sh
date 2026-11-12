#!/bin/bash
# Batch 13: prompt-space baselines (v2/steer/scripts/mk_prompt_arms.py) against the unedited model and the explore-more
# edit, on one served endpoint. Arms are "<harness>:<adapter>"; results in results/steer_eval/b13_<harness>_<adapter>.
# Usage: b13_prompt.sh PORT
R=/path/to/gemma-v2; P=$R/tii-venv/bin/python; PORT=$1; B=b13
cd $R/tii-steer
for ha in w5:base pexp:base pedit:base pboth:base w5:bneg_t0.5 pedit:bneg_t0.5 pboth:bneg_t0.5 w5:bnm25_s0; do
  h=${ha%%:*}; a=${ha##*:}; n=$a; [ $h != w5 ] && n=${h}_$a
  O=$R/results/steer_eval/${B}_$n
  if [ -e $O ]; then echo "exists: $O"; continue; fi
  nohup $P -m v2.eval.golden_ab --golden v2 --arm $R/harnesses/steer_ext_$h --external-api-base http://127.0.0.1:$PORT/v1 \
    --served-adapter $a --out $O --seeds 1 --concurrency-per-gpu 4 --keep-images > $R/results/steer_eval/logs/${B}_$n.log 2>&1 < /dev/null &
  echo "started $n"
done
