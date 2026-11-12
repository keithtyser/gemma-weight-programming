#!/bin/bash
# Round-2 candidate edits on the base model: think briefly (th), read not run (rr), stacked with explore-more.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=$S/edits; A=$S/acts_based/acts_base_rl.pt
b() { n=$1; shift; [ -f $E/$n/adapter_config.json ] || $PY -m v2.steer.neuron --acts $A --mode const "$@" --out $E/$n > $E/$n.log 2>&1; echo "$n $(tail -1 $E/$n.log | cut -c1-200)"; }
for t in 0.25 0.5 1.0; do b bth_t$t --adapter $E/zero --tags th@0 --layers 20,22,26 --beta 0.6 --sign -1 --target $t; done
b bthmatch_t0.5 --adapter $E/zero --tags th@0 --layers 20,22,26 --beta 0.6 --sign -1 --target 0.5 --control match --seed 0
for t in 0.5 1.0; do b brr_t$t --adapter $E/zero --tags rr@0 --layers 18,20,22 --beta 0.6 --sign 1 --target $t; done
b bcombo_n25_th50 --adapter $E/bneg_t0.25 --tags th@0 --layers 24,26,28 --beta 0.6 --sign -1 --target 0.5
echo BUILD_DONE
