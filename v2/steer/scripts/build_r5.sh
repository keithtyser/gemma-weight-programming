#!/bin/bash
# Round 5 (screen s1): low-dose verify-more, layer-band and conditional variants of explore-more, explore+verify stack.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=$S/edits; Z=$E/zero; AD=$S/acts_based/acts_base.pt; AS=$S/acts_bases/acts_base.pt
b() { n=$1; shift; [ -f $E/$n/adapter_config.json ] && { echo "have $n"; return; }
  $PY -m v2.steer.neuron --base $R/models/gemma-4-e4b-it-qat-w4a16-ct --out $E/$n --beta 0.6 "$@" > $E/$n.log 2>&1 && echo "built $n" || echo "FAILED $n"; }
b bvfyD_t0.25 --adapter $Z --acts $AS --tags sv@0 --layers 18,20,22 --sign -1 --mode const --target 0.25 &
b bvfyD_t0.5  --adapter $Z --acts $AS --tags sv@0 --layers 18,20,22 --sign -1 --mode const --target 0.5 &
b bnegL_t0.5  --adapter $Z --acts $AD --tags edit@0 --layers 10,12,14 --sign -1 --mode const --target 0.5 &
b bnegH_t0.5  --adapter $Z --acts $AD --tags edit@0 --layers 24,26,28 --sign -1 --mode const --target 0.5 &
b bnegc_t1.0  --adapter $Z --acts $AD --tags edit@0 --layers 18,20,22 --sign -1 --mode cond --read-tags edit@0 --read-sign 1 --target 1.0 &
b bnv25       --adapter $E/bneg_t0.25 --acts $AS --tags sv@0 --layers 24,26,28 --sign -1 --mode const --target 0.25 &
wait
echo BUILD_R5_DONE
