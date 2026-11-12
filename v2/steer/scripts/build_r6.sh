#!/bin/bash
# Round 6 (screen s2): wide low-dose explore-more, explore-more read at other token positions, edit-succeeds direction,
# think-longer (the reverse of think-briefly).
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=$S/edits; Z=$E/zero; AD=$S/acts_based/acts_base.pt; AS=$S/acts_bases/acts_base.pt; AR=$S/acts_based/acts_base_rl.pt
b() { n=$1; shift; [ -f $E/$n/adapter_config.json ] && { echo "have $n"; return; }
  $PY -m v2.steer.neuron --base $R/models/gemma-4-e4b-it-qat-w4a16-ct --out $E/$n --beta 0.6 "$@" > $E/$n.log 2>&1 && echo "built $n" || echo "FAILED $n"; }
b bnegW_t0.125 --adapter $Z --acts $AD --tags edit@0 --layers 10,14,18,22,26 --sign -1 --mode const --target 0.125 &
b bnegW_t0.25  --adapter $Z --acts $AD --tags edit@0 --layers 10,14,18,22,26 --sign -1 --mode const --target 0.25 &
b bneg32_t0.5  --adapter $Z --acts $AD --tags edit@32 --layers 18,20,22 --sign -1 --mode const --target 0.5 &
b bnegte_t0.5  --adapter $Z --acts $AD --tags edit@te --layers 18,20,22 --sign -1 --mode const --target 0.5 &
wait
b bok_t0.5     --adapter $Z --acts $AS --tags ok@0 --layers 18,20,22 --sign 1 --mode const --target 0.5 &
b bokN_t0.5    --adapter $Z --acts $AS --tags ok@0 --layers 18,20,22 --sign -1 --mode const --target 0.5 &
b bthP_t0.25   --adapter $Z --acts $AR --tags th@0 --layers 20,22,26 --sign 1 --mode const --target 0.25 &
b bthP_t0.5    --adapter $Z --acts $AR --tags th@0 --layers 20,22,26 --sign 1 --mode const --target 0.5 &
wait
echo BUILD_R6_DONE
