#!/bin/bash
# Round-3 edits on the base model, aimed at failed edit_file calls (see v2.steer.states5).
#   brf_c*      IF "my last edit failed" (read ef) THEN "read the file again" (write rf@te), conditional neuron
#   brfconst    the same write, always on (shows what the gate buys)
#   brfmatch    the conditional neuron with a matched random write (control)
#   beok        "this edit will land" (eok@te), always on
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
E=${E:-/dev/shm/e3}; A=${A:-/dev/shm/ef/acts/acts_base.pt}; Z=${Z:-/dev/shm/ef/edits/zero}
mkdir -p $E
b() { n=$1; shift; [ -f $E/$n/adapter_config.json ] || $PY -m v2.steer.neuron --acts $A "$@" --out $E/$n > $E/$n.log 2>&1; echo "$n $(grep -c pred_shift $E/$n.log) layers"; }
RF="--tags rf@te --layers 16,24,26 --beta 0.6 --sign 1"
for t in 0.5 1.0 2.0; do b brf_c$t --adapter $Z $RF --mode cond --read-tags ef@0,ef@te --read-sign 1 --target $t; done
b brfconst_t0.5 --adapter $Z $RF --mode const --target 0.5
b brfmatch_c1.0 --adapter $Z $RF --mode cond --read-tags ef@0,ef@te --read-sign 1 --target 1.0 --control match --seed 0
for t in 0.5 1.0; do b beok_t$t --adapter $Z --tags eok@te --layers 26,29,30 --beta 0.6 --sign 1 --mode const --target $t; done

#   bse         "quote less text in old_string" (se@te, relabelled eok rows; v2.steer.relabel2), always on
A=/dev/shm/ef/acts/acts_base_se.pt
SE="--tags se@te --layers 10,14,18 --beta 0.6 --sign 1 --mode const"
for t in 0.5 1.0; do b bse_t$t --adapter $Z $SE --target $t; done
b bsematch_t1.0 --adapter $Z $SE --target 1.0 --control match --seed 0
echo BUILD_SE_DONE
