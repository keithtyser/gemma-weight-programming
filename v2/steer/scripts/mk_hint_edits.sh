#!/bin/bash
# Compile prompt rules into one constant neuron at layer 18: direction = mean residual shift the rule causes at
# decisions (v2.steer.states8 contrast), strength in units of that shift (1 = what the prompt rule does on average).
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
A=${1:-/dev/shm/ef/acts_hint/acts_base.pt}; Z=${2:-/dev/shm/ef/edits/zero}; O=${3:-/dev/shm/ef/edits}
mk() { # name tags target control
  [ -f $O/$1/adapter_model.safetensors ] || $PY -m v2.steer.neuron --adapter $Z --acts $A --tags $2 --layers 18 --beta 0.6 --target $3 --control $4 --out $O/$1 > $O/$1.log 2>&1
  $PY - <<PYEOF
import json; r = json.load(open("$O/$1/neuron_report.json"))["layers"]["18"]
print("$1", {k: round(r[k], 3) for k in ("gap", "heldout_auc", "pred_shift_over_gap", "beta", "natural_write_scale")})
PYEOF
}
mk hpe18_t1 pedit@0 1 none; mk hpe18_t3 pedit@0 3 none; mk hpe18_t6 pedit@0 6 none
mk hpx18_t1 pexp@0 1 none; mk hpx18_t3 pexp@0 3 none; mk hpx18_t6 pexp@0 6 none
mk hpb18_t3 pedit@0,pexp@0 3 none; mk hpb18_t6 pedit@0,pexp@0 6 none
mk hpematch_t3 pedit@0 3 match; mk hpxmatch_t3 pexp@0 3 match
