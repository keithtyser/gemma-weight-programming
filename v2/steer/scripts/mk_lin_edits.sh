#!/bin/bash
# Compile prompt rules into 16 linear neurons at layer 18 (v2.steer.linmap): the rule's context-dependent residual
# shift as a rank-16 map of the MLP input; dose 1 = the fitted shift, 2 = twice it. Matched controls at dose 2.
. /path/to/gemma-v2/env.sh
cd $R/tii-steer
A=${1:-/dev/shm/ef/acts_hint/acts_base.pt}; Z=${2:-/dev/shm/ef/edits/zero}; O=${3:-/dev/shm/ef/edits}
mk() { # name tags scale control
  [ -f $O/$1/adapter_model.safetensors ] || $PY -m v2.steer.linmap --adapter $Z --acts $A --tags $2 --layer 18 --rank 16 --scale $3 --control $4 --out $O/$1 > $O/$1.log 2>&1
  $PY - <<PYEOF
import json; r = json.load(open("$O/$1/linmap_report.json"))
print("$1", {k: round(v, 3) for k, v in r["heldout_r2"].items()}, {k: round(r[k], 3) for k in ("write_norm_pairs_mean", "beta_mean", "cos_write_fit_heldout", "up_down_balance_c", "up_proj_rows_rel_err", "down_cols_rel_err")})
PYEOF
}
mk lme16_t1 pedit@0 1 none; mk lme16_t2 pedit@0 2 none; mk lmx16_t1 pexp@0 1 none; mk lmx16_t2 pexp@0 2 none
mk lmb16_t1 pedit@0,pexp@0 1 none; mk lmb16_t2 pedit@0,pexp@0 2 none
mk lmematch_t2 pedit@0 2 match; mk lmxmatch_t2 pexp@0 2 match; mk lmbmatch_t2 pedit@0,pexp@0 2 match
