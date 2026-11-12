#!/bin/bash
# Build the round-3 wide-screen edits (author claude), publish them as ac3_<name>, and check each adapter.
. /path/to/gemma-v2/env.sh
C=$R/wsp_arena/claude; E=$S/edits
cd $C/code
one() { n=$1; [ -f $C/edits/$n/report.json ] || $PY build_r3.py --edit $n > $C/work/r3_$n.log 2>&1; }
mkdir -p $C/work
EDITS="u33 u36 u63 u33_L5 u33_c6 late30 late60 pu pe_lo pe_hi pe_L5 pe_L14 pe_L20 pe_L22 pe_c6 pre15 pre35"
i=0
for n in $EDITS; do one $n & i=$((i+1)); [ $((i % 6)) -eq 0 ] && wait; done; wait
one u33_ctrl & one pe_ctrl & wait
for n in $EDITS u33_ctrl pe_ctrl; do
  if [ -f $C/edits/$n/report.json ]; then
    mkdir -p $E/ac3_$n; cp $C/edits/$n/adapter_config.json $C/edits/$n/adapter_model.safetensors $E/ac3_$n/
    echo "ac3_$n $($PY $R/wsp_arena/check_adapter.py $E/ac3_$n 2>&1 | tail -1 | cut -c1-120)"
  else echo "ac3_$n FAILED $(tail -1 $C/work/r3_$n.log | cut -c1-160)"; fi
done
echo R3_BUILD_DONE
