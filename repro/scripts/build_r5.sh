#!/bin/bash
# Build the round-5 edits (author claude), publish them as ac5_<name>, and check each adapter.
. /path/to/gemma-v2/env.sh
C=$R/wsp_arena/claude; E=$S/edits
cd $C/code
EDITS="late90 late60_x3 late60_x5 late60_c6 late100_c6 late60_ctrl"
for n in $EDITS; do [ -f $C/edits/$n/report.json ] || $PY build_r5.py --edit $n > $C/work/r5_$n.log 2>&1 & done; wait
for n in $EDITS; do
  if [ -f $C/edits/$n/report.json ]; then
    mkdir -p $E/ac5_$n; cp $C/edits/$n/adapter_config.json $C/edits/$n/adapter_model.safetensors $E/ac5_$n/
    echo "ac5_$n $($PY $R/wsp_arena/check_adapter.py $E/ac5_$n 2>&1 | tail -1 | cut -c1-120)"
  else echo "ac5_$n FAILED $(tail -1 $C/work/r5_$n.log | cut -c1-160)"; fi
done
echo R5_BUILD_DONE
