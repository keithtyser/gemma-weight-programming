#!/bin/bash
# Start batch 13 on the port-18000 endpoint as soon as batch 10 has no evaluation process left.
until [ $(ps aux | grep "[g]olden_ab" | grep -c "steer_eval/b10_") -eq 0 ]; do sleep 60; done
curl -s -m 20 http://127.0.0.1:18000/v1/models | grep -q bneg_t0.5 || { echo "B13_NO_SERVER $(date)"; exit 1; }
echo "B13_STARTED $(TZ=America/New_York date +%H:%M)"
/path/to/gemma-v2/tii-steer/v2/steer/scripts/b13_prompt.sh 18000
