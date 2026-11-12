#!/bin/bash
cd /path/to/gemma-v2
M=""; for n in brf_c0.5 brf_c1.0 brf_c2.0 brfconst_t0.5 brfmatch_c1.0 beok_t0.5 beok_t1.0; do M="$M $n=/dev/shm/e3/$n"; done; M="$M bneg_t0.25=/dev/shm/e2/bneg_t0.25"
exec /path/to/gemma-v2/tii-venv/bin/python -m vllm.entrypoints.openai.api_server --model /path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct --host 127.0.0.1 --port 18000 --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.60 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --reasoning-config "{\"reasoning_start_str\": \"<|channel>\", \"reasoning_end_str\": \"<channel|>\"}" --no-scheduler-reserve-full-isl --logprobs-mode processed_logprobs --enable-prefix-caching --enable-lora --max-loras 8 --max-cpu-loras 32 --max-lora-rank 32 --lora-modules $M
