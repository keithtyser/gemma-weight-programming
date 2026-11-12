#!/bin/bash
# vLLM server with the production flags (v2/eval/gpu_runner.py build_cmd) for edited-adapter evals on evalhost.
# Usage: serve_w.sh GPU PORT GPU_UTIL name=path [name=path ...]
. /path/to/gemma-v2/env.sh
G=$1; P=$2; U=$3; shift 3
cd $R/tii-steer
CUDA_VISIBLE_DEVICES=$G exec $PY -m vllm.entrypoints.openai.api_server --model $R/models/gemma-4-e4b-it-qat-w4a16-ct --host 127.0.0.1 --port $P \
  --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization $U --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4 --reasoning-config "{\"reasoning_start_str\": \"<|channel>\", \"reasoning_end_str\": \"<channel|>\"}" \
  --no-scheduler-reserve-full-isl --logprobs-mode processed_logprobs --enable-prefix-caching \
  --enable-lora --max-loras 8 --max-cpu-loras 32 --max-lora-rank 32 --lora-modules "$@"
