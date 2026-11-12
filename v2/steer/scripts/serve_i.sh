#!/bin/bash
# Serve the INT4 base model with the listed edits as LoRA modules. Usage: serve_i.sh GPU PORT arm...
G=$1; P=$2; shift 2
cd "${GWP_ROOT:-/path/to/gemma-v2}"
E=/path/to/gemma-v2/results/steer/edits
M="base=$E/zero"; for n in "$@"; do [ -d "$E/$n" ] || { echo "missing edit $n"; exit 1; }; M="$M $n=$E/$n"; done
export CUDA_VISIBLE_DEVICES=$G
exec /path/to/gemma-v2/tii-venv/bin/python -m vllm.entrypoints.openai.api_server --model /path/to/gemma-v2/models/gemma-4-e4b-it-qat-w4a16-ct --host 127.0.0.1 --port $P --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.60 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --reasoning-config "{\"reasoning_start_str\": \"<|channel>\", \"reasoning_end_str\": \"<channel|>\"}" --no-scheduler-reserve-full-isl --logprobs-mode processed_logprobs --enable-prefix-caching --enable-lora --max-loras 8 --max-cpu-loras 32 --max-lora-rank 32 --lora-modules $M
