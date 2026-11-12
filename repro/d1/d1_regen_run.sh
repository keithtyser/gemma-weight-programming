#!/bin/bash
# usage: d1_regen_run.sh PAIR OUT [args]   (GPU 1 of the serving box; d1_fidelity.py must sit beside d1_regen.py)
cd /path/to/gemma-v2/tii-steer
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.
pair=$1; out=$2; shift 2
exec /path/to/gemma-v2/tii-venv/bin/python /path/to/gemma-v2/d1/d1_regen.py --points /path/to/gemma-v2/d1/points.jsonl --pair "$pair" --out /path/to/gemma-v2/d1/"$out" "$@"
