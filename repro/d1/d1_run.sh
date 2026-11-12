#!/bin/bash
# usage: d1_run.sh OUTDIR [extra args]   (GPU 1 only)
cd /path/to/gemma-v2/tii-steer
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. TII_MARLIN_SO=/path/to/gemma-v2/tii-dev/v2/tii/engine/tii_marlin/build_k128n64L128t3/tii_marlin_k128n64L128t3.so
out=$1; shift
exec /path/to/gemma-v2/tii-venv/bin/python /path/to/gemma-v2/d1/d1_fidelity.py --points /path/to/gemma-v2/d1/points.jsonl --out /path/to/gemma-v2/d1/$out "$@"
