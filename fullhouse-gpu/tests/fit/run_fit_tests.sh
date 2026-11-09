#!/bin/bash
# Run the fit unit tests T1-T6 (test_gpu_fit.py) with the environment the CUDA extension
# needs: torch's cpp_extension loader requires ninja on PATH even for a cached build.
# Environment (defaults): PYTHON=$HOME/venvs/poker/bin/python, CUDA_HOME=/usr/local/cuda,
#   CUDA_VISIBLE_DEVICES=0. T2/T5 need JAX in the same environment (fp32 oracle).
set -u
PYTHON=${PYTHON:-$HOME/venvs/poker/bin/python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")"
"$PYTHON" test_gpu_fit.py "$@"
