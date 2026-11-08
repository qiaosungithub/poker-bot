#!/bin/bash
# Launcher: environment for the trainer (the fit package's CUDA extension needs ninja +
# nvcc on PATH). Records launch / exit epochs around the training process.
# usage: ./run.sh CONFIG.json [--run-name NAME]   (logs to stdout; wrap with nohup/setsid yourself)
# Environment (defaults): PYTHON=$HOME/venvs/poker/bin/python, CUDA_HOME=/usr/local/cuda,
#   FULLHOUSE_SOURCE=<repo>/fullhouse-repro/source, FULLHOUSE_GPU_RUNTIME=<repo>/fullhouse-gpu/.runtime
set -u
PYTHON=${PYTHON:-$HOME/venvs/poker/bin/python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
cd "$(dirname "$0")"
echo "LAUNCH_EPOCH $(date +%s.%N)"
"$PYTHON" -X faulthandler -u train_fast.py "$@"
rc=$?
echo "EXIT $rc END_EPOCH $(date +%s.%N)"
exit $rc
