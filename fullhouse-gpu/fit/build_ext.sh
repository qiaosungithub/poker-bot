#!/bin/bash
# Build (or load) the fit package's CUDA extension once, before any timed run.
# Output: fit/.runtime/torch_ext (about 1-2 min on first build; needs nvcc + ninja).
# Environment (defaults): PYTHON=$HOME/venvs/poker/bin/python, CUDA_HOME=/usr/local/cuda
set -euo pipefail
PYTHON=${PYTHON:-$HOME/venvs/poker/bin/python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
cd "$(dirname "$0")"
sha256sum -c MANIFEST.sha256
"$PYTHON" -c "
import sys, time; sys.path.insert(0, '.')
t = time.time(); import fast_step; fast_step.ext(); print('extension built/loaded in %.1fs at' % (time.time() - t), fast_step._EXT.__file__)"
