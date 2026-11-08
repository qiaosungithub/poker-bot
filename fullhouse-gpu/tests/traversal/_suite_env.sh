# Shared settings for the run_*_suite.sh scripts (source it; do not execute).
# Environment (defaults):
#   PYTHON=$HOME/venvs/poker/bin/python
#   FULLHOUSE_SOURCE=<repo>/fullhouse-repro/source   (released net: $FULLHOUSE_SOURCE/data/deep_cfr_model.npz)
#   OUT=<repo>/fullhouse-gpu/.runtime/test-results/traversal   (new results; the committed records
#       behind the reports are in fullhouse-gpu/results/traversal and are never overwritten)
#   ITER4_NPZ=<a partly trained advantage NPZ, e.g. iteration 4 of a run>; iter-4 cases are skipped if unset
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-$HOME/venvs/poker/bin/python}
SRC=${FULLHOUSE_SOURCE:-$(cd ../../.. && pwd)/fullhouse-repro/source}
REL=$SRC/data/deep_cfr_model.npz
R=${OUT:-$(cd ../.. && pwd)/.runtime/test-results/traversal}
IT4=${ITER4_NPZ:-}
mkdir -p "$R"
