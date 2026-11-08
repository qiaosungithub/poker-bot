#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
poker_python="${1:-python3.12}"
git submodule update --init -- external/bay-area-entry external/denfujita-poker external/james-nolimit
mkdir -p .runtime/tmp .runtime/cache/pip
export TMPDIR="$project_root/.runtime/tmp"
export PIP_CACHE_DIR="$project_root/.runtime/cache/pip"
if [[ ! -x .runtime/envs/poker-bay-area312/bin/python ]]; then
  "$poker_python" -m venv .runtime/envs/poker-bay-area312
fi
poker_python="$project_root/.runtime/envs/poker-bay-area312/bin/python"
"$poker_python" -m pip install -r bay_area/requirements.txt
"$poker_python" -m pip install --no-build-isolation --no-deps \
  -e external/bay-area-entry/kit/sdk/agent-protocol \
  -e external/bay-area-entry/kit/sdk/engine \
  -e external/bay-area-entry/kit/sdk/agent-sdk
"$poker_python" -m pytest bay_area/tests -q
