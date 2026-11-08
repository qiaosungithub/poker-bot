#!/bin/bash
# Statistical agreement vs unmodified upstream (production RNG), 6 x 30k traversals per side
source "$(dirname "$0")/_suite_env.sh"
set -x
$PY compare_stats.py --net released --eps 0.02 --reps 6 --json $R/stats_released.json
[ -n "$IT4" ] && ITER4_NPZ="$IT4" $PY compare_stats.py --net iter4 --eps 0.0988 --reps 6 --json $R/stats_iter4.json
$PY compare_stats.py --net random:0 --eps 0.1 --reps 6 --json $R/stats_random.json
echo STATS_DONE
