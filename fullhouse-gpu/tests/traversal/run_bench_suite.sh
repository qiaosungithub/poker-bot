#!/bin/bash
# Final benchmark: released net (late-run tree size), eps 0.02, 30k traversals
source "$(dirname "$0")/_suite_env.sh"
set -x
$PY bench.py --reps 8 --groups 2 --json $R/bench_final_highest_g2.json
$PY bench.py --reps 8 --groups 1 --json $R/bench_final_highest_g1.json
$PY bench.py --reps 8 --groups 3 --json $R/bench_final_highest_g3.json
$PY bench.py --reps 8 --groups 2 --precision tf32 --json $R/bench_final_tf32_g2.json
$PY bench.py --reps 4 --profile --json $R/bench_final_profile.json
for T in 12 24 36; do $PY bench.py --reps 6 --groups 2 --threads $T --json $R/bench_final_threads$T.json; done
$PY bench.py --reps 6 --groups 2 --to-device cuda:1 --json $R/bench_final_todevice.json
[ -n "$IT4" ] && $PY bench.py --reps 6 --groups 2 --net "$IT4" --eps 0.0988 --json $R/bench_final_iter4.json
$PY bench.py --reps 6 --groups 2 --net random:0 --eps 0.1 --json $R/bench_final_random.json
$PY example_integration.py > $R/example_integration.log 2>&1
echo BENCH_DONE
