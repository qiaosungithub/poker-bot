#!/bin/bash
# Exact-equality suite (numba forward = bit-exact tree logic; torch = GPU tolerance)
source "$(dirname "$0")/_suite_env.sh"
set -x
$PY test_exact.py --n 30000 --net random:0 --forward numba --traverser 0 --eps 0.1 --seed 1001 --json $R/exact_numba_random.json
[ -n "$IT4" ] && $PY test_exact.py --n 30000 --net $IT4 --forward numba --traverser 3 --eps 0.0988 --seed 1002 --json $R/exact_numba_iter4.json
$PY test_exact.py --n 30000 --net $REL --forward numba --traverser 5 --eps 0.02 --seed 1003 --json $R/exact_numba_released.json
$PY test_exact.py --n 30000 --net $REL --forward numba --groups 3 --traverser 1 --eps 0.05 --seed 2001 --json $R/exact_numba_released_g3.json
$PY test_exact.py --n 5000 --net $REL --forward numba --traverser 5 --eps 0.02 --seed 1003 --negative-control --json $R/exact_numba_negctrl.json
for P in highest tf32 bf16; do
$PY test_exact.py --n 30000 --net random:0 --forward torch --precision $P --traverser 0 --eps 0.1 --seed 1001 --json $R/exact_torch_${P}_random.json
[ -n "$IT4" ] && $PY test_exact.py --n 30000 --net $IT4 --forward torch --precision $P --traverser 3 --eps 0.0988 --seed 1002 --json $R/exact_torch_${P}_iter4.json
$PY test_exact.py --n 30000 --net $REL --forward torch --precision $P --traverser 5 --eps 0.02 --seed 1003 --json $R/exact_torch_${P}_released.json
done
echo SUITE_DONE
