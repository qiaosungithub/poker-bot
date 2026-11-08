"""Invariance checks: results must not depend on the number of pipeline
groups or numba threads, and parallel deals must be thread-count independent."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import sys

import numba as nb
import numpy as np

import common
from gpu_traverse import BatchedTraverser, deal_hands_parallel
from test_exact import compare


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a.as_worker_result(),
                                                   b.as_worker_result()))


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else str(common.SOURCE / "data" / "deep_cfr_model.npz")
    params = common.params_from_npz(net)
    n = 30000
    nb.set_num_threads(48)
    d48 = deal_hands_parallel(n, 6, 777, True)
    nb.set_num_threads(7)
    d7 = deal_hands_parallel(n, 6, 777, True)
    nb.set_num_threads(48)
    ok_deal = all(np.array_equal(x, y) for x, y in zip(d48, d7))
    print("parallel deals identical for 48 vs 7 threads:", ok_deal)
    d_other = deal_hands_parallel(n, 6, 778, True)
    print("negative control (seed 778 differs):",
          not np.array_equal(d48[1], d_other[1]))
    # dealer / stack sanity
    st = d48[0]
    print("dealer histogram:", np.bincount(st[:, 43], minlength=6).tolist())
    print("initial stacks sum per hand unique:", np.unique(st[:, 53:59].sum(1)).tolist())

    trav = BatchedTraverser("cuda:1", "highest", "torch", n_groups=1)
    r1 = trav.run(params, 2, n, 0.05, 4242)
    r2 = trav.run(params, 2, n, 0.05, 4242, n_groups=2)
    r3 = trav.run(params, 2, n, 0.05, 4242, n_groups=3)
    nb.set_num_threads(13)
    r4 = trav.run(params, 2, n, 0.05, 4242, n_groups=2)
    nb.set_num_threads(48)
    r5 = trav.run(params, 2, n, 0.05, 4243, n_groups=2)
    print("groups 1 vs 2 identical:", same(r1, r2))
    c = compare(r1.as_worker_result(), r2.as_worker_result(), r1.per_trav_adv,
                r2.per_trav_adv)
    print("  groups 1 vs 2 (torch fwd, batch-size dependent cuBLAS rounding):",
          {k: c[k] for k in ("n_diverged_traversals", "n_bitexact_traversals",
                             "tolerance_on_aligned")})
    tn = BatchedTraverser("cuda:1", "highest", "numba", n_groups=1)
    small = 4000
    dl = deal_hands_parallel(small, 6, 99, True)
    q1 = tn.run(params, 2, small, 0.05, 4242, deals=dl, n_groups=1)
    q3 = tn.run(params, 2, small, 0.05, 4242, deals=dl, n_groups=3)
    print("numba fwd: groups 1 vs 3 identical (pipelining logic):", same(q1, q3))
    print("groups 1 vs 3 identical:", same(r1, r3))
    print("threads 48 vs 13 identical:", same(r2, r4))
    print("negative control seed+1 differs:", not same(r2, r5))
    print("n_adv", r1.features.shape[0], "rounds", r1.n_rounds, "nodes", r1.n_nodes)


if __name__ == "__main__":
    main()
