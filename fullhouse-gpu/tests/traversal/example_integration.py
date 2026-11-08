"""Integration smoke test: feed the batched traversal into the UPSTREAM buffer
path (``train._collect_worker_batches`` + ``ReservoirBuffer``) for a few
iterations, exactly as ``train.train_deep_cfr`` does with worker results.

Usage pattern for the integrator:

    from gpu_traverse import BatchedTraverser
    import common
    trav = BatchedTraverser(device="cuda:1", precision="highest", n_groups=2)
    params = common.params_from_torch(net)          # or params_from_npz(path)
    res = trav.run(params, traverser=(it - 1) % 6, n_traversals=30000,
                   eps=_exploration_epsilon(it, ...), seed=base_seed * 1000 + it)
    _collect_worker_batches([res.as_worker_result()], buffers[traverser],
                            strategy_buffer, it)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import time

import numpy as np

import common
from gpu_traverse import BatchedTraverser
from training.buffer import ReservoirBuffer
from training.train import _collect_worker_batches, _exploration_epsilon


def main():
    trav = BatchedTraverser(device="cuda:1", precision="highest", n_groups=2)
    buffers = [ReservoirBuffer(5_000_000, 51, 5) for _ in range(6)]
    strategy_buffer = ReservoirBuffer(5_000_000, 51, 5, store_sample_weights=True)
    for it in range(1, 4):
        params = common.params_random_init(it)  # stand-in for the fitted net
        traverser = (it - 1) % 6
        eps = _exploration_epsilon(it, 250, 0.1, 0.02, 250)
        t0 = time.perf_counter()
        res = trav.run(params, traverser, 30000, eps, 20261107 * 1000 + it)
        t1 = time.perf_counter()
        n_s, n_ss = _collect_worker_batches([res.as_worker_result()],
                                            buffers[traverser], strategy_buffer, it)
        t2 = time.perf_counter()
        kept = int((res.strat_weights > 1e-8).sum())
        assert n_s == res.features.shape[0], (n_s, res.features.shape)
        assert n_ss == kept, (n_ss, kept)
        assert all(x.dtype == np.float32 for x in res.as_worker_result())
        print(f"iter {it}: traverser {traverser} eps {eps:.4f} traversal {t1 - t0:.3f}s "
              f"adv {n_s} strat kept {n_ss} buffer add {t2 - t1:.3f}s "
              f"buf sizes {[len(b) for b in buffers]} strat {len(strategy_buffer)}")
    print("integration OK")


if __name__ == "__main__":
    main()
