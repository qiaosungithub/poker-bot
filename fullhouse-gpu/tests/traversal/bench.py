"""Benchmark: seconds per iteration of 30k traversals and the time split."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import argparse
import json
import time

import numpy as np

import common
from gpu_traverse import BatchedTraverser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--net", default=str(common.SOURCE / "data" / "deep_cfr_model.npz"))
    ap.add_argument("--precision", default="highest")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--to-device", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.net.startswith("random"):
        params = common.params_random_init(int(args.net.split(":")[1]))
    else:
        params = common.params_from_npz(args.net)
    trav = BatchedTraverser(args.device, args.precision, "torch",
                            n_threads=args.threads, n_groups=args.groups)
    trav.run(params, 0, 2000, args.eps, 1, profile=args.profile)  # compile/warm
    rows = []
    for r in range(args.reps):
        traverser = r % 6
        t0 = time.perf_counter()
        res = trav.run(params, traverser, args.n, args.eps, 100 + r,
                       profile=args.profile, to_device=args.to_device)
        wall = time.perf_counter() - t0
        row = dict(rep=r, traverser=traverser, wall=wall, rounds=res.n_rounds,
                   nodes=res.n_nodes, n_adv=int(res.features.shape[0]),
                   n_strat_kept=int((res.strat_weights > 1e-8).sum()),
                   **{k: round(v, 4) for k, v in res.timing.items()})
        rows.append(row)
        print(json.dumps(row))
    walls = np.array([r["wall"] for r in rows])
    summ = dict(
        mean_wall=float(walls.mean()), min_wall=float(walls.min()),
        trav_per_s=float(args.n / walls.mean()),
        mean_split={k: float(np.mean([r[k] for r in rows]))
                    for k in rows[0] if k not in ("rep", "traverser")},
        config=vars(args),
        peak_gpu_mem_gb=__import__("torch").cuda.max_memory_allocated(args.device) / 1e9,
    )
    print(json.dumps(summ, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(rows=rows, summary=summ), f, indent=1)


if __name__ == "__main__":
    main()
