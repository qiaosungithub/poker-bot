"""Exact-equality test: batched traversal vs the recursive reference copy.

Both sides use the same deals and the same node-keyed hash uniforms
(det_rng), so with the same forward they must agree bit for bit.

  --forward numba : batched traversal uses the upstream numba forward per row
                    -> every output array must be bit-identical (tree logic).
  --forward torch : GPU forward (fp32 'highest' by default) -> report the
                    float tolerance and how many traversals diverged because a
                    float difference flipped a sampled action / a zero clamp.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import argparse
import json
import time

import numpy as np

import common
from gpu_traverse import BatchedTraverser, deal_hands
from ref_det import parallel_ref_hash

NAMES = ("features", "advantages", "strat_features", "strategies", "strat_weights")


def load_params(spec: str):
    if spec.startswith("random"):
        seed = int(spec.split(":")[1]) if ":" in spec else 0
        return common.params_random_init(seed)
    return common.params_from_npz(spec)


def segments(per_trav):
    ends = np.cumsum(per_trav)
    return ends - per_trav, ends


def compare(ref, mine, ref_per, mine_per):
    out = {}
    out["n_adv_ref"] = int(ref[0].shape[0])
    out["n_adv_mine"] = int(mine[0].shape[0])
    out["exact_all"] = bool(
        all(r.shape == m.shape and np.array_equal(r, m) for r, m in zip(ref, mine))
        and np.array_equal(ref_per, mine_per)
    )
    for name, r, m in zip(NAMES, ref, mine):
        out[f"exact_{name}"] = bool(r.shape == m.shape and np.array_equal(r, m))
    # per-traversal alignment (adv and strat samples are one per traverser node)
    rs, re_ = segments(ref_per)
    ms, me = segments(mine_per)
    diverged = []
    rows_r = []
    rows_m = []
    for t in range(len(ref_per)):
        if ref_per[t] != mine_per[t] or not np.array_equal(
                ref[0][rs[t]:re_[t]], mine[0][ms[t]:me[t]]):
            diverged.append(t)
            continue
        rows_r.append(np.arange(rs[t], re_[t]))
        rows_m.append(np.arange(ms[t], me[t]))
    rows_r = np.concatenate(rows_r) if rows_r else np.zeros(0, np.int64)
    rows_m = np.concatenate(rows_m) if rows_m else np.zeros(0, np.int64)
    out["n_traversals"] = int(len(ref_per))
    out["n_diverged_traversals"] = len(diverged)
    out["diverged_examples"] = diverged[:10]
    out["n_aligned_samples"] = int(rows_r.size)
    tol = {}
    bitexact_rows = np.ones(rows_r.size, dtype=bool)
    for k, name in enumerate(NAMES):
        x = ref[k][rows_r].astype(np.float64)
        y = mine[k][rows_m].astype(np.float64)
        d = np.abs(x - y)
        if d.ndim == 2:
            dr = d.max(axis=1)
        else:
            dr = d
        bitexact_rows &= dr == 0
        scale = np.maximum(np.abs(x), 1e-30)
        tol[name] = dict(
            max_abs=float(d.max()) if d.size else 0.0,
            max_rel_where_ref_gt_1em3=float((d / scale)[np.abs(x) > 1e-3].max())
            if (np.abs(x) > 1e-3).any() else 0.0,
            frac_rows_nonzero_diff=float((dr > 0).mean()) if dr.size else 0.0,
            n_rows_diff_gt_1em4=int((dr > 1e-4).sum()),
            n_rows_diff_gt_1em2=int((dr > 1e-2).sum()),
            p9999_abs=float(np.quantile(dr, 0.9999)) if dr.size else 0.0,
        )
    out["tolerance_on_aligned"] = tol
    out["n_bitexact_samples"] = int(bitexact_rows.sum())
    # strategy-buffer filter (weight > 1e-8) decisions that differ
    wr = ref[4][rows_r] > 1e-8
    wm = mine[4][rows_m] > 1e-8
    out["n_strat_keep_flips"] = int((wr != wm).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--net", default="random:0")
    ap.add_argument("--forward", default="numba", choices=["numba", "torch"])
    ap.add_argument("--precision", default="highest")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--traverser", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=46)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--json", default=None)
    ap.add_argument("--negative-control", action="store_true",
                    help="run the batched side with seed+1 uniforms (must FAIL)")
    args = ap.parse_args()

    params = load_params(args.net)
    deals = deal_hands(args.n, common.N_PLAYERS, args.seed, True)

    t0 = time.perf_counter()
    (ref, ref_stats, ref_per) = parallel_ref_hash(
        params, deals, args.traverser, args.seed, args.eps, n_workers=args.workers)
    t_ref = time.perf_counter() - t0

    trav = BatchedTraverser(args.device, args.precision, args.forward,
                            n_threads=args.threads, n_groups=args.groups)
    mseed = args.seed + 1 if args.negative_control else args.seed
    res = trav.run(params, args.traverser, args.n, args.eps, mseed,
                   deals=deals, collect_stats=True)  # warm-up / compile
    t0 = time.perf_counter()
    res = trav.run(params, args.traverser, args.n, args.eps, mseed,
                   deals=deals, collect_stats=True)
    t_mine = time.perf_counter() - t0
    mine = res.as_worker_result()

    out = compare(ref, mine, ref_per, res.per_trav_adv)
    out["stats_equal"] = bool(np.array_equal(ref_stats, res.stats))
    out["ref_stats"] = ref_stats.tolist()
    out["mine_stats"] = res.stats.tolist()
    out["config"] = vars(args)
    out["t_ref_s"] = t_ref
    out["t_mine_s"] = t_mine
    out["rounds"] = res.n_rounds
    out["nodes"] = res.n_nodes
    keep_ref = int((ref[4] > 1e-8).sum())
    keep_mine = int((mine[4] > 1e-8).sum())
    out["strat_kept_ref"] = keep_ref
    out["strat_kept_mine"] = keep_mine
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
