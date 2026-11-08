"""Statistical agreement: batched GPU traversal vs UNMODIFIED upstream traversal.

Reference side: upstream ``training.traverse_numba.worker_run_jit`` run exactly
like ``train._run_parallel_traversals`` (46 spawn workers, seeds
iteration*1000 + worker, np.random RNG, numba CPU forward).  The instrumented
test copy ``ref_det.run_ref_np`` is run with the same seeds; its 5 output
arrays are asserted bit-identical to upstream, and it additionally provides
per-traversal sample counts and per-street node counts.

Batched side: ``gpu_traverse.BatchedTraverser`` with production settings
(hash RNG, parallel deals, 2 pipeline groups) at fp32 'highest' and
optionally TF32 / bf16 forward.

R replicates of 30,000 traversals per side; for every metric we report the
replicate mean +- sd on both sides and Welch's t.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import argparse
import json
import os
import time

import numpy as np

import common
from ref_det import parallel_ref_np

N_TRAV = 30000


def _upstream_worker(args):
    import common as c  # noqa: F401  (sys.path in the spawned child)
    from training.traverse_numba import worker_run_jit

    return worker_run_jit(args)


def run_upstream(params, n_traversals, traverser, iteration, eps, n_workers=46):
    import multiprocessing as mp

    tables = common.load_tables()
    weights, biases = common.worker_lists(params)
    per = n_traversals // n_workers
    rem = n_traversals % n_workers
    counts = [per + (1 if i < rem else 0) for i in range(n_workers)]
    args = [
        {
            "seed": iteration * 1000 + i,
            "n_traversals": c,
            "traverser": traverser,
            "n_players": common.N_PLAYERS,
            "weights": weights,
            "biases": biases,
            "flush_lut": tables[0],
            "nf_keys": tables[1],
            "nf_vals": tables[2],
            "eq_paired": tables[3],
            "eq_suited": tables[4],
            "eq_offsuit": tables[5],
            "opponent_exploration": eps,
            "randomize_stacks": True,
        }
        for i, c in enumerate(counts)
    ]
    with mp.get_context("spawn").Pool(n_workers) as pool:
        results = pool.map(_upstream_worker, args)
    cap_hits = sum(int(r[0].shape[0] >= c * 60 or r[2].shape[0] >= c * 60)
                   for r, c in zip(results, counts))
    outs = tuple(np.concatenate([r[k] for r in results]) for k in range(5))
    return outs, cap_hits


def metrics(res5, n_trav, per_trav=None, stats=None):
    f, a, sf, s, w = res5
    legal = f[:, 46:51] > 0
    m = {}
    m["adv_samples_per_trav"] = f.shape[0] / n_trav
    kept = w > np.float32(1e-8)
    m["strat_kept_per_trav"] = kept.sum() / n_trav
    m["strat_kept_frac"] = kept.mean()
    for k in range(5):
        x = a[legal[:, k], k].astype(np.float64)
        m[f"adv_mean_a{k}"] = x.mean()
        m[f"adv_std_a{k}"] = x.std()
        m[f"legal_frac_a{k}"] = legal[:, k].mean()
        m[f"sigma_mean_a{k}"] = s[:, k].astype(np.float64).mean()
        m[f"sigma_zero_frac_a{k}"] = ((s[:, k] == 0) & legal[:, k]).mean()
    m["adv_abs_mean"] = np.abs(a[legal]).astype(np.float64).mean()
    for st in range(4):
        m[f"trav_nodes_street{st}_per_trav"] = (f[:, 17 + st] == 1).sum() / n_trav
    for j in range(46):
        x = f[:, j].astype(np.float64)
        m[f"feat_mean_d{j:02d}"] = x.mean()
        m[f"feat_std_d{j:02d}"] = x.std()
    wk = w[kept].astype(np.float64)
    m["reach_mean_kept"] = wk.mean()
    m["reach_logmean_kept"] = np.log(wk).mean()
    for q in (0.1, 0.5, 0.9):
        m[f"reach_q{int(q * 100):02d}_kept"] = np.quantile(wk, q)
    m["reach_eq1_frac"] = (w == 1.0).mean()
    m["reach_weighted_sigma_a1"] = (wk * s[kept, 1]).sum() / wk.sum()
    if per_trav is not None:
        pt = per_trav.astype(np.float64)
        m["per_trav_adv_std"] = pt.std()
        m["per_trav_zero_frac"] = (pt == 0).mean()
        for q in (0.5, 0.9, 0.99):
            m[f"per_trav_adv_q{int(q * 100):02d}"] = np.quantile(pt, q)
    if stats is not None:
        names = ("opp", "trav", "term")
        for st in range(4):
            for kind in range(3):
                m[f"nodes_{names[kind]}_street{st}_per_trav"] = stats[st, kind] / n_trav
    return {k: float(v) for k, v in m.items()}


def welch(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    mx, my = xs.mean(), ys.mean()
    vx = xs.var(ddof=1) / len(xs) if len(xs) > 1 else 0.0
    vy = ys.var(ddof=1) / len(ys) if len(ys) > 1 else 0.0
    se = np.sqrt(vx + vy)
    if se == 0:
        return 0.0 if mx == my else float("inf")
    return float((my - mx) / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="released")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--precisions", default="highest,tf32,bf16")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift iteration seeds (ref) and traversal seeds (mine)")
    args = ap.parse_args()

    nets = {
        "released": str(common.SOURCE / "data" / "deep_cfr_model.npz"),
        "iter4": os.environ.get("ITER4_NPZ", "iter4-npz-not-set"),  # baseline iteration-4 advantage NPZ (not committed)
    }
    if args.net.startswith("random"):
        params = common.params_random_init(int(args.net.split(":")[1]))
    else:
        params = common.params_from_npz(nets.get(args.net, args.net))

    ref_m = []
    upstream_equal = []
    cap_hits = []
    t_ref = []
    for r in range(args.reps):
        traverser = r % 6
        iteration = 200 + r + args.seed_offset
        t0 = time.perf_counter()
        up, hits = run_upstream(params, N_TRAV, traverser, iteration, args.eps)
        t_ref.append(time.perf_counter() - t0)
        cap_hits.append(hits)
        inst, stats, per_trav = parallel_ref_np(params, N_TRAV, traverser,
                                                iteration, args.eps)
        eq = all(np.array_equal(x, y) for x, y in zip(up, inst))
        upstream_equal.append(bool(eq))
        ref_m.append(metrics(up, N_TRAV, per_trav, stats))
        print(f"ref rep {r}: upstream==instrumented copy: {eq}, "
              f"adv/trav {ref_m[-1]['adv_samples_per_trav']:.3f}, "
              f"{t_ref[-1]:.1f}s", flush=True)

    from gpu_traverse import BatchedTraverser

    out = dict(config=vars(args), upstream_equal_instrumented=upstream_equal,
               upstream_cap_hits=cap_hits, t_ref_s=t_ref, variants={})
    for prec in args.precisions.split(","):
        trav = BatchedTraverser(args.device, prec, "torch", n_groups=2)
        trav.run(params, 0, 2000, args.eps, 1)
        mine_m = []
        walls = []
        for r in range(args.reps):
            t0 = time.perf_counter()
            res = trav.run(params, r % 6, N_TRAV, args.eps,
                           900000 + 1000 * (r + args.seed_offset),
                           collect_stats=True)
            walls.append(time.perf_counter() - t0)
            mine_m.append(metrics(res.as_worker_result(), N_TRAV,
                                  res.per_trav_adv, res.stats))
        table = {}
        for k in ref_m[0]:
            xs = [m[k] for m in ref_m]
            ys = [m[k] for m in mine_m]
            table[k] = dict(ref_mean=float(np.mean(xs)), ref_sd=float(np.std(xs, ddof=1)),
                            mine_mean=float(np.mean(ys)), mine_sd=float(np.std(ys, ddof=1)),
                            t=welch(xs, ys))
        ts = np.array([abs(v["t"]) for v in table.values()])
        summ = dict(n_metrics=len(ts), max_abs_t=float(ts.max()),
                    n_abs_t_gt_3=int((ts > 3).sum()), n_abs_t_gt_4=int((ts > 4).sum()),
                    worst=sorted(table, key=lambda k: -abs(table[k]["t"]))[:8],
                    mean_wall_s=float(np.mean(walls)))
        out["variants"][prec] = dict(summary=summ, table=table)
        print(prec, json.dumps(summ), flush=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
