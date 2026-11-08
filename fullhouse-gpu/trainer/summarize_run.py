"""Summarise a trainer run's per-iteration metrics (time split, samples, losses).

usage: python summarize_run.py runs/full [--baseline LOG]
       python summarize_run.py ../results/runs/full_v2     (the committed run-2 records)
--baseline: a train.log of the original upstream code (not committed).
Prints a markdown block suitable for REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    recs = [json.loads(l) for l in Path(str(run) + "_metrics.jsonl").read_text().splitlines()]
    summ = json.loads(Path(str(run) + "_summary.json").read_text())
    n = len(recs)
    print(f"iterations: {n}")
    for k in ("loop_seconds_compute", "loop_seconds_incl_final_ckpt", "setup_seconds",
              "process_seconds_to_final_ckpt", "final_buffer_save_seconds", "process_seconds_total",
              "strategy_busy_seconds", "writer_busy_seconds"):
        if k in summ:
            print(f"{k}: {summ[k]:.1f}")
    print("\n| phase | total s | mean s | median s | p95 s | max s |\n|---|---:|---:|---:|---:|---:|")
    for k in ("t_traversal", "t_add", "t_fit", "t_ckpt", "t_iter"):
        xs = [r[k] for r in recs]
        print(f"| {k[2:]} | {sum(xs):.1f} | {st.mean(xs):.3f} | {st.median(xs):.3f} | "
              f"{q(xs, .95):.3f} | {max(xs):.3f} |")
    other = [r["t_iter"] - r["t_traversal"] - r["t_add"] - r["t_fit"] - r["t_ckpt"] for r in recs]
    print(f"| other (logging etc.) | {sum(other):.1f} | {st.mean(other):.4f} | | | {max(other):.4f} |")
    splits = {}
    for r in recs:
        for k, v in r["trav_split"].items():
            splits.setdefault(k, []).append(v)
    print("\ntraversal internal split (mean s/iter): " +
          ", ".join(f"{k} {st.mean(v):.3f}" for k, v in sorted(splits.items(), key=lambda kv: -st.mean(kv[1]))))
    adv = [r["adv_samples"] for r in recs]
    strat = [r["strat_samples"] for r in recs]
    print(f"\nadv samples total {sum(adv):,} (mean {st.mean(adv):,.0f}/iter, "
          f"{sum(adv) / (n * recs[0].get('n_traversals', 30000)):.2f}/traversal); "
          f"strategy samples kept total {sum(strat):,}")
    print("\n| iter | adv_samples | strat_samples | adv_loss | buffer | strat_buffer | t_iter |\n"
          "|---:|---:|---:|---:|---:|---:|---:|")
    for r in recs:
        if r["iteration"] <= 4 or r["iteration"] % 25 == 0:
            print(f"| {r['iteration']} | {r['adv_samples']:,} | {r['strat_samples']:,} | "
                  f"{r['adv_loss']:.4f} | {r['buffer']:,} | {r['strategy_buffer']:,} | {r['t_iter']:.2f} |")
    print("\nstrategy fits: " + ", ".join(
        f"it{k}: loss {v['loss']:.4f} ({v['seconds']:.2f}s)" for k, v in summ["strategy_fits"].items()))
    if a.baseline:
        pat = re.compile(r"\[Iter\s+(\d+)/\d+\].*adv_samples=([\d,]+) strat_samples=([\d,]+) "
                         r"adv_loss=([\d.]+) strat_loss=([\d.]+)")
        print("\nbaseline (original code on this VM):")
        for m in pat.finditer(Path(a.baseline).read_text()):
            print(f"  iter {m[1]}: adv {m[2]} strat {m[3]} adv_loss {m[4]} strat_loss {m[5]}")


if __name__ == "__main__":
    main()
