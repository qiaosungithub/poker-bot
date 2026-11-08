"""Measure released and reproduced policies on the retained six-player corpus.

This measures the research decision function, not the official SDK or IPC path.
Existing result directories are never overwritten.
"""
import runtime_paths
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, defaultdict
import datetime
import gc
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'hu-benchmark'))
import run_benchmark as R
from fullhouse_onnx import ONNXLookup


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64) / 1e6
    return dict(n=len(x), p50_ms=float(np.quantile(x, .5)),
                p99_ms=float(np.quantile(x, .99)), p999_ms=float(np.quantile(x, .999)),
                max_ms=float(x.max()), over_1ms=int((x > 1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    corpus = ROOT.parent / 'hu-benchmark/multiway_latency_states.jsonl.gz'
    with gzip.open(corpus, 'rt') as stream:
        states = [json.loads(line) for line in stream]
    assert len(states) == 10000
    assert Counter(s['street'] for s in states) == dict.fromkeys(('preflop','flop','turn','river'), 2500)
    models = {
        'released': ROOT.parent / 'latency-survey/fullhouse_fp32.onnx',
        'reproduced_iter210': ROOT / 'eval/iter_00210.onnx',
    }
    R.initialize()
    sessions = {name: ONNXLookup(path) for name, path in models.items()}
    timings = {name: dict(wall=[], thread_cpu=[], by_street=defaultdict(list)) for name in models}
    legal_kinds = {'fold','check','call','raise','all_in'}
    started = time.perf_counter()
    for repeat in range(args.repeats):
        # Alternate order to reduce a fixed reference-first scheduling bias.
        names = list(models) if repeat % 2 == 0 else list(reversed(models))
        for name in names:
            R.HERO._DEEP_CFR = sessions[name]
            R.HERO._RNG = random.Random(20261108 + repeat)
            for state in states[:200]:
                assert R.HERO.decide(state)['action'] in legal_kinds
            gc.collect()
            was_enabled = gc.isenabled()
            gc.disable()
            try:
                for state in states:
                    begin = time.perf_counter_ns()
                    cpu_begin = time.thread_time_ns()
                    action = R.HERO.decide(state)
                    cpu_ns = time.thread_time_ns() - cpu_begin
                    wall_ns = time.perf_counter_ns() - begin
                    assert action['action'] in legal_kinds
                    timings[name]['wall'].append(wall_ns)
                    timings[name]['thread_cpu'].append(cpu_ns)
                    timings[name]['by_street'][state['street']].append(wall_ns)
            finally:
                if was_enabled:
                    gc.enable()
            print(json.dumps(dict(pass_number=repeat+1, model=name,
                                  measured=len(timings[name]['wall']))), flush=True)
    arrays = {f'{name}_{kind}_ns': np.asarray(values[kind], dtype=np.int64)
              for name, values in timings.items() for kind in ('wall','thread_cpu')}
    np.savez_compressed(output / 'timings.npz', **arrays)
    cpu_line = next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name')), 'unknown')
    result = dict(
        measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,
        environment=dict(cpu_model=cpu_line, logical_cpu=args.cpu, python=sys.version,
                         platform=platform.platform(), load_average=os.getloadavg(),
                         packages={p: importlib.metadata.version(p) for p in ('numpy','onnxruntime','eval7','pokerkit')}),
        protocol=dict(states=len(states), street_counts=dict(Counter(s['street'] for s in states)),
                      repeats=args.repeats, warmup_per_pass=200, gc_enabled_during_timing=False,
                      ort_intra_threads=1, ort_inter_threads=1, alternating_model_order=True,
                      scope='Warm state-dict to action-dict: feature construction, FP32 inference, policy/action logic. No official SDK, memory serialization, JSON, IPC, startup, game simulation or deadline penalties.',
                      legality='Only action-kind validation; the frozen strength adapter handles action bounds. Not an official legality/timeout acceptance test.'),
        corpus=dict(path=str(corpus),sha256=sha(corpus)),
        source_hashes={str(p):sha(p) for p in [Path(__file__), ROOT.parent/'hu-benchmark/run_benchmark.py',
                      ROOT.parent/'latency-survey/fullhouse_onnx.py', ROOT.parent/'latency-survey/fullhouse-bot/bot/bot.py',
                      ROOT.parent/'latency-survey/fullhouse-bot/bot/features.py']},
        results={name:dict(model=str(models[name]),model_sha256=sha(models[name]),
                          wall=stats(values['wall']),thread_cpu=stats(values['thread_cpu']),
                          by_street={street:stats(ns) for street,ns in values['by_street'].items()})
                 for name,values in timings.items()})
    with (output/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
