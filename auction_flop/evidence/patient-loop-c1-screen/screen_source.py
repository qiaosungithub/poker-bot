"""Independent-seed tuning screen using the frozen v1 engine and opponents."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from auction_flop.benchmark import run

_CONFIGS = None
_CACHE = {}


def initialize(configs):
    global _CONFIGS
    _CONFIGS = configs


def block(task):
    name, panel, index = task
    key = name, panel
    config = _CONFIGS[name][panel]
    if key not in _CACHE:
        run.arena.initialize(config)
        _CACHE[key] = run.arena._WORKER
    run.arena._WORKER = _CACHE[key]
    rows = run.arena.block((index, f"{config['seed']}:{index}",
                            [e['name'] for e in config['entries']]))
    for row in rows:
        row['panel'] = panel
    return name, rows


def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
                  {**os.environ, 'PYTHONHASHSEED': '0'})
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=16)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--cpus', type=Path, default=Path('.runtime/simple-tuning-cpus.json'))
    parser.add_argument('--seed', default='simple-tuning-v2-screen-20261110')
    args = parser.parse_args()
    assert args.blocks >= 2 and args.workers >= 1
    cpus = json.loads(args.cpus.read_text())
    os.sched_setaffinity(0, cpus)
    spec = run.verify_lock()
    assert args.seed != spec['evaluation']['seed']
    started = time.monotonic()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'
    registry = run.read_json(run.ROOT / 'strategies.json')
    hashes = {}
    for name, definition in registry.items():
        if 'path' in definition:
            hashes[name] = run.copy_source(run.ROOT / definition['path'], sources / name)
    arms = run.read_json(args.arms)
    configs = {}
    for name, source in arms.items():
        target = sources / ('candidate_' + name)
        hashes['candidate_' + name] = run.copy_source(run.ROOT / source, target)
        configs[name] = []
        for panel, opponents in enumerate(spec['evaluation']['panels']):
            entries = [{'name': 'candidate', 'policy': 'candidate', 'snapshot': str(target)}]
            entries += [run.entry(f'opponent_{i}', policy, sources)
                        for i, policy in enumerate(opponents)]
            configs[name].append(dict(seed=f'{args.seed}:panel:{panel}',
                blocks=args.blocks, hand_cap=250, entries=entries, compare=['candidate']))
    manifest = dict(kind='tuning screen, not permanent v1 score',
        started=datetime.now(timezone.utc).isoformat(), benchmark=spec['id'],
        benchmark_sha256=run.digest(run.ROOT/'benchmark/v1.json'),
        arms=arms, source_hashes=hashes, panels=configs, blocks=args.blocks,
        workers=args.workers, cpus=cpus, python=sys.version,
        dependencies=spec['dependencies'], runner_sha256=run.digest(__file__),
        timing='Untimed direct engine. Screen arms share the process pool.')
    run.write_json(out / 'manifest.json', manifest)
    (out/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    tasks = [(name, p, b) for p in range(4) for b in range(args.blocks) for name in arms]
    rows = {name: [] for name in arms}
    with ExitStack() as stack:
        streams = {name: stack.enter_context(gzip.open(out/f'{name}.jsonl.gz', 'wt')) for name in arms}
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(configs,)))
        futures = [pool.submit(block, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            name, results = future.result()
            rows[name].extend(results)
            for row in results:
                streams[name].write(json.dumps(row, separators=(',', ':')) + '\n')
            if done % 64 == 0 or done == len(tasks):
                print(json.dumps(dict(blocks=done, total_blocks=len(tasks),
                                      seconds=round(time.monotonic()-started, 2))), flush=True)
    scores = []
    for name, records in rows.items():
        run.audit_rows(records, configs[name], args.blocks)
        failures = sum(e['failures'] for r in records for e in r['entries'])
        hero = [next(e for e in r['entries'] if e['name']=='candidate') for r in records]
        result = run.estimate(run.candidate_blocks(records))
        result.update(name=name, tables=len(records), total_failures=failures,
            candidate_decisions=sum(e['decisions'] for e in hero),
            mean_spent_flopchips=sum(e['spent_flopchips'] for e in hero)/len(hero),
            mean_auctions=sum(e['auction_calls'] for e in hero)/len(hero),
            table_sha256=run.digest(out/f'{name}.jsonl.gz'))
        scores.append(result)
    for name, expected in hashes.items():
        assert run.source_hashes(sources/name) == expected
    summary = dict(scores=sorted(scores, key=lambda s:(-s['mean_points'], s['name'])),
        seconds=time.monotonic()-started, total_tables=sum(len(r) for r in rows.values()),
        all_scores_audited=True, selection_only=True,
        confidence_scope='Unadjusted descriptive intervals; arm selection is optimistic.')
    run.write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if any(s['total_failures'] for s in scores):
        raise SystemExit('Screen contains failures; inspect raw records.')


if __name__ == '__main__':
    main()
