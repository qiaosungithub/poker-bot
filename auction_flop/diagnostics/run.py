"""Separate Hboth diagnostic replay, with mandatory parity to retained v1 tables."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

from ..benchmark import run as benchmark
from . import observe


def semantic_hash(row):
    return hashlib.sha256(json.dumps({k: v for k, v in row.items() if k != 'seconds'},
                                    sort_keys=True).encode()).hexdigest()


def execute_block(config, task, directory, digest_only):
    observe.initialize(config, directory, digest_only)
    return observe.block(task)


def action_hashes(directory):
    result = {}
    for path in Path(directory).glob('*.jsonl.gz'):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'table':
                    result[row['panel'], row['block'], row['rotation']] = row['action_sha256']
    return result


def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('Set PYTHONHASHSEED=0, matching the frozen runner.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks-per-panel', type=int, default=8)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--validate-actions', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.blocks_per_panel <= 128 and args.workers > 0
    benchmark.verify_lock()
    reference = benchmark.ROOT / 'evidence/candidate-hybrid_both'
    ref_manifest, reference_rows = benchmark.audit(reference)
    expected_sources = next(x for x in benchmark.read_json(
        benchmark.ROOT / 'candidate_catalog_followup_v1.json')['candidates'] if x['name'] == 'hybrid_both')['sha256']
    assert ref_manifest['source_hashes']['candidate'] == expected_sources
    expected = {(r['panel'], r['block'], r['rotation']): semantic_hash(r) for r in reference_rows
                if r['block'] < args.blocks_per_panel}
    del reference_rows
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'trace').mkdir()
    (out / 'harness').mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source, out / 'harness' / source.name)
    for name, hashes in ref_manifest['source_hashes'].items():
        assert benchmark.copy_source(reference / 'sources' / name, out / 'sources' / name) == hashes
    panels = json.loads(json.dumps(ref_manifest['panels']))
    for panel in panels:
        panel['blocks'] = args.blocks_per_panel
        for entry in panel['entries']:
            for key in ('snapshot', 'auction_snapshot'):
                if key in entry:
                    entry[key] = str(out / 'sources' / Path(entry[key]).name)
    manifest = {**ref_manifest, 'benchmark': 'auction-flop-diagnostics-v1',
                'official_score': False, 'diagnostic_only': True,
                'blocks_per_panel': args.blocks_per_panel, 'panels': panels,
                'started': datetime.now(timezone.utc).isoformat(), 'workers': args.workers,
                'reference': 'evidence/candidate-hybrid_both',
                'reference_raw_sha256': ref_manifest['tables_sha256'],
                'harness_sha256': benchmark.source_hashes(out / 'harness'),
                'notes': 'Known-seed descriptive replay. No new ranking score or policy change. Intermediate ICM is not reward or causal regret.'}
    for stale in ('completed', 'tables_sha256'):
        manifest.pop(stale, None)
    benchmark.write_json(out / 'manifest.json', manifest)
    tasks = [(panel, (i, block, f"{panel['seed']}:{block}", [e['name'] for e in panel['entries']]))
             for i, panel in enumerate(panels) for block in range(args.blocks_per_panel)]
    rows, traces = [], {}
    with gzip.open(out / 'tables.jsonl.gz', 'wt', compresslevel=1) as stream:
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(execute_block, config, task, out / 'trace', False) for config, task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                result, filename, digest = future.result()
                traces[filename] = digest
                for row in result:
                    key = row['panel'], row['block'], row['rotation']
                    if semantic_hash(row) != expected[key]:
                        raise ValueError(f'Instrumentation changed retained result: {key}')
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                stream.flush()
                rows.extend(result)
                print(f'{i}/{len(tasks)} blocks; {len(rows)} exact retained-table matches', flush=True)
    benchmark.audit_rows(rows, panels, args.blocks_per_panel)
    assert len(rows) == len(expected)
    if args.validate_actions:
        (out / 'actions_only').mkdir()
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(execute_block, config, task, out / 'actions_only', True) for config, task in tasks]
            for future in as_completed(futures):
                result, _, _ = future.result()
                for row in result:
                    assert semantic_hash(row) == expected[row['panel'], row['block'], row['rotation']]
        if action_hashes(out / 'trace') != action_hashes(out / 'actions_only'):
            raise ValueError('Diagnostic feature computation changed the action stream.')
        manifest['action_parity_tables'] = len(rows)
    summary = benchmark.estimate(benchmark.candidate_blocks(rows))
    summary.update(tables=len(rows), candidate_failures=sum(
        e['failures'] for r in rows for e in r['entries'] if e['name'] == 'candidate'),
        total_failures=sum(e['failures'] for r in rows for e in r['entries']),
        official_score=False, diagnostic_only=True, retained_table_parity=len(rows))
    assert summary['total_failures'] == 0
    manifest.update(completed=datetime.now(timezone.utc).isoformat(),
                    tables_sha256=benchmark.digest(out / 'tables.jsonl.gz'), trace_sha256=traces,
                    retained_table_parity=len(rows))
    benchmark.write_json(out / 'manifest.json', manifest)
    benchmark.write_json(out / 'summary.json', summary)
    benchmark.audit(out)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
