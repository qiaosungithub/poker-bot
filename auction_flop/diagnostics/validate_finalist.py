"""Frozen, separately named fresh-seed and style-field transfer checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
import os
from pathlib import Path

from ..benchmark import run


def worker(config, task):
    run.arena.initialize(config)
    return run.arena.block(task)


def audit(directory, protocol):
    manifest = run.read_json(directory / 'manifest.json')
    spec = run.read_json(protocol)
    assert run.digest(protocol) == protocol.with_suffix('.sha256').read_text().strip()
    assert manifest['protocol_sha256'] == run.digest(protocol)
    assert run.digest(directory / 'tables.jsonl.gz') == manifest['tables_sha256']
    for name, entry in spec['sources'].items():
        assert run.source_hashes(directory / 'sources' / name) == entry['sha256']
    rows = run.read_rows(directory)
    run.audit_rows(rows, manifest['panels'], spec['blocks_per_panel'])
    assert not any(e['failures'] for r in rows for e in r['entries'])
    result = run.estimate(run.candidate_blocks(rows))
    summary = run.read_json(directory / 'summary.json')
    for key in result: assert summary[key] == result[key]
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--field', choices=['fresh_v1_field', 'style_field'], required=True)
    parser.add_argument('--candidate', choices=['hybrid_both', 'prior_hybrid_both_v1'], required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(); assert os.environ.get('PYTHONHASHSEED') == '0'
    run.verify_lock(); spec = run.read_json(args.protocol)
    assert run.digest(args.protocol) == args.protocol.with_suffix('.sha256').read_text().strip()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    for name, entry in spec['sources'].items():
        assert run.copy_source(run.ROOT / entry['path'], out / 'sources' / name) == entry['sha256']
    def entry(name, policy):
        definition = spec['policies'][policy]
        result = {'name': name, 'policy': policy, 'snapshot': str(out / 'sources' / definition['betting'])}
        if 'auction' in definition: result['auction_snapshot'] = str(out / 'sources' / definition['auction'])
        return result
    field = spec['fields'][args.field]; panels = []
    for i, opponents in enumerate(field['panels']):
        panels.append({'seed': field['seed'] + f':panel:{i}', 'blocks': spec['blocks_per_panel'],
                       'hand_cap': 250, 'entries': [entry('candidate', args.candidate)] +
                       [entry(f'opponent_{j}', policy) for j, policy in enumerate(opponents)]})
    manifest = {'protocol': spec['id'], 'protocol_sha256': run.digest(args.protocol), 'field': args.field,
                'candidate': args.candidate, 'panels': panels, 'started': datetime.now(timezone.utc).isoformat(),
                'source_hashes': {k: v['sha256'] for k, v in spec['sources'].items()},
                'timing': 'Untimed strength only', 'candidate_frozen_before_unblinding': True}
    run.write_json(out / 'manifest.json', manifest)
    tasks = [(i, config, (b, config['seed'] + f':{b}', [e['name'] for e in config['entries']]))
             for i, config in enumerate(panels) for b in range(spec['blocks_per_panel'])]
    rows = []
    with gzip.open(out / 'tables.jsonl.gz', 'wt', compresslevel=1) as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(worker, config, task): i for i, config, task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                for row in future.result():
                    row['panel'] = futures[future]; rows.append(row)
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                if count % 32 == 0: stream.flush(); print(count, '/', len(tasks), 'blocks', flush=True)
    run.audit_rows(rows, panels, spec['blocks_per_panel'])
    summary = run.estimate(run.candidate_blocks(rows))
    summary.update(tables=len(rows), total_failures=sum(e['failures'] for r in rows for e in r['entries']))
    assert summary['total_failures'] == 0
    manifest.update(completed=datetime.now(timezone.utc).isoformat(), tables_sha256=run.digest(out / 'tables.jsonl.gz'))
    run.write_json(out / 'manifest.json', manifest); run.write_json(out / 'summary.json', summary)
    audit(out, args.protocol); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
