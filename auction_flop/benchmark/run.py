"""Versioned, source-locked Auction Flop regression benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
from statistics import fmean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
# Always load the released SDK in this checkout, even with another editable SDK installed.
for package in ('agent-protocol', 'engine', 'agent-sdk'):
    sys.path.insert(0, str(ROOT / 'vendor' / 'sdk' / package / 'src'))
from . import arena


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and not any(part.startswith('.') for part in p.relative_to(path).parts)
            and p.suffix not in ('.pyc', '.pyo')}


def verify_lock(root=ROOT):
    spec_path = root / 'benchmark' / 'v1.json'
    if digest(spec_path) != (root / 'benchmark' / 'v1.sha256').read_text().strip():
        raise ValueError('Benchmark definition changed; create a new version.')
    spec = read_json(spec_path)
    for name, expected in spec['sha256'].items():
        if not (root / name).is_file() or digest(root / name) != expected:
            raise ValueError(f'Frozen file changed or missing: {name}')
    for name in spec['source_directories']:
        expected = {p[len(name) + 1:]: h for p, h in spec['sha256'].items()
                    if p.startswith(name + '/')}
        if source_hashes(root / name) != expected:
            raise ValueError(f'Frozen directory changed: {name}')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Benchmark v1 requires Python 3.12.')
    for name, expected in spec['dependencies'].items():
        if version(name) != expected:
            raise ValueError(f'{name} must be {expected}, found {version(name)}')
    actual_rules = json.loads(json.dumps(asdict(arena.TableRules(
        variant=arena.VARIANT, players=8, hand_cap=250,
        decision_ms=60000, time_bank_ms=0))))
    if actual_rules != spec['engine_rules']:
        raise ValueError('Engine rules do not match v1.')
    return spec


def entry(name, policy, sources):
    registry = read_json(ROOT / 'strategies.json')
    if policy == 'candidate':
        return {'name': name, 'policy': policy, 'snapshot': str(sources / policy)}
    definition = registry[policy]
    if 'betting' in definition:
        return {'name': name, 'policy': policy,
                'snapshot': str(sources / definition['betting']),
                'auction_snapshot': str(sources / definition['auction'])}
    return {'name': name, 'policy': policy, 'snapshot': str(sources / policy)}


def copy_source(source, target):
    hashes = source_hashes(source)
    if 'agent.py' not in hashes:
        raise ValueError(f'Missing agent.py: {source}')
    target.mkdir(parents=True, exist_ok=False)
    for rel in hashes:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
    if source_hashes(target) != hashes:
        raise ValueError('Source changed during snapshot; retry with a stable source.')
    return hashes


def read_rows(directory):
    with gzip.open(Path(directory) / 'tables.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def audit_rows(rows, panels, blocks):
    """Validate the full schedule and independently recompute every payout."""
    if len(rows) != len(panels) * blocks * 8:
        raise ValueError('Incomplete result: missing tables.')
    seen = set()
    for row in rows:
        panel, block, rotation = row['panel'], row['block'], row['rotation']
        if not (0 <= panel < len(panels) and 0 <= block < blocks and 0 <= rotation < 8):
            raise ValueError('Out-of-range schedule coordinates.')
        key = panel, block, rotation
        if key in seen:
            raise ValueError('Duplicate table.')
        seen.add(key)
        config = panels[panel]
        seed = f"{config['seed']}:{block}"
        names = [e['name'] for e in config['entries']]
        random.Random(seed + ':seats').shuffle(names)
        names = names[rotation:] + names[:rotation]
        if row['seed'] != seed or [e['name'] for e in row['entries']] != names:
            raise ValueError('Seed or seat rotation mismatch.')
        if [e['seat'] for e in row['entries']] != list(range(8)):
            raise ValueError('Incorrect physical seat IDs.')
        stacks = [e['stack'] for e in row['entries']]
        if any(s < 0 for s in stacks) or sum(stacks) != 16000:
            raise ValueError('Chip conservation failed.')
        if list(row['payouts']) != [50, 30, 20] or not (1 <= row['hands'] <= 250):
            raise ValueError('Wrong payouts or hand cap.')
        if len(set(row['bust_order'])) != len(row['bust_order']):
            raise ValueError('Duplicate eliminated seat.')
        if set(row['bust_order']) != {i for i, stack in enumerate(stacks) if stack == 0}:
            raise ValueError('Elimination order disagrees with stacks.')
        scores = arena.independent_scores(stacks, row['bust_order'], row['ties'], row['payouts'])
        if abs(sum(scores) - 100) > 1e-7:
            raise ValueError('Payout conservation failed.')
        for e, score in zip(row['entries'], scores):
            if not math.isfinite(e['points']) or abs(e['points'] - score) > 1e-7:
                raise ValueError('Independent score audit failed.')
            if not 0 <= e['failures'] <= e['decisions']:
                raise ValueError('Invalid decision counters.')
            if e['spent_flopchips'] + e['remaining_flopchips'] != 1000:
                raise ValueError('Auction currency conservation failed.')
            if min(e['spent_flopchips'], e['remaining_flopchips']) < 0:
                raise ValueError('Negative auction currency.')
    return True


def candidate_blocks(rows):
    values = {}
    for row in rows:
        hero = next(e for e in row['entries'] if e['name'] == 'candidate')
        values.setdefault((row['panel'], row['block']), []).append(hero['points'])
    return {key: fmean(v) for key, v in values.items()}


def estimate(values):
    """Equal panel weights, independent blocks within each panel."""
    panels = {}
    for (panel, _), value in values.items():
        panels.setdefault(panel, []).append(value)
    mean = fmean(fmean(v) for v in panels.values())
    se = (math.sqrt(sum(stdev(v) ** 2 / len(v) for v in panels.values())) / len(panels)
          if all(len(v) > 1 for v in panels.values()) else None)
    return {'mean_points': mean,
            'ci95': [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            'independent_blocks': len(values)}


def execute(args):
    spec = verify_lock()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.smoke_blocks is not None and args.smoke_blocks < 1:
        raise ValueError('smoke-blocks must be positive')
    out = args.out.resolve()
    # Snapshot before starting workers; never evaluate a moving worktree.
    if args.candidate and (out == args.candidate.resolve() or args.candidate.resolve() in out.parents):
        raise ValueError('Output must be outside the candidate source folder.')
    out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'
    registry = read_json(ROOT / 'strategies.json')
    hashes = {}
    for policy, definition in registry.items():
        if 'path' in definition:
            hashes[policy] = copy_source(ROOT / definition['path'], sources / policy)
    if args.candidate:
        hashes['candidate'] = copy_source(args.candidate.resolve(), sources / 'candidate')
    if args.command == 'selection':
        schedule = spec['selection']
        blocks = args.smoke_blocks or schedule['blocks']
        panels = [{'seed': schedule['seed'], 'blocks': blocks, 'hand_cap': 250,
                   'entries': [entry(n, n, sources) for n in schedule['roster']],
                   'compare': schedule['candidates']}]
    else:
        schedule = spec['evaluation']
        blocks = args.smoke_blocks or schedule['blocks_per_panel']
        panels = []
        for i, opponents in enumerate(schedule['panels']):
            entries = [entry('candidate', args.reference or 'candidate', sources)]
            entries += [entry(f'opponent_{j}', n, sources) for j, n in enumerate(opponents)]
            panels.append({'seed': f"{schedule['seed']}:panel:{i}", 'blocks': blocks,
                           'hand_cap': 250, 'entries': entries, 'compare': ['candidate']})
    manifest = {'benchmark': spec['id'], 'benchmark_sha256': digest(ROOT / 'benchmark/v1.json'),
                'official_score': args.smoke_blocks is None,
                'mode': args.command, 'candidate': args.reference or 'external',
                'blocks_per_panel': blocks, 'panels': panels, 'source_hashes': hashes,
                'started': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'dependencies': spec['dependencies'],
                'workers': args.workers, 'timing': 'Direct engine; no clock penalties.'}
    write_json(out / 'manifest.json', manifest)
    rows = []
    with gzip.open(out / 'tables.jsonl.gz', 'wt') as stream:
        for i, config in enumerate(panels):
            names = [e['name'] for e in config['entries']]
            tasks = [(b, f"{config['seed']}:{b}", names) for b in range(blocks)]
            with ProcessPoolExecutor(max_workers=args.workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=arena.initialize, initargs=(config,)) as pool:
                futures = [pool.submit(arena.block, task) for task in tasks]
                done = 0
                for future in as_completed(futures):
                    result = future.result()
                    for row in result:
                        row['panel'] = i
                        stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                    stream.flush()
                    rows.extend(result)
                    done += 1
                    if done % 32 == 0 or done == blocks:
                        print(f'panel {i + 1}/{len(panels)}: {done}/{blocks} blocks', flush=True)
    audit_rows(rows, panels, blocks)
    failures = sum(e['failures'] for r in rows for e in r['entries'])
    if args.command == 'selection':
        summary = arena.summarize(rows, panels[0])
    else:
        summary = estimate(candidate_blocks(rows))
        heroes = [next(e for e in r['entries'] if e['name'] == 'candidate') for r in rows]
        summary.update({key: fmean(e[source] for e in heroes) for key, source in (
            ('mean_chips', 'stack'), ('mean_spent_flopchips', 'spent_flopchips'),
            ('mean_remaining_flopchips', 'remaining_flopchips'))})
        summary['candidate_decisions'] = sum(e['decisions'] for e in heroes)
        summary['candidate_failures'] = sum(e['failures'] for e in heroes)
    summary.update(tables=len(rows), total_failures=failures,
                   valid=failures == 0, official_score=manifest['official_score'],
                   all_scores_audited=True)
    manifest.update(completed=datetime.now(timezone.utc).isoformat(),
                    tables_sha256=digest(out / 'tables.jsonl.gz'))
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit('Invalid benchmark: failed decisions; inspect raw records.')


def audit(directory):
    spec = verify_lock()
    manifest = read_json(directory / 'manifest.json')
    if manifest['benchmark_sha256'] != digest(ROOT / 'benchmark/v1.json'):
        raise ValueError('Result belongs to a different benchmark.')
    if digest(directory / 'tables.jsonl.gz') != manifest['tables_sha256']:
        raise ValueError('Raw records changed.')
    for name, expected in manifest['source_hashes'].items():
        if source_hashes(directory / 'sources' / name) != expected:
            raise ValueError(f'Run source changed: {name}')
    # Check source fingerprints against v1, not just a self-reported run manifest.
    registry = read_json(ROOT / 'strategies.json')
    for name, definition in registry.items():
        if 'path' in definition and manifest['source_hashes'].get(name) != definition['sha256']:
            raise ValueError(f'Opponent is not the frozen version: {name}')
    expected_blocks = (spec['selection']['blocks'] if manifest['mode'] == 'selection'
                       else spec['evaluation']['blocks_per_panel'])
    if manifest['official_score'] and manifest['blocks_per_panel'] != expected_blocks:
        raise ValueError('Official result used the wrong sample size.')
    if manifest['mode'] == 'evaluate':
        if len(manifest['panels']) != len(spec['evaluation']['panels']):
            raise ValueError('Wrong number of panels.')
        for i, panel in enumerate(manifest['panels']):
            if panel['seed'] != f"{spec['evaluation']['seed']}:panel:{i}":
                raise ValueError('Wrong evaluation seed prefix.')
            entries = panel['entries']
            if [e['name'] for e in entries] != ['candidate'] + [f'opponent_{j}' for j in range(7)]:
                raise ValueError('Wrong slot identities.')
            if [e['policy'] for e in entries[1:]] != spec['evaluation']['panels'][i]:
                raise ValueError('Opponent mixture changed.')
    elif manifest['mode'] == 'selection':
        if len(manifest['panels']) != 1:
            raise ValueError('Selection requires one panel.')
        panel = manifest['panels'][0]
        if (panel['seed'] != spec['selection']['seed'] or
                [e['policy'] for e in panel['entries']] != spec['selection']['roster']):
            raise ValueError('Selection schedule changed.')
    else:
        raise ValueError('Unknown run mode.')
    for panel in manifest['panels']:
        for e in panel['entries']:
            expected = entry(e['name'], e['policy'], Path('sources'))
            for key in ('snapshot', 'auction_snapshot'):
                if (Path(e[key]).name if key in e else None) != (
                        Path(expected[key]).name if key in expected else None):
                    raise ValueError('Source mapping does not match the declared policy.')
    rows = read_rows(directory)
    audit_rows(rows, manifest['panels'], manifest['blocks_per_panel'])
    if any(e['failures'] for r in rows for e in r['entries']):
        raise ValueError('Failed decisions invalidate this score.')
    summary = read_json(directory / 'summary.json')
    if manifest['mode'] == 'evaluate':
        recomputed = estimate(candidate_blocks(rows))
        for key in ('mean_points', 'ci95', 'independent_blocks'):
            if summary.get(key) != recomputed[key]:
                raise ValueError(f'Summary does not match the raw records: {key}')
    return manifest, rows


def compare(left, right):
    lm, lr = audit(left)
    rm, rr = audit(right)
    if any(m['mode'] != 'evaluate' or not m['official_score'] for m in (lm, rm)):
        raise ValueError('Paired comparison requires two complete evaluation runs.')
    lv, rv = candidate_blocks(lr), candidate_blocks(rr)
    if lv.keys() != rv.keys():
        raise ValueError('Mismatched schedules.')
    result = estimate({k: lv[k] - rv[k] for k in lv})
    result['difference_left_minus_right'] = result.pop('mean_points')
    result['scope'] = 'One prespecified paired comparison; 95% normal interval over seed blocks.'
    return result


def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, '-m', 'auction_flop.benchmark.run',
                                  *sys.argv[1:]], {**os.environ, 'PYTHONHASHSEED': '0'})
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ('evaluate', 'selection'):
        p = subs.add_parser(command)
        if command == 'evaluate':
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument('--candidate', type=Path)
            g.add_argument('--reference', choices=list(read_json(ROOT / 'strategies.json')))
        else:
            p.set_defaults(candidate=None, reference=None)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--smoke-blocks', type=int, help='Non-score smoke test only; cannot be compared.')
    subs.add_parser('verify')
    p = subs.add_parser('audit'); p.add_argument('directory', type=Path)
    p = subs.add_parser('compare'); p.add_argument('left', type=Path); p.add_argument('right', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps({'verified': verify_lock()['id']}))
    elif args.command == 'audit':
        manifest, rows = audit(args.directory)
        print(json.dumps({'verified': manifest['benchmark'], 'tables': len(rows)}))
    elif args.command == 'compare':
        print(json.dumps(compare(args.left, args.right), indent=2))
    else:
        execute(args)


if __name__ == '__main__':
    main()
