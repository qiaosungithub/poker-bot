"""Recompute retained selection/history scores without the original worktree."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from . import run


def audit(directory):
    manifest = run.read_json(directory / 'manifest.json')
    data = gzip.decompress((directory / 'tables.jsonl.gz').read_bytes())
    if hashlib.sha256(data).hexdigest() != manifest['tables_sha256']:
        raise ValueError(f'{directory.name}: original raw-table hash mismatch')
    if run.digest(directory / 'runner_source.py') != manifest['runner_sha256']:
        raise ValueError(f'{directory.name}: original runner hash mismatch')
    rows = [json.loads(line) for line in data.splitlines()]
    config = manifest['config']
    recomputed = run.arena.summarize(rows, config)
    recorded = run.read_json(directory / 'summary.json')
    for key in ('scores', 'pairs', 'tables', 'independent_blocks'):
        if recomputed[key] != recorded[key]:
            raise ValueError(f'{directory.name}: summary mismatch in {key}')
    for row in rows:
        row['panel'] = 0
    run.audit_rows(rows, [config], config['blocks'])
    archive = directory / 'sources.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name, provenance in manifest['provenance'].items():
                for rel, expected in provenance['sha256'].items():
                    if hashlib.sha256(z.read(f'sources/{name}/{rel}')).hexdigest() != expected:
                        raise ValueError(f'{directory.name}: original policy hash mismatch')
    else:
        for name, provenance in manifest['provenance'].items():
            if run.source_hashes(directory / 'sources' / name) != provenance['sha256']:
                raise ValueError(f'{directory.name}: original policy hash mismatch')
    return {'tables': len(rows), 'independent_scores': True, 'schedule': True,
            'raw_sha256': True, 'original_runner_sha256': True, 'source_sha256': True}


def main():
    run.verify_lock()
    results = {}
    for directory in sorted((run.ROOT / 'evidence').iterdir()):
        if directory.is_dir() and (directory / 'runner_source.py').exists():
            results[directory.name] = audit(directory)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
