"""Compose SQA's full opening prior with two unchanged postflop policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def hashes(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob('*.py'))}


def build(source: Path, output: Path, revision: str) -> dict:
    donors = source / 'auction_flop' / 'candidates'
    prior = donors / 'practice_prior_full_v1'
    records = {}
    for target, base in [('sqa_prior_classic_v1', 'classic_hybrid_v1'),
                         ('sqa_prior_fragile_v1', 'fragile_both_v1')]:
        destination = output / target
        shutil.copytree(donors / base, destination)
        before = hashes(destination)
        for name in ('independent_preflop.py', 'distilled_prior.py'):
            shutil.copyfile(prior / name, destination / name)
        after = hashes(destination)
        changed = sorted(name for name in before if before[name] != after[name])
        added = sorted(set(after) - set(before))
        assert changed == ['independent_preflop.py'], changed
        assert added == ['distilled_prior.py'], added
        records[target] = {
            'base': base,
            'preflop_donor': 'practice_prior_full_v1',
            'source_revision': revision,
            'changed_files': changed,
            'added_files': added,
            'source_hashes': after,
            'base_hashes': before,
        }
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    records = build(args.source_root, args.output, args.revision)
    args.manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')
