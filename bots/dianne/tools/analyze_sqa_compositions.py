"""Audit the two declared compositions and compare four paired contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', required=True, type=Path)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark_root.resolve()))
    from auction_flop.benchmark.run import audit, candidate_blocks, estimate, source_hashes, verify_lock
    spec = verify_lock()
    results = args.repo_root / 'bots/dianne/results'
    evidence = args.benchmark_root / 'auction_flop/evidence'
    directories = {
        'prior_classic': results / 'sqa-prior-classic-v1-macos',
        'prior_fragile': results / 'sqa-prior-fragile-v1-macos',
        'full_prior': evidence / 'candidate-practice_prior_full_v1',
        'classic': evidence / 'candidate-classic_hybrid_v1',
        'fragile': evidence / 'candidate-fragile_both_v1',
    }
    live_sources = {
        'prior_classic': args.repo_root / 'bots/dianne/sqa_prior_classic_v1',
        'prior_fragile': args.repo_root / 'bots/dianne/sqa_prior_fragile_v1',
        'full_prior': args.benchmark_root / 'auction_flop/candidates/practice_prior_full_v1',
        'classic': args.benchmark_root / 'auction_flop/candidates/classic_hybrid_v1',
        'fragile': args.benchmark_root / 'auction_flop/candidates/fragile_both_v1',
    }
    runs, blocks = {}, {}
    for name, directory in directories.items():
        manifest, rows = audit(directory)
        assert manifest['official_score'] and manifest['mode'] == 'evaluate'
        assert manifest['source_hashes']['candidate'] == source_hashes(live_sources[name]), name
        blocks[name] = candidate_blocks(rows)
        runs[name] = {
            'summary': json.loads((directory / 'summary.json').read_text()),
            'audit_passed': True,
            'tables_sha256': manifest['tables_sha256'],
            'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
            'candidate_source_hashes': manifest['source_hashes']['candidate'],
            'platform': 'macOS affinity adapter' if name.startswith('prior_') else 'SQA published Linux result',
        }
    contrasts = [('prior_classic', 'full_prior'), ('prior_fragile', 'full_prior'),
                 ('prior_classic', 'classic'), ('prior_fragile', 'fragile')]
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * len(contrasts)))
    comparisons = {}
    for left, right in contrasts:
        assert blocks[left].keys() == blocks[right].keys()
        stat = estimate({key: blocks[left][key] - blocks[right][key] for key in blocks[left]})
        mean = stat.pop('mean_points')
        se = (stat['ci95'][1] - stat['ci95'][0]) / (2 * 1.96)
        comparisons[f'{left}_minus_{right}'] = {
            **stat, 'difference_left_minus_right': mean,
            'ci95_simultaneous_four_contrasts': [mean - critical * se, mean + critical * se],
        }
    output = {
        'benchmark': spec['id'],
        'source_revision': 'c62cdaccb66cec9f0b49adcaa43327b88d2419b8',
        'runs': runs,
        'comparisons': comparisons,
        'scope': 'Four declared paired contrasts, panel-stratified seed blocks; fixed public evaluation seeds. No fresh-seed or live-field superiority claim.',
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value['summary']['mean_points'] for key, value in runs.items()}))


if __name__ == '__main__':
    main()
