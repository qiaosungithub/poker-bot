"""Read back screening artifacts; reject source, score or schedule corruption."""
import copy
import gzip
import json
from pathlib import Path
import sys

from auction_flop.benchmark import run


def audit(directory):
    directory = Path(directory)
    run.verify_lock()
    manifest = run.read_json(directory/'manifest.json')
    summary = run.read_json(directory/'summary.json')
    assert manifest['benchmark_sha256'] == run.digest(run.ROOT/'benchmark/v1.json')
    for name, expected in manifest['source_hashes'].items():
        assert run.source_hashes(directory/'sources'/name) == expected
    for result in summary['scores']:
        path = directory / (result['name'] + '.jsonl.gz')
        assert run.digest(path) == result['table_sha256']
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        panels = manifest['panels'][result['name']]
        run.audit_rows(rows, panels, manifest['blocks'])
        assert not any(e['failures'] for row in rows for e in row['entries'])
        recomputed = run.estimate(run.candidate_blocks(rows))
        assert all(result[key] == recomputed[key] for key in recomputed)
        # A corrupt score must be rejected, never silently normalized.
        bad = copy.deepcopy(rows)
        bad[0]['entries'][0]['points'] += 1
        try:
            run.audit_rows(bad, panels, manifest['blocks'])
        except ValueError as exc:
            assert 'score audit' in str(exc)
        else:
            raise AssertionError('Corrupt score accepted')
    return {'audited_arms': len(summary['scores']), 'tables': summary['total_tables'],
            'corrupt_score_rejected_for_each_arm': True}


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1]), indent=2))
