"""Independently bind transfer results to the predeclared field and source manifest."""
import argparse
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

from ..benchmark import run
from .validate_finalist import audit


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    args = parser.parse_args(); directory = args.directory; protocol = directory / 'protocol.json'
    spec = run.read_json(protocol); run.verify_lock(); output = {}
    z = NormalDist().inv_cdf(1 - .05 / 4)
    for field, field_spec in spec['fields'].items():
        values, summaries = {}, {}
        for candidate in (spec['control'], spec['candidate']):
            path = directory / (field + '-' + candidate)
            manifest, rows = audit(path, protocol)
            assert manifest['candidate'] == candidate and manifest['field'] == field
            assert datetime.fromisoformat(manifest['started']) >= datetime.fromisoformat(spec['frozen_at'])
            assert len(manifest['panels']) == len(field_spec['panels'])
            for i, (panel, opponents) in enumerate(zip(manifest['panels'], field_spec['panels'])):
                assert panel['seed'] == field_spec['seed'] + f':panel:{i}'
                assert panel['blocks'] == spec['blocks_per_panel'] and panel['hand_cap'] == 250
                assert [e['name'] for e in panel['entries']] == ['candidate'] + [f'opponent_{j}' for j in range(7)]
                assert [e['policy'] for e in panel['entries']] == [candidate] + opponents
                for entry in panel['entries']:
                    expected = spec['policies'][entry['policy']]
                    assert Path(entry['snapshot']).name == expected['betting']
                    assert (Path(entry['auction_snapshot']).name if 'auction_snapshot' in entry else None) == expected.get('auction')
                assert not panel['seed'].startswith(run.verify_lock()['evaluation']['seed'])
            values[candidate] = run.candidate_blocks(rows)
            summaries[candidate] = run.read_json(path / 'summary.json')
        control, candidate = values[spec['control']], values[spec['candidate']]
        assert control.keys() == candidate.keys()
        result = run.estimate({k: candidate[k] - control[k] for k in control})
        mean = result['mean_points']; se = (result['ci95'][1] - mean) / 1.96
        output[field] = {'summaries': summaries, 'difference_candidate_minus_control': mean,
                         'ci95': result['ci95'], 'simultaneous_ci95': [mean - z * se, mean + z * se],
                         'independent_blocks': len(control)}
    result = {'protocol': spec['id'], 'protocol_sha256': run.digest(protocol),
              'frozen_at': spec['frozen_at'], 'fields': output,
              'all_sources_rules_schedules_scores_audited': True,
              'intervals': spec['intervals'], 'interpretation': spec['interpretation']}
    run.write_json(directory / 'paired-transfer-comparison.json', result)
    print(result, flush=True)


if __name__ == '__main__': main()
