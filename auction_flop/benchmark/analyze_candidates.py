"""Audit the first immutable candidate batch and paired prespecified contrasts."""
from statistics import NormalDist

from . import run


def main():
    catalog = run.read_json(run.ROOT / 'candidate_catalog.json')['candidates']
    summaries, values = {}, {}
    for name, candidate in catalog.items():
        directory = run.ROOT / 'evidence' / ('candidate-' + name)
        manifest, rows = run.audit(directory)
        if manifest['source_hashes']['candidate'] != candidate['sha256']:
            raise ValueError(f'{name}: source differs from the published catalog')
        summaries[name] = run.read_json(directory / 'summary.json')
        values[name] = run.candidate_blocks(rows)
        print(f"{name}: {summaries[name]['mean_points']:.5f}, audited {len(rows)} tables", flush=True)
        del rows
    baseline = values['strong_modular_baseline']
    z = NormalDist().inv_cdf(1 - .05 / (2 * 8))
    comparisons = []
    for name, vals in values.items():
        if name == 'strong_modular_baseline':
            continue
        if vals.keys() != baseline.keys():
            raise ValueError('Mismatched paired blocks')
        result = run.estimate({key: vals[key] - baseline[key] for key in vals})
        mean = result['mean_points']
        se = (result['ci95'][1] - mean) / 1.96
        comparisons.append({'candidate': name, 'difference_vs_modular_baseline': mean,
                            'simultaneous_ci95': [mean - z * se, mean + z * se]})
    _, reference_rows = run.audit(run.ROOT / 'evidence/calibration-hybrid_tournament_auction')
    frozen = run.candidate_blocks(reference_rows)
    control = run.estimate({key: baseline[key] - frozen[key] for key in baseline})
    result = {'benchmark': 'auction-flop-v1', 'tables_per_candidate': 4096,
              'comparisons': 8, 'interval_method': 'Paired panel-stratified blocks; Bonferroni normal 95%',
              'ranking': sorted([{'candidate': name, **summary} for name, summary in summaries.items()],
                                key=lambda s: (-s['mean_points'], s['candidate'])),
              'paired_vs_modular_baseline': comparisons,
              'modular_minus_frozen_hybrid_control': control,
              'all_sources_match_published_catalog': True, 'all_scores_audited': True}
    run.write_json(run.ROOT / 'evidence/candidate-comparison.json', result)


if __name__ == '__main__':
    main()
