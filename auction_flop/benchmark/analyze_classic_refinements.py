"""Audit the two isolated classic refinements against the retained control."""
from statistics import NormalDist

from . import run


def main():
    names = ['classic_txnujk32_v1', 'classic_price_v1', 'classic_contribution_v1']
    catalogs = ['classic_v1', 'classic_price_v1', 'classic_contribution_v1']
    values, summaries = {}, {}
    for name, catalog in zip(names, catalogs):
        expected = run.read_json(run.ROOT / f'candidate_catalog_{catalog}.json')
        directory = run.ROOT / 'evidence' / ('candidate-' + name)
        manifest, rows = run.audit(directory)
        if manifest['source_hashes']['candidate'] != expected['sha256']:
            raise ValueError(f'{name}: source differs from the published catalog')
        values[name] = run.candidate_blocks(rows)
        summaries[name] = run.read_json(directory / 'summary.json')
    baseline = values[names[0]]
    z = NormalDist().inv_cdf(1 - .05 / (2 * 2))
    contrasts = []
    for name in names[1:]:
        if values[name].keys() != baseline.keys():
            raise ValueError('Mismatched paired blocks')
        result = run.estimate({k: values[name][k] - baseline[k] for k in baseline})
        mean = result['mean_points']
        se = (result['ci95'][1] - mean) / 1.96
        contrasts.append({'candidate': name, 'difference_vs_classic_control': mean,
                          'ci95': result['ci95'],
                          'simultaneous_ci95': [mean - z * se, mean + z * se]})
    output = {'benchmark': 'auction-flop-v1', 'tables_per_candidate': 4096,
              'control_reused': names[0], 'independent_blocks': 512,
              'interval_method': 'Paired panel-stratified blocks; Bonferroni normal 95% for these two contrasts only, not the ongoing strategy search',
              'summaries': summaries, 'paired_vs_classic_control': contrasts,
              'all_sources_match_published_catalog': True, 'all_scores_audited': True}
    run.write_json(run.ROOT / 'evidence/classic-refinements-v1-comparison.json', output)
    print(output, flush=True)


if __name__ == '__main__':
    main()
