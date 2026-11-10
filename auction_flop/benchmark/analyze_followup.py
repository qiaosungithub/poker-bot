"""Audit the retained controls, three follow-ups and hybrid refinement 2x2."""
from statistics import NormalDist

from . import run


def intervals(contrasts):
    z = NormalDist().inv_cdf(1 - .05 / (2 * len(contrasts)))
    output = {}
    for name, blocks in contrasts.items():
        result = run.estimate(blocks)
        mean = result['mean_points']
        se = (result['ci95'][1] - mean) / 1.96
        output[name] = {'difference': mean, 'ci95': result['ci95'],
                        'simultaneous_ci95': [mean - z * se, mean + z * se]}
    return output


def main():
    catalog = run.read_json(run.ROOT / 'candidate_catalog_followup_v1.json')
    expected = {x['name']: x['sha256'] for x in catalog['candidates']}
    for name, path in [('classic_hybrid_v1', 'candidate_catalog_classic_hybrid_v1.json'),
                       ('classic_txnujk32_v1', 'candidate_catalog_classic_v1.json')]:
        expected[name] = run.read_json(run.ROOT / path)['sha256']
    expected['corrected_preflop_loose_v2'] = run.read_json(
        run.ROOT / 'candidate_catalog_corrected_v2.json')['corrected-preflop-loose-v2']['source_sha256']
    values, summaries = {}, {}
    for name, hashes in expected.items():
        directory = run.ROOT / 'evidence' / ('candidate-' + name)
        manifest, rows = run.audit(directory)
        if manifest['source_hashes']['candidate'] != hashes:
            raise ValueError(f'{name}: source differs from the published catalog')
        values[name] = run.candidate_blocks(rows)
        summaries[name] = run.read_json(directory / 'summary.json')
        print(name, summaries[name]['mean_points'], 'audited', flush=True)
    keys = values['classic_hybrid_v1'].keys()
    if any(v.keys() != keys for v in values.values()):
        raise ValueError('Mismatched paired blocks')
    pairs = [('classic_tournament_v1', 'classic_hybrid_v1'),
             ('classic_tournament_v1', 'classic_txnujk32_v1'),
             ('classic_flop_value_v1', 'corrected_preflop_loose_v2'),
             ('preflop_price_v1', 'corrected_preflop_loose_v2')]
    followup = {a + '_minus_' + b: {k: values[a][k] - values[b][k] for k in keys}
                for a, b in pairs}
    c, p, b, pb = [values[n] for n in (
        'classic_hybrid_v1', 'hybrid_price', 'hybrid_contribution', 'hybrid_both')]
    factorial = {
        'price_minus_control': {k: p[k] - c[k] for k in keys},
        'contribution_minus_control': {k: b[k] - c[k] for k in keys},
        'both_minus_control': {k: pb[k] - c[k] for k in keys},
        'average_price_main_effect': {k: (p[k] - c[k] + pb[k] - b[k]) / 2 for k in keys},
        'average_contribution_main_effect': {k: (b[k] - c[k] + pb[k] - p[k]) / 2 for k in keys},
        'price_by_contribution_interaction': {k: pb[k] - p[k] - b[k] + c[k] for k in keys},
    }
    output = {'benchmark': 'auction-flop-v1', 'tables_per_candidate': 4096,
              'independent_blocks': 512, 'summaries': summaries,
              'followup_four': intervals(followup), 'hybrid_six': intervals(factorial),
              'interval_method': catalog['intervals'],
              'all_sources_match_published_catalog': True, 'all_scores_audited': True,
              'control_reuse': catalog['control_reuse']}
    run.write_json(run.ROOT / 'evidence/followup-v1-comparison.json', output)
    print(output['followup_four'], output['hybrid_six'], flush=True)


if __name__ == '__main__':
    main()
