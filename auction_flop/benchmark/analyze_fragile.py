"""Audit the 2x2 fragile-hand experiment on the frozen v1 schedule."""
from statistics import NormalDist

from . import run


def main():
    catalog = run.read_json(run.ROOT / 'candidate_catalog_fragile_v1.json')
    control = run.read_json(run.ROOT / 'candidate_catalog_corrected_v2.json')['corrected-preflop-loose-v2']
    sources = {'corrected_preflop_loose_v2': control['source_sha256']}
    sources.update({x['name']: x['sources'] for x in catalog['candidates']})
    values, summaries, table_values = {}, {}, {}
    for name, hashes in sources.items():
        directory = run.ROOT / 'evidence' / ('candidate-' + name)
        manifest, rows = run.audit(directory)
        if manifest['source_hashes']['candidate'] != hashes:
            raise ValueError(f'{name}: source differs from the published catalog')
        values[name] = run.candidate_blocks(rows)
        table_values[name] = {
            (r['panel'], r['block'], r['rotation']):
                next(e['points'] for e in r['entries'] if e['name'] == 'candidate')
            for r in rows}
        summaries[name] = run.read_json(directory / 'summary.json')
    control, flush, pair, both = [values[n] for n in (
        'corrected_preflop_loose_v2', 'fragile_flush_v1', 'fragile_pair_v1', 'fragile_both_v1')]
    if not control.keys() == flush.keys() == pair.keys() == both.keys():
        raise ValueError('Mismatched paired blocks')
    contrasts = {
        'flush_minus_control': {k: flush[k] - control[k] for k in control},
        'pair_minus_control': {k: pair[k] - control[k] for k in control},
        'both_minus_control': {k: both[k] - control[k] for k in control},
        'average_flush_main_effect': {k: (flush[k] - control[k] + both[k] - pair[k]) / 2 for k in control},
        'average_pair_main_effect': {k: (pair[k] - control[k] + both[k] - flush[k]) / 2 for k in control},
        'flush_by_pair_interaction': {k: both[k] - flush[k] - pair[k] + control[k] for k in control},
    }
    z = NormalDist().inv_cdf(1 - .05 / (2 * len(contrasts)))
    results = {}
    for name, blocks in contrasts.items():
        result = run.estimate(blocks)
        mean = result['mean_points']
        se = (result['ci95'][1] - mean) / 1.96
        results[name] = {'difference': mean, 'ci95': result['ci95'],
                         'simultaneous_ci95': [mean - z * se, mean + z * se]}
    tc, tf, tp, tb = [table_values[n] for n in (
        'corrected_preflop_loose_v2', 'fragile_flush_v1', 'fragile_pair_v1', 'fragile_both_v1')]
    changed_flush = {k for k in tc if tf[k] != tc[k]}
    changed_pair = {k for k in tc if tp[k] != tc[k]}
    overlap = {'flush_changed_payout_tables': len(changed_flush),
               'pair_changed_payout_tables': len(changed_pair),
               'changed_payout_overlap_tables': len(changed_flush & changed_pair),
               'nonzero_interaction_tables': sum(tb[k] - tf[k] - tp[k] + tc[k] != 0 for k in tc),
               'interpretation': 'An observed zero interaction and zero empirical interval do not establish zero interaction on unseen deals or other fields.'}
    output = {'benchmark': 'auction-flop-v1', 'tables_per_candidate': 4096,
              'independent_blocks': 512, 'summaries': summaries,
              'contrasts': results, 'control_reused': 'corrected_preflop_loose_v2',
              'observed_payout_overlap': overlap,
              'interval_method': 'Paired panel-stratified blocks; Bonferroni normal 95% for six contrasts in this family only, not the ongoing strategy search',
              'all_sources_match_published_catalog': True, 'all_scores_audited': True}
    run.write_json(run.ROOT / 'evidence/fragile-v1-comparison.json', output)
    print(results, flush=True)


if __name__ == '__main__':
    main()
