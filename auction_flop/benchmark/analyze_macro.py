"""Audit the four macro hypotheses and the isolated objective ablation."""
from statistics import NormalDist
import hashlib
import json
from . import run


def main():
    catalog = run.read_json(run.ROOT / 'candidate_catalog_macro_v1.json')['candidates']
    expected = {x['name']: x['sha256'] for x in catalog}
    prior = run.read_json(run.ROOT / 'candidate_catalog_prior_hybrid_both_v1.json')
    expected[prior['name']] = prior['sha256']
    h = next(x for x in run.read_json(run.ROOT / 'candidate_catalog_followup_v1.json')['candidates']
             if x['name'] == 'hybrid_both')
    expected['hybrid_both'] = h['sha256']
    values, summaries, record_hashes = {}, {}, {}
    for name, hashes in expected.items():
        directory = run.ROOT / 'evidence' / ('candidate-' + name)
        manifest, rows = run.audit(directory)
        if manifest['source_hashes']['candidate'] != hashes:
            raise ValueError(f'{name}: measured source differs from catalog')
        values[name] = run.candidate_blocks(rows)
        summaries[name] = run.read_json(directory / 'summary.json')
        if name in ('macro_tournament_utility_v1', 'macro_tournament_utility_v2'):
            record_hashes[name] = {(r['panel'], r['block'], r['rotation']):
                hashlib.sha256(json.dumps({k: v for k, v in r.items() if k != 'seconds'},
                                         sort_keys=True).encode()).hexdigest() for r in rows}
        print(name, summaries[name]['mean_points'], 'audited', flush=True)
    base = values['hybrid_both']; contrasts = {}
    z = NormalDist().inv_cdf(1 - .05 / 8)
    for name in ['prior_hybrid_both_v1', 'macro_auction_value_v1',
                 'macro_tournament_utility_v1', 'macro_action_value_v1']:
        assert values[name].keys() == base.keys()
        result = run.estimate({k: values[name][k] - base[k] for k in base})
        mean = result['mean_points']; se = (result['ci95'][1] - mean) / 1.96
        contrasts[name] = {'difference_vs_hboth': mean, 'ci95': result['ci95'],
                           'simultaneous_ci95': [mean - z * se, mean + z * se]}
    icm = values['macro_tournament_utility_v2']; chip = values['macro_chip_utility_control_v1']
    assert icm.keys() == chip.keys()
    assert [k for k in expected['macro_tournament_utility_v2'] if
            expected['macro_tournament_utility_v2'][k] != expected['macro_chip_utility_control_v1'][k]] == ['tournament_utility.py']
    ablation = run.estimate({k: icm[k] - chip[k] for k in icm})
    first, second = [record_hashes[k] for k in ('macro_tournament_utility_v1', 'macro_tournament_utility_v2')]
    assert first.keys() == second.keys()
    changed = [list(k) for k in first if first[k] != second[k]]
    output = {'benchmark': 'auction-flop-v1', 'tables_per_candidate': 4096,
              'summaries': summaries, 'macro_four_vs_hboth': contrasts,
              'interval_scope': 'Paired panel-stratified normal intervals; Bonferroni within four macro hypotheses only, not the entire adaptive search.',
              'icm_minus_linear_chip_objective': ablation,
              'ablation_scope': 'One separately authorized paired objective contrast: full ICM v2 versus linear control, differing only in tournament_utility.py. All other source files identical.',
              'v2_v1_retained_outcome_parity': {'tables': len(first), 'changed_tables': changed,
                  'exact_excluding_wallclock': not changed,
                  'scope': 'Complete retained table records include scores, stacks, busts, decisions, phases and auction outcomes. Original full-v1 action streams were not retained; this is not a claim about every literal action.'},
              'all_sources_and_scores_audited': True}
    run.write_json(run.ROOT / 'evidence/macro-program-comparison.json', output)
    print(contrasts, ablation, flush=True)


if __name__ == '__main__': main()
