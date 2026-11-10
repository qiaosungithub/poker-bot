"""Descriptive summaries with explicit decision, hand and table denominators."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from ..benchmark import run


def lead(share):
    return 'behind' if share < .75 else 'ahead' if share > 1.25 else 'near_equal'


def depth(bb):
    return '<=10BB' if bb <= 10 else '10-25BB' if bb <= 25 else '25-60BB' if bb <= 60 else '>60BB'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    args = parser.parse_args(); directory = args.directory
    manifest, _ = run.audit(directory)
    assert manifest['action_parity_tables'] == manifest['retained_table_parity']
    groups = defaultdict(lambda: {'decisions': 0, 'cost': 0, 'hands': {}, 'tables': {}})
    checkpoints = defaultdict(list); heads_up = defaultdict(list); examples = defaultdict(list)
    totals = {'tables': 0, 'decisions': 0, 'hands': 0, 'outright_wins': 0, 'cap_tables': 0}
    def add(key, decision, hand, table, identity):
        g = groups[key]; g['decisions'] += 1; g['cost'] += decision['cost']
        g['hands'][(*identity, decision['hand'])] = hand
        g['tables'][identity] = table
    for filename, digest in manifest['trace_sha256'].items():
        path = directory / 'trace' / filename
        assert run.digest(path) == digest
        decisions, hands = [], {}
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'decision': decisions.append(row)
                elif row['kind'] == 'hand': hands[row['hand']] = row
                elif row['kind'] == 'table':
                    identity = (row['panel'], row['block'], row['rotation'])
                    totals['tables'] += 1; totals['decisions'] += len(decisions); totals['hands'] += len(hands)
                    totals['outright_wins'] += row['outright_win']; totals['cap_tables'] += row['cap_reached']
                    for checkpoint in row['checkpoints']:
                        label = 'busted' if checkpoint['stack'] == 0 else lead(checkpoint['relative_equal_share'])
                        checkpoints[f"hand{checkpoint['hand']}/{label}"].append((checkpoint, row))
                        checkpoints[f"hand{checkpoint['hand']}/all"].append((checkpoint, row))
                    for entry in row['survivor_entries']:
                        label = 'busted' if entry['stack'] == 0 else lead(entry['relative_equal_share'])
                        checkpoints[f"first_le_{entry['threshold']}/{label}"].append((entry, row))
                    if row['hu_entry']:
                        entry = row['hu_entry']; s = entry['share']
                        label = '<25%' if s < .25 else '25-50%' if s < .5 else '50-75%' if s < .75 else '>=75%'
                        heads_up[label].append((entry, row))
                    for decision in decisions:
                        hand = hands[decision['hand']]
                        label = lead(decision['hand_start_stack']['relative_equal_share'])
                        action = decision['action']['kind']
                        cost = 'free' if decision['call_cost'] == 0 else 'cheap<=5%' if decision['eligible_price'] <= .05 else 'normal'
                        keys = [f"street/{decision['street']}/{action}",
                                f"lead/{label}/{decision['street']}/{action}",
                                f"depth/{depth(decision['effective_bb'])}/{decision['street']}/{action}",
                                f"position/{decision['position']}/{decision['street']}/{action}",
                                f"price/{cost}/{decision['street']}/{action}"]
                        keys += ['flag/' + f for f in decision['flags']]
                        for key in keys: add(key, decision, hand, row, identity)
                        for flag in decision.get('example_for', []):
                            if len(examples[flag]) < 3:
                                examples[flag].append({'decision': decision, 'settled_hand': hand,
                                                       'final_points': row['final_points']})
                    decisions, hands = [], {}
        assert not decisions and not hands
    def aggregate(g):
        hs = list(g['hands'].values()); ts = list(g['tables'].values())
        return {'decisions': g['decisions'], 'distinct_hands': len(hs), 'distinct_tables': len(ts),
                'normal_chip_cost_sum': g['cost'],
                'mean_hand_icm_change': sum(h['icm_delta'] for h in hs) / len(hs),
                'sum_hand_icm_change': sum(h['icm_delta'] for h in hs),
                'mean_hand_chip_change': sum(h['chip_delta'] for h in hs) / len(hs),
                'losing_hand_fraction': sum(h['icm_delta'] < 0 for h in hs) / len(hs),
                'mean_final_points_distinct_tables': sum(t['final_points'] for t in ts) / len(ts)}
    def checkpoint_summary(pairs):
        n = len(pairs)
        return {'tables': n, 'alive': sum(c['stack'] > 0 for c, t in pairs),
                'absorbed': sum(c.get('absorbed', False) for c, t in pairs),
                'mean_checkpoint_icm': sum(c['icm'] for c, t in pairs) / n,
                'mean_final_points': sum(t['final_points'] for c, t in pairs) / n,
                'mean_final_minus_checkpoint_icm': sum(t['final_points']-c['icm'] for c, t in pairs) / n,
                'outright_wins': sum(t['outright_win'] for c, t in pairs),
                'cap_tables': sum(t['cap_reached'] for c, t in pairs)}
    output = {'scope': 'Known-seed Hboth descriptive replay; not a new strength score or causal regret estimate.',
              'denominators': totals, 'action_and_result_parity_tables': manifest['retained_table_parity'],
              'groups': {k: aggregate(v) for k, v in groups.items()},
              'checkpoints': {k: checkpoint_summary(v) for k, v in checkpoints.items()},
              'heads_up_entry': {k: checkpoint_summary(v) for k, v in heads_up.items()},
              'caveats': ['All signs of hand outcomes included; losing hands are not the analysis population.',
                         'Hand ICM logs are rounded to4 decimals; checkpoints use exact settled-state ICM.',
                         'Flags overlap; never sum their hand changes as a total loss or improvement opportunity.',
                         'A32-sample equity estimate and a uniform opponent model do not prove a call/check is wrong.',
                         'Fixed-hand checkpoints carry absorbed final states for early table endings; alive/absorbed counts are explicit.']}
    run.write_json(directory / 'descriptive-summary.json', output)
    run.write_json(directory / 'native-examples.json', dict(examples))
    print(json.dumps({'denominators': totals, 'flags': {k: v for k, v in output['groups'].items() if k.startswith('flag/')},
                      'heads_up_entry': output['heads_up_entry']}, indent=2), flush=True)


if __name__ == '__main__': main()
