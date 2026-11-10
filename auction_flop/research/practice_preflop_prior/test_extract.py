"""Causality and chip-accounting tests for public preflop data extraction."""
from copy import deepcopy
import unittest
from .extract import decisions, context_key
from .fit import fit


def fixture():
    return {'stacks_before': [2000] * 3, 'button': 0, 'blinds': [10, 20, 0],
            'events': [{'kind': 'blind', 'seat': 1, 'amount': 10},
                       {'kind': 'blind', 'seat': 2, 'amount': 20},
                       {'kind': 'raise', 'seat': 0, 'amount': 60},
                       {'kind': 'call', 'seat': 1, 'amount': 50},
                       {'kind': 'raise', 'seat': 2, 'amount': 180},
                       {'kind': 'call', 'seat': 0, 'amount': 120},
                       {'kind': 'fold', 'seat': 1},
                       {'kind': 'auction', 'cards': ['As', 'Ks', 'Qs', 'Js', 'Ts']}]}


class ExtractTests(unittest.TestCase):
    def test_negative_control_and_future_card_invariance(self):
        hand = fixture()
        reference = list(decisions(hand))
        altered = deepcopy(hand)
        altered['cards'] = [['As', 'Ah']] * 3
        altered['stacks_after'] = [6000, 0, 0]
        altered['events'][-1]['cards'] = ['2s', '3s', '4s', '5s', '6s']
        altered['events'].append({'kind': 'raise', 'seat': 0, 'amount': 2000})
        self.assertEqual(reference, list(decisions(altered)))
        altered['events'][2]['amount'] = 80
        self.assertNotEqual(reference, list(decisions(altered)))

    def test_raise_totals_call_increments_and_first_action_only(self):
        rows = list(decisions(fixture()))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['to_call_bb'] for r in rows], [1, 2.5, 2])
        self.assertEqual([r['context'] for r in rows],
                         ['3|BTN|60+|unopened', '3|SB|60+|raised', '3|BB|60+|raised'])

    def test_folded_cover_is_excluded_and_heads_up_position(self):
        hand = fixture()
        hand['stacks_before'] = [10000, 2000, 100]
        hand['events'][2] = {'kind': 'fold', 'seat': 0}
        rows = list(decisions(hand))
        self.assertEqual(rows[1]['effective_bb'], 5)
        self.assertEqual(context_key([0, 2], 0, 0, 3, 100, 0, 0), '2|SB|60+|unopened')
        self.assertEqual(context_key([0, 2], 0, 2, 3, 100, 0, 1), '2|BB|60+|limped')

    def test_held_out_actions_never_enter_fitted_probabilities(self):
        data = {'teachers': [{'team_name': 'a'}, {'team_name': 'b'}],
                'tournament_id': 'fixture', 'history_sha256': 'fixture',
                'split': 'fixture', 'coverage': {}, 'selection': 'fixture',
                'counts': [{'split': split, 'team': team, 'context': '7-10|BTN|60+|unopened',
                            'actions': {'fold': 80, 'raise': 20}}
                           for split in ('train', 'test') for team in ('a', 'b')]}
        before = fit(data)
        for row in data['counts']:
            if row['split'] == 'test':
                row['actions'] = {'fold': 0, 'raise': 100}
        after = fit(data)
        self.assertEqual(before['priors'], after['priors'])
        self.assertNotEqual(before['metrics'], after['metrics'])


if __name__ == '__main__':
    unittest.main()
