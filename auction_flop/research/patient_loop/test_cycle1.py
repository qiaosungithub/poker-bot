"""Regression examples for price-sensitive defense without new bluffing."""
from dataclasses import replace
import gzip
import json
import unittest

from auction_flop.benchmark import run
from auction_flop.research.patient_value_v1.test_policy import PolicyTests
from poker_agent_protocol.game import Event, Seat, LegalAction, BlindLevel, state_from_dict, is_legal


class PriceTests(PolicyTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.patient = run.arena.import_agent(run.ROOT/'candidates/patient_price_v2', 'price_test')()

    def test_observed_aces_call_at_twenty_percent_price(self):
        examples = json.load(gzip.open(run.ROOT/'research/patient_loop/diagnosis-v1/examples.json.gz','rt'))
        state = next(state_from_dict(x['state']) for x in examples['made_value_fold']
                     if x['state']['cards'] == ['Ah', 'Ad'] and len(x['state']['board']) == 3)
        old = run.arena.import_agent(run.ROOT/'candidates/patient_value_v1', 'control_test')()
        self.assertEqual(old.act(state,b'')[0].kind, 'fold')
        action = self.patient.act(state,b'')[0]
        self.assertEqual(action.kind, 'call')
        self.assertTrue(is_legal(action, state.legal_actions))


class ShortCallTests(PolicyTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.patient = run.arena.import_agent(run.ROOT/'candidates/patient_shortcall_v2', 'short_test')()

    def test_heads_up_cheap_all_in_call_and_full_table_scope(self):
        history = (Event('hand',amount=1), Event('blind',0,10), Event('blind',1,20),
                   Event('raise',1,200))
        state = replace(self.base,seat=0,button=0,cards=('Qs','Js'),board=(),
            players_remaining=2,blinds=BlindLevel(10,20),to_call=190,pot=210,
            seats=(Seat(0,40,10,'active'),Seat(1,0,200,'all_in'))+
                  tuple(Seat(i,0,0,'busted') for i in range(2,8)),
            history=history,events=history,
            legal_actions=(LegalAction('fold'),LegalAction('call')),phase='betting')
        old = run.arena.import_agent(run.ROOT/'candidates/patient_value_v1', 'control_short')()
        self.assertEqual(old.act(state,b'')[0].kind,'fold')
        self.assertEqual(self.patient.act(state,b'')[0].kind,'call')
        # Deliberately alter only tournament player count to test the scope guard.
        state = replace(state, players_remaining=8)
        self.assertEqual(self.patient.act(state,b'')[0],old.act(state,b'')[0])


class CombinedTests(ShortCallTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.patient = run.arena.import_agent(run.ROOT/'candidates/patient_defend_v2', 'combined_test')()


if __name__ == '__main__':
    unittest.main()
