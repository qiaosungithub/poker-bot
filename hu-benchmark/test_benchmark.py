import unittest
import run_benchmark as bench
from engine.game import PokerEngine as Upstream
from engine_adapter import PokerEngine


class EngineTests(unittest.TestCase):
    def make(self, cls=PokerEngine, stacks=(5000, 10000)):
        return cls('test', ['p0','p1'], 0, dict(zip(('p0','p1'), stacks)), 812)

    def test_negative_control_upstream_requests_action_after_matched_allin(self):
        e = self.make(Upstream)
        e.start_hand()
        e.apply_action(0, {'action':'all_in'})
        state = e.apply_action(1, {'action':'call'})
        self.assertEqual(state['type'], 'action_request')  # reproduced upstream defect

    def test_matched_allin_finishes(self):
        e = self.make(); e.start_hand()
        e.apply_action(0, {'action':'all_in'})
        state = e.apply_action(1, {'action':'call'})
        self.assertEqual(state['type'], 'hand_complete')
        self.assertEqual(len(state['community_cards']), 5)
        self.assertEqual(sum(state['final_stacks'].values()), 15000)
        self.assertGreaterEqual(state['final_stacks']['p1'], 5000)  # uncalled chips retained

    def test_short_allin_still_requires_call_or_fold(self):
        e=self.make(stacks=(250,10000));e.start_hand()
        e.apply_action(0,{'action':'call'})
        e.apply_action(1,{'action':'raise','amount':200})
        state=e.apply_action(0,{'action':'all_in'})
        self.assertEqual(state['type'],'action_request')
        self.assertEqual(state['seat_to_act'],1)
        self.assertEqual(state['amount_owed'],50)
        result=e.apply_action(1,{'action':'fold'})
        self.assertEqual(result['final_stacks'],{'p0':450,'p1':9800})

    def test_blind_allin_forced_runout(self):
        e = self.make(stacks=(40, 10000)); state = e.start_hand()
        self.assertEqual(state['type'], 'hand_complete')
        self.assertEqual(sum(state['final_stacks'].values()), 10040)

    def test_short_blind_call_does_not_request_redundant_big_blind_action(self):
        e=self.make(stacks=(51,19949));e.start_hand()
        state=e.apply_action(0,{'action':'call'})
        self.assertEqual(state['type'],'hand_complete')
        self.assertEqual(sum(state['final_stacks'].values()),20000)

    def test_fold_settlement_and_position(self):
        e=self.make(stacks=(10000,10000)); state=e.start_hand()
        self.assertEqual(state['seat_to_act'], 0)
        self.assertEqual(state['amount_owed'],50)
        state=e.apply_action(0,{'action':'fold'})
        self.assertEqual(state['final_stacks'], {'p0':9950,'p1':10050})

    def test_postflop_order_and_hidden_cards(self):
        e=self.make(stacks=(10000,10000));e.start_hand()
        e.apply_action(0,{'action':'call'})
        state=e.apply_action(1,{'action':'check'})
        self.assertEqual(state['street'],'flop')
        self.assertEqual(state['seat_to_act'],1)
        self.assertTrue(all(p['hole_cards'] is None for p in state['players']))

    def test_mirror_and_reproducibility(self):
        bench.initialize()
        a=bench.play_pair(('mirror', 911, 200))
        b=bench.play_pair(('mirror', 911, 200))
        self.assertNotIn('error',a)
        self.assertEqual(a['score'],50)
        self.assertEqual(a['legs'],b['legs'])


if __name__ == '__main__':
    unittest.main()
