"""Validate policy access and information boundaries of the LBR opponent."""
import unittest
import numpy as np
import run_benchmark as bench
from engine_adapter import PokerEngine
from lbr_baseline import LocalBestResponse, CARDS, CARD_ID
from bot.features import encode_state_dict


class LBRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench.initialize()

    def make(self, stacks=(10000,10000)):
        e=PokerEngine('parity',['p0','p1'],0,dict(zip(('p0','p1'),stacks)),31)
        state=e.start_hand()
        lbr=LocalBestResponse(.25)
        own=[str(c) for c in e.players[1].hole_cards]
        lbr.new_hand(1,own)
        return e,state,lbr,own

    def test_blockers_and_no_private_card_access(self):
        e,state,lbr,own=self.make()
        self.assertEqual(len(lbr.combos),1225)
        for c in own:self.assertFalse((lbr.combos==CARD_ID[c]).any())
        with self.assertRaises(AssertionError):
            lbr.observe(state,{'action':'call'})
        public=dict(state);public.pop('your_cards')
        lbr.observe(public,bench.HERO.decide(state))
        self.assertAlmostEqual(float(lbr.weights.sum()),1.)

    def test_batched_probabilities_and_features_match_original(self):
        e,state,lbr,own=self.make()
        checked=0
        for stage in range(4):
            public=dict(state);public.pop('your_cards')
            probs,actions=lbr.probabilities(public)
            for i in range(0,len(lbr.combos),27):
                a,b=lbr.combos[i];cards=[CARDS[a],CARDS[b]]
                if set(cards)&set(state['community_cards']):continue
                s=dict(state,your_cards=cards)
                original=bench.HERO._apply_legal_mask(bench.HERO._DEEP_CFR.get_strategy(s),s)
                np.testing.assert_allclose(probs[i],original,atol=2e-5,rtol=2e-4)
                np.testing.assert_array_equal(lbr.card_features(state['community_cards'])[i],encode_state_dict(s)[0][:17])
                checked+=1
            while state['type']=='action_request' and {'preflop':0,'flop':1,'turn':2,'river':3}[state['street']]==stage:
                state=e.apply_action(state['seat_to_act'],{'action':'check' if state['can_check'] else 'call'})
        self.assertGreater(checked,100)

    def test_short_stack_overlay(self):
        e,state,lbr,own=self.make((1000,1000))
        public=dict(state);public.pop('your_cards')
        probs,actions=lbr.probabilities(public)
        for i,(a,b) in enumerate(lbr.combos):
            s=dict(state,your_cards=[CARDS[a],CARDS[b]])
            expected=bench.HERO._push_fold(s)
            self.assertEqual(actions[int(probs[i].argmax())],expected)
            self.assertEqual(float(probs[i].max()),1.)


if __name__=='__main__':
    unittest.main()
