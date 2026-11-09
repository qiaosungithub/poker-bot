import importlib.util
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('strategy', Path(__file__).parents[1] / 'bot/strategy.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
def cards(text):
    return tuple(s.card_code(c) for c in text.split())

class StrategyTests(unittest.TestCase):
    def test_categories_and_private_contribution(self):
        self.assertEqual(s.hand_rank(cards('As Ah Ac 7d 2s'))[0], 3)
        self.assertEqual(s.hand_rank(cards('9s 9h 9d 9c 2s'))[0], 7)
        self.assertEqual(s.hand_rank(cards('As Ks Qs Js Ts'))[0], 8)
        self.assertEqual(s.hand_rank(cards('As 2d 3c 4h 5s'))[:2], (4, 5))
        self.assertGreater(s.board_quality(cards('As Ah'), cards('Ac 7d 2s')), 0.8)
        self.assertGreater(s.board_quality(cards('9s 9h'), cards('9d 9c 2s')), 0.98)
        self.assertLess(s.board_quality(cards('2s 3h'), cards('Ah Ad Ks Kd Qc')), 0.35)
        self.assertLess(s.board_quality(cards('2s 3h'), cards('Ah Ad Ac As Kc')), 0.5)

    def test_auction_budget_and_synergy(self):
        hole = cards('As Ks')
        pool = cards('Qs Js Ts 2d 7c')
        bids = s.auction_bids(hole,pool,100,hand_number=1,hand_cap=250,
                              stack=2000,big_blind=20,pot=100,opponents=5)
        self.assertTrue(all(type(b) is int and b >= 0 for b in bids))
        self.assertLessEqual(sum(bids),100)
        self.assertGreater(sum(bids[:3]),0)
        self.assertEqual(bids[3:],(0,0))
        late = s.auction_bids(hole,pool,100,hand_number=249,hand_cap=250,
                              stack=2000,big_blind=20,pot=100,opponents=5)
        self.assertGreaterEqual(sum(late),sum(bids))
        rng=random.Random(1)
        for _ in range(100):
            dealt=rng.sample(range(52),7);budget=rng.randrange(101)
            bids=s.auction_bids(dealt[:2],dealt[2:],budget,hand_number=20,
                 hand_cap=250,stack=700,big_blind=40,pot=300,opponents=3)
            self.assertLessEqual(sum(bids),budget)
            self.assertTrue(all(type(b) is int and b >= 0 for b in bids))

    def test_risk_is_zero_for_winner_take_all(self):
        self.assertEqual(s.risk_premium(100,200,[100,200,50],[100,0,0],100),0)
        self.assertGreater(s.risk_premium(100,200,[100,200,50],[50,30,20],100),0)

if __name__ == '__main__':
    unittest.main()
