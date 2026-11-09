import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bot'))
from equity_core import ids, CARDS, preflop_strength, showdown_share, auction_equities, auction_bids, TRIPLES
import eval7

class CoreTests(unittest.TestCase):
    def test_evaluator_order(self):
        self.assertGreater(eval7.evaluate([eval7.Card(c) for c in 'As Ah Ac Ad 2s'.split()]), eval7.evaluate([eval7.Card(c) for c in 'Ks Kh Kc 3d 2s'.split()]))
    def test_nuts_and_board_split(self):
        self.assertEqual(showdown_share(ids(('As','Ks')),ids(('Qs','Js','Ts','2c','3d')),7,15),1)
        self.assertAlmostEqual(showdown_share(ids(('2c','3d')),ids(('As','Ks','Qs','Js','Ts')),3,15),.25)
    def test_preflop_order(self):
        self.assertGreater(preflop_strength(ids(('As','Ah'))),preflop_strength(ids(('7s','2h'))))
    def test_auction_structural_choice(self):
        values=auction_equities(ids(('As','Ks')),ids(('Qs','Js','Ts','2c','3d')),3,42,8)
        self.assertEqual(TRIPLES[max(range(10),key=values.__getitem__)],(0,1,2))
    def test_budget(self):
        values=(.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)
        for budget in (0,1,2,10,500):
            bids=auction_bids(values,budget,20,10,40)
            self.assertLessEqual(sum(bids),budget)
            self.assertTrue(all(type(v) is int and v>=0 for v in bids))
        self.assertEqual(auction_bids((.5,)*10,500,20,10,40),(0,)*5)

if __name__=='__main__':unittest.main()
