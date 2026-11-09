import itertools
import unittest
from auction import TRIPLES, chosen_indices, combination_utilities, forcing_bid, settle, choose_bids

class AuctionTest(unittest.TestCase):
    def test_zero_tie_order(self):
        self.assertEqual(chosen_indices([0]*5), (0,1,2))
        self.assertEqual(chosen_indices([0]*5, (4,3,2,1,0)), (2,3,4))

    def test_only_selected_paid(self):
        selected, left = settle([(0,0,0,5,0),(8,8,8,0,0)],[5,24])
        self.assertEqual(selected,(0,1,2));self.assertEqual(left,(5,0))
        with self.assertRaises(ValueError):settle([(1,1,1,1,1)],[4])

    def test_minimal_forcing_all_small_totals(self):
        for other in itertools.product(range(3), repeat=5):
            for target in TRIPLES:
                bid=forcing_bid(target,other)
                self.assertEqual(chosen_indices([x+y for x,y in zip(other,bid)]),target)
                for i,v in enumerate(bid):
                    if v:
                        smaller=list(bid);smaller[i]-=1
                        self.assertNotEqual(chosen_indices([x+y for x,y in zip(other,smaller)]),target)

    def test_private_set_beats_overpair_and_air(self):
        u=combination_utilities(['As','Ah'],['Ac','7d','2c','Ks','Qh'])
        self.assertGreater(u[TRIPLES.index((0,1,2))],u[TRIPLES.index((1,2,3))])
        v=combination_utilities(['Js','Th'],['Ac','7d','2c','Ks','Qh'])
        self.assertGreater(v[TRIPLES.index((0,3,4))],v[TRIPLES.index((0,1,2))])

    def test_complementarity_and_no_waste(self):
        utilities=[0.]*10;utilities[TRIPLES.index((2,3,4))]=1.
        bids=choose_bids(utilities,100,[],hands_left=1)
        self.assertEqual(chosen_indices(bids),(2,3,4))
        self.assertLessEqual(sum(bids),3)
        self.assertEqual(choose_bids([.5]*10,100,[100]),(0,)*5)
        self.assertEqual(choose_bids(utilities,0,[100]),(0,)*5)

    def test_budget_and_determinism(self):
        utilities=(.2,.8,.5,.3,.7,.6,.1,.4,.9,.2)
        for budget in range(21):
            bids=choose_bids(utilities,budget,[5,17,36],hands_left=20)
            self.assertLessEqual(sum(bids),budget)
            self.assertTrue(all(type(b) is int and b>=0 for b in bids))
            self.assertEqual(bids,choose_bids(utilities,budget,[5,17,36],hands_left=20))

if __name__=='__main__':unittest.main()
