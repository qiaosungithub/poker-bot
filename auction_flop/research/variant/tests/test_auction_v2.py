import itertools
import random
import unittest
from auction import TRIPLES, chosen_indices, combination_utilities
from auction_v2 import ZERO, bid_metrics, prune_nonpivotal, choose_with_scenarios, choose_bids

class PivotalAuctionTest(unittest.TestCase):
    def test_free_rides_on_sufficient_opponent_support(self):
        utilities=[.1]*10;utilities[TRIPLES.index((0,1,4))]=.9
        scenarios=(((9,8,0,0,10),1.),)
        self.assertEqual(choose_with_scenarios(utilities,100,scenarios,.01),ZERO)
        # Paying on an already-selected card buys no change and is removable.
        self.assertEqual(prune_nonpivotal((20,20,0,0,20),scenarios),ZERO)

    def test_complementary_card_needs_one_pivotal_vote(self):
        utilities=[.1]*10;utilities[TRIPLES.index((0,1,4))]=.9
        scenarios=(((10,9,2,0,0),1.),)
        bid=choose_with_scenarios(utilities,100,scenarios,.01)
        self.assertEqual(bid,(0,0,0,0,3))
        self.assertEqual(bid_metrics(utilities,bid,scenarios),(.8,3.,1.,0.))

    def test_cheap_partial_gain_beats_expensive_top_board(self):
        utilities=[.2]*10
        utilities[TRIPLES.index((0,1,3))]=.65
        utilities[TRIPLES.index((2,3,4))]=.95
        scenarios=(((100,90,1,0,0),1.),)
        bid=choose_with_scenarios(utilities,20,scenarios,.02)
        self.assertEqual(bid,(0,0,0,2,0))

    def test_losing_bids_have_no_realized_payment(self):
        scenarios=(((100,100,100,0,0),1.),)
        metrics=bid_metrics([.5]*10,(0,0,0,7,8),scenarios)
        self.assertEqual(metrics,(0.,0.,0.,0.))
        self.assertEqual(prune_nonpivotal((0,0,0,7,8),scenarios),ZERO)

    def test_avoid_bad_cross_scenario_externality(self):
        utilities=[.4]*10;utilities[TRIPLES.index((2,3,4))]=.8
        utilities[TRIPLES.index((0,3,4))]=.01
        scenarios=(((10,0,0,1,1),.7),((0,0,0,0,0),.3))
        bid=choose_with_scenarios(utilities,30,scenarios,.03)
        gain,paid,_,down=bid_metrics(utilities,bid,scenarios)
        self.assertGreaterEqual(gain-.03*paid-.15*down,-1e-12)

    def test_prune_preserves_all_modeled_boards_and_reduces_cost(self):
        rng=random.Random(901111)
        for _ in range(250):
            bid=tuple(rng.randrange(20) for _ in range(5))
            scenarios=tuple((tuple(rng.randrange(20) for _ in range(5)),1/3) for _ in range(3))
            reduced=prune_nonpivotal(bid,scenarios)
            self.assertTrue(all(a<=b for a,b in zip(reduced,bid)))
            for opposition,_ in scenarios:
                self.assertEqual(chosen_indices([a+b for a,b in zip(opposition,bid)]),chosen_indices([a+b for a,b in zip(opposition,reduced)]))
            self.assertLessEqual(bid_metrics([.5]*10,reduced,scenarios)[1],bid_metrics([.5]*10,bid,scenarios)[1])

    def test_full_budget_vector_not_just_paid_subset(self):
        for budget in range(25):
            bid=choose_bids((.1,.2,.3,.4,.5,.6,.7,.8,.9,.99),budget,[35,70,105])
            self.assertLessEqual(sum(bid),budget)
            self.assertTrue(all(type(x)is int and x>=0 for x in bid))

if __name__=='__main__':unittest.main()
