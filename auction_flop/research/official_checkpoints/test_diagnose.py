from copy import deepcopy
from itertools import permutations
import unittest
from .diagnose import icm_features, checkpoint, final_result, Moments


class DiagnosticTests(unittest.TestCase):
    def test_icm_against_full_permutation_enumeration(self):
        stacks = [100, 250, 300, 750]
        values = dict(points=0., top3=0., rank=0.)
        for order in permutations(range(4)):
            probability=1.;remaining=sum(stacks)
            for seat in order:
                probability *= stacks[seat]/remaining;remaining -= stacks[seat]
            rank = order.index(0)+1
            values['points'] += probability * ([50,30,20,0][rank-1])
            values['top3'] += probability * (rank<=3)
            values['rank'] += probability*rank
        actual = icm_features(stacks,0)
        self.assertAlmostEqual(values['points'],actual['computed_icm_points'])
        self.assertAlmostEqual(values['top3'],actual['expected_top3'])
        self.assertAlmostEqual(values['rank'],actual['expected_rank'])
        self.assertNotAlmostEqual(values['rank'],2.5)

    def test_future_outcomes_and_cards_do_not_change_checkpoint(self):
        hand={'hand':21,'stacks_before':[1000,3000], 'blinds':[10,20,0],
              'equity_before':[35,45], 'flop_chips_before':[200,700]}
        before=checkpoint(hand,0)
        altered=deepcopy(hand);altered.update(cards=[['As','Ah'],['Ks','Kh']],
            stacks_after=[4000,0],events=[{'kind':'win','seat':0,'amount':4000}])
        self.assertEqual(before,checkpoint(altered,0))
        altered['stacks_before']=[2000,2000]
        with self.assertRaises(AssertionError):checkpoint(altered,0)

    def test_simultaneous_bust_order_and_cap_rank(self):
        hand={'stacks_before':[100,200,700], 'stacks_after':[0,0,1000],
              'equity_after':[20,30,50], 'flop_chips_after':[100]*3}
        table={'hands':[hand]}
        self.assertEqual(final_result(table,0)['final_rank'],3)
        self.assertEqual(final_result(table,1)['final_rank'],2)
        hand['stacks_before']=[100,100,800]
        self.assertEqual(final_result(table,0)['final_rank'],2)
        hand['stacks_after']=[100,400,500]
        self.assertEqual(final_result(table,1)['final_rank'],2)
        self.assertEqual(final_result(table,1)['capped'],1)

    def test_unreached_stages_are_zero_in_per_initial_table_accounting(self):
        m=Moments();m.add({'points_change':10});m.add({'points_change':20})
        self.assertEqual(m.report()['metrics']['points_change']['mean'],15)
        self.assertEqual(m.report(total=4)['metrics']['points_change']['mean'],7.5)


if __name__=='__main__':unittest.main()
