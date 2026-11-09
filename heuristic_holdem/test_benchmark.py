import ast
import copy
import random
import unittest

from . import benchmark as B
from . import policies as P
from .analyze import audit_leg


class Tests(unittest.TestCase):
    def test_private_memory_and_rng(self):
        a=P.Policy('vikram_32',12);b=P.Policy('vikram_32',12)
        self.assertIsNot(a.namespace['_TRACKER'],b.namespace['_TRACKER'])
        a.namespace['_TRACKER'].stats['sentinel']={}
        self.assertNotIn('sentinel',b.namespace['_TRACKER'].stats)
        a=P.Policy('txnujk_32',13);b=P.Policy('txnujk_32',13)
        self.assertIsNot(a.namespace['_RNG'],b.namespace['_RNG'])
        self.assertEqual(a.namespace['_RNG'].getstate(),b.namespace['_RNG'].getstate())
        self.assertEqual(a.namespace['PREFLOP_EQ'],b.namespace['PREFLOP_EQ'])

    def test_all_sampling_caps(self):
        for name in ['hydra_32','vikram_32','txnujk_32','pantheon_32','cfr_lite_32','fold_equity_32']:
            p=P.Policy(name,2);meta=P.metadata(name)
            self.assertEqual(meta['sampling_cap'],32)
            if name=='pantheon_32':
                self.assertEqual(p.namespace['_equity_vs_one_range'].__defaults__[1],32)
                self.assertEqual(p.namespace['_equity_vs_multi_range'].__defaults__[1],32)
                self.assertTrue(all(n==32 for _,n in p.namespace['_EQ_STREET_BUDGET'].values()))

    def test_anonymous_inputs_and_hidden_cards(self):
        class Check:
            def before(self,policy,state):
                for player in state['players']:
                    assert player['bot_id'].startswith('seat_')
                    assert player['hole_cards'] is None
                assert all(e['bot_id'].startswith('seat_') for e in state['match_action_log'])
            def after(self,*args):pass
        leg=B.match(['solver_like']*6,31000001,25,Check())
        audit_leg(leg)
        broken=copy.deepcopy(leg);broken['scores'][0]+=1
        with self.assertRaises(AssertionError):audit_leg(broken)

    def test_same_seed_replay(self):
        names=['solver_like','tournament_specialist','cfr_lite_32','hydra_32','txnujk_32','vikram_32']
        a=B.match(names,31000002,30);b=B.match(names,31000002,30)
        self.assertEqual(a,b)
        self.assertFalse(a['exceptions'])

    def test_identical_policy_rotations(self):
        row=B.block((['solver_like']*6,31000003,35,'mirror'))
        for score in row['scores']:self.assertAlmostEqual(score,235/6,9)

    def test_diagnostics_detect_swallowed_exception(self):
        p=P.Policy('solver_like',1)
        code=compile(ast.fix_missing_locations(P.Diagnostics().visit(ast.parse('def test():\n try: 1/0\n except Exception: return 7\n'))),'<control>','exec')
        ns={'_record_exception':p.record_exception};exec(code,ns)
        p.enter()
        try:self.assertEqual(ns['test'](),7)
        finally:p.leave()
        self.assertEqual(len(p.exceptions),1)
        self.assertIn('ZeroDivisionError',p.exceptions[0])


if __name__=='__main__':unittest.main()
