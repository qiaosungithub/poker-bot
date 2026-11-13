"""Concrete poker examples, legal bid budgets and isolated auction ablation."""
from dataclasses import replace
import random
import unittest

from auction_flop.benchmark import run
from poker_agent_protocol.game import Event, Seat, LegalAction, BlindLevel, is_legal


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = run.arena.import_agent(run.ROOT/'candidates/simple_tuned_v2', 'old')()
        cls.key = run.arena.import_agent(run.ROOT/'candidates/simple_keycard_v1', 'key')()
        cls.patient = run.arena.import_agent(run.ROOT/'candidates/patient_value_v1', 'patient')()
        cls.auction_plan = staticmethod(cls.key.act.__globals__['auction'].__globals__['auction_plan'])
        cls.made_hand = staticmethod(cls.patient.act.__globals__['postflop'].__globals__['made_hand'])

    def setUp(self):
        table = run.arena.SitAndGo(rules=run.arena.TableRules(variant=run.arena.VARIANT, players=8),
            variant=run.arena.load_variant(run.arena.VARIANT), table_id='patient-tests', seed='patient-tests')
        self.base = table.decision()

    def auction(self, hole=('7s','7h'), cards=('7c','Kd','Qh','8s','2c'), balance=1000):
        return replace(self.base, seat=0, cards=hole, phase='auction', board=(),
            seats=tuple(Seat(i,1000,0,'active' if i<2 else 'busted') for i in range(8)),
            legal_actions=(LegalAction('bid',0,balance),),variant={'auction_cards':cards})

    def postflop(self, hole, board, bet=0):
        history=(Event('hand',amount=1),Event('blind',0,100),Event('blind',1,100),
                 Event('board',cards=board[:3]))
        for card in board[3:]: history += (Event('board',cards=(card,)),)
        if bet: history += (Event('raise',1,bet),)
        legal=(LegalAction('fold'),LegalAction('call')) if bet else (LegalAction('check'),)
        legal += (LegalAction('raise',max(20,2*bet),1000),)
        seats=(Seat(0,1000,0,'active'),Seat(1,1000-bet,bet,'active'))
        seats += tuple(Seat(i,0,0,'busted') for i in range(2,8))
        return replace(self.base,seat=0,cards=hole,board=board,pot=200+bet,to_call=bet,
            history=history,events=history,seats=seats,legal_actions=legal,phase='betting',
            players_remaining=2,variant={})

    def test_single_critical_seven_gets_entire_budget(self):
        state=self.auction()
        plan=self.auction_plan(state)
        self.assertGreater(plan['budget'],0)
        self.assertEqual(plan['bids'],(plan['budget'],0,0,0,0))
        self.assertEqual(plan['budget'],sum(self.old.act(state,b'')[0].cards))

    def test_three_royal_flush_cards_each_have_value(self):
        plan=self.auction_plan(self.auction(('As','Ks'),('Qs','Js','Ts','2h','3c')))
        self.assertTrue(all(x>0 for x in plan['bids'][:3]))
        self.assertEqual(plan['bids'][3:],(0,0))

    def test_random_auctions_keep_total_spend_and_legal_balances(self):
        rng=random.Random(111010)
        deck=[r+s for r in '23456789TJQKA' for s in 'shdc']
        for _ in range(1000):
            drawn=rng.sample(deck,7);balance=rng.choice((0,1,2,3,17,1000))
            state=self.auction(tuple(drawn[:2]),tuple(drawn[2:]),balance)
            old=self.old.act(state,b'')[0];new=self.key.act(state,b'')[0]
            self.assertEqual(sum(old.cards),sum(new.cards))
            self.assertEqual(new,self.key.act(state,b'')[0])
            self.assertEqual(new,self.patient.act(state,b'')[0])
            self.assertEqual(len(new.cards),5)
            self.assertTrue(all(type(x) is int and x>=0 for x in new.cards))
            self.assertLessEqual(sum(new.cards),balance)

    def test_private_royal_raises_even_when_facing_a_bet(self):
        for bet in (0,100):
            state=self.postflop(('As','Ks'),('Qs','Js','Ts','2h','3d'),bet)
            action=self.patient.act(state,b'')[0]
            self.assertEqual(action.kind,'raise')
            self.assertGreater(action.amount,bet)
            self.assertTrue(is_legal(action,state.legal_actions))

    def test_strong_flush_bets_all_three_streets(self):
        full=('Qh','Jh','2h','7c','8d')
        for n in (3,4,5):
            state=self.postflop(('Ah','Kh'),full[:n])
            self.assertEqual(self.patient.act(state,b'')[0].kind,'raise')

    def test_shared_strength_is_not_private_strength(self):
        self.assertFalse(self.made_hand(('Ah','Kc'),('Qs','Qh','Qd','4c','2s'))[1])
        self.assertTrue(self.made_hand(('As','Ah'),('Kd','Kc','8s'))[2])
        state=self.postflop(('2c','3d'),('As','Ks','Qs','Js','Ts'),400)
        self.assertEqual(self.patient.act(state,b'')[0].kind,'call')

    def test_air_checks_and_folds_to_pressure(self):
        for bet,expected in ((0,'check'),(400,'fold')):
            state=self.postflop(('7s','2h'),('Ac','Kd','9h','4c','Js'),bet)
            self.assertEqual(self.patient.act(state,b'')[0].kind,expected)

    def test_short_handed_qjs_does_not_wait_for_premiums(self):
        state=replace(self.base,seat=0,button=0,cards=('Qs','Js'),board=(),
            players_remaining=2,blinds=BlindLevel(10,20),to_call=10,pot=30,
            seats=(Seat(0,190,10,'active'),Seat(1,180,20,'active'))+
                  tuple(Seat(i,0,0,'busted') for i in range(2,8)),
            history=(Event('blind',0,10),Event('blind',1,20)),
            legal_actions=(LegalAction('fold'),LegalAction('call'),LegalAction('raise',40,200)),
            phase='betting')
        self.assertEqual(self.old.act(state,b'')[0].kind,'fold')
        self.assertEqual(self.patient.act(state,b'')[0].kind,'raise')


if __name__=='__main__':
    unittest.main()
