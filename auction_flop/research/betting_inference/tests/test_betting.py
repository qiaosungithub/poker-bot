import dataclasses
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poker_agent_protocol.game import Action, BlindLevel, Event, GameState, LegalAction, Seat, TableConfig
from betting_baseline import choose_baseline_bet
from betting_policy import choose_bet, choose_bet_without_inference, infer_pressure, public_unseen_cards


def make_state(board=('Ac', '7h', '2d'), cards=('As', 'Td'), to_call=75, pot=100):
    blind = BlindLevel(10, 20)
    config = TableConfig('fixture', 0, 2, 'flop_auction_holdem', 2000, (blind,), 20, 250, (50,30), 1, 65536, 100)
    actions = (LegalAction('fold'), LegalAction('call'), LegalAction('raise', 150, 1000)) if to_call else (LegalAction('check'), LegalAction('raise', 20, 1000))
    return GameState((), 10, 0, cards, board, pot, to_call, actions,
        (Event('hand', amount=10), Event('auction', cards=board[:3]+('3c','4d')), Event('board', cards=board[:3])),
        (Seat(0,1000,0,'active'), Seat(1,1000,to_call,'active')), 1,0,blind,10,2,(),config,
        {'auction_cards': board[:3]+('3c','4d'), 'flop_chips_by_seat':[1000,500]}, 'betting',100.)


def features(state, paid=500, n=64, dead=('3c','4d')):
    stats=SimpleNamespace(auctions=n,paid_mean=25.,fraction_mean=.025,fraction_observations=n)
    return SimpleNamespace(auction_settled=True,paid_this_hand=(0,paid),balance_before_hand=(1000,1000),
        history=(stats,stats),auction_cards=state.board[:3]+dead,dead_cards=dead)


class BettingTests(unittest.TestCase):
    def test_inference_changes_real_decision(self):
        state=make_state(); f=features(state)
        self.assertEqual(choose_bet_without_inference(state,f).kind,'call')
        self.assertGreater(infer_pressure(state,f).penalty,.012)
        self.assertEqual(choose_bet(state,f).kind,'fold')

    def test_unknown_and_zero_are_not_weakness(self):
        state=make_state()
        for paid in (None,0):
            f=features(state,paid)
            self.assertEqual(infer_pressure(state,f).penalty,0.)
            self.assertEqual(choose_bet(state,f),choose_baseline_bet(state))
        f=features(state); f.auction_settled=False
        self.assertEqual(infer_pressure(state,f).penalty,0.)

    def test_sparse_evidence_is_capped(self):
        state=make_state()
        thin=infer_pressure(state,features(state,n=0)).penalty
        rich=infer_pressure(state,features(state,n=64)).penalty
        self.assertGreater(thin,0.)
        self.assertLess(thin,.02)
        self.assertGreater(rich,thin)
        self.assertLessEqual(rich,.055)

    def test_draw_preference_not_made_hand_certainty(self):
        dry=make_state(board=('Ks','9d','2c'))
        wet=make_state(board=('Kh','Qh','7c'))
        a=infer_pressure(dry,features(dry))
        b=infer_pressure(wet,features(wet))
        self.assertGreater(b.draw_weight,a.draw_weight)
        self.assertLess(b.penalty,a.penalty)
        missed=make_state(board=('Kh','Qh','7c','2d','5s'))
        completed=make_state(board=('Kh','Qh','7c','2h','5s'))
        self.assertLess(infer_pressure(missed,features(missed)).penalty,infer_pressure(completed,features(completed)).penalty)

    def test_dead_cards_excluded_and_block_range_support(self):
        state=make_state(board=('9s','9d','2c'))
        f=features(state,dead=('9h','9c'))
        unseen=public_unseen_cards(state,f)
        for c in state.cards+state.board+f.dead_cards+tuple(state.variant['auction_cards']):
            self.assertNotIn(c,unseen)
        unblocked=features(state,dead=('3h','4c'))
        # Update native known candidates too; both inputs are public facts.
        state_blocked=dataclasses.replace(state,variant={'auction_cards':state.board+('9h','9c')})
        state_open=dataclasses.replace(state,variant={'auction_cards':state.board+('3h','4c')})
        self.assertLess(infer_pressure(state_blocked,f).made_weight,infer_pressure(state_open,unblocked).made_weight)

    def test_hidden_fields_are_not_read(self):
        state=make_state(); f=features(state)
        before=choose_bet(state,f)
        poisoned=dataclasses.replace(state,variant={**state.variant,'hidden_bids':object(),'opponent_cards':object(),'scores':object()})
        self.assertEqual(before,choose_bet(poisoned,f))

    def test_folded_opponent_and_auction_are_not_inferred(self):
        state=make_state(); f=features(state)
        folded=dataclasses.replace(state,seats=(state.seats[0],dataclasses.replace(state.seats[1],status='folded')))
        self.assertEqual(infer_pressure(folded,f).penalty,0.)
        self.assertEqual(infer_pressure(dataclasses.replace(state,phase='auction'),f).penalty,0.)

    def test_current_payment_excluded_from_history_calibration(self):
        state=make_state(); f=features(state,n=1)
        only_current=SimpleNamespace(auctions=1,paid_total=500,paid_mean=500.,last_paid=500,
                                    fraction_mean=.5,fraction_observations=1)
        f.history=(only_current,only_current)
        self.assertAlmostEqual(infer_pressure(state,f).penalty,infer_pressure(state,features(state,n=0)).penalty)

    def test_original_baseline_action_parity(self):
        # Pinned fresh event source, outside submitted module. Not a research bot.
        reference=ROOT.parent/'variant'
        sys.path.insert(0,str(reference))
        spec=importlib.util.spec_from_file_location('frozen_variant_reference',reference/'agent.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        agent=mod.Agent(); rng=random.Random(812009)
        deck=[r+s for r in '23456789TJQKA' for s in 'cdhs']
        for i in range(2000):
            cards=rng.sample(deck,7); street=(0,3,4,5)[i%4]
            state=make_state(tuple(cards[2:2+street]),tuple(cards[:2]),to_call=(0,20,50,100,250)[i%5],pot=100+i%500)
            state=dataclasses.replace(state,hand_number=i+1,players_remaining=(2,3,5,8)[i%4])
            original,_=agent.act(state,b'')
            self.assertEqual(original,choose_bet_without_inference(state,None),(i,state,original))

if __name__=='__main__':unittest.main()
