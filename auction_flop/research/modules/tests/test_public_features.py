import json
import sys
from pathlib import Path
from dataclasses import replace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bot'))
from public_features import observe
from poker_agent_protocol.game import Action, BlindLevel, Event, GameState, LegalAction, Seat, TableConfig

CANDIDATES = ('As','Kd','7c','2h','9s')
FLOP = CANDIDATES[:3]


def state(*, hand=1, table='a', balances=(1000,1000,1000), board=(), phase='betting', events=(), history=None, ownpaid=None):
    config = TableConfig(table,0,3,'flop_auction_holdem',2000,(BlindLevel(10,20),),20,250,(50,30,20),1,65536,100)
    variant = {'flop_chips':balances[0],'flop_chips_by_seat':list(balances)}
    if phase == 'auction' or board:variant['auction_cards']=CANDIDATES
    if ownpaid is not None:variant['flop_chips_paid']=ownpaid
    legal=(LegalAction('bid',0,balances[0]),) if phase=='auction' else (LegalAction('check'),LegalAction('raise',20,2000))
    hist=tuple(e for e in events if e.kind!='hand') if history is None else tuple(history)
    return GameState(tuple(events),hand,0,('Ah','Qh'),tuple(board),60,0,legal,hist,
        tuple(Seat(i,1980,20,'active') for i in range(3)),2,1,BlindLevel(10,20),19,3,(),config,variant,phase,100)


def hand_start(hand=1):
    return (Event('hand',amount=hand),Event('blind',1,10),Event('blind',2,20))


def auction_events():
    return (Event('auction',cards=CANDIDATES), Event('bid',0),Event('bid',1),Event('bid',2),Event('board',cards=FLOP))


class FeatureTests(unittest.TestCase):
    def test_selected_payments_and_dead_cards_and_replay(self):
        s0=state(events=hand_start());_,m0=observe(s0,b'')
        auction=state(phase='auction',events=(Event('auction',cards=CANDIDATES),),history=s0.history+(Event('auction',cards=CANDIDATES),))
        fa,ma=observe(auction,m0)
        self.assertEqual(fa.paid_this_hand,(None,None,None))
        self.assertEqual([h.auctions for h in fa.history],[0,0,0])
        s1=state(balances=(990,940,1000),board=FLOP,events=auction_events()[1:],history=auction.history+auction_events()[1:],ownpaid=10)
        f,m=observe(s1,ma)
        self.assertEqual(f.paid_this_hand,(10,60,0))
        self.assertEqual(f.dead_cards,('2h','9s'))
        self.assertEqual(f.history[1].auctions,1)
        self.assertEqual(f.history[1].paid_total,60)
        self.assertEqual(f.history[2].positive_observations,1)
        self.assertEqual(f.history[2].positive_auctions,0)
        self.assertEqual(f.auction_participants,(0,1,2))
        self.assertEqual(observe(s1,ma),(f,m)) # timeout replay from accepted memory
        self.assertEqual(observe(s1,m),(f,m)) # repeated same snapshot idempotent
        damaged=json.loads(m);damaged['rows'][1][1]+=1
        self.assertNotEqual(observe(s1,json.dumps(damaged).encode())[0].history[1].paid_total,60) # audit detects corruption

    def test_after_preflop_fold_observe_payment_next_hand(self):
        s0=state(events=hand_start());_,m=observe(s0,b'')
        ev=(Event('fold',0),Event('auction',cards=CANDIDATES),Event('bid',1),Event('bid',2),Event('board',cards=FLOP))+hand_start(2)
        s1=state(hand=2,balances=(1000,900,980),events=ev,history=hand_start(2)[1:])
        f,m=observe(s1,m)
        self.assertEqual(f.paid_this_hand,(None,None,None))
        self.assertEqual(f.balance_before_hand,(1000,900,980))
        self.assertEqual([x.auctions for x in f.history],[0,1,1])
        self.assertEqual([x.paid_total for x in f.history],[0,100,20])

    def test_multiple_auctions_are_not_fabricated_individual_bids(self):
        s0=state(events=hand_start());_,m=observe(s0,b'')
        ev=auction_events()+hand_start(2)+auction_events()+hand_start(3)
        s1=state(hand=3,balances=(950,900,1000),events=ev,history=hand_start(3)[1:])
        f,_=observe(s1,m)
        self.assertEqual(f.history[1].auctions,2)
        self.assertEqual(f.history[1].paid_total,100)
        self.assertEqual(f.history[1].paid_mean,50)
        self.assertEqual(f.history[1].positive_observations,0)
        self.assertIsNone(f.history[1].last_paid)

    def test_missing_memory_midhand_does_not_invent_payment(self):
        hist=hand_start()[1:]+auction_events()
        s=state(balances=(700,600,500),board=FLOP,events=(Event('check',1),),history=hist,ownpaid=12)
        for memory in (b'',b'{',b'{"v":1}',b'[]',b'null'):
            f,m=observe(s,memory)
            self.assertTrue(f.memory_reset)
            self.assertEqual(f.paid_this_hand,(12,None,None))
            self.assertEqual([x.paid_total for x in f.history],[0,0,0])
            self.assertEqual([x.auctions for x in f.history],[0,0,0])
            self.assertEqual(f.auction_participants,(0,1,2))
            self.assertLess(len(m),8192)

    def test_truncated_memory_during_auction_recovers_current_cursor(self):
        hist=hand_start()[1:]+(Event('auction',cards=CANDIDATES),Event('bid',1))
        sa=state(phase='auction',events=(Event('bid',1),),history=hist)
        _,m=observe(sa,b'{')
        tail=(Event('bid',0),Event('bid',2),Event('board',cards=FLOP))
        sf=state(balances=(990,980,970),board=FLOP,events=tail,history=hist+tail,ownpaid=10)
        f,_=observe(sf,m)
        self.assertEqual(f.paid_this_hand,(10,20,30))
        self.assertEqual([x.auctions for x in f.history],[1,1,1])

    def test_table_isolation_and_currency_noninterference(self):
        s=state(events=hand_start());_,m=observe(s,b'')
        other=state(table='b',balances=(600,700,800),events=hand_start())
        f,_=observe(other,m)
        self.assertTrue(f.memory_reset)
        self.assertEqual([x.paid_total for x in f.history],[0,0,0])
        normal=replace(s,pot=9999,seats=tuple(replace(seat,stack=1,bet=1000) for seat in s.seats))
        self.assertEqual(observe(normal,b'')[0].balances,observe(s,b'')[0].balances)


if __name__=='__main__':unittest.main()
