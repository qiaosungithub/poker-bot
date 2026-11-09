import unittest
from dataclasses import replace
from poker_agent_sdk import BlindLevel, TableConfig, Seat, LegalAction, GameState
from agent import Agent, hand_quality


def state_fixture(phase='betting'):
    cfg=TableConfig('test',0,2,'boston',2000,(BlindLevel(10,20),),20,250,(50,30,20),1,65536,100)
    return GameState((),1,0,('As','Ah'),('Ac','7d','2s'),500,100,
        (LegalAction('fold'),LegalAction('call'),LegalAction('raise',200,2000)),(),
        (Seat(0,2000,0,'active'),Seat(1,1900,100,'active')),1,0,BlindLevel(10,20),19,2,(),cfg,
        {'flop_chips':1000,'flop_chips_by_seat':[1000,1000],'auction_cards':['Ac','7d','2s','Kh','Qd']},phase,100)

class AgentTest(unittest.TestCase):
    def test_strong_categories(self):
        self.assertEqual(hand_quality(('As','Ah'),('Ac','7d','2s'))[1],3)
        self.assertEqual(hand_quality(('9s','9h'),('9d','9c','2s'))[1],7)
        self.assertLess(hand_quality(('2s','3h'),('Ah','Ad','Ks','Kd','Qc'))[0],.2)
        self.assertNotEqual(Agent().act(state_fixture(),b'broken memory')[0].kind,'fold')

    def test_official_auction_wire(self):
        state=replace(state_fixture('auction'),board=(),legal_actions=(LegalAction('bid',0,1000),),to_call=0)
        action,memory=Agent().act(state,b'')
        self.assertEqual(action.kind,'bid');self.assertEqual(action.amount,0)
        self.assertEqual(len(action.cards),5);self.assertLessEqual(sum(action.cards),1000)
        self.assertEqual(memory,b'')
        all_in=replace(state,seats=(Seat(0,0,2000,'all_in'),Seat(1,0,2000,'all_in')))
        self.assertEqual(Agent().act(all_in,b'')[0].kind,'bid')

    def test_no_raising_when_not_legal(self):
        s=replace(state_fixture(),legal_actions=(LegalAction('fold'),LegalAction('call')))
        self.assertEqual(Agent().act(s,b'')[0].kind,'call')

if __name__=='__main__':unittest.main()
