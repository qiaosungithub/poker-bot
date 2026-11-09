import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bot'))
from agent import Agent as Bot
from poker_agent_sdk import *
from poker_agent_protocol.game import is_legal

BLIND=BlindLevel(10,20)
CONFIG=TableConfig('test',0,8,'flop_auction_holdem',2000,(BLIND,),20,250,(50,30,20,0,0,0,0,0),1,65536,100)
SEATS=tuple(Seat(i,1980,20,'active') for i in range(8))
STATE=GameState((),1,0,('As','Ks'),(),160,0,(LegalAction('bid',0,1000),),(),SEATS,7,0,BLIND,19,8,(),CONFIG,{'auction_cards':['Qs','Js','Ts','2d','3c'],'flop_chips':1000},'auction',100)

if __name__=='__main__':
    agent=Bot()
    for memory in (b'',b'corrupted',bytes(65536)):
        a,m=agent.act(STATE,memory)
        assert a.kind=='bid' and a.amount==0 and len(a.cards)==5 and sum(a.cards)<=1000
        assert m==b''
    a,_=agent.act(replace(STATE,legal_actions=(LegalAction('bid',0,0),),variant={**STATE.variant,'flop_chips':0}),b'')
    assert a.cards==(0,0,0,0,0)
    for board in ((),('Qs','Js','Ts'),('Qs','Js','Ts','8c'),('Qs','Js','Ts','8c','2h')):
        st=replace(STATE,phase='betting',board=board,legal_actions=(LegalAction('check'),LegalAction('raise',40,2000)))
        a,m=agent.act(st,b'junk')
        assert is_legal(a,st.legal_actions),(board,a)
    print('native SDK adapter tests passed')
