"""Local full act timing; excludes SDK serialization/IPC and engine."""
import sys,time,random,statistics,json
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bot'))
from test_agent import STATE
from agent import Agent
from range_core import DECK
from poker_agent_sdk import LegalAction
rng=random.Random(923118)
states=[]
for i in range(200):
    c=tuple(rng.sample(DECK,9))
    auction=replace(STATE,cards=c[:2],variant={'auction_cards':list(c[2:7]),'flop_chips':1000},hand_number=i+1)
    states.append(auction)
agent=Agent();results={}
for phase in ('auction','flop','turn','river'):
    times=[]
    for iteration in range(3):
        for state in states:
            if phase!='auction':
                n={'flop':3,'turn':4,'river':5}[phase]
                board=tuple(state.variant['auction_cards'][:3])
                extra=tuple(c for c in DECK if c not in state.cards and c not in state.variant['auction_cards'])[:n-3]
                state=replace(state,board=board+extra,phase='betting',legal_actions=(LegalAction('check'),LegalAction('raise',40,2000)))
            start=time.perf_counter_ns();agent.act(state,b'');times.append((time.perf_counter_ns()-start)/1e6)
    ordered=sorted(times)
    results[phase]={'calls':len(times),'p50_ms':statistics.median(times),'p99_ms':ordered[int(0.99*len(times))],'max_ms':max(times),'over_1ms':sum(x>1 for x in times)}
print(json.dumps({'scope':'act including view and action construction, no serialization/IPC; research server provisional env','results':results},indent=2))
