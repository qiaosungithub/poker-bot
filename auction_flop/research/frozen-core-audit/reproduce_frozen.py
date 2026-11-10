"""Read-only frozen-source correctness probes; no policy/engine changes."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[3]/'event-20261109'
SNAPS={'hybrid':('7e7aa88','bot/'),'inference':('6d12db2',''),'joint':('3a74011','bot/')}
CODE=r'''
import itertools,json,sys
from dataclasses import replace
from poker_agent_sdk import GameState,TableConfig,BlindLevel,Seat,LegalAction,Event
import eval7
label=sys.argv[1]
if label=='hybrid':
 from agent import Agent
 from variant_agent import hand_quality
 act=lambda s:Agent().act(s,b'')[0]
elif label=='inference':
 from betting_inference_core import choose_bet
 from betting_baseline import hand_quality
 act=lambda s:choose_bet(s,None)
else:
 from joint_policy import choose_postflop
 from joint_betting_base import hand_quality
 act=lambda s:choose_postflop(s,None)
def fixture(hole,board,players=2,pot=300,call=100,stack=1000):
 cfg=TableConfig('audit',0,players,'boston',2000,(BlindLevel(10,20),),20,250,(50,30,20),1,65536,100)
 seats=(Seat(0,stack,0,'active'),)+tuple(Seat(i,900,call,'active') for i in range(1,players))
 return GameState((),10,0,hole,board,pot,call,(LegalAction('fold'),LegalAction('call'),LegalAction('raise',call*2,stack)),(Event('board',cards=board),Event('raise',1,call)),seats,1,0,BlindLevel(10,20),10,players,(),cfg,{'flop_chips':900,'flop_chips_by_seat':[900]*players,'auction_cards':list(board[:3])+['8c','9d']},'betting',100)
def exact_share(state):
 known=set(state.cards+state.board+tuple(state.variant['auction_cards']))
 deck=[eval7.Card(r+s) for r in '23456789TJQKA' for s in 'cdhs' if r+s not in known]
 board=[eval7.Card(c) for c in state.board];ours=eval7.evaluate(board+[eval7.Card(c) for c in state.cards])
 vals=[]
 for other in itertools.combinations(deck,2):
  rank=eval7.evaluate(board+list(other));vals.append(1 if ours>rank else .5 if ours==rank else 0)
 return sum(vals)/len(vals)
royal=fixture(('2c','3d'),('As','Ks','Qs','Js','Ts'))
quads=fixture(('3c','4d'),('As','Ah','Ad','Ac','2s'))
side=fixture(('Kc','2d'),('Ks','Qd','9c','7h','4d'),3,2000,100,100)
side=replace(side,to_call=1000,seats=(Seat(0,100,0,'active'),Seat(1,0,1000,'all_in'),Seat(2,0,1000,'all_in')),legal_actions=(LegalAction('fold'),LegalAction('call')),history=(Event('board',cards=side.board),Event('raise',1,1000),Event('call',2,1000)))
rows={'royal_board':{'quality':hand_quality(royal.cards,royal.board),'action':repr(act(royal)),'exact_hu_showdown_share':exact_share(royal)},'public_quads_low_kicker':{'quality':hand_quality(quads.cards,quads.board),'action':repr(act(quads)),'exact_hu_showdown_share':exact_share(quads)},'inaccessible_sidepot':{'action_raw_pot':repr(act(side)),'action_accessible_pot_control':repr(act(replace(side,pot=200))),'raw_pot_odds':100/2100,'eligible_pot_odds':100/300,'accessible_opponents_contributions':[100,100],'inaccessible_sidepot':1800}}
assert rows['royal_board']['action'].startswith("Action(kind='fold'")
assert rows['royal_board']['exact_hu_showdown_share']==.5
assert rows['public_quads_low_kicker']['quality'][0]>.9
assert rows['public_quads_low_kicker']['exact_hu_showdown_share']<.03
assert rows['inaccessible_sidepot']['action_raw_pot'].startswith("Action(kind='call'")
assert rows['inaccessible_sidepot']['action_accessible_pot_control'].startswith("Action(kind='fold'")
print(json.dumps(rows))
'''
results={}
for label,(commit,prefix) in SNAPS.items():
 folder=ROOT/'.audit_sources'/label;folder.mkdir(parents=True,exist_ok=True)
 names=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',commit],text=True).splitlines()
 for name in names:
  if name.startswith(prefix) and name.endswith('.py') and '/' not in name[len(prefix):]:
   (folder/Path(name).name).write_bytes(subprocess.check_output(['git','-C',str(REPO),'show',f'{commit}:{name}']))
 r=subprocess.run([sys.executable,'-c',CODE,label],cwd=folder,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 results[label]={'commit':commit,'findings':json.loads(r.stdout)}
print(json.dumps(results,indent=2))
(ROOT/'audit/frozen_findings.json').write_text(json.dumps(results,indent=2)+'\n')
