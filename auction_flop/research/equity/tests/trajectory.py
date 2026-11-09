"""Validate full Agent calls on official engine trajectories; no IPC in timings."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bot'))
from agent import Agent
from poker_engine.table import SitAndGo,TableRules
from poker_engine.variants.flop_auction_holdem import FlopAuctionHoldem
samples={};failed=0; tables=[]; agent=Agent()
for k in range(4):
 t=SitAndGo(rules=TableRules(players=8,hand_cap=100),variant=FlopAuctionHoldem(),table_id=f'eq-{k}',seed=f'equity-trajectory-{k}')
 while (state:=t.decision()) is not None:
  start=time.perf_counter_ns();action,memory=agent.act(state,b'');elapsed=(time.perf_counter_ns()-start)/1e6
  label='auction' if state.phase=='auction' else 'betting-'+str(len(state.board))
  samples.setdefault(label,[]).append(elapsed)
  failed+=not t.apply(action)
 tables.append({'hands':t.hands_played,'stacks':t.stacks})
report={'scope':'Exact official SDK state -> Agent.act -> Action. Official engine trajectories; excludes process IPC and state construction. Local shared pinned Python 3.12 environment.','failed_actions':failed,'tables':tables,'latency':{k:{'n':len(v),'p99_ms':sorted(v)[min(len(v)-1,int(.99*len(v)))],'max_ms':max(v),'over_1ms':sum(x>1 for x in v)} for k,v in samples.items()}}
Path('trajectory-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert failed==0
