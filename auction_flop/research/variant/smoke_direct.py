"""Development-only official-engine smoke and policy-path timing."""
import json,time,statistics
from pathlib import Path
from collections import defaultdict
from poker_engine.table import SitAndGo,TableRules
from poker_engine.variant import load_variant
from poker_engine.scoring import table_points
from poker_agent_sdk import Action
from agent import Agent

timings=defaultdict(list);records=[];bot=Agent()
for players in (2,3,8):
 for seed in range(4):
  rules=TableRules(variant='flop_auction_holdem',players=players,hand_cap=120)
  table=SitAndGo(rules=rules,variant=load_variant('flop_auction_holdem'),table_id=f'variant-smoke-{players}-{seed}',seed=f'variant-v1-{players}-{seed}')
  decisions=[0]*players;failures=[0]*players
  while (state:=table.decision()) is not None:
   start=time.perf_counter_ns();action,memory=bot.act(state,b'');elapsed=time.perf_counter_ns()-start
   timings[state.phase if state.phase!='betting' else f'betting_{len(state.board)}'].append(elapsed)
   decisions[state.seat]+=1
   if not table.apply(action):failures[state.seat]+=1
  result=table.result(tuple(failures),tuple(decisions))
  records.append({'players':players,'seed':seed,'hands':result.hands_played,'decisions':decisions,'failures':failures,'points':table_points(result,table.payouts),'stacks':result.final_stacks})
summary={k:{'n':len(v),'p50_ms':statistics.median(v)/1e6,'p99_ms':sorted(v)[max(0,int(len(v)*.99)-1)]/1e6,'max_ms':max(v)/1e6,'above_1ms':sum(t>1e6 for t in v)} for k,v in timings.items()}
out={'scope':'official engine direct calls, no SDK process or IPC; all seats independently stateless same policy','summary':summary,'tables':records,'timings_ns':dict(timings)}
Path('evidence/direct_smoke.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'summary':summary,'tables':len(records),'failed':sum(sum(x['failures']) for x in records)},indent=2))
assert not any(any(x['failures']) for x in records)
