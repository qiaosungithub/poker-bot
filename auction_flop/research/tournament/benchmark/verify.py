"""Negative score audit and official direct/SDK process settlement parity."""
import asyncio, copy, json, sys
from pathlib import Path
import run
from poker_agent_protocol.process import BotPool, CLEAN_ENV
from poker_engine.play import play_table
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

async def main():
    directory=Path(sys.argv[1]);config=json.loads((directory/'manifest.json').read_text())['config']
    rows=[json.loads(x) for x in (directory/'tables.jsonl').read_text().splitlines()]
    broken=copy.deepcopy(rows);broken[0]['entries'][0]['points']+=1
    try:run.summarize(broken,config)
    except AssertionError:negative=True
    else:raise AssertionError('Corrupted score was accepted')
    # Parity excludes zero-bid wrappers so SDK gets exactly the frozen sources.
    for e in config['entries']:e['auction_zero']=False
    run.initialize(config);names=[e['name'] for e in config['entries']]
    seed='boston-20261109-sdk-parity-v1'
    direct=run.run_table(seed,names)
    pools=[BotPool([sys.executable,'-m','poker_agent_sdk.guest',e['snapshot']],copies=1,
                   cwd=e['snapshot'],env={**CLEAN_ENV,'PYTHONDONTWRITEBYTECODE':'1'}) for e in config['entries']]
    timing={e['name']:[] for e in config['entries']}
    class Recorded:
        def __init__(self,pool,name):self.pool=pool;self.name=name
        async def act(self,state,memory):
            answer=await self.pool.act(state,memory)
            timing[self.name].append({'ms':answer.elapsed_ms,'phase':state.phase,'board':len(state.board)})
            return answer
    try:
        table=SitAndGo(rules=TableRules(variant=run.VARIANT,players=len(names),decision_ms=60000,time_bank_ms=0),
                       variant=load_variant(run.VARIANT),table_id=seed,seed=seed)
        result=await play_table(table,[Recorded(p,n) for p,n in zip(pools,names)])
        assert list(result.final_stacks)==[e['stack'] for e in direct['entries']]
        assert list(result.decisions)==[e['decisions'] for e in direct['entries']]
        assert not any(result.failures)
        assert result.hands_played==direct['hands']
        errors=[p.take_errors() for p in pools]
    finally:
        for p in pools:await p.stop()
    report={'negative_score_control_rejected':negative,'direct_sdk_exact_stacks_decisions_hands':True,
            'sdk_failures':result.failures,'sdk_errors':errors,'seed':seed,'timings':timing,
            'scope':'Official SDK host/guest IPC, relaxed60s deadline for deterministic protocol parity; not deadline certification'}
    (directory/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='timings'},indent=2))

if __name__=='__main__':asyncio.run(main())
