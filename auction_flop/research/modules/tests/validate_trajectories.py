"""Development-only official-engine parity and observer ground-truth audit."""
import json,sys,time,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bot'))
from agent import Agent
from tournament_agent import Agent as AuctionBaseline
from variant_agent import Agent as BettingBaseline
from public_features import observe
from poker_engine.table import SitAndGo,TableRules
from poker_engine.variant import load_variant

records=[];observer_times=[];max_memory=0;decisions=0
for number in range(8):
    rules=TableRules(variant='flop_auction_holdem',players=8)
    table=SitAndGo(rules=rules,variant=load_variant(rules.variant),table_id=f'module-parity-{number}',seed=f'modules-public-v1-{number}')
    memories=[b'']*8;failures=[0]*8;counts=[0]*8
    while (state:=table.decision()) is not None:
        seat=state.seat
        expected,_=(AuctionBaseline() if state.phase=='auction' else BettingBaseline()).act(state,b'')
        started=time.perf_counter_ns();features,new_memory=observe(state,memories[seat]);observer_times.append(time.perf_counter_ns()-started)
        action,memory=Agent().act(state,memories[seat])
        assert memory==new_memory
        assert action==expected,(state.phase,action,expected)
        # Privileged engine values are used ONLY by this development audit.
        # Production observer sees only GameState and its bytes memory.
        for i,h in enumerate(features.history):
            assert h.paid_total==1000-features.balances[i],(number,state.hand_number,seat,i,h.paid_total,features.balances[i])
        if features.auction_settled:
            expected_paid=[0]*8
            for position,global_seat in enumerate(table._positions):
                expected_paid[global_seat]=table._hand._paid[position] or 0
            assert features.paid_this_hand==tuple(expected_paid),(features.paid_this_hand,expected_paid)
        memories[seat]=memory;max_memory=max(max_memory,len(memory));counts[seat]+=1;decisions+=1
        if not table.apply(action):failures[seat]+=1
    result=table.result(tuple(failures),tuple(counts))
    records.append({'seed':number,'hands':result.hands_played,'decisions':counts,'failures':failures})
summary={'tables':8,'decisions':decisions,'failures':sum(sum(r['failures']) for r in records),'max_memory_bytes':max_memory,'observer_p99_ms':sorted(observer_times)[int(.99*(len(observer_times)-1))]/1e6,'observer_max_ms':max(observer_times)/1e6,'scope':'observer wall time only during direct engine audit; shared-host load not controlled; no SDK/IPC','records':records}
Path('evidence/trajectory-parity-v1.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
assert summary['failures']==0
