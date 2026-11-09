"""Event-time official-engine benchmark. All bot inputs are real public GameState views."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
from statistics import fmean, stdev, NormalDist
import subprocess
import sys
import time
import traceback

from poker_agent_protocol.game import Action, action_from_dict
from poker_agent_sdk.guest import reply
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant
from poker_engine.scoring import table_points

VARIANT = 'flop_auction_holdem'
_WORKER = None


def import_agent(path, name):
    """Isolate identically named local helper modules across source snapshots."""
    path=Path(path)
    local={p.stem for p in path.glob('*.py')}
    previous={n:sys.modules.pop(n) for n in local if n in sys.modules}
    old_path=sys.path[:]
    sys.path.insert(0,str(path))
    module_name='event_bench_'+name.replace('-','_')
    spec=importlib.util.spec_from_file_location(module_name,path/'agent.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=module
    try:
        spec.loader.exec_module(module)
        return module.Agent
    finally:
        sys.path[:]=old_path
        for n in local:
            sys.modules.pop(n,None)
        sys.modules.update(previous)


def initialize(config):
    global _WORKER
    # Numerical libraries must remain single-threaded; choose affinity before work.
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cpus=sorted(os.sched_getaffinity(0))
    ordinal=multiprocessing.current_process()._identity
    if cpus and ordinal:
        os.sched_setaffinity(0,{cpus[(ordinal[0]-1)%len(cpus)]})
    classes={}
    for e in config['entries']:
        classes[e['name']]=import_agent(e['snapshot'],e['name'])
        if e.get('auction_snapshot'):
            classes[e['name']+'__auction']=import_agent(e['auction_snapshot'],e['name']+'__auction')
    _WORKER=(config,classes)


def run_table(seed, names):
    config,classes=_WORKER
    registry={e['name']:e for e in config['entries']}
    rules=TableRules(variant=VARIANT,players=len(names),hand_cap=config.get('hand_cap',250),
                    decision_ms=60000,time_bank_ms=0)
    table=SitAndGo(rules=rules,variant=load_variant(VARIANT),table_id=seed,seed=seed)
    agents=[classes[n]() for n in names]
    alternates=[classes[n+'__auction']() if registry[n].get('auction_snapshot') else None for n in names]
    memory=[b'']*len(names); other_memory=[b'']*len(names)
    failures=[0]*len(names); decisions=[0]*len(names)
    phase_counts=[{} for _ in names]; errors=[]
    offered=[0]*len(names); auctions=[0]*len(names)
    started=time.monotonic()
    while (state:=table.decision()) is not None:
        seat=state.seat; decisions[seat]+=1
        phase_counts[seat][state.phase]=phase_counts[seat].get(state.phase,0)+1
        action=None;answered=False
        try:
            answer=agents[seat].act(state,memory[seat])
            raw,new_memory=reply(answer)
            primary=action_from_dict(raw)
            if alternates[seat] is not None:
                alt_raw,alt_mem=reply(alternates[seat].act(state,other_memory[seat]))
                alternate=action_from_dict(alt_raw)
                other_memory[seat]=alt_mem[:rules.memory_bytes]
                action=alternate if state.phase=='auction' else primary
            else:
                action=primary
            if state.phase=='auction' and registry[names[seat]].get('auction_zero'):
                action=Action('bid',cards=(0,0,0,0,0))
            memory[seat]=new_memory[:rules.memory_bytes]
            answered=True
            if state.phase=='auction':
                offered[seat]+=sum(action.cards);auctions[seat]+=1
        except Exception as exc:
            errors.append({'seat':seat,'hand':state.hand_number,'phase':state.phase,
                           'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()[-3000:]})
        if not table.apply(action,answered=answered,bank_ms=0.0):
            failures[seat]+=1
            if answered:
                errors.append({'seat':seat,'hand':state.hand_number,'phase':state.phase,
                               'error':'illegal','action':asdict(action)})
    result=table.result(tuple(failures),tuple(decisions))
    points=table_points(result,table.payouts)
    if sum(result.final_stacks)!=rules.starting_stack*len(names):
        raise AssertionError('Chip conservation')
    if abs(sum(points)-sum(table.payouts))>1e-7:
        raise AssertionError('Score conservation')
    spend=[0]*len(names); outcomes=[]; remaining=[1000]*len(names)
    for hand, record in enumerate(table.log,1):
        before=record.get('flop_chips_before',[1000]*len(names))
        after=record.get('flop_chips_after',before)
        paid=[a-b for a,b in zip(before,after)]
        spend=[a+b for a,b in zip(spend,paid)];remaining=after
        if any(paid):
            outcomes.append({'hand':hand,'spent':paid,
              'chip_delta':[b-a for a,b in zip(record['stacks_before'],record['stacks_after'])],
              'equity_delta':[b-a for a,b in zip(record['equity_before'],record['equity_after'])]})
    entries=[{'name':n,'seat':i,'points':points[i],'stack':result.final_stacks[i],
              'decisions':decisions[i],'failures':failures[i],'phase_counts':phase_counts[i],
              'auction_calls':auctions[i],'offered_flopchips':offered[i],
              'spent_flopchips':spend[i],'remaining_flopchips':remaining[i]} for i,n in enumerate(names)]
    return {'seed':seed,'hands':result.hands_played,'payouts':table.payouts,
            'bust_order':result.bust_order,'ties':result.ties,'entries':entries,
            'errors':errors,'spend_outcomes':outcomes,'seconds':time.monotonic()-started}


def block(task):
    index,seed,names=task
    rng=random.Random(seed+':seats');names=names[:];rng.shuffle(names)
    rows=[]
    for rotation in range(len(names)):
        seats=names[rotation:]+names[:rotation]
        row=run_table(seed,seats);row.update(block=index,rotation=rotation);rows.append(row)
    return rows


def independent_scores(stacks,bust_order,ties,payouts):
    values=[0.0]*len(stacks);groups=[]
    for i,seat in enumerate(bust_order):
        if i in ties and groups:groups[-1].append((i,seat))
        else:groups.append([(i,seat)])
    for group in groups:
        paid=sum(payouts[len(stacks)-1-i] if len(stacks)-1-i<len(payouts) else 0 for i,_ in group)/len(group)
        for _,seat in group:values[seat]=paid
    live=tuple(i for i,x in enumerate(stacks) if x)
    def recurse(alive,place,prob):
        if not alive or place>=min(len(live),len(payouts)):return
        total=sum(stacks[i] for i in alive)
        for i in alive:
            p=prob*stacks[i]/total;values[i]+=p*payouts[place]
            recurse(tuple(j for j in alive if i!=j),place+1,p)
    recurse(live,0,1.0)
    return values


def summarize(rows,config):
    names=[e['name'] for e in config['entries']];blocks=config['blocks']
    grouped={n:[[] for _ in range(blocks)] for n in names}
    totals={n:{'decisions':0,'failures':0,'spent':0,'remaining':0,'auctions':0,'chips':0,'tables':0} for n in names}
    seen=set()
    for row in rows:
        key=row['block'],row['rotation'];assert key not in seen;seen.add(key)
        stacks=[e['stack'] for e in row['entries']]
        oracle=independent_scores(stacks,row['bust_order'],row['ties'],row['payouts'])
        for e,expected in zip(row['entries'],oracle):
            assert abs(expected-e['points'])<1e-7
            n=e['name'];grouped[n][row['block']].append(e['points']);t=totals[n]
            for target,source in [('decisions','decisions'),('failures','failures'),('spent','spent_flopchips'),
                                  ('remaining','remaining_flopchips'),('auctions','auction_calls'),('chips','stack')]:
                t[target]+=e[source]
            t['tables']+=1
    block_means={n:[fmean(v) for v in grouped[n] if v] for n in names}
    scores=[]
    for n in names:
        vals=block_means[n];mean=fmean(vals);se=stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else None;t=totals[n]
        scores.append({'name':n,'mean_points':mean,'ci95':[mean-1.96*se,mean+1.96*se] if se is not None else None,
          'blocks':len(vals),'tables':t['tables'],'decisions':t['decisions'],'failures':t['failures'],
          'mean_chips':t['chips']/t['tables'],'mean_spent_flopchips':t['spent']/t['tables'],
          'mean_remaining_flopchips':t['remaining']/t['tables'],
          'mean_spend_per_auction':t['spent']/max(1,t['auctions'])})
    candidates=config.get('compare',names)
    count=max(1,len(candidates)*(len(candidates)-1)//2)
    z=NormalDist().inv_cdf(1-.05/(2*count));pairs=[]
    for a,b in itertools.combinations(candidates,2):
        vals=[x-y for x,y in zip(block_means[a],block_means[b])];mean=fmean(vals)
        se=stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else None
        pairs.append({'a':a,'b':b,'difference':mean,'simultaneous_ci95':[mean-z*se,mean+z*se] if se is not None else None})
    return {'scores':sorted(scores,key=lambda s:-s['mean_points']),'pairs':pairs,
            'independent_blocks':len(rows)//len(names),'tables':len(rows),'all_scores_audited':True,
            'timing_scope':'Untimed direct official engine, separate latency tests required',
            'confidence_scope':'Independent seed block means of cyclic seat rotations'}


def snapshot(source,target,include=None,revision=None):
    target.mkdir(parents=True,exist_ok=False)
    source=Path(source).resolve();hashes={}
    git=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
    top=subprocess.run(['git','-C',str(source),'rev-parse','--show-toplevel'],capture_output=True,text=True)
    dirty=subprocess.run(['git','-C',str(source),'status','--short'],capture_output=True,text=True).stdout
    if revision:
        repo=Path(top.stdout.strip());prefix=source.relative_to(repo)
        listing=subprocess.run(['git','-C',str(repo),'ls-tree','-r','--name-only',revision,'--',str(prefix)],capture_output=True,text=True,check=True)
        files=[Path(line).relative_to(prefix) for line in listing.stdout.splitlines()]
    else:
        files=[p.relative_to(source) for p in source.rglob('*') if p.is_file()]
    for rel in files:
        if '__pycache__' in rel.parts or any(x.startswith('.') for x in rel.parts):continue
        if include is not None and str(rel) not in include:continue
        if revision:
            data=subprocess.run(['git','-C',str(repo),'show',revision+':'+str(prefix/rel)],capture_output=True,check=True).stdout
        else:data=(source/rel).read_bytes()
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        hashes[str(rel)]=hashlib.sha256(data).hexdigest()
    if not (target/'agent.py').is_file():raise ValueError('Snapshot lacks agent.py')
    return {'path':str(source),'git_head':git.stdout.strip(),'frozen_revision':revision,
            'source_dirty_status':dirty,'sha256':hashes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    config=json.loads(args.config.read_text());names=[e['name'] for e in config['entries']]
    assert 2<=len(names)<=10 and len(names)==len(set(names))
    args.out.mkdir(parents=True,exist_ok=False);args.out=args.out.resolve()
    provenance={}
    for e in config['entries']:
        target=args.out/'sources'/e['name'];provenance[e['name']]=snapshot(e['path'],target,e.get('files'),e.get('revision'));e['snapshot']=str(target)
        if e.get('auction_path'):
            alt=args.out/'sources'/(e['name']+'__auction');provenance[e['name']+'__auction']=snapshot(e['auction_path'],alt,e.get('auction_files'),e.get('auction_revision'));e['auction_snapshot']=str(alt)
    manifest={'config':config,'provenance':provenance,'started':datetime.now(timezone.utc).isoformat(),
              'workers':args.workers,'python':sys.version,'engine_rules':asdict(TableRules(variant=VARIANT,players=len(names),hand_cap=config.get('hand_cap',250),decision_ms=60000,time_bank_ms=0)),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mode':'direct, no clock penalties, real official SDK observations',
              'hybrid_memory':'Each donor observes every actual decision; separate bytes per donor'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(b,f"{config['seed']}:{b}",names) for b in range(config['blocks'])]
    rows=[];started=time.monotonic()
    with (args.out/'tables.jsonl').open('x') as output:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(config,)) as pool:
            futures=[pool.submit(block,t) for t in tasks]
            for future in as_completed(futures):
                result=future.result();rows.extend(result)
                for row in result:output.write(json.dumps(row,separators=(',',':'))+'\n')
                output.flush()
                print(json.dumps({'blocks':len(rows)//len(names),'total_blocks':config['blocks'],'seconds':round(time.monotonic()-started,2)}),flush=True)
    summary=summarize(rows,config);summary['seconds']=time.monotonic()-started
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest.update(completed=datetime.now(timezone.utc).isoformat(),seconds=summary['seconds'],
                    tables_sha256=hashlib.sha256((args.out/'tables.jsonl').read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
