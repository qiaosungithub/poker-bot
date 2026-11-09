"""Latency screening and paired six-player tournaments for pinned policies."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gc
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from . import policies as P
import numpy as np

ROOT=P.ROOT
H=P.engine
CORPUS=ROOT/'hu-benchmark/multiway_latency_states.jsonl.gz'


def dump(path,obj):
    path=Path(path)
    with path.open('x') as f:json.dump(obj,f,indent=2)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_stats(values):
    x=np.asarray(values)/1e6
    return {'n':len(x),'p50_ms':float(np.quantile(x,.5)),'p99_ms':float(np.quantile(x,.99)),
            'p999_ms':float(np.quantile(x,.999)),'max_ms':float(x.max()),'over_1ms':int((x>1).sum())}


def load_states(per_street):
    buckets={s:[] for s in P.engine.C.STREETS}
    with gzip.open(CORPUS,'rt') as f:
        for line in f:
            s=json.loads(line)
            if len(buckets[s['street']])<per_street:buckets[s['street']].append(s)
    assert all(len(v)==per_street for v in buckets.values())
    states=[s for group in buckets.values() for s in group]
    random.Random(31009000).shuffle(states)
    # Do not expose strategy names from the historical corpus.
    for s in states:
        for player in s['players']:player['bot_id']='seat_'+str(player['seat'])
        for event in s.get('match_action_log',[]):event['bot_id']='seat_'+str(event['seat'])
    return states


def latency(args):
    os.sched_setaffinity(0,{args.cpu})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    states=load_states(args.per_street)
    names=args.names.split(',') if args.names else P.CANDIDATES
    records={};arrays={}
    for index,name in enumerate(names):
        try:
            policy=P.Policy(name,31009900+index)
            for s in states[:min(200,len(states))]:
                policy.enter()
                try:policy.act(copy.deepcopy(s))
                finally:policy.leave()
            policy.exceptions.clear()
            wall=[];cpu=[];phases=[];errors=[]
            gc.collect();gc.disable()
            for repeat in range(args.repeats):
                for source in states:
                    s=copy.deepcopy(source)
                    policy.enter()
                    c=time.thread_time_ns();start=time.perf_counter_ns()
                    try:a=policy.act(s)
                    except Exception as ex:
                        errors.append(repr(ex));a=None
                    elapsed=time.perf_counter_ns()-start;cpu_elapsed=time.thread_time_ns()-c
                    policy.leave()
                    wall.append(elapsed);cpu.append(cpu_elapsed);phases.append(P.engine.C.STREETS.index(s['street']))
                    if not isinstance(a,dict) or a.get('action') not in ('fold','check','call','raise','all_in'):
                        errors.append('Invalid action: '+repr(a));break
                if errors:break
            gc.enable()
            by_street={st:time_stats([v for v,p in zip(wall,phases) if p==i]) for i,st in enumerate(P.engine.C.STREETS) if i in phases}
            stats=time_stats(wall)
            eligible=(not errors and not policy.exceptions and len(wall)==len(states)*args.repeats and
                      stats['p99_ms']<1 and all(v['p99_ms']<1 for v in by_street.values()) and
                      stats['over_1ms']/stats['n']<=.001)
            records[name]={'wall':stats,'cpu':time_stats(cpu),'by_street':by_street,'eligible':eligible,
                           'exceptions':dict(Counter(policy.exceptions)),'errors':errors,'policy':P.metadata(name)}
            arrays[name+'__wall']=np.asarray(wall,dtype=np.int64)
            arrays[name+'__cpu']=np.asarray(cpu,dtype=np.int64)
            arrays[name+'__phase']=np.asarray(phases,dtype=np.uint8)
        except Exception as ex:
            import traceback
            records[name]={'eligible':False,'error':traceback.format_exc()}
        finally:gc.enable()
        dump(output/(name+'.json'),records[name])
        print(json.dumps({'bot':name,'eligible':records[name]['eligible'],'wall':records[name].get('wall'),'errors':records[name].get('error',records[name].get('exceptions'))}),flush=True)
    np.savez_compressed(output/'timings.npz',**arrays)
    result={'config':vars(args),'corpus':str(CORPUS.relative_to(ROOT)),'corpus_sha256':sha(CORPUS),
            'raw_sha256':sha(output/'timings.npz'),'results':records,'scope':'Complete decision function, warm CPU, GC disabled; no engine, startup, JSON/IPC or RNG isolation.'}
    dump(output/'result.json',result)


def match(names,seed,hands=200,measure=None):
    assert len(names)==6
    ids=[f'seat_{i}' for i in range(6)]
    policies={slot:P.Policy(name,seed*101+i) for i,(slot,name) in enumerate(zip(ids,names))}
    stacks={i:10000 for i in ids};locked={};history=[];dealer=0
    corrections=Counter();decisions=Counter();actions=Counter();completed=0;eliminations=[]
    for hand_index in range(hands):
        alive=[i for i in ids if stacks[i]>0]
        if len(alive)==1:break
        while not stacks[ids[dealer]]:dealer=(dealer+1)%6
        before=[stacks[i] for i in alive]
        hand=H.Hand(alive,before,alive.index(ids[dealer]),seed*1000003+hand_index,f'holdem_{seed}_{hand_index}')
        state=hand.state();steps=0
        while state is not None:
            slot=alive[state['seat_to_act']];policy=policies[slot]
            state['match_action_log']=history[-200:]
            if measure is not None:measure.before(policy,state)
            policy.enter()
            if measure is None:
                try:action=policy.act(state)
                finally:policy.leave()
            else:
                c=time.thread_time_ns();t=time.perf_counter_ns()
                try:action=policy.act(state)
                finally:
                    dt=time.perf_counter_ns()-t;ct=time.thread_time_ns()-c
                    policy.leave()
                measure.after(policy,state,dt,ct)
            decisions[slot]+=1;actions[slot+'_'+action['action']]+=1
            entry=hand.act(action)
            history.append(dict(entry,bot_id=slot,hand_num=hand_index))
            state=hand.state();steps+=1;assert steps<400
        final=hand.final_stacks()
        for slot,value in zip(alive,final):stacks[slot]=value
        awarded=H.elimination_scores(alive,before,final)
        if awarded:eliminations.append({'alive':alive,'before':before,'after':final,'awarded':awarded})
        locked.update(awarded);corrections.update(hand.normalizations)
        assert sum(stacks.values())==60000
        completed+=1;dealer=(dealer+1)%6
    alive=[i for i in ids if stacks[i]>0]
    scores=dict(locked)
    scores.update(zip(alive,H.icm([stacks[i] for i in alive],H.PAYOUTS[:len(alive)])))
    assert abs(sum(scores.values())-235)<1e-8
    return {'roster':names,'scores':[scores[i] for i in ids],'stacks':[stacks[i] for i in ids],
            'hands':completed,'decisions':dict(decisions),'normalizations':dict(corrections),
            'actions':dict(actions),'eliminations':eliminations,
            'exceptions':{i:dict(Counter(p.exceptions)) for i,p in policies.items() if p.exceptions}}


def block(job):
    names,seed,hands,label=job
    order=list(range(6));random.Random(seed).shuffle(order)
    legs=[]
    for rotation in range(6):
        perm=order[rotation:]+order[:rotation]
        leg=match([names[i] for i in perm],seed,hands)
        leg['slot_order']=perm
        legs.append(leg)
    result={'seed':seed,'label':label,'names':names,'legs':legs,
            'scores':[statistics.mean(leg['scores'][leg['slot_order'].index(i)] for leg in legs) for i in range(6)]}
    # Isolated modules have function/global cycles; collect between blocks only.
    gc.collect()
    return result


def initialize_worker():
    # Keep each worker on its own logical CPU; parent remains unconstrained.
    available=sorted(os.sched_getaffinity(0))
    identity=multiprocessing.current_process()._identity
    if identity:os.sched_setaffinity(0,{available[(identity[0]-1)%len(available)]})
    P.initialize_fullhouse()
    gc.disable()


def arena(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    names=args.names.split(',')
    manifest={'config':vars(args),'payouts':H.PAYOUTS,'policies':{n:P.metadata(n) for n in set(names+P.CONTROLS)},
              'engine_sha256':sha(ROOT/'hu-benchmark/multiway_benchmark.py'),
              'runner_sha256':{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}}
    dump(output/'manifest.json',manifest)
    jobs=[]
    if args.mode=='screen':
        for candidate in names:
            jobs.extend(([candidate]+P.CONTROLS,args.seed+b,args.hands,candidate) for b in range(args.blocks))
    else:
        assert len(names)==6
        jobs=[(names,args.seed+b,args.hands,'confirmation') for b in range(args.blocks)]
    start=time.perf_counter();results=[]
    with (output/'blocks.jsonl').open('x') as stream:
        with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            for i,row in enumerate(pool.map(block,jobs,chunksize=1)):
                stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
                results.append(row)
                if (i+1)%64==0:print(json.dumps({'completed_blocks':i+1,'total':len(jobs),'seconds':time.perf_counter()-start}),flush=True)
    groups=defaultdict(list)
    for row in results:groups[row['label']].append(row)
    summary={}
    for label,rows in groups.items():
        group={}
        for i,name in enumerate(rows[0]['names']):
            xs=[r['scores'][i] for r in rows]
            mean=statistics.mean(xs);se=statistics.stdev(xs)/len(xs)**.5 if len(xs)>1 else None
            group[f'{i}:{name}']={'mean':mean,'ci95':[mean-1.96*se,mean+1.96*se] if se is not None else None}
        summary[label]=group
    dump(output/'summary.json',{'results':summary,'elapsed_seconds':time.perf_counter()-start,
                               'blocks':len(results),'tables':6*len(results),'raw_sha256':sha(output/'blocks.jsonl')})
    print(json.dumps({'summary':summary,'seconds':time.perf_counter()-start}),flush=True)


class Recorder:
    def __init__(self):self.values=defaultdict(list);self.cpu=defaultdict(list);self.phases=defaultdict(list);self.history=defaultdict(list)
    def before(self,policy,state):pass
    def after(self,policy,state,wall,cpu):
        name=policy.name
        self.values[name].append(wall);self.cpu[name].append(cpu)
        self.phases[name].append(H.C.STREETS.index(state['street']))
        self.history[name].append(len(state['match_action_log']))


def trajectory_latency(args):
    os.sched_setaffinity(0,{args.cpu})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    names=args.names.split(',');assert len(names)==6
    # Warm a whole table. Subsequent imports/table resets are outside decision timing.
    match(names,args.seed-1,20)
    rec=Recorder();tables=[]
    gc.disable()
    try:
        for b in range(args.blocks):
            seed=args.seed+b;order=list(range(6));random.Random(seed).shuffle(order)
            for r in range(6):
                perm=order[r:]+order[:r]
                tables.append(match([names[i] for i in perm],seed,args.hands,rec))
            print(json.dumps({'timed_blocks':b+1,'decisions':sum(map(len,rec.values.values()))}),flush=True)
    finally:gc.enable()
    result={};arrays={}
    for name,values in rec.values.items():
        by={s:time_stats([v for v,p in zip(values,rec.phases[name]) if p==i]) for i,s in enumerate(H.C.STREETS) if i in rec.phases[name]}
        stats=time_stats(values)
        result[name]={'wall':stats,'cpu':time_stats(rec.cpu[name]),'by_street':by,
                      'max_history_actions':max(rec.history[name]),
                      'eligible':stats['p99_ms']<1 and all(v['p99_ms']<1 for v in by.values()) and stats['over_1ms']/len(values)<=.001}
        for kind,source in [('wall',rec.values),('cpu',rec.cpu),('phase',rec.phases),('history',rec.history)]:arrays[name+'__'+kind]=np.asarray(source[name],dtype=np.int64)
    np.savez_compressed(output/'timings.npz',**arrays)
    dump(output/'tables.json',tables)
    dump(output/'result.json',{'config':vars(args),'results':result,'raw_sha256':sha(output/'timings.npz')})
    print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['latency','screen','confirmation','trajectory'])
    ap.add_argument('--names');ap.add_argument('--output',required=True)
    ap.add_argument('--cpu',type=int,default=2);ap.add_argument('--per-street',type=int,default=100)
    ap.add_argument('--repeats',type=int,default=1);ap.add_argument('--blocks',type=int,default=128)
    ap.add_argument('--seed',type=int,default=31100000);ap.add_argument('--hands',type=int,default=200)
    ap.add_argument('--workers',type=int,default=24)
    args=ap.parse_args()
    if args.mode=='latency':latency(args)
    elif args.mode=='trajectory':trajectory_latency(args)
    else:arena(args)


if __name__=='__main__':main()
