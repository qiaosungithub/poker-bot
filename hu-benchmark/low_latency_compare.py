"""Compare existing policies under the frozen HU engine and scoring protocol."""
import run_benchmark as R
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gc
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
import numpy as np
from engine_adapter import PokerEngine

HERE = Path(__file__).resolve().parent
R.PATHS['txnujk'] = HERE.parent / 'txnujk-bot/bots/floorfix/bot.py'
STREETS = ('preflop', 'flop', 'turn', 'river')


def make_bot(name):
    if name == 'cfr_lite64':
        fn=R.opponent('cfr_lite')
        config=copy.deepcopy(fn.__globals__['CONFIG'])
        config['equity_iterations'].update(flop=64,turn=64,river=64,heads_up_bonus=0,multiway_penalty=0,multiway_floor=64)
        fn.__globals__['CONFIG']=config
        return fn
    if name == 'hydra32':
        fn=R.opponent('adaptive_hydra')
        tree=ast.parse(R.PATHS['adaptive_hydra'].read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mc_equity')
        changes=0
        for assignment in ast.walk(node):
            if isinstance(assignment,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='max_iters' for t in assignment.targets):
                assert assignment.value.value==300
                assignment.value=ast.Constant(32);changes+=1
        assert changes==1
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(R.PATHS['adaptive_hydra']),'exec'),fn.__globals__)
        return fn
    return R.HERO.decide if name == 'fullhouse' else R.opponent(name)


def check_action(s, a):
    assert a['action'] in ('fold','check','call','raise','all_in'), a
    assert a['action'] != 'check' or s['can_check'], a
    if a['action'] == 'raise':
        assert min(s['min_raise_to'], s['your_stack']+s['your_bet_this_street']) <= a['amount'] <= s['your_stack']+s['your_bet_this_street'], (s,a)


def corpus(per_street, seed):
    rng=random.Random(seed); buckets={s:[] for s in STREETS}; hand=0
    while min(map(len,buckets.values())) < per_street:
        stacks={f'p{i}':rng.choice((500,1000,2000,5000,10000,20000)) for i in range(2)}
        e=PokerEngine(f'latency_{seed}_{hand}', ['p0','p1'], hand%2, stacks, seed+hand)
        s=e.start_hand(); passive=hand%3==0
        while s['type']=='action_request':
            s['match_action_log']=[dict(a,hand_num=hand,bot_id=f"p{a['seat']}") for a in s['action_log']][-200:]
            if len(buckets[s['street']])<per_street:buckets[s['street']].append(copy.deepcopy(s))
            x=rng.random()
            if passive or x<.65:a={'action':'check' if s['can_check'] else 'call'}
            elif x<.78:a={'action':'fold'}
            else:
                total=s['your_bet_this_street']+s['your_stack']
                amount=min(total,max(s['min_raise_to'],s['current_bet']+int(rng.choice((.33,.5,1,2))*(s['pot']+s['amount_owed']))))
                a={'action':'raise','amount':amount}
            s=e.apply_action(s['seat_to_act'],a)
        hand+=1
    states=[s for b in buckets.values() for s in b];rng.shuffle(states)
    return states


def timing_stats(values):
    x=np.array(values)/1e6
    return dict(n=len(x),p50_ms=float(np.quantile(x,.5)),p99_ms=float(np.quantile(x,.99)),
                p999_ms=float(np.quantile(x,.999)),max_ms=float(x.max()),over_1ms=int((x>1).sum()))


def latency(args):
    os.sched_setaffinity(0,{args.cpu});R.initialize()
    states=corpus(args.per_street,args.seed)
    output={}
    for name in args.names.split(','):
        fn=make_bot(name)
        for s in states[:min(200,len(states))]:check_action(s,fn(s))
        gc.collect();gc.disable()
        wall=[];cpu=[];by=defaultdict(list);json_wall=[];errors=[]
        for _ in range(args.repeats):
            for s in states:
                start=time.perf_counter_ns();cs=time.thread_time_ns()
                try:a=fn(s)
                except Exception as ex:errors.append(repr(ex));break
                used=time.thread_time_ns()-cs;elapsed=time.perf_counter_ns()-start
                check_action(s,a);wall.append(elapsed);cpu.append(used);by[s['street']].append(elapsed)
        # Separate JSON cost, after the core measurement.
        for s in states:
            raw=json.dumps(s);start=time.perf_counter_ns()
            json.dumps(fn(json.loads(raw)))
            json_wall.append(time.perf_counter_ns()-start)
        output[name]=dict(wall=timing_stats(wall),thread_cpu=timing_stats(cpu),
                          by_street={k:timing_stats(v) for k,v in by.items()},
                          json_roundtrip=timing_stats(json_wall),errors=errors)
        print(json.dumps({name:output[name]}),flush=True)
    return dict(config=vars(args),cpu_model='AMD EPYC 7B12',gc_enabled=False,
                scope='Warm complete state-dict to action-dict, one CPU core, one ORT thread; JSON measured separately; no IPC or process startup.',
                corpus='Legal engine-generated heads-up states, equal street counts, 5-200 BB stacks',results=output)


def game(a,b,seed,swap,hands):
    names=[a,b] if not swap else [b,a]
    funcs=[make_bot(n) for n in names]
    rngs=[random.Random(seed*101+i) for i in (0,1)]
    global_states=[random.Random(seed*13+(1-i)).getstate() for i in (0,1)]
    stacks=[10000,10000];log=[];counts=Counter()
    for h in range(hands):
        if min(stacks)==0:break
        e=PokerEngine(f'{seed}_{int(swap)}_{h}',['p0','p1'],h%2,dict(zip(('p0','p1'),stacks)),seed*1000003+h)
        s=e.start_hand();steps=0
        while s['type']=='action_request':
            seat=s['seat_to_act'];s['match_action_log']=log[-200:]
            R.HERO._RNG=rngs[seat];random.setstate(global_states[seat])
            action=funcs[seat](s);global_states[seat]=random.getstate()
            check_action(s,action);counts[names[seat]+'_'+action['action']]+=1
            s=e.apply_action(seat,action)
            log.append(dict(e.action_log[-1],hand_num=h,bot_id=f'p{seat}'))
            steps+=1;assert steps<300
        stacks=[p.stack for p in e.players]
        assert sum(stacks)==20000 and min(stacks)>=0
    return dict(score=stacks[int(swap)]/200,hands=h if min(stacks)==0 and h<hands-1 else h+1,actions=dict(counts))


def pair(job):
    a,b,seed,hands=job
    try:
        legs=[game(a,b,seed,s,hands) for s in (False,True)]
        return dict(a=a,b=b,seed=seed,legs=legs,score=sum(g['score'] for g in legs)/2)
    except Exception:
        import traceback
        return dict(a=a,b=b,seed=seed,error=traceback.format_exc())


def arena(args):
    names=args.names.split(',')
    jobs=[(a,b,args.seed+i,args.hands) for a,b in itertools.combinations(names,2) for i in range(args.pairs)]
    raw=HERE/args.output.replace('.json','.jsonl')
    rows=[];start=time.perf_counter();last=start
    with raw.open('x') as f,ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=R.initialize) as pool:
        for row in pool.map(pair,jobs,chunksize=8):
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if 'error' in row:print(json.dumps(row),flush=True)
            if time.perf_counter()-last>20:
                print(json.dumps(dict(completed=len(rows),total=len(jobs),seconds=time.perf_counter()-start)),flush=True);last=time.perf_counter()
    results={}
    for a,b in itertools.combinations(names,2):
        rr=[r for r in rows if r['a']==a and r['b']==b];good=[r for r in rr if 'error' not in r]
        results[a+' vs '+b]=dict(a=a,b=b,pairs=len(good),errors=len(rr)-len(good),
                                score_a=R.mean_ci([r['score'] for r in good]),
                                hands=sum(g['hands'] for r in good for g in r['legs']))
    return dict(config=vars(args),elapsed_seconds=time.perf_counter()-start,results=results,
                scoring='100BB each; fixed blinds 50/100; max 200 hands; score=100*final chip share; paired decks and swapped seats; no timeout penalties')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['latency','arena']);p.add_argument('--names',required=True)
    p.add_argument('--per-street',type=int,default=200);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--cpu',type=int,default=2);p.add_argument('--seed',type=int,default=8100000)
    p.add_argument('--pairs',type=int,default=64);p.add_argument('--hands',type=int,default=200);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--output',required=True);args=p.parse_args()
    path=HERE/args.output
    if path.exists():raise FileExistsError(path)
    result=(latency if args.mode=='latency' else arena)(args)
    result['source_hashes']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),HERE/'engine_adapter.py',HERE/'run_benchmark.py']+[R.PATHS[n] for n in args.names.split(',') if n in R.PATHS]}
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
