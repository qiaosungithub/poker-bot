"""Six existing policies at one table; PokerKit owns all poker rules.

100 BB per bot, fixed 50/100 blinds, max 200 hands. Payouts [100,60,40,25,10,0].
Eliminations lock in payouts; survivors receive exact ICM of remaining payouts.
An independent block contains six cyclic seat rotations of a random roster.
"""
import low_latency_compare as C
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gc
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
import warnings
import numpy as np
from pokerkit import Automation,NoLimitTexasHoldem

HERE=Path(__file__).resolve().parent
NAMES=['fullhouse','cfr_lite64','hydra32','solver_like','maniac','station']
PAYOUTS=[100.,60.,40.,25.,10.,0.]
AUTOMATIONS=tuple(a for a in Automation if a.name not in
                 ('HOLE_DEALING','BOARD_DEALING','RUNOUT_COUNT_SELECTION','CARD_BURNING'))


class Hand:
    def __init__(self,ids,stacks,dealer,seed,hand_id):
        self.ids=ids;self.n=len(ids);self.start=list(stacks);self.dealer=dealer;self.id=hand_id
        # PokerKit: position 0 is SB for 3+, BB for heads-up.
        self.order=[(dealer+j)%self.n for j in (range(1,self.n+1) if self.n>2 else (1,0))]
        self.inverse={seat:i for i,seat in enumerate(self.order)}
        self.k=NoLimitTexasHoldem.create_state(AUTOMATIONS,True,0,(50,100),100,
                                             [stacks[s] for s in self.order],self.n)
        deck=[r+s for r in '23456789TJQKA' for s in 'shdc'];random.Random(seed).shuffle(deck)
        self.holes=[deck[2*i:2*i+2] for i in range(self.n)]
        self.board_deck=deck[2*self.n:2*self.n+5];self.board=[]
        for i,seat in enumerate(self.order):self.k.deal_hole(''.join(self.holes[seat]),i)
        sb=dealer if self.n==2 else (dealer+1)%self.n;bb=(dealer+1)%self.n if self.n==2 else (dealer+2)%self.n
        self.log=[dict(seat=sb,action='small_blind',amount=min(50,stacks[sb])),
                  dict(seat=bb,action='big_blind',amount=min(100,stacks[bb]))]
        self.normalizations=Counter();self.advance()

    def advance(self):
        while self.k.can_burn_card() or self.k.can_deal_board():
            if self.k.can_burn_card():self.k.burn_card('??')
            else:
                count=self.k.board_dealing_count
                cards=self.board_deck[len(self.board):len(self.board)+count]
                self.k.deal_board(''.join(cards));self.board.extend(cards)

    def state(self):
        k=self.k
        if not k.status:return None
        ki=k.actor_index;assert ki is not None
        seat=self.order[ki];players=[]
        for s in range(self.n):
            j=self.inverse[s];folded=not k.statuses[j];allin=k.statuses[j] and k.stacks[j]==0
            players.append(dict(seat=s,bot_id=self.ids[s],stack=k.stacks[j],
                                state='folded' if folded else 'all_in' if allin else 'active',
                                is_folded=folded,is_all_in=allin,bet_this_street=k.bets[j],hole_cards=None))
        can_raise=k.can_complete_bet_or_raise_to()
        return dict(type='action_request',hand_id=self.id,street=C.STREETS[k.street_index],dealer=self.dealer,
                    seat_to_act=seat,pot=sum(self.start)-sum(k.stacks),community_cards=list(self.board),
                    current_bet=max(k.bets),min_raise_to=k.min_completion_betting_or_raising_to_amount if can_raise else 0,
                    can_raise=can_raise,amount_owed=max(k.bets)-k.bets[ki],can_check=k.checking_or_calling_amount==0,
                    your_cards=self.holes[seat],your_stack=k.stacks[ki],your_bet_this_street=k.bets[ki],
                    players=players,action_log=list(self.log))

    def act(self,raw):
        k=self.k;ki=k.actor_index;seat=self.order[ki];name=self.ids[seat]
        action=raw['action'];assert action in ('fold','check','call','raise','all_in'),raw
        if action=='fold' and k.can_fold():
            k.fold();entry=dict(seat=seat,action='fold',amount=0)
        elif action in ('fold','check','call'):
            if action=='fold':self.normalizations[name+'_free_fold_to_check']+=1
            paid=k.checking_or_calling_amount
            if action=='check' and paid:self.normalizations[name+'_illegal_check_to_call']+=1
            k.check_or_call();entry=dict(seat=seat,action='call' if paid else 'check',amount=paid)
        else:
            amount=k.bets[ki]+k.stacks[ki] if action=='all_in' else int(raw['amount'])
            if amount<=max(k.bets) or not k.can_complete_bet_or_raise_to():
                if amount>max(k.bets):self.normalizations[name+'_raise_not_reopened_to_call']+=1
                paid=k.checking_or_calling_amount;k.check_or_call()
                entry=dict(seat=seat,action='call' if paid else 'check',amount=paid)
            else:
                low=k.min_completion_betting_or_raising_to_amount;high=k.max_completion_betting_or_raising_to_amount
                total=max(low,min(high,amount))
                if total!=amount:self.normalizations[name+'_raise_clamped']+=1
                allin=total==k.bets[ki]+k.stacks[ki]
                k.complete_bet_or_raise_to(total)
                entry=dict(seat=seat,action='all_in' if allin else 'raise',amount=total)
        self.log.append(entry);self.advance()
        return entry

    def final_stacks(self):
        assert not self.k.status
        stacks=[self.k.stacks[self.inverse[s]] for s in range(self.n)]
        assert sum(stacks)==sum(self.start) and min(stacks)>=0
        return stacks


def icm(stacks,payouts):
    assert len(stacks)==len(payouts) and all(s>0 for s in stacks)
    n=len(stacks)
    @lru_cache(None)
    def solve(mask):
        out=[0.]*n;rank=n-mask.bit_count();total=sum(stacks[i] for i in range(n) if mask>>i&1)
        for i in range(n):
            if not mask>>i&1:continue
            prob=stacks[i]/total;out[i]+=prob*payouts[rank]
            if mask!=(1<<i):
                following=solve(mask^(1<<i))
                for j in range(n):out[j]+=prob*following[j]
        return tuple(out)
    return solve((1<<n)-1)


def elimination_scores(alive,before,final):
    eliminated=[i for i,x in enumerate(final) if x==0]
    groups={size:[i for i in eliminated if before[i]==size] for size in sorted({before[i] for i in eliminated})}
    bottom=len(alive);out={}
    for size,group in groups.items():
        prize=sum(PAYOUTS[bottom-len(group):bottom])/len(group)
        for i in group:out[alive[i]]=prize
        bottom-=len(group)
    return out


def match(roster,seed,hands=200,collect=None,mirror=False):
    funcs={name:C.make_bot('fullhouse' if mirror else name) for name in roster}
    rngs={name:random.Random(seed*101+seat) for seat,name in enumerate(roster)}
    globals_rng={name:random.Random(seed*13+seat).getstate() for seat,name in enumerate(roster)}
    stacks={name:10000 for name in roster};locked={};log=[];dealer_physical=0;corrections=Counter();decisions=Counter()
    completed=0
    for h in range(hands):
        alive=[name for name in roster if stacks[name]>0]
        if len(alive)==1:break
        while stacks[roster[dealer_physical]]==0:dealer_physical=(dealer_physical+1)%6
        dealer=alive.index(roster[dealer_physical]);before=[stacks[name] for name in alive]
        hand=Hand(alive,before,dealer,seed*1000003+h,f'multi_{seed}_{h}')
        s=hand.state();steps=0
        while s is not None:
            name=alive[s['seat_to_act']];s['match_action_log']=log[-200:]
            if collect is not None:collect(s)
            C.R.HERO._RNG=rngs[name];random.setstate(globals_rng[name])
            a=funcs[name](s);globals_rng[name]=random.getstate();decisions[name]+=1
            entry=hand.act(a);log.append(dict(entry,bot_id=name,hand_num=h))
            s=hand.state();steps+=1;assert steps<400
        final=hand.final_stacks();corrections.update(hand.normalizations)
        for name,x in zip(alive,final):stacks[name]=x
        # Larger stack at hand start wins the tie for simultaneous elimination.
        locked.update(elimination_scores(alive,before,final))
        assert sum(stacks.values())==60000
        completed+=1;dealer_physical=(dealer_physical+1)%6
    alive=[name for name in roster if stacks[name]>0]
    scores=dict(locked)
    scores.update(zip(alive,icm([stacks[name] for name in alive],PAYOUTS[:len(alive)])))
    assert abs(sum(scores.values())-sum(PAYOUTS))<1e-8
    return dict(scores=scores,stacks=stacks,hands=completed,survivors=len(alive),normalizations=dict(corrections),decisions=dict(decisions))


def block(job):
    seed,hands=job;roster=list(NAMES);random.Random(seed).shuffle(roster)
    try:
        legs=[match(roster[r:]+roster[:r],seed,hands) for r in range(6)]
        return dict(seed=seed,roster=roster,legs=legs,scores={name:sum(x['scores'][name] for x in legs)/6 for name in NAMES})
    except Exception:
        import traceback
        return dict(seed=seed,error=traceback.format_exc())


def latency(args):
    os.sched_setaffinity(0,{args.cpu});C.R.initialize()
    rng=random.Random(args.seed);buckets={s:[] for s in C.STREETS};h=0
    while min(map(len,buckets.values()))<args.per_street:
        hand=Hand(NAMES,[rng.choice((500,1000,2000,5000,10000,20000)) for _ in NAMES],h%6,args.seed+h,f'latency6_{h}')
        s=hand.state()
        while s is not None:
            s['match_action_log']=[dict(e,bot_id=NAMES[e['seat']],hand_num=h) for e in hand.log]
            if len(buckets[s['street']])<args.per_street:buckets[s['street']].append(s)
            x=rng.random()
            if h%3==0 or x<.65:a={'action':'check' if s['can_check'] else 'call'}
            elif x<.80:a={'action':'fold'}
            else:a={'action':'raise','amount':s['current_bet']+max(100,int((s['pot']+s['amount_owed'])*rng.choice((.5,1,2))))}
            hand.act(a);s=hand.state()
        h+=1
    states=[s for b in buckets.values() for s in b];rng.shuffle(states)
    with gzip.open(HERE/'multiway_latency_states.jsonl.gz','xt') as f:
        for s in states:f.write(json.dumps(s)+'\n')
    out={}
    for name in NAMES:
        fn=C.make_bot(name)
        for s in states[:200]:fn(s)
        gc.collect();gc.disable();wall=[];cpu=[];by=defaultdict(list)
        for _ in range(args.repeats):
            for s in states:
                t=time.perf_counter_ns();c=time.thread_time_ns();a=fn(s)
                used=time.thread_time_ns()-c;dt=time.perf_counter_ns()-t
                assert a['action'] in ('fold','check','call','raise','all_in')
                wall.append(dt);cpu.append(used);by[s['street']].append(dt)
        out[name]=dict(wall=C.timing_stats(wall),thread_cpu=C.timing_stats(cpu),by_street={s:C.timing_stats(v) for s,v in by.items()})
        print(json.dumps({name:out[name]}),flush=True)
    return dict(config=vars(args),results=out,states=len(states),scope='Six-seat legal PokerKit states; equal street counts; 5-200BB stacks; warm full decision function, GC disabled; no IPC or JSON.')


def arena(args):
    rows=[];start=time.perf_counter();last=start
    with (HERE/args.output.replace('.json','.jsonl')).open('x') as raw,ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=C.R.initialize) as pool:
        for row in pool.map(block,[(args.seed+i,args.hands) for i in range(args.blocks)]):
            rows.append(row);raw.write(json.dumps(row)+'\n');raw.flush()
            if 'error' in row:print(json.dumps(row),flush=True)
            if time.perf_counter()-last>20:
                print(json.dumps(dict(blocks=len(rows),total=args.blocks,seconds=time.perf_counter()-start)),flush=True);last=time.perf_counter()
    good=[r for r in rows if 'error' not in r]
    return dict(config=vars(args),elapsed_seconds=time.perf_counter()-start,errors=len(rows)-len(good),
                matches=len(good)*6,hands=sum(g['hands'] for r in good for g in r['legs']),
                results={name:C.R.mean_ci([r['scores'][name] for r in good]) for name in NAMES})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['latency','arena']);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=9100000);p.add_argument('--blocks',type=int,default=1024)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--hands',type=int,default=200)
    p.add_argument('--per-street',type=int,default=2500);p.add_argument('--repeats',type=int,default=3);p.add_argument('--cpu',type=int,default=2)
    args=p.parse_args();path=HERE/args.output
    if path.exists():raise FileExistsError(path)
    result=(latency if args.mode=='latency' else arena)(args)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
