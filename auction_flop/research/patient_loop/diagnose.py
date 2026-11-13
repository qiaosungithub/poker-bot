"""Collect public-state decisions in a fresh fixed-field diagnostic sample."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from auction_flop.benchmark import run
from poker_agent_protocol.game import state_to_dict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',required=True)
    parser.add_argument('--tables',type=int,default=64)
    args=parser.parse_args()
    spec=run.verify_lock();args.out.mkdir(parents=True,exist_ok=False)
    # Use retained frozen sources; only the candidate is new input.
    frozen=run.ROOT/'evidence/candidate-patient_value_v1/sources'
    candidate=args.out.resolve()/'source'
    hashes=run.copy_source(args.candidate.resolve(),candidate)
    counts=Counter();examples={};actions=Counter();rows=[]
    def observe(state, action):
        phase='auction' if state.phase=='auction' else 'postflop' if state.board else 'preflop'
        actions[f'{phase}:{action.kind}']+=1
        if phase=='auction': return
        me=state.seats[state.seat]
        bb=max(1,state.blinds.big_blind)
        tags=[]
        if phase=='preflop':
            if state.players_remaining<=3:
                counts['short_handed_preflop']+=1
                if action.kind=='fold': counts['short_handed_preflop_folds']+=1
                if state.to_call and state.can('call'):
                    counts['short_handed_facing_price']+=1
                    if action.kind=='fold': counts['short_handed_facing_price_folds']+=1
            if action.kind=='fold' and (me.stack+me.bet)<=12*bb:
                tags.append('short_stack_preflop_fold')
            cost=min(me.stack,state.to_call)
            odds=cost/max(1,state.pot+cost)
            if action.kind=='fold' and odds<=.15: tags.append('cheap_preflop_fold')
        else:
            function=agent_class.act.__globals__['postflop']
            g=function.__globals__
            category,genuine,top=g['made_hand'](state.cards,state.board)
            price=g['eligible_call'](state)
            if action.kind=='fold' and (genuine or top):
                tags.append('made_value_fold')
            if action.kind=='fold' and price.threshold<=.15:
                tags.append('cheap_postflop_fold')
            if state.can('check') and genuine:
                counts['private_two_pair_plus_free']+=1
                counts['private_two_pair_plus_free_'+action.kind]+=1
        for tag in tags:
            counts[tag]+=1
            if len(examples.setdefault(tag,[]))<40:
                examples[tag].append({'action':{'kind':action.kind,'amount':action.amount},
                                      'state':state_to_dict(state)})
    for index in range(args.tables):
        panel=index%4
        entries=[{'name':'candidate','snapshot':str(candidate)}]
        entries += [run.entry(f'opponent_{j}',p,frozen) for j,p in enumerate(spec['evaluation']['panels'][panel])]
        config={'entries':entries,'hand_cap':250}
        run.arena.initialize(config)
        config,classes=run.arena._WORKER
        agent_class=classes['candidate']
        class Observed:
            def __init__(self): self.bot=agent_class()
            def act(self,state,memory):
                action,new_memory=self.bot.act(state,memory)
                observe(state,action)
                return action,new_memory
        classes['candidate']=Observed
        names=[e['name'] for e in entries]
        rotation=(index//4)%8
        names=names[rotation:]+names[:rotation]
        result=run.arena.run_table(f'{args.seed}:{index//32}:{panel}',names)
        assert not any(e['failures'] for e in result['entries'])
        rows.append(result)
    with gzip.open(args.out/'examples.json.gz','wt') as f: json.dump(examples,f)
    with gzip.open(args.out/'tables.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result={'tables':args.tables,'seed':args.seed,'started_scope':'fresh diagnostic, not strength score',
            'source_hashes':hashes,'counts':dict(counts),'actions':dict(actions),
            'recorded':datetime.now(timezone.utc).isoformat()}
    run.write_json(args.out/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
