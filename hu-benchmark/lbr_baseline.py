"""White-box local best response to Fullhouse, following Lisy & Bowling (2017).

Exact enumeration of legal opponent hole-card combinations and Bayesian updates.
Monte Carlo (16 runouts per combo) replaces exhaustive rollout equity; candidate
bets are 0.5, 1, 2 pots and all-in. This is our implementation, not a released
champion. It knows the fixed policy, NEVER the opponent's actual cards or deck.
Source: https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf
"""
import hashlib
import itertools
from pathlib import Path
import random

import eval7
import numpy as np
import onnxruntime as ort
from bot import bot
from bot import features as F

HERE = Path(__file__).resolve().parent
CARDS = [r+s for s in 'cdhs' for r in '23456789TJQKA']
EVAL_CARDS = [eval7.Card(c) for c in CARDS]
CARD_ID = {c:i for i,c in enumerate(CARDS)}
SESSION = None


def export_batched():
    import onnx
    from onnx import helper, numpy_helper
    src = HERE.parent/'latency-survey/fullhouse_fp32.onnx'
    dst = HERE/'fullhouse_batch.onnx'
    model = onnx.load(src)
    for value, width in zip(model.graph.input, [51, 5]):
        value.type.tensor_type.shape.CopyFrom(helper.make_tensor_value_info('x', 1, ['batch',width]).type.tensor_type.shape)
    model.graph.output[0].type.tensor_type.shape.CopyFrom(
        helper.make_tensor_value_info('p', 1, ['batch',5]).type.tensor_type.shape)
    model.graph.initializer.append(numpy_helper.from_array(np.array([-1],dtype=np.int64),'last_axis'))
    for node in model.graph.node:
        if node.op_type == 'ReduceSum':
            node.input.append('last_axis')
    onnx.checker.check_model(model); onnx.save(model, dst)
    return dst


def get_session():
    global SESSION
    if SESSION is None:
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        SESSION=ort.InferenceSession(str(HERE/'fullhouse_batch.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
    return SESSION


class LocalBestResponse:
    def __init__(self, risk_power=1.0):
        self.session=get_session()
        # Optional local adaptation for finite-bankroll matches: concave utility
        # avoids marginal stack-off gambles. This is NOT part of original LBR.
        self.risk_power=risk_power
        self.stats={'updates':0, 'zero_posterior':0, 'decisions':0}

    def new_hand(self, seat, own_cards):
        self.seat=seat;self.hole=list(own_cards)
        available=[i for i,c in enumerate(CARDS) if c not in self.hole]
        self.combos=np.array(list(itertools.combinations(available,2)),dtype=int)
        self.weights=np.ones(len(self.combos),dtype=float)/len(self.combos)
        self.card_cache={};self.equity_cache={}

    def condition_board(self, board):
        for c in board:
            self.weights[(self.combos==CARD_ID[c]).any(axis=1)]=0
        total=self.weights.sum()
        if total <= 1e-20:
            raise RuntimeError('Posterior mass vanished on a public board')
        self.weights/=total

    def card_features(self, board):
        key=tuple(board)
        if key in self.card_cache:
            return self.card_cache[key]
        bi=[CARD_ID[c] for c in board]
        street={0:0,3:1,4:2,5:3}[len(board)]
        out=np.zeros((len(self.combos),17),dtype=np.float32)
        paired,suited,offsuit=F.load_preflop_equity()
        bp,bm,bw,sp=F.board_texture(bi)
        for i,(a,b) in enumerate(self.combos.tolist()):
            if a in bi or b in bi:continue
            ra,rb=a%13,b%13;sa,sb=a//13,b//13;hi,lo=max(ra,rb),min(ra,rb)
            out[i,0]=paired[ra] if ra==rb else suited[hi,lo] if sa==sb else offsuit[hi,lo]
            if board:out[i,1]=F._evaluate_hand_strength(a,b,bi)
            fd,oe,gs,outs,nfd=F.detect_draws([a,b],bi,street)
            out[i,2:12]=[fd,oe,gs,outs/20,nfd,hi/12,lo/12,sa==sb,ra==rb,(hi-lo-1)/12 if hi!=lo else 0]
            if board:out[i,12:17]=[bp,bm,sum(c%13>hi for c in bi)/5,bw,sp]
        self.card_cache[key]=out
        return out

    def probabilities(self, public_state):
        # Input must contain no actual opponent private cards.
        s=dict(public_state)
        idx=int(np.argmax(self.weights)); a,b=self.combos[idx]
        s['your_cards']=[CARDS[a],CARDS[b]]
        base,legal=F.encode_state_dict(s)
        x=np.broadcast_to(base,(len(self.combos),51)).copy()
        x[:,:17]=self.card_features(s['community_cards'])
        mask=np.broadcast_to(legal.astype(np.float32),(len(self.combos),5)).copy()
        probabilities=self.session.run(None,{'features':x,'legal':mask})[0]
        probabilities/=probabilities.sum(axis=1,keepdims=True)
        actions=[bot._to_engine_action(i,s) for i in range(5)]
        # The deterministic short-stack preflop overlay takes precedence.
        if s['street']=='preflop' and min(p['stack'] for p in s['players'])<=1200:
            for i,(a,b) in enumerate(self.combos.tolist()):
                s['your_cards']=[CARDS[a],CARDS[b]]
                pf=bot._push_fold(s)
                if pf is not None:
                    probabilities[i]=0
                    slot=next(j for j,act in enumerate(actions) if act==pf)
                    probabilities[i,slot]=1
        return probabilities,actions

    def observe(self, public_state, action):
        assert 'your_cards' not in public_state
        if action['action']=='fold':return
        self.condition_board(public_state['community_cards'])
        probs,actions=self.probabilities(public_state)
        slots=[j for j,a in enumerate(actions) if a==action]
        if not slots:raise RuntimeError(f'Observed action not in fixed policy menu: {action}, {actions}')
        new=self.weights*probs[:,slots].sum(axis=1)
        if new.sum()<=1e-20:
            self.stats['zero_posterior']+=1
            raise RuntimeError('Observed action has zero posterior probability')
        self.weights=new/new.sum();self.stats['updates']+=1

    def equities(self, board):
        key=tuple(board)
        if key in self.equity_cache:return self.equity_cache[key]
        own=[eval7.Card(c) for c in self.hole];bc=[eval7.Card(c) for c in board]
        known={CARD_ID[c] for c in self.hole+board}
        available=[i for i in range(52) if i not in known]
        seed=int.from_bytes(hashlib.sha256(repr((self.hole,board)).encode()).digest()[:8],'big')
        rng=random.Random(seed);nneed=5-len(board)
        eq=np.zeros(len(self.combos))
        for i,(a,b) in enumerate(self.combos.tolist()):
            if a in known or b in known:continue
            deck=[c for c in available if c!=a and c!=b]
            opp=[EVAL_CARDS[a],EVAL_CARDS[b]]
            n=16 if nneed else 1
            for _ in range(n):
                runout=bc+[EVAL_CARDS[c] for c in rng.sample(deck,nneed)]
                h=eval7.evaluate(own+runout);v=eval7.evaluate(opp+runout)
                eq[i]+=(h>v)+.5*(h==v)
            eq[i]/=n
        self.equity_cache[key]=eq
        return eq

    def __call__(self,s):
        self.stats['decisions']+=1
        self.condition_board(s['community_cards'])
        eq=self.equities(s['community_cards']);w=self.weights
        pot=s['pot'];owed=s['amount_owed'];cost=min(owed,s['your_stack'])
        effective_pot=pot-max(0,owed-cost)
        pwin=float(w@eq)
        stack=s['your_stack']
        def gain(delta):
            return max(0.,stack+delta)**self.risk_power-stack**self.risk_power
        best_value=pwin*gain(effective_pot)+(1-pwin)*gain(-cost)
        best={'action':'check' if s['can_check'] else 'call'}
        if best_value<0:best_value=0.;best={'action':'fold'}
        other=next(p for p in s['players'] if p['seat']!=self.seat)
        max_to=min(s['your_bet_this_street']+s['your_stack'],other['bet_this_street']+other['stack'])
        if other['stack']==0 or max_to<=s['current_bet']:return best
        fractions=(.5,1,2) if self.risk_power==1 else (.125,.25,.5,1,2)
        totals={min(max_to,max(s['min_raise_to'],s['current_bet']+int(frac*(pot+owed)))) for frac in fractions}
        totals.add(max_to)
        for total in sorted(totals):
            if total<s['min_raise_to'] and total<s['your_bet_this_street']+s['your_stack']:continue
            paid=total-s['your_bet_this_street'];response_cost=total-other['bet_this_street']
            action={'action':'all_in'} if paid==s['your_stack'] else {'action':'raise','amount':total}
            players=[dict(p) for p in s['players']]
            players[self.seat]['stack']-=paid;players[self.seat]['bet_this_street']=total
            players[self.seat]['is_all_in']=players[self.seat]['stack']==0
            players[self.seat]['state']='all_in' if players[self.seat]['is_all_in'] else 'active'
            public=dict(s,seat_to_act=other['seat'],players=players,pot=pot+paid,
                        current_bet=total,min_raise_to=total+max(total-s['current_bet'],s['min_raise_to']-s['current_bet']),
                        amount_owed=response_cost,can_check=False,your_stack=other['stack'],
                        your_bet_this_street=other['bet_this_street'],
                        action_log=s['action_log']+[dict(seat=self.seat,action=action['action'],amount=total)])
            public.pop('your_cards',None)
            probs,aa=self.probabilities(public)
            fold=probs[:,[i for i,a in enumerate(aa) if a['action']=='fold']].sum(axis=1)
            value=float(w@(fold*gain(pot)+(1-fold)*(eq*gain(pot+response_cost)+(1-eq)*gain(-paid))))
            if value>best_value:best_value=value;best=action
        return best


if __name__=='__main__':
    print(export_batched())
