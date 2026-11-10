"""Bounded public-range, action-conditioned postflop expected-chip policy.

Uniform complete deals are importance weighted by public action likelihoods.
Candidate action values integrate independent opponent continue probabilities;
flop/turn values discount the contested share for future betting. This is a
transparent small-model approximation, not a claim of equilibrium solution.
"""
import math
import random
import eval7
from poker_agent_sdk import Action
from pot_accounting import eligible_call
from hand_contribution import shared_nut_action
from classic_policy import CARDS, DECK, RANKS, EQUITY, class_key

SAMPLES = 20
RANK = {c: RANKS.index(c[0]) for c in DECK}
SUIT = {c: 'shdc'.index(c[1]) for c in DECK}
WINDOWS = tuple(31 << low for low in range(9)) + ((1 << 12) | 15,)
STRAIGHT_DRAW = tuple(any((mask & window).bit_count() == 4 for window in WINDOWS) for mask in range(8192))
PREFLOP = {(a,b): EQUITY[class_key((a,b))] for a in DECK for b in DECK if a != b}

def clip(x, low, high):
    return low if x < low else high if x > high else x

def chip_utility(state, action, estimate):
    """Swappable utility hook. Estimate contains chips EV and loss/gain branches."""
    return estimate['chip_ev']

def _seed(state):
    # No process RNG/global state, table identity or private opponent fields.
    seed = 0x5A17 ^ (state.hand_number * 1009 + state.seat * 7919 + len(state.history) * 104729)
    for c in state.cards + state.board + tuple(state.variant.get('auction_cards', ())):
        seed = ((seed * 16777619) ^ (RANKS.index(c[0]) * 4 + 'shdc'.index(c[1]))) & 0xffffffff
    return seed

def _texture(board):
    ranks = tuple(RANK[c] for c in board)
    suits = tuple(sum(SUIT[c] == suit for c in board) for suit in range(4))
    mask = sum(1 << r for r in set(ranks))
    board_value = eval7.evaluate([CARDS[c] for c in board]) if len(board) == 5 else -1
    return ranks, suits, mask, board_value

def _strength(hole, board, texture, value):
    """Continuous range likelihood feature; never directly maps to an action."""
    br, bs, boardmask, board_value = texture
    a,b = RANK[hole[0]],RANK[hole[1]]
    category = value >> 24
    private_pair = a == b
    matching = max(a if a in br else -1,b if b in br else -1)
    high = max(a,b)
    if category == 0:
        made = .12 + .16 * high / 12
    elif category == 1:
        if private_pair:
            made = .48 + .23 * sum(a > r for r in br) / len(br)
        elif matching >= 0:
            made = .47 + .22 * sum(matching >= r for r in br) / len(br) + .035 * high / 12
        else:
            made = .18 + .15 * high / 12
    elif category == 2:
        made = .77 if matching >= 0 or private_pair else .28 + .12*high/12
    elif category == 3:
        made = .88 if br.count(matching) == 2 or private_pair else .33 + .18*high/12
    else:
        made = (.91, .94, .975, .99, .998)[min(4, category-4)]
    if value == board_value:
        made = min(made, .38)
    if len(board) < 5:
        sa,sb=SUIT[hole[0]],SUIT[hole[1]]
        if bs[sa]+1+(sa==sb)==4 or bs[sb]+1+(sa==sb)==4: made += .18
        if STRAIGHT_DRAW[boardmask | (1<<a) | (1<<b)] and (a not in br or b not in br): made += .11
    return min(made,.995)

def _public_context(state, opponents, prepared):
    """Only actual actions and their contemporaneous board/pot enter likelihoods."""
    n=len(state.seats); street=0; pot=0; bets=[0]*n
    pre=[0]*n; evidence=[[] for _ in state.seats]; last=[None]*n
    current_raises=0
    for event in state.history[-128:]:
        kind=event.kind; seat=event.seat
        if kind=='board':
            street += len(event.cards); bets=[0]*n; current_raises=0
        elif kind in ('ante','blind','call','raise') and seat is not None:
            delta=max(0,event.amount-bets[seat]) if kind=='raise' else event.amount
            fraction=delta/max(1,pot)
            pot+=delta
            if kind!='ante': bets[seat]+=delta
            if street==0 and kind in ('call','raise'):
                pre[seat]=max(pre[seat], 2 if kind=='raise' else 1)
            elif street>=3 and kind in ('call','raise'):
                # Retain last action on each street plus the number of raises.
                if evidence[seat] and evidence[seat][-1][0]==street:
                    evidence[seat][-1]=(street,kind,min(2.,fraction))
                else: evidence[seat].append((street,kind,min(2.,fraction)))
            if kind=='raise': current_raises+=1
            last[seat]=kind
        elif kind=='check' and seat is not None and street>=3:
            if not evidence[seat] or evidence[seat][-1][0]!=street:
                evidence[seat].append((street,kind,0.))
            last[seat]=kind
    rows=prepared.get('classic',{}).get('rows',()) if prepared else ()
    profiles=[]
    for seat in opponents:
        row=rows[seat.seat] if len(rows)==n else (0,0,0,0)
        raises,calls,folds,checks=row; total=sum(row)
        # Shrink to a broad population prior; raw counts are action frequencies.
        aggression=(raises+4)/(total+24)
        looseness=(calls+raises+10)/(raises+calls+folds+20)
        order=(seat.seat-state.button-1)%n/max(1,n-1)
        profiles.append((pre[seat.seat], evidence[seat.seat][-3:], aggression, looseness, order, last[seat.seat]))
    clockwise=lambda s:(s-state.button-1)%n
    after=sum(clockwise(s.seat)>clockwise(state.seat) and s.status=='active' for s in opponents)
    return profiles,after,current_raises

def _likelihood(hole, strengths, profile, exponent):
    pre,evidence,aggression,looseness,position,last=profile
    prior=PREFLOP[hole]
    threshold=.57-.035*position + (.025 if pre==2 else -.015)
    if pre:
        preweight=clip(1 + 4.8*(prior-threshold), .24, 2.0)
    else:
        preweight=1.
    weight=preweight
    for street,kind,fraction in evidence:
        strength=strengths[street]
        if kind=='raise':
            # A large bet is polarized: strong value plus a bounded bluff floor.
            floor=.20+.55*aggression
            threshold=.45+.12*min(1.5,fraction)
            likelihood=floor+2.1*clip((strength-threshold+.15)/.55,0.,1.)
        elif kind=='call':
            likelihood=.30+1.7*clip((strength-.30-.10*min(1.,fraction))/.55,0.,1.)
        else:
            likelihood=1.14-.30*strength
        weight *= clip(likelihood,.25,2.3)
    # Mixture with unconditioned prior + bounded weights avoid brittle inference.
    weight=clip(weight,.16,4.5)
    return .28 + .72*weight**exponent

def _worlds(state, opponents, profiles):
    known=set(state.cards)|set(state.board)|set(state.variant.get('auction_cards',()))
    for seat in state.seats:
        known.update(seat.shown_cards)
    deck=[c for c in DECK if c not in known]
    unknown=sum(len(s.shown_cards)!=2 for s in opponents)
    missing=5-len(state.board)
    rng=random.Random(_seed(state))
    textures={street:_texture(state.board[:street]) for street in range(3,len(state.board)+1)}
    current_cards=[CARDS[c] for c in state.board]
    hero_cards=[CARDS[c] for c in state.cards]
    samples=SAMPLES if len(opponents)==1 else 14 if len(opponents)==2 else 10
    if any(s.status=='all_in' for s in opponents) and len(opponents)>1: samples=min(samples,10)
    exponent=1./max(1.,math.sqrt(len(opponents)))
    board_by_street={street:[CARDS[c] for c in state.board[:street]] for street in textures}
    worlds=[]
    for _ in range(samples):
        drawn=rng.sample(deck,missing+2*unknown)
        full=current_cards+[CARDS[c] for c in drawn[:missing]]
        hero=eval7.evaluate(hero_cards+full); offset=missing; scores=[]; strengths=[]; weight=1.
        for seat,profile in zip(opponents,profiles):
            if len(seat.shown_cards)==2: hole=seat.shown_cards
            else:
                hole=tuple(drawn[offset:offset+2]);offset+=2
            hc=[CARDS[c] for c in hole]
            score=eval7.evaluate(hc+full)
            current=score if not missing else eval7.evaluate(hc+current_cards)
            observed_strength={len(state.board):_strength(hole,state.board,textures[len(state.board)],current)}
            for street,_,_ in profile[1]:
                if street not in observed_strength:
                    value=eval7.evaluate(hc+board_by_street[street])
                    observed_strength[street]=_strength(hole,state.board[:street],textures[street],value)
            strengths.append(observed_strength[len(state.board)])
            scores.append(score)
            if len(seat.shown_cards)!=2:
                weight*=_likelihood(hole,observed_strength,profile,exponent)
        worlds.append((hero,tuple(scores),tuple(strengths),clip(weight,.12,7.)))
    total=sum(w[3] for w in worlds)
    ess=total*total/sum(w[3]*w[3] for w in worlds)
    if ess < max(8., .45*samples):
        # Degenerate importance samples explicitly shrink back toward uniform.
        worlds=[(h,s,q,.5+.5*w*samples/total) for h,s,q,w in worlds]
    return worlds

def _continue(strength, added, pot, profile, raising, allin):
    if allin or added<=0: return 1.
    _,_,aggression,looseness,_,last=profile
    odds=added/max(1,pot+added)
    threshold=.30+.70*odds + (.07 if raising else 0.) + (.035 if last=='raise' else 0.)
    threshold -= .20*(looseness-.5)+.08*(aggression-.17)
    signal=clip(.5+2.5*(strength-threshold),.025,.985)
    # A 12% population-prior component resists pretending exact known responses.
    population=clip(.72-.65*odds-(.10 if raising else 0.),.25,.75)
    return .88*signal+.12*population

def _share(hero,scores,probabilities):
    """E[showdown share] and E[share*caller_i], exact for independent calls/ties."""
    if len(scores)==1:
        q=probabilities[0]
        if scores[0]<hero: return 1.,(q,)
        if scores[0]>hero: return 1-q,(0.,)
        return 1-.5*q,(.5*q,)
    stronger=1.; ties=[]
    for i,(score,q) in enumerate(zip(scores,probabilities)):
        if score>hero: stronger*=1-q
        elif score==hero: ties.append(i)
    if not ties:
        return stronger,tuple(stronger*q if score<hero else 0. for score,q in zip(scores,probabilities))
    # Enumerate only tied opponents, normally zero; public-board ties max 128.
    share=0.; contribution=[0.]*len(scores)
    for mask in range(1<<len(ties)):
        probability=stronger; count=0
        for j,i in enumerate(ties):
            called=bool(mask&(1<<j)); q=probabilities[i]
            probability*=q if called else 1-q;count+=called
        amount=probability/(1+count);share+=amount
        for j,i in enumerate(ties):
            if mask&(1<<j): contribution[i]+=amount
    for i,(score,q) in enumerate(zip(scores,probabilities)):
        if score<hero: contribution[i]=share*q
    return share,tuple(contribution)

def _pot_layers(contributions, hero, added, opponents, opp_add):
    """Exact chip eligibility layers, including folded money and uncalled returns."""
    old=list(contributions); final=old[:];final[hero]+=added
    for seat,amount in zip(opponents,opp_add): final[seat.seat]+=amount
    cap=final[hero]
    if all(final[s.seat]>=cap for s in opponents): return None
    levels=sorted({0,cap}|{min(cap,x) for x in old+final})
    layers=[]
    for low,high in zip(levels,levels[1:]):
        if high<=low: continue
        # Hero's new money is certain; opponent's new money is conditional.
        base=sum(max(0,min(high,x)-low) for x in old)+max(0,min(high,final[hero])-max(low,old[hero]))
        ids=tuple(i for i,s in enumerate(opponents) if final[s.seat]>=high)
        extra=tuple(max(0,min(high,final[opponents[i].seat])-max(low,old[opponents[i].seat])) for i in ids)
        layers.append((base,ids,extra))
    return layers

def evaluate_actions(state, prepared=None):
    """Return public-only EV diagnostics, useful for a tournament utility adapter."""
    me=state.seats[state.seat]
    opponents=[s for s in state.seats if s.seat!=state.seat and s.status in ('active','all_in')]
    price=eligible_call(state); cost=price.call_cost
    profiles,after,raises=_public_context(state,opponents,prepared)
    worlds=_worlds(state,opponents,profiles)
    weight_sum=sum(w[3] for w in worlds)
    remain=5-len(state.board)
    actions=[]
    if state.can('fold'): actions.append(Action.fold())
    if state.can('check'): actions.append(Action.check())
    if state.can('call'): actions.append(Action.call())
    legal=state.can('raise')
    if legal:
        current=me.bet+state.to_call
        pot=max(state.blinds.big_blind,price.eligible_pot_after_call)
        # Raises use pot after calling; all targets are street totals.
        cap=min(legal.max_amount,max((s.bet+s.stack for s in opponents),default=legal.max_amount))
        targets={max(legal.min_amount,min(cap,current+round(pot*f))) for f in (.40,.75,1.15)}
        if cap<=current+1.6*pot: targets.add(max(legal.min_amount,cap))
        actions.extend(Action.raise_to(t) for t in sorted(targets) if legal.min_amount<=t<=legal.max_amount)
    estimates=[]
    for action in actions:
        if action.kind=='fold':
            estimates.append((action,{'chip_ev':0.,'risk':0.,'gain':0.,'win_probability':0.,'fold_probability':0.,'showdown_share':0.}));continue
        aggressive=action.kind=='raise'
        effective_target=min(action.amount,max((s.bet+s.stack for s in opponents),default=action.amount)) if aggressive else 0
        added=max(0,effective_target-me.bet) if aggressive else cost if action.kind=='call' else 0
        target=me.bet+added
        # Eligible existing money expands with hero's contribution cap.
        if price.exact:
            eligible=sum(min(x,price.contributions[state.seat]+added) for x in price.contributions)
        else: eligible=price.eligible_pot_before_call
        opp_add=[min(s.stack,max(0,target-s.bet)) for s in opponents]
        layers=_pot_layers(price.contributions,state.seat,added,opponents,opp_add) if price.exact else None
        ev=fold_mean=share_mean=gain_mean=0.
        active=sum(s.stack>amount for s,amount in zip(opponents,opp_add))
        future=remain if me.stack>added and active else 0
        loss_rate=(.025 if aggressive else .045)+.018*min(after,2)+.010*max(0,len(opponents)-1)
        realization=1-future*loss_rate
        for hero,scores,strengths,weight in worlds:
            probabilities=tuple(_continue(strength,amount,eligible+added,p,raises>0 or cost>0,s.status=='all_in')
                              for strength,amount,p,s in zip(strengths,opp_add,profiles,opponents))
            fold=math.prod(1-q for q in probabilities)
            share,contrib=_share(hero,scores,probabilities)
            if layers is None:
                gross=share*(eligible+added)+sum(amount*x for amount,x in zip(opp_add,contrib))
            else:
                gross=0.
                for base,ids,extra in layers:
                    part,part_contrib=_share(hero,tuple(scores[i] for i in ids),tuple(probabilities[i] for i in ids))
                    gross+=part*base+sum(amount*x for amount,x in zip(extra,part_contrib))
            # Current equity is not action EV: future bets deny realization,
            # especially OOP and when taking a passive line. All-ins and river
            # have exact showdown realization in this approximation.
            # All-fold reward has no future-street discount.
            contested=max(0.,gross-fold*(eligible+added))
            gross=fold*(eligible+added)+contested*realization
            # Continuing ranges can re-raise: bounded reserve for the portion of
            # a bet exposed to stronger holdings; no charge when already all-in.
            reraise=0.
            if aggressive and me.stack>added:
                pressure=sum(q*max(0.,strength-.79) for q,strength in zip(probabilities,strengths))
                reraise=min(.18,pressure*.25)*added*(1-share)
            ev+=weight*(gross-added-reraise)
            fold_mean+=weight*fold;share_mean+=weight*share
            gain_mean+=weight*max(0.,gross-added)
        estimate={'chip_ev':ev/weight_sum,'risk':float(added),'gain':gain_mean/weight_sum,
                  'win_probability':share_mean/weight_sum,'fold_probability':fold_mean/weight_sum,
                  'showdown_share':share_mean/weight_sum,'samples':len(worlds),
                  'effective_samples':weight_sum**2/sum(w[3]*w[3] for w in worlds),
                  'eligible_pot':eligible,'continuation_streets':remain}
        estimates.append((action,estimate))
    return estimates

def postflop(state, prepared=None, utility=chip_utility):
    forced=shared_nut_action(state)
    if forced is not None: return forced
    estimates=evaluate_actions(state,prepared)
    # Small EV differences favor the passive line, controlling sampling noise.
    best=None;score=-float('inf')
    for action,estimate in estimates:
        value=utility(state,action,estimate)
        if action.kind=='raise': value-=max(1.,.025*estimate['eligible_pot'])
        if value>score:
            best,score=action,value
    return best
