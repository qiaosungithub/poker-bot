"""Independent preflop alternatives for the released Boston five-stage bot.

Imports only frozen event-time variant helpers. Auction balances and private
future cards are intentionally unused. All other stage sources stay frozen.
"""
from poker_agent_sdk import Action
from distilled_prior import opening_threshold
from pot_accounting import eligible_call
from variant_agent import Agent as ControlAgent, preflop_quality, legal_raise, safe_pass


def choose_preflop_control(state, features):
    """Exact strongest event-time preflop control, including its edge cases."""
    return ControlAgent().act(state, b'')[0]


def _context(state):
    own = next(s for s in state.seats if s.seat == state.seat)
    opponents = tuple(s for s in state.seats
                      if s.seat != state.seat and s.status in ('active', 'all_in'))
    bb = max(1, state.blinds.big_blind)
    alive = [s.seat for s in state.seats if s.status != 'busted']
    clockwise = sorted(alive, key=lambda i:(i-state.button-1)%len(state.seats))
    small = state.button if len(alive)==2 else clockwise[0]
    big = clockwise[0] if len(alive)==2 else clockwise[1]
    position = clockwise.index(state.seat)/max(1,len(clockwise)-1)
    raises = 0
    for event in reversed(state.history):
        if event.kind in ('board','auction'):
            break
        if event.kind=='raise':
            raises += 1
    limpers = sum(e.kind=='call' for e in state.history) if raises==0 else 0
    own_total = own.stack+own.bet
    covered = max((s.stack+s.bet for s in opponents),default=own_total)
    effective_bb = min(own_total,covered)/bb
    return own,opponents,bb,small,big,position,raises,limpers,effective_bb


def choose_preflop_with(state, features, *, strength=1.0, late_open_discount=0.0):
    """strength=0 is the exact control; nonzero selects a declared alternative.

    late_open_discount is the sole difference between the two alternatives:
    0.0 for discipline and 0.035 for position pressure. It applies only when
    unopened/unlimped, in late position outside the blinds, and at least 3-handed.
    """
    if strength == 0:
        return choose_preflop_control(state,features)
    if state.phase!='betting' or state.board or len(state.cards)!=2:
        return choose_preflop_control(state,features)
    own,opponents,bb,small,big,position,raises,limpers,effective_bb = _context(state)
    quality = preflop_quality(state.cards)
    to_call = min(own.stack,state.to_call)
    own_bb = own.stack/bb
    # Position modifies opening requirements smoothly. Blinds do not inherit
    # button stealing thresholds; heads-up SB and BB are handled explicitly.
    open_threshold = 0.69-0.12*position
    if state.seat==small:
        open_threshold=0.63
    elif state.seat==big:
        open_threshold=0.64
    if state.players_remaining==2:
        open_threshold=0.50 if state.seat==small else 0.56
    elif state.players_remaining==3:
        open_threshold-=0.045
    open_threshold += min(0.03,0.01*limpers)
    late = position>=0.65 and state.seat not in (small,big)
    if raises==0 and limpers==0 and late and state.players_remaining>=3:
        open_threshold -= late_open_discount
    # Effective stacks determine commitment, but a short caller does not make
    # a deep multiway pot short when another covering opponent is still live.
    if effective_bb<=10 and quality>=open_threshold+0.015 and raises<2:
        target=min(own.stack+own.bet,max(bb,round(effective_bb*bb)))
        return legal_raise(state,target)
    if raises==0 and limpers==0 and effective_bb>10:
        open_threshold = opening_threshold(state, open_threshold, effective_bb)
    if raises==0 and quality>=open_threshold:
        return legal_raise(state,bb*(2.4+min(3,limpers)*0.8))
    if quality>=0.89 or (quality>=0.83 and raises==1 and state.players_remaining<=4):
        return legal_raise(state,max(bb*7,(own.bet+state.to_call)*2.7))
    if state.can('check'):
        return Action.check()
    required=0.56 if to_call<=bb else 0.65 if to_call<=2.5*bb else 0.73 if to_call<=6*bb else 0.81
    if raises>1:
        required+=0.04
    if state.players_remaining<=3:
        required-=0.055
    # This shared adjustment prices normal-chip calls only. Flop-chips are
    # neither money in the pot nor a reason to defend a marginal preflop hand.
    required += 0.025
    odds=eligible_call(state).threshold
    if odds<0.12:
        required-=0.025
    if to_call>own.stack*0.45 and own_bb>10:
        required=max(required,0.82)
    paid_places=sum(p>0 for p in state.config.payouts)
    shorter=sum(0<s.stack+s.bet<own.stack+own.bet for s in state.seats
                if s.seat!=state.seat and s.status!='busted')
    if state.players_remaining==paid_places+1 and shorter and to_call>own.stack*0.25:
        required+=0.035
    if quality>=required and state.can('call'):
        return Action.call()
    return safe_pass(state)


def choose_preflop(state, features):
    return choose_preflop_with(state,features)


def choose_preflop_tight(state, features):
    return choose_preflop_with(state,features,late_open_discount=0.0)


def choose_preflop_loose(state, features):
    return choose_preflop_with(state,features,late_open_discount=0.035)
