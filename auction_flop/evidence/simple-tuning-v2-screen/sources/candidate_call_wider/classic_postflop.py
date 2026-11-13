"""Parameterized postflop functions from classic_txnujk32_v1/classic_policy.py.

Bet size, flop air-stab frequency and sampling budget are configurable. Calls use
archetype='unknown': this baseline has no opponent profiling or table memory.
"""
from collections import Counter
import random
import eval7
from poker_agent_sdk import Action
from hand_contribution import shared_nut_action, public_quads_share
from pot_accounting import eligible_call

RANKS = '23456789TJQKA'
DECK = tuple(r + s for r in RANKS for s in 'shdc')
CARDS = {card: eval7.Card(card) for card in DECK}
from config import EQUITY_SAMPLES, BET_SCALE, BLUFF_FREQUENCY
SAMPLES = EQUITY_SAMPLES

def passive(state):
    if state.can('check'):
        return Action.check()
    return Action.fold() if state.can('fold') else Action.call()

def call(state):
    return Action.call() if state.can('call') else passive(state)

def raise_to(state, target):
    legal = state.can('raise')
    if legal is None:
        return call(state)
    return Action.raise_to(min(legal.max_amount, max(legal.min_amount, int(target))))

def players_after(state):
    order = sorted((s.seat for s in state.seats if s.status != 'busted'),
                   key=lambda seat: (seat - state.button - 1) % len(state.seats))
    return len(order) - 1 - order.index(state.seat)

def hand_features(hole, board):
    """Separate value-betting category from willingness to commit a stack."""
    value = eval7.evaluate([CARDS[c] for c in hole + board])
    made = value >> 24
    hr = [RANKS.index(c[0]) for c in hole]
    br = [RANKS.index(c[0]) for c in board]
    paired = len(set(br)) < len(br)
    if made >= 2:
        category = 4 if made >= 6 else 3
    elif made == 1:
        if hr[0] == hr[1]:
            category = 2 if hr[0] > max(br) else 1
        else:
            matching = set(hr).intersection(br)
            category = (2 if max(matching) == max(br) else 1) if matching else 0
    else:
        category = 0
    if len(board) == 5 and value == eval7.evaluate([CARDS[c] for c in board]):
        category = made = 0
    if made >= 3:
        commitment = 2
    elif made == 2:
        commitment = 2 if len(set(hr).intersection(br)) == 2 and not paired else 1
    else:
        commitment = int(made == 1 and category == 2)
    flush_draw = straight_draw = False
    if len(board) < 5:
        flush_draw = 4 in Counter(c[1] for c in hole + board).values()
        ranks = set(hr + br)
        if 12 in ranks:
            ranks.add(-1)
        straight_draw = any(all(r in ranks for r in range(lo, lo + 4)) for lo in range(-1, 9))
    return category, commitment, flush_draw, straight_draw

def random_stream(state):
    value = 0xC0FFEE ^ (state.hand_number * 1009 + state.seat * 7919 + len(state.history) * 104729)
    for card in state.cards + state.board:
        value = ((value * 16777619) ^ (RANKS.index(card[0]) * 4 + 'shdc'.index(card[1]))) & 0xFFFFFFFF
    return random.Random(value)

def equity(state, stream):
    """32 complete multiway deals; dead auction cards and shown cards excluded."""
    hero = [CARDS[c] for c in state.cards]
    board = [CARDS[c] for c in state.board]
    opponents = [s for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in')]
    known = set(state.cards) | set(state.board) | set(state.variant.get('auction_cards', ()))
    fixed = []
    unknown = 0
    for seat in opponents:
        if len(seat.shown_cards) == 2:
            fixed.append([CARDS[c] for c in seat.shown_cards])
            known.update(seat.shown_cards)
        else:
            unknown += 1
    if not opponents:
        return 1.
    deck = [CARDS[c] for c in DECK if c not in known]
    need = 5 - len(board)
    total = 0.
    for _ in range(SAMPLES):
        sampled = stream.sample(deck, need + unknown * 2)
        full = board + sampled[:need]
        hero_value = eval7.evaluate(hero + full)
        scores = [eval7.evaluate(cards + full) for cards in fixed]
        scores.extend(eval7.evaluate(sampled[i:i + 2] + full) for i in range(need, len(sampled), 2))
        best = max(scores)
        total += 1. if hero_value > best else 1. / (scores.count(best) + 1) if hero_value == best else 0.
    return total / SAMPLES

def postflop(state, archetype='unknown'):
    forced_tie = shared_nut_action(state)
    if forced_tie is not None:
        return forced_tie
    price = eligible_call(state)
    quad_share = public_quads_share(state.cards, state.board, tuple(state.variant.get('auction_cards', ())))
    if quad_share is not None:
        if state.can('check'):
            return Action.check()
        opponents = sum(s.status in ('active', 'all_in') for s in state.seats) - 1
        return call(state) if quad_share ** max(1, opponents) >= price.threshold + .06 else passive(state)
    stream = random_stream(state)
    eq = equity(state, stream)
    category, commitment, fd, sd = hand_features(state.cards, state.board)
    draw = fd or sd
    outs = 15 if fd and sd else 9 if fd else 8 if sd else 0
    # Original 2/4 rule, reduced for visibly dead cards completing the draw.
    if outs:
        unavailable = set(state.variant.get('auction_cards', ())) - set(state.board)
        ranks = set(RANKS.index(c[0]) for c in state.cards + state.board)
        if 12 in ranks:
            ranks.add(-1)
        suits = Counter(c[1] for c in state.cards + state.board)
        for card in unavailable:
            r = RANKS.index(card[0])
            augmented = ranks | {r} | ({-1} if r == 12 else set())
            completes_straight = sd and any(all(x in augmented for x in range(lo, lo + 5)) for lo in range(-1, 9))
            if fd and suits[card[1]] == 4 or completes_straight:
                outs -= 1
    draw_equity = min(.9, max(0, outs) * (.04 if len(state.board) == 3 else .02))
    me = state.seats[state.seat]
    opponents = sum(s.status in ('active', 'all_in') for s in state.seats) - 1
    if state.can('check'):
        if commitment >= 2:
            return raise_to(state, round(.75 * BET_SCALE * state.pot))
        if archetype != 'maniac':
            if draw and eq >= .34 and opponents <= 2 and len(state.board) < 5:
                return raise_to(state, round(.55 * BET_SCALE * state.pot))
            if len(state.board) == 3:
                if category >= 2:
                    return raise_to(state, round(.5 * BET_SCALE * state.pot))
                if category == 0 and opponents <= 2 and players_after(state) <= 1 and archetype != 'station' and stream.random() < BLUFF_FREQUENCY:
                    return raise_to(state, round(.5 * BET_SCALE * state.pot))
        return Action.check()
    fraction = price.call_cost / max(1, me.stack + me.bet)
    stackoff = fraction >= .55
    big = not stackoff and (fraction >= .28 or price.call_cost >= .70 * price.eligible_pot_before_call)
    adjusted = eq - (0. if archetype == 'maniac' else .13 if big or stackoff else .06)
    if commitment >= 2 and not (big or stackoff) and adjusted >= .60:
        return raise_to(state, me.bet + state.to_call + round(.8 * BET_SCALE * state.pot))
    if stackoff:
        return call(state) if commitment >= 2 else passive(state)
    if big:
        return call(state) if commitment >= 2 or draw and draw_equity >= price.threshold + .03 else passive(state)
    if adjusted >= price.threshold + (.02 if archetype == 'maniac' else .06):
        return call(state)
    if draw and draw_equity >= price.threshold and fraction <= .20:
        return call(state)
    if archetype == 'maniac' and category >= 1 and eq >= price.threshold:
        return call(state)
    return passive(state)
