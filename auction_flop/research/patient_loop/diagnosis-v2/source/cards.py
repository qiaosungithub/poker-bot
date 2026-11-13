"""Unchanged legal-action and 32-deal equity helpers from simple_tuned_v2."""
import random
import eval7
from poker_agent_sdk import Action
RANKS = '23456789TJQKA'
DECK = tuple(r+s for r in RANKS for s in 'shdc')
CARDS = {c:eval7.Card(c) for c in DECK}
SAMPLES = 32

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
