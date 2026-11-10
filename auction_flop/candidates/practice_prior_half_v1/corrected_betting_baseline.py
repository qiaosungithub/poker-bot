"""Betting-only extraction of frozen event-time variant-v1 (6af47e4).

Inference OFF preserves its actions. The only extension is an explicit bounded
quality_penalty argument; auction currency never modifies pot or stack amounts.
"""
import eval7
from poker_agent_sdk import Action
from pot_accounting import eligible_call
from hand_contribution import adjust_hand_quality, shared_nut_action
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
STRAIGHT_MASKS = tuple(sum(1 << r for r in range(high - 4, high + 1)) for high in range(5, 15))
def parse_card(card):
    return RANKS.index(card[0]) + 2, SUITS.index(card[1].lower())
CARDS = {r + s: eval7.Card(r + s) for r in RANKS for s in SUITS}
RANK = {r: i + 2 for i, r in enumerate(RANKS)}
eval7.evaluate([CARDS[c] for c in ('As', 'Ah', 'Ac', 'Kd', 'Qd')])

def preflop_quality(cards):
    (a, b) = sorted((RANK[c[0]] for c in cards), reverse=True)
    if a == b:
        return 0.48 + (a - 2) * 0.04
    value = 0.23 + (a - 2) * 0.025 + (b - 2) * 0.015
    if a == 14:
        value += 0.08
    if cards[0][1] == cards[1][1]:
        value += 0.065
    if a - b == 1:
        value += 0.035
    elif a - b == 2:
        value += 0.015
    elif a - b > 4 and b < 9:
        value -= 0.04
    return value

def _legacy_hand_quality(cards, board):
    """Strength proxy and draw flag; category is numeric eval7, not label text."""
    parsed = tuple((parse_card(c) for c in cards + board))
    hr = (parsed[0][0], parsed[1][0])
    br = [c[0] for c in parsed[2:]]
    rank = eval7.evaluate([CARDS[c] for c in cards + board])
    category = rank >> 24
    if len(board) == 5 and rank == eval7.evaluate([CARDS[c] for c in board]):
        return (0.1 + max(hr) / 150, category, False)
    groups = {r: br.count(r) for r in set(br)}
    combined = {r: br.count(r) + hr.count(r) for r in set(br + list(hr))}
    top = max(br)
    own_pair = max((r for r in hr if combined[r] >= 2), default=0)
    if category >= 6:
        quality = 0.97
    elif category == 5:
        quality = 0.91
    elif category == 4:
        quality = 0.86
    elif category == 3:
        trip = max((r for r in combined if combined[r] >= 3))
        quality = 0.83 if trip in hr else 0.28 + max(hr) / 100
    elif category == 2:
        private_pairs = len({r for r in hr if combined[r] >= 2})
        quality = 0.78 if private_pairs == 2 else 0.59 + max(hr) / 200
    elif own_pair:
        if hr[0] == hr[1] and hr[0] > top:
            quality = 0.72
        elif own_pair == top:
            quality = 0.61 + min(hr) / 200
        elif own_pair >= sorted(set(br), reverse=True)[min(1, len(set(br)) - 1)]:
            quality = 0.49
        else:
            quality = 0.36
    else:
        quality = 0.15 + max(hr) / 130
    suits = [c[1] for c in parsed]
    board_suits = suits[2:]
    danger = max((board_suits.count(s) for s in set(board_suits)))
    if danger >= 3 and category < 5:
        flush_suit = max(set(board_suits), key=board_suits.count)
        own_suit = [c[0] for c in parsed[:2] if c[1] == flush_suit]
        if not own_suit:
            quality -= 0.08 if danger == 3 else 0.22
    draw = False
    if len(board) < 5:
        for suit in (parsed[0][1], parsed[1][1]):
            if suits.count(suit) == 4:
                quality += 0.12
                draw = True
                break
        ranks = set((c[0] for c in parsed))
        if 14 in ranks:
            ranks.add(1)
        mask = sum((1 << r for r in ranks))
        if category < 4 and max(((mask & m).bit_count() for m in STRAIGHT_MASKS)) == 4:
            quality += 0.065
            draw = True
    return (max(0.01, min(0.995, quality)), category, draw)

def hand_quality(cards, board, dead_cards=()):
    """Original proxy except proven shared-nut/public-quads corrections."""
    return adjust_hand_quality(cards, board, _legacy_hand_quality(cards, board), dead_cards)

def legal_raise(state, target):
    legal = state.can('raise')
    if legal:
        return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(target))))
    if state.can('check'):
        return Action.check()
    if state.can('call'):
        return Action.call()
    return Action.fold()

def safe_pass(state):
    if state.can('check'):
        return Action.check()
    if state.can('fold'):
        return Action.fold()
    legal = state.legal_actions[0]
    return Action(legal.kind, legal.min_amount)

def choose_baseline_bet(state, quality_penalty=0.0):
    seats = state.seats
    own = next((s for s in seats if s.seat == state.seat))
    opponents = tuple((s for s in seats if s.seat != state.seat and s.status in ('active', 'all_in')))
    bb = max(1, state.blinds.big_blind)
    if state.phase != 'betting' or len(state.cards) != 2:
        return safe_pass(state)
    # A proven shared nut hand cannot lose; never let heuristic penalties fold it.
    tied = shared_nut_action(state)
    if tied is not None:
        return tied
    players = len(opponents) + 1
    stack_bb = own.stack / bb
    to_call = min(own.stack, state.to_call)
    active_seats = [s.seat for s in seats if s.status != 'busted']
    ordered = sorted(active_seats, key=lambda s: (s - state.button - 1) % len(seats))
    late = state.seat in ordered[-3:]
    quality = preflop_quality(state.cards) if not state.board else 0.0
    raises = 0
    for event in reversed(state.history):
        if event.kind in ('board', 'auction'):
            break
        if event.kind == 'raise':
            raises += 1
    if not state.board:
        open_threshold = 0.59 if late else 0.66
        if state.players_remaining <= 3:
            open_threshold -= 0.075
        if stack_bb <= 10 and quality >= open_threshold - 0.02 and (raises < 2):
            return legal_raise(state, own.stack + own.bet)
        if raises == 0 and quality >= open_threshold:
            limpers = sum((e.kind == 'call' for e in state.history))
            return legal_raise(state, bb * (2.4 + min(3, limpers) * 0.8))
        if quality >= 0.89 or (quality >= 0.83 and raises == 1 and (state.players_remaining <= 4)):
            return legal_raise(state, max(bb * 7, (own.bet + state.to_call) * 2.7))
        if state.can('check'):
            return Action.check()
        required = 0.56 if to_call <= bb else 0.65 if to_call <= 2.5 * bb else 0.73 if to_call <= 6 * bb else 0.81
        if raises > 1:
            required += 0.04
        if state.players_remaining <= 3:
            required -= 0.055
        if to_call > own.stack * 0.45 and stack_bb > 10:
            required = max(required, 0.82)
        if quality >= required and state.can('call'):
            return Action.call()
        return safe_pass(state)
    (quality, category, draw) = hand_quality(state.cards, state.board, tuple(state.variant.get('auction_cards', ())))
    quality -= quality_penalty
    quality -= min(0.16, max(0, players - 2) * 0.035)
    pot = max(bb, state.pot)
    paid_places = sum((p > 0 for p in state.config.payouts))
    risk = 0.03 if state.players_remaining == paid_places + 1 else 0.0
    if state.can('check'):
        threshold = 0.68 if players <= 3 else 0.75
        if quality >= threshold:
            size = 0.72 if quality >= 0.85 else 0.5
            target = own.bet + max(bb, int(pot * size))
            if quality >= 0.88 and own.stack <= pot * 2:
                target = own.bet + own.stack
            return legal_raise(state, target)
        mix = (state.hand_number * 73 + state.seat * 29 + RANK[state.cards[0][0]] * 17 + len(state.history) * 11) % 100
        if players <= 2 and len(state.board) <= 4 and (mix < (14 if draw else 5)):
            return legal_raise(state, own.bet + max(bb, int(pot * 0.45)))
        return Action.check()
    odds = eligible_call(state).threshold
    defend = quality - 0.22 - risk - min(0.1, max(0, raises - 1) * 0.05)
    if category >= 3:
        defend += 0.1
    if to_call > own.stack * 0.5 and category < 2 and (not draw):
        defend -= 0.12
    if quality >= 0.9 and state.can('raise'):
        return legal_raise(state, own.bet + to_call + int((pot + to_call) * 0.75))
    if defend >= odds and state.can('call'):
        return Action.call()
    return safe_pass(state)
