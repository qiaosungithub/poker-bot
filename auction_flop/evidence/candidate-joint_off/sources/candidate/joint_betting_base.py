"""Boston variant specialist: combinatorial auctions and bounded hand rules.

Written during the 2026-11-09 event. No trained model or cross-table state.
"""
import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState
from joint_value_base import combination_utilities, choose_bids, parse_card, STRAIGHT_MASKS

CARDS = {r + s: eval7.Card(r + s) for r in '23456789TJQKA' for s in 'cdhs'}
RANK = {r: i + 2 for i, r in enumerate('23456789TJQKA')}
# A tiny import-time evaluator warm-up; all global data is immutable lookup data.
eval7.evaluate([CARDS[c] for c in ('As', 'Ah', 'Ac', 'Kd', 'Qd')])


def preflop_quality(cards):
    a, b = sorted((RANK[c[0]] for c in cards), reverse=True)
    if a == b:
        return .48 + (a - 2) * .04
    value = .23 + (a - 2) * .025 + (b - 2) * .015
    if a == 14:
        value += .08
    if cards[0][1] == cards[1][1]:
        value += .065
    if a - b == 1:
        value += .035
    elif a - b == 2:
        value += .015
    elif a - b > 4 and b < 9:
        value -= .04
    return value


def hand_quality(cards, board):
    """Strength proxy and draw flag; category is numeric eval7, not label text."""
    parsed = tuple(parse_card(c) for c in cards + board)
    hr = (parsed[0][0], parsed[1][0])
    br = [c[0] for c in parsed[2:]]
    rank = eval7.evaluate([CARDS[c] for c in cards + board])
    category = rank >> 24
    if len(board) == 5 and rank == eval7.evaluate([CARDS[c] for c in board]):
        return .10 + max(hr) / 150, category, False
    groups = {r: br.count(r) for r in set(br)}
    combined = {r: br.count(r) + hr.count(r) for r in set(br + list(hr))}
    top = max(br)
    own_pair = max((r for r in hr if combined[r] >= 2), default=0)
    if category >= 6:
        quality = .97
    elif category == 5:
        quality = .91
    elif category == 4:
        quality = .86
    elif category == 3:
        trip = max(r for r in combined if combined[r] >= 3)
        quality = .83 if trip in hr else .28 + max(hr) / 100
    elif category == 2:
        private_pairs = len({r for r in hr if combined[r] >= 2})
        quality = .78 if private_pairs == 2 else .59 + max(hr) / 200
    elif own_pair:
        if hr[0] == hr[1] and hr[0] > top:
            quality = .72
        elif own_pair == top:
            quality = .61 + min(hr) / 200
        elif own_pair >= sorted(set(br), reverse=True)[min(1, len(set(br)) - 1)]:
            quality = .49
        else:
            quality = .36
    else:
        quality = .15 + max(hr) / 130
    suits = [c[1] for c in parsed]
    board_suits = suits[2:]
    danger = max(board_suits.count(s) for s in set(board_suits))
    if danger >= 3 and category < 5:
        flush_suit = max(set(board_suits), key=board_suits.count)
        own_suit = [c[0] for c in parsed[:2] if c[1] == flush_suit]
        if not own_suit:
            quality -= .08 if danger == 3 else .22
    draw = False
    if len(board) < 5:
        for suit in (parsed[0][1], parsed[1][1]):
            if suits.count(suit) == 4:
                quality += .12
                draw = True
                break
        ranks = set(c[0] for c in parsed)
        if 14 in ranks:
            ranks.add(1)
        mask = sum(1 << r for r in ranks)
        if category < 4 and max((mask & m).bit_count() for m in STRAIGHT_MASKS) == 4:
            quality += .065
            draw = True
    return max(.01, min(.995, quality)), category, draw


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


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes):
        seats = state.seats
        own = next(s for s in seats if s.seat == state.seat)
        opponents = tuple(s for s in seats if s.seat != state.seat and s.status in ('active', 'all_in'))
        bb = max(1, state.blinds.big_blind)
        if state.phase == 'auction':
            legal = state.can('bid')
            budget = min(int(state.variant.get('flop_chips', 0)), legal.max_amount if legal else 0)
            cards = tuple(state.variant.get('auction_cards', ()))
            if budget <= 0 or len(cards) != 5 or len(state.cards) != 2:
                return Action('bid', cards=(0, 0, 0, 0, 0)), b''
            balances = state.variant.get('flop_chips_by_seat', ())
            opposition = tuple(int(balances[s.seat]) for s in opponents if s.seat < len(balances))
            utilities = combination_utilities(state.cards, cards)
            bids = choose_bids(utilities, budget, opposition,
                               hands_left=max(1, state.config.hand_cap - state.hand_number + 1),
                               stack_bb=own.stack / bb, pot_bb=state.pot / bb,
                               all_in=own.status == 'all_in')
            return Action('bid', cards=bids), b''
        if state.phase != 'betting' or len(state.cards) != 2:
            return safe_pass(state), b''
        players = len(opponents) + 1
        stack_bb = own.stack / bb
        to_call = min(own.stack, state.to_call)
        active_seats = [s.seat for s in seats if s.status != 'busted']
        ordered = sorted(active_seats, key=lambda s: (s - state.button - 1) % len(seats))
        late = state.seat in ordered[-3:]
        quality = preflop_quality(state.cards) if not state.board else 0.
        # Count only this street's raises; auction itself is not a betting raise.
        raises = 0
        for event in reversed(state.history):
            if event.kind in ('board', 'auction'):
                break
            if event.kind == 'raise':
                raises += 1
        if not state.board:
            open_threshold = .59 if late else .66
            if state.players_remaining <= 3:
                open_threshold -= .075
            if stack_bb <= 10 and quality >= open_threshold - .02 and raises < 2:
                return legal_raise(state, own.stack + own.bet), b''
            if raises == 0 and quality >= open_threshold:
                limpers = sum(e.kind == 'call' for e in state.history)
                return legal_raise(state, bb * (2.4 + min(3, limpers) * .8)), b''
            if quality >= .89 or quality >= .83 and raises == 1 and state.players_remaining <= 4:
                return legal_raise(state, max(bb * 7, (own.bet + state.to_call) * 2.7)), b''
            if state.can('check'):
                return Action.check(), b''
            required = .56 if to_call <= bb else .65 if to_call <= 2.5 * bb else .73 if to_call <= 6 * bb else .81
            if raises > 1:
                required += .04
            if state.players_remaining <= 3:
                required -= .055
            if to_call > own.stack * .45 and stack_bb > 10:
                required = max(required, .82)
            if quality >= required and state.can('call'):
                return Action.call(), b''
            return safe_pass(state), b''
        quality, category, draw = hand_quality(state.cards, state.board)
        quality -= min(.16, max(0, players - 2) * .035)
        pot = max(bb, state.pot)
        paid_places = sum(p > 0 for p in state.config.payouts)
        risk = .03 if state.players_remaining == paid_places + 1 else 0.
        if state.can('check'):
            threshold = .68 if players <= 3 else .75
            if quality >= threshold:
                size = .72 if quality >= .85 else .5
                target = own.bet + max(bb, int(pot * size))
                if quality >= .88 and own.stack <= pot * 2:
                    target = own.bet + own.stack
                return legal_raise(state, target), b''
            # Deterministic, table-local mixing from public state and own cards.
            mix = (state.hand_number * 73 + state.seat * 29 + RANK[state.cards[0][0]] * 17 + len(state.history) * 11) % 100
            if players <= 2 and len(state.board) <= 4 and mix < (14 if draw else 5):
                return legal_raise(state, own.bet + max(bb, int(pot * .45))), b''
            return Action.check(), b''
        odds = to_call / max(1, pot + to_call)
        # Proxy quality is not an actual equity estimate; apply explicit margins.
        defend = quality - .22 - risk - min(.10, max(0, raises - 1) * .05)
        if category >= 3:
            defend += .10
        if to_call > own.stack * .5 and category < 2 and not draw:
            defend -= .12
        if quality >= .90 and state.can('raise'):
            return legal_raise(state, own.bet + to_call + int((pot + to_call) * .75)), b''
        if defend >= odds and state.can('call'):
            return Action.call(), b''
        return safe_pass(state), b''
