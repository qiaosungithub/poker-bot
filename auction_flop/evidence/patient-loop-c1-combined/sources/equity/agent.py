"""Boston event equity policy. Fresh implementation after the live reveal."""
import poker_agent_sdk
from poker_agent_sdk import Action
from equity_core import (ids, stable_seed, preflop_strength, board_texture,
                         showdown_share, auction_equities, auction_competitive)


def _passive(state):
    if state.can('check'):
        return Action.check()
    if state.can('fold'):
        return Action.fold()
    if state.can('call'):
        return Action.call()
    action = state.legal_actions[0]
    return Action(action.kind, action.min_amount)


def _raise(state, total):
    legal = state.can('raise')
    if legal is None:
        return Action.call() if state.can('call') else _passive(state)
    return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(total))))


def _position(state):
    alive = sorted(s.seat for s in state.seats if s.status != 'busted')
    distance = (state.seat - state.button) % len(state.seats)
    before = sum(0 < (s - state.button) % len(state.seats) < distance for s in alive)
    # Button is best postflop; this coarse rank is only a range adjustment.
    if state.seat == state.button:
        return 1.0
    return before / max(1, len(alive) - 1)


def _eligible_pot(state, call):
    own = next(s for s in state.seats if s.seat == state.seat)
    if call < own.stack and not any(s.status == 'all_in' for s in state.seats):
        return state.pot + call
    totals = [0] * len(state.seats)
    street = [0] * len(state.seats)
    for event in state.history:
        if event.kind == 'board':
            street = [0] * len(state.seats)
        elif event.seat is not None and event.kind in ('blind', 'ante', 'call', 'raise'):
            seat = event.seat
            extra = max(0, event.amount - street[seat]) if event.kind == 'raise' else event.amount
            totals[seat] += extra
            if event.kind != 'ante':
                street[seat] += extra
    if sum(totals) != state.pot:
        # Rare refund/event variants: use a conservative current-street cap.
        return min(state.pot + call, (own.bet + call) * max(2, sum(s.status != 'busted' for s in state.seats)))
    cap = totals[state.seat] + call
    return call + sum(min(v, cap) for v in totals)


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        own = next(s for s in state.seats if s.seat == state.seat)
        hole = ids(state.cards)
        board = ids(state.board)
        live = [s for s in state.seats if s.status in ('active', 'all_in') and s.seat != state.seat]
        opponents = max(1, len(live))
        bb = max(1, state.blinds.big_blind)
        stack_bb = own.stack / bb
        seed = stable_seed(hole, board, state.hand_number * 211 + state.seat)
        if state.phase == 'auction':
            candidates = ids(state.variant['auction_cards'])
            legal = state.can('bid')
            budget = min(int(state.variant['flop_chips']), legal.max_amount)
            if budget == 0:
                return Action('bid', cards=(0,) * 5), b''
            values = auction_equities(hole, candidates, opponents, seed, 8, True)
            balances = state.variant['flop_chips_by_seat']
            bids = auction_competitive(values, budget, stack_bb, state.pot / bb,
                max(1, state.config.hand_cap - state.hand_number),
                [balances[s.seat] for s in live], seed, own.status == 'all_in')
            return Action('bid', cards=bids), b''
        if state.phase != 'betting':
            return _passive(state), b''
        call = min(own.stack, state.to_call)
        price = call / max(1, _eligible_pot(state, call))
        position = _position(state)
        if not board:
            strength = preflop_strength(hole)
            raises = sum(e.kind == 'raise' for e in state.history)
            pair = hole[0] // 4 == hole[1] // 4
            rank = max(hole[0] // 4, hole[1] // 4) + 2
            paid_places = sum(v > 0 for v in state.config.payouts)
            bubble = state.players_remaining == paid_places + 1 and own.stack > 8 * bb
            opening = .62 - .13 * position - .05 * (state.players_remaining <= 3)
            if raises:
                threshold = .61 + .075 * min(3, raises) + .08 * (call > 8 * bb) + .04 * bubble
                if strength >= .90 or (pair and rank >= 11):
                    if call >= .28 * max(1, own.stack) or stack_bb <= 18:
                        return _raise(state, own.bet + own.stack), b''
                    return _raise(state, max(state.to_call * 3 + own.bet, 7 * bb)), b''
                if strength >= threshold and price <= .37 and call <= own.stack * (.5 if stack_bb < 12 else .20):
                    return Action.call() if state.can('call') else _passive(state), b''
                if pair and call <= min(3 * bb, own.stack * .06) and price < .29:
                    return Action.call() if state.can('call') else _passive(state), b''
                return _passive(state), b''
            if stack_bb <= 10 and strength >= opening + .06:
                return _raise(state, own.bet + own.stack), b''
            if strength >= opening:
                return _raise(state, (2.3 + .25 * (opponents > 3)) * bb), b''
            if price <= .13 and strength >= .40 and call <= bb:
                return Action.call() if state.can('call') else _passive(state), b''
            return _passive(state), b''
        candidates = ids(state.variant.get('auction_cards', ()))
        dead = tuple(c for c in candidates if c not in board)
        equity = showdown_share(hole, board, opponents, seed, 24, dead)
        top, over, flush_draw, straight_draw, paired, wet = board_texture(hole, board)
        price_margin = .035 + .025 * (opponents > 1) + .05 * (call > own.stack * .45)
        if len(board) < 5:
            equity *= .91 if opponents > 1 else .96
        if call:
            # Avoid turning a uniform-opponent estimate into a large weak bluff catch.
            if not (top or over) and call > state.pot * .6:
                price_margin += .035
            if equity < price + price_margin:
                return _passive(state), b''
            if equity > max(.70, .53 + .06 * opponents) and state.can('raise'):
                return _raise(state, own.bet + call + .70 * (state.pot + call)), b''
            return Action.call() if state.can('call') else _passive(state), b''
        value_threshold = .60 if opponents == 1 else .49
        if equity > value_threshold and equity > 1.25 / (opponents + 1):
            size = .68 if equity > .80 or wet else .42
            return _raise(state, own.bet + max(bb, size * state.pot)), b''
        # Bounded semi-bluff with actual draw contribution and low opposition.
        if opponents == 1 and len(board) < 5 and (flush_draw or straight_draw) and (seed % 100) < 22:
            return _raise(state, own.bet + max(bb, .38 * state.pot)), b''
        if opponents == 1 and position > .7 and len(board) == 3 and equity > .35 and seed % 100 < 18:
            return _raise(state, own.bet + max(bb, .30 * state.pot)), b''
        return _passive(state), b''
