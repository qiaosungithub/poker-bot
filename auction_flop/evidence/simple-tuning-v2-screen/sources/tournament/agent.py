"""Tournament-aware Auction Flop policy, freshly implemented during Boston event."""
import poker_agent_sdk
from poker_agent_sdk import Action
from strategy import card_code, preflop_quality, board_quality, risk_premium, auction_bids


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        hole = tuple(card_code(c) for c in state.cards)
        me = state.seats[state.seat]
        live = [s for s in state.seats if s.status in ('active', 'all_in')]
        opponents = max(1, len(live) - 1)
        bb = max(1, state.blinds.big_blind)
        if state.phase == 'auction':
            legal = state.can('bid')
            budget = legal.max_amount if legal else 0
            bids = auction_bids(hole, tuple(card_code(c) for c in state.variant['auction_cards']),
                budget, hand_number=state.hand_number, hand_cap=state.config.hand_cap,
                stack=me.stack, big_blind=bb, pot=state.pot, opponents=opponents,
                all_in=me.status == 'all_in')
            return Action('bid', cards=bids), b''
        check, call, raise_ = state.can('check'), state.can('call'), state.can('raise')
        passive = Action.check() if check else (Action.fold() if state.can('fold') else Action.call())
        if len(hole) != 2:
            return passive, b''
        stack_bb = (me.stack + me.bet) / bb
        other_stacks = [s.stack + s.bet for s in state.seats if s.seat != state.seat and s.status != 'busted']
        risk = risk_premium(me.stack, max(other_stacks, default=0),
                            [s.stack + s.bet for s in state.seats], state.config.payouts, state.to_call)
        current = []
        for e in reversed(state.history):
            if e.kind == 'board':
                break
            current.append(e)
        raises = sum(e.kind == 'raise' for e in current)
        active_order = sorted((s.seat for s in state.seats if s.status != 'busted'),
                              key=lambda seat: (seat - state.button - 1) % len(state.seats))
        position = active_order.index(state.seat) / max(1, len(active_order) - 1)
        mix = ((state.hand_number * 1103515245 + state.seat * 12345 + sum(hole) * 2654435761 +
                len(state.board) * 101) & 65535) / 65536.0

        def raise_to(value):
            if raise_:
                return Action.raise_to(max(raise_.min_amount, min(raise_.max_amount, int(value))))
            return Action.call() if call else passive

        if not state.board:
            q = preflop_quality(hole)
            open_threshold = 0.60 - 0.13 * position - (0.09 if state.players_remaining <= 3 else 0)
            if raises == 0:
                if q >= open_threshold:
                    if stack_bb <= 10 and q >= max(0.56, open_threshold):
                        return raise_to(me.stack + me.bet), b''
                    callers = sum(e.kind == 'call' for e in current)
                    return raise_to(bb * (2.3 + 0.7 * callers)), b''
                if call and q >= open_threshold - 0.065 and state.to_call <= bb * 0.6:
                    return Action.call(), b''
                return passive, b''
            price_bb = state.to_call / bb
            threshold = 0.52 + min(0.22, price_bb * 0.018) + 0.025 * max(0, raises - 1) + risk
            if q >= 0.85 or (stack_bb <= 10 and q >= 0.68):
                return raise_to(me.stack + me.bet if stack_bb <= 28 else me.bet + 3.2 * state.to_call), b''
            if call and q >= threshold:
                return Action.call(), b''
            return passive, b''

        q = board_quality(hole, tuple(card_code(c) for c in state.board))
        equity_proxy = q ** (1 + 0.24 * max(0, opponents - 1))
        odds = state.to_call / max(1, state.pot + state.to_call)
        if state.to_call:
            required = odds + 0.06 + 0.035 * max(0, opponents - 1) + risk
            # Large bets against a marginal pair require a clearer strength edge.
            if q < 0.64 and state.to_call > 0.5 * state.pot:
                required += 0.12
            if q >= 0.90 and raises < 2:
                return raise_to(me.bet + state.to_call + 0.80 * (state.pot + state.to_call)), b''
            if call and equity_proxy >= required:
                return Action.call(), b''
            return passive, b''
        value = q >= (0.70 if opponents <= 2 else 0.79)
        probe = opponents == 1 and len(state.board) == 3 and mix < 0.12 and q > 0.24
        if raise_ and (value or probe):
            fraction = 0.68 if q >= 0.83 else 0.43
            if stack_bb <= 10 and q >= 0.84:
                return raise_to(me.stack + me.bet), b''
            return raise_to(me.bet + max(bb, fraction * state.pot)), b''
        return passive, b''
