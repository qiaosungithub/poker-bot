"""A bounded, closing-action cheap-call exception to corrected loose v2.

No equity estimate is claimed: a random-opponent HU table is not an all-in
range posterior. Preserve every existing non-fold, and every ordinary price.
The exception is limited to an unseen preflop board against one all-in player,
at <=1% reachable-pot price, <=0.1 BB, and <=5% of remaining hero chips.
"""
from poker_agent_sdk import Action
from independent_preflop import choose_preflop_loose
from pot_accounting import eligible_call

CARDS = frozenset(r + s for r in '23456789TJQKA' for s in 'shdc')


def cheap_closed_call(state):
    """All valid preflop holdings retain win/tie paths on the unknown board.

This is a deliberately narrow price rule, not a claim that any random-HU
equity is realizable multiway or after pressure. No remaining active opponent
can charge another normal-chip bet; auction-card control is still uncertain.
"""
    if (state.phase != 'betting' or state.board or not state.can('call')
            or len(state.cards) != 2 or len(set(state.cards)) != 2
            or not set(state.cards) <= CARDS
            or state.variant.get('auction_cards')):
        return False
    opponents = [s for s in state.seats if s.seat != state.seat
                 and s.status in ('active', 'all_in')]
    if len(opponents) != 1 or opponents[0].status != 'all_in':
        return False
    # A shown holding is usable only if it is a complete, disjoint legal pair.
    # It does not remove the unseen-board win/tie paths of a valid preflop hand.
    shown = opponents[0].shown_cards
    if shown and (len(shown) != 2 or len(set(shown)) != 2
                  or not set(shown) <= CARDS or set(shown) & set(state.cards)):
        return False
    own = state.seats[state.seat]
    price = eligible_call(state)
    return (price.exact and price.call_cost > 0
            and price.eligible_pot_after_call > 0
            and price.threshold <= .01
            and 10 * price.call_cost <= max(1, state.blinds.big_blind)
            and 20 * price.call_cost <= own.stack)


def choose_preflop_with(state, features, *, enabled=True):
    action = choose_preflop_loose(state, features)
    if enabled and action.kind == 'fold' and cheap_closed_call(state):
        return Action.call()
    return action


def choose_preflop(state, features):
    return choose_preflop_with(state, features)


def choose_preflop_off(state, features):
    return choose_preflop_with(state, features, enabled=False)
