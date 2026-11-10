"""Classic-inspired, isolated flop top-pair value opportunity."""
from postflop_policy import choose_postflop
from corrected_betting_baseline import hand_quality, legal_raise, RANK


def eligible_value_check(state):
    """A modest value bet can be called by worse pairs or unmade draws.

    This is a hypothesis about possible calls, not an opponent range estimate.
    Restrict it to a single live opponent and a non-monotone, unpaired flop.
    """
    if (state.phase != 'betting' or len(state.board) != 3 or
            len(state.cards) != 2 or state.to_call != 0 or
            state.can('check') is None or state.can('raise') is None):
        return False
    opponents = [s for s in state.seats
                 if s.seat != state.seat and s.status in ('active', 'all_in')]
    if len(opponents) != 1 or opponents[0].status != 'active' or opponents[0].stack <= 0:
        return False
    own = next(s for s in state.seats if s.seat == state.seat)
    if own.bet != 0:
        return False
    br = [RANK[c[0]] for c in state.board]
    hr = [RANK[c[0]] for c in state.cards]
    if (len(set(br)) != 3 or len({c[1] for c in state.board}) == 1 or
            hr[0] == hr[1] or max(br) not in hr):
        return False
    _, category, draw = hand_quality(state.cards, state.board,
                                     tuple(state.variant.get('auction_cards', ())))
    return category == 1 and not draw


def choose_flop(state, features, enabled=True):
    action = choose_postflop(state, features)
    if enabled and action.kind == 'check' and eligible_value_check(state):
        own = next(s for s in state.seats if s.seat == state.seat)
        return legal_raise(state, own.bet + round(.5 * state.pot))
    return action
