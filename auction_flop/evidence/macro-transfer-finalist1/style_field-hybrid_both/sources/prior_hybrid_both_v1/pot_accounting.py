"""Public-history normal-chip accounting for eligible call pots.

No poker engine imports and no hidden cards/state. Boston uses trimmed antes;
normal-chip contributions are distinct from the auction's flop-chip currency.
"""
from typing import NamedTuple

MAX_EXACT_HISTORY_EVENTS = 128


class CallPot(NamedTuple):
    call_cost: int
    eligible_pot_before_call: int
    eligible_pot_after_call: int
    threshold: float
    exact: bool
    reason: str
    contributions: tuple[int, ...]
    refunded: int


def _refund_uncalled(round_bets, contributions):
    """At collection, PokerKit returns a unique highest bet's unmatched excess."""
    high = second = 0
    index = -1
    for i, amount in enumerate(round_bets):
        if amount > high:
            second, high, index = high, amount, i
        elif amount > second:
            second = amount
    refund = high - second
    if refund:
        contributions[index] -= refund
    return refund


def _fallback(state, call_cost, reason):
    """Use only visible current-street bets: a lower bound on reachable money.

    Missing earlier contributions also lower hero's cap, so this never invents
    cheap odds from unknown prior-street money. It may fold profitable calls.
    """
    current = tuple(max(0, int(s.bet)) for s in state.seats)
    cap = current[state.seat] + call_cost
    before = min(max(0, int(state.pot)), sum(min(value, cap) for value in current))
    after = before + call_cost
    return CallPot(call_cost, before, after, call_cost / after if after else 0.0,
                   False, reason, current, 0)


def eligible_call(state):
    """Return cost / reachable pot *after* the call, including hero's old money.

    `eligible_pot_before_call` includes already committed hero chips, each
    opponent's reachable contribution and reachable folded chips. Only the
    new call is excluded. Bids and flop-chip payments never count as pot money.
    All normal money is reconstructed from the current-hand public history.
    Uncalled refunds have no public event in the released engine: they are
    reconstructed at street/ante collection and checked against state.pot.
    """
    n = len(state.seats)
    hero = state.seat
    call_cost = min(max(0, int(state.to_call)), max(0, int(state.seats[hero].stack)))
    if len(state.history) > MAX_EXACT_HISTORY_EVENTS:
        return _fallback(state, call_cost, 'history_budget_exceeded')
    contributions = [0] * n
    street = [0] * n
    antes = [0] * n
    ante_open = False
    refunded = 0
    for event in state.history:
        kind = event.kind
        if ante_open and kind != 'ante':
            refunded += _refund_uncalled(antes, contributions)
            ante_open = False
        if kind == 'board':
            refunded += _refund_uncalled(street, contributions)
            street = [0] * n
        elif kind in ('ante', 'blind', 'call', 'raise'):
            seat, amount = event.seat, event.amount
            if type(seat) is not int or not 0 <= seat < n or type(amount) is not int or amount < 0:
                return _fallback(state, call_cost, 'invalid_money_event')
            if kind == 'ante':
                antes[seat] += amount
                contributions[seat] += amount
                ante_open = True
            elif kind == 'raise':
                delta = amount - street[seat]
                if delta < 0:
                    return _fallback(state, call_cost, 'raise_below_recorded_street_bet')
                contributions[seat] += delta
                street[seat] = amount
            else:
                contributions[seat] += amount
                street[seat] += amount
        elif kind in ('win', 'stack'):
            return _fallback(state, call_cost, 'settled_hand_in_current_history')
        elif kind not in ('hand', 'fold', 'check', 'show', 'muck', 'auction', 'bid',
                          'flop_chips', 'flop_chips_paid'):
            return _fallback(state, call_cost, 'unknown_history_event')
    if ante_open:
        refunded += _refund_uncalled(antes, contributions)
    actual_bets = [max(0, int(seat.bet)) for seat in state.seats]
    if street != actual_bets:
        if not any(actual_bets):
            # Most notably the preflop auction: betting is collected but the
            # flop's board event has not arrived yet.
            refunded += _refund_uncalled(street, contributions)
        else:
            return _fallback(state, call_cost, 'current_street_bets_disagree')
    if any(value < 0 for value in contributions) or sum(contributions) != state.pot:
        return _fallback(state, call_cost, 'normal_chip_total_disagrees')
    cap = contributions[hero] + call_cost
    before = sum(min(value, cap) for value in contributions)
    after = before + call_cost
    return CallPot(call_cost, before, after, call_cost / after if after else 0.0,
                   True, 'complete_public_history', tuple(contributions), refunded)
