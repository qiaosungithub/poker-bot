"""Original ICM v2 postflop policy, edited independently."""

from .common import (Action, call_threshold, contestable_pot, hand_quality,
                     preflop_quality, raise_safely, risk_premium)


def postflop_decision(state, memory):
    own = state.seats[state.seat]
    opponents = [seat for seat in state.seats
                 if seat.seat != state.seat and seat.status in ("active", "all_in")]
    big_blind = max(1, state.blinds.big_blind)
    cost = min(state.to_call, own.stack)
    eligible_pot = contestable_pot(state, cost)
    passive = Action.check() if state.can("check") else Action.fold()
    premium = risk_premium(state)
    required = call_threshold(state, opponents, cost, premium)
    raising = state.can("raise")
    stack_bb = (own.stack + own.bet) / big_blind
    late = state.seat == state.button
    covering = all(own.stack + own.bet >= seat.stack + seat.bet for seat in opponents)


    quality, category = hand_quality(tuple(state.cards), tuple(state.board))
    quality *= 0.89 ** max(0, len(opponents) - 1)
    if raising:
        value_cutoff = 0.73 + premium + 0.02 * max(0, len(opponents) - 1)
        if quality >= value_cutoff and (cost < eligible_pot * 0.6 or category >= 3):
            fraction = 0.48 if premium > 0.07 else 0.65
            target = own.bet + cost + max(big_blind, (eligible_pot + cost) * fraction)
            if quality >= 0.92 or stack_bb < 7 and quality > 0.79 + premium:
                target = own.bet + own.stack
            return raise_safely(state, target), b""
    if state.can("check"):
        return Action.check(), b""
    if state.can("call") and quality >= required:
        if cost < own.stack * 0.45 or quality > 0.67 + premium or category >= 3:
            return Action.call(), b""
    return passive, b""
