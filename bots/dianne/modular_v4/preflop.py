"""Independent preflop phase alternative."""

from .common import (Action, CARDS, RANKS, eval7, call_threshold, contestable_pot,
                     preflop_quality, raise_safely, risk_premium,
                     cheap_preflop_continue, postflop_hand_quality)


def preflop_decision(state, memory):
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

    quality = preflop_quality(state.cards)
    if cheap_preflop_continue(state, opponents, cost, eligible_pot, required):
        return Action.call(), b""
    raised = any(event.kind == "raise" for event in state.history)
    opening = 0.62 - 0.03 * max(0, 6 - state.players_remaining) - 0.045 * late
    opening += premium * 0.5
    if covering and state.players_remaining <= 4:
        opening -= 0.045
    short_threshold = 0.58 + 0.018 * min(5, len(opponents)) + premium
    if stack_bb <= 9 and quality >= short_threshold and not raised:
        return raise_safely(state, own.bet + own.stack), b""
    if quality >= 0.87 and raising:
        target = own.bet + cost + max(2 * big_blind, eligible_pot * 0.65)
        return raise_safely(state, target), b""
    if not raised and raising and quality >= opening:
        return raise_safely(state, max(2.2 * big_blind, own.bet + cost + big_blind)), b""
    if state.can("call"):
        threshold = 0.62 + premium + min(0.17, cost / big_blind * 0.012)
        if cost >= own.stack * 0.55:
            threshold = max(threshold, required + 0.10)
        if quality >= threshold:
            return Action.call(), b""
    return passive, b""
