"""Independent postflop phase alternative."""

from .common import (Action, CARDS, RANKS, eval7, call_threshold, contestable_pot,
                     preflop_quality, raise_safely, risk_premium,
                     cheap_preflop_continue, postflop_hand_quality)


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


    dead = tuple(card for card in state.variant.get("auction_cards", ()) if card not in state.board)
    quality, category = postflop_hand_quality(tuple(state.cards), tuple(state.board), dead)
    if (category == 2 and state.cards[0][0] == state.cards[1][0]
            and any(RANKS[card[0]] > RANKS[state.cards[0][0]] for card in state.board)):
        # Repeated postflop betting shifts this bluff-catcher toward ranges
        # connecting with the board. Count streets, rather than raise events.
        betting_streets = set()
        street = 0
        for event in state.history:
            if event.kind == "board":
                street += 1
            elif event.kind == "raise" and event.seat != state.seat and street:
                betting_streets.add(street)
        if len(betting_streets) >= 2:
            quality *= max(0.40, 1.0 - 0.18 * len(betting_streets))
    if len(state.board) == 5:
        board_score = eval7.evaluate([CARDS[card] for card in state.board])
        board_category = board_score >> 24
        # Guaranteed shared nuts retain positive value when calling any bet.
        royal = board_category == 8 and {card[0] for card in state.board} == set("TJQKA")
        broadway = (board_category == 4 and {card[0] for card in state.board} == set("TJQKA")
                    and max(sum(card[1] == suit for card in state.board) for suit in "cdhs") < 3)
        board_ranks = [card[0] for card in state.board]
        board_quads_ace = (board_category == 7 and
                           (board_ranks.count("A") == 1 or board_ranks.count("A") == 4 and "K" in board_ranks))
        if royal or broadway or board_quads_ace:
            return (Action.check() if state.can("check") else Action.call()), b""
    quality *= (quality if quality >= 0.90 else 0.89) ** max(0, len(opponents) - 1)
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
