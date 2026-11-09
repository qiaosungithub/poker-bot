"""Fast original Boston policy using made hands, texture and scarce auction bids."""

from itertools import combinations

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action


RANKS = {rank: index + 2 for index, rank in enumerate("23456789TJQKA")}
CARDS = {rank + suit: eval7.Card(rank + suit)
         for rank in RANKS for suit in "cdhs"}
FLOPS = tuple(combinations(range(5), 3))


def preflop_quality(hole):
    high, low = sorted((RANKS[card[0]] for card in hole), reverse=True)
    if high == low:
        return 0.47 + high * 0.032
    value = 0.15 + (high - 2) * 0.034 + (low - 2) * 0.018
    value += 0.065 * (hole[0][1] == hole[1][1])
    gap = high - low
    value += 0.04 if gap == 1 else 0.02 if gap == 2 else 0.0
    value -= min(0.08, max(0, gap - 3) * 0.018)
    return value


def hand_quality(hole, board):
    """Bounded heuristic equity versus one opponent, plus immediate hand class."""
    all_cards = hole + board
    ranks = [RANKS[card[0]] for card in all_cards]
    board_ranks = [RANKS[card[0]] for card in board]
    hole_ranks = [RANKS[card[0]] for card in hole]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    board_counts = {rank: board_ranks.count(rank) for rank in set(board_ranks)}
    encoded = eval7.evaluate([CARDS[card] for card in all_cards])
    category = encoded >> 24
    quality = (0.22, 0.48, 0.73, 0.82, 0.86, 0.90, 0.95, 0.985, 0.999)[category]

    if category == 0:
        quality += 0.025 * sum(rank > max(board_ranks) for rank in hole_ranks)
        quality += 0.025 * (14 in hole_ranks)
    elif category == 1:
        pair = max(rank for rank, count in rank_counts.items() if count == 2)
        if pair in hole_ranks:
            higher = sum(rank > pair for rank in set(board_ranks))
            quality = (0.70, 0.54, 0.43, 0.36, 0.31)[min(higher, 4)]
            quality += max(hole_ranks) * 0.002
            quality += 0.035 * (hole_ranks[0] == hole_ranks[1])
        else:
            quality = 0.22 + max(hole_ranks) * 0.008
    elif category == 2:
        private_pairs = sum(rank_counts.get(rank, 0) >= 2 for rank in set(hole_ranks))
        quality = 0.84 if private_pairs == 2 else 0.68 if private_pairs else 0.30
    elif category == 3:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        quality = 0.90 if hole_ranks.count(trips) == 2 else 0.83 if trips in hole_ranks else 0.35
    elif category == 6:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        if board_counts.get(trips, 0) >= 3:
            quality = 0.84

    if len(board) == 5:
        board_score = eval7.evaluate([CARDS[card] for card in board])
        if encoded == board_score:
            quality = 0.48 if category >= 4 else 0.22
    else:
        suits = [card[1] for card in all_cards]
        flush_draw = any(suits.count(suit) == 4 and any(card[1] == suit for card in hole)
                         for suit in "cdhs")
        distinct = set(ranks)
        if 14 in distinct:
            distinct.add(1)
        four_run = any(len(distinct.intersection(range(start, start + 5))) == 4
                       for start in range(1, 11))
        # Draws add option value, while strong made hands retain their baseline.
        improvement = 0.14 * flush_draw + 0.075 * four_run
        quality += improvement * (1.0 - quality)
    return min(0.999, quality), category


def auction_bid(state):
    legal = state.can("bid")
    budget = legal.max_amount if legal else 0
    cards = tuple(state.variant.get("auction_cards", ()))
    if budget <= 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    hole = tuple(state.cards)
    options = [(hand_quality(hole, tuple(cards[index] for index in combo))[0], combo)
               for combo in FLOPS]
    best_quality, best = max(options)
    default_quality = options[0][0]
    gain = best_quality - default_quality
    if gain < 0.07:
        return Action("bid", cards=(0, 0, 0, 0, 0))

    remaining = max(12, min(70, state.config.hand_cap - state.hand_number))
    big_blind = max(1, state.blinds.big_blind)
    pot_factor = min(3.5, max(0.65, state.pot / (big_blind * 8)))
    spend = int(budget / remaining * pot_factor * min(4.0, gain * 9))
    if best_quality >= 0.86:
        spend = max(spend, min(budget // 8, int(25 * pot_factor)))
    own = state.seats[state.seat]
    if own.status == "all_in":
        spend = max(spend, min(budget // 3, int(gain * budget)))
    spend = min(budget, max(1, spend))
    # Positive bids on all three target cards preserve a coherent selected flop.
    bids = [0] * 5
    quotient, remainder = divmod(spend, 3)
    for offset, index in enumerate(best):
        bids[index] = quotient + (offset < remainder)
    return Action("bid", cards=tuple(bids))


def raise_safely(state, target):
    legal = state.can("raise")
    if legal:
        return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(target))))
    return Action.call() if state.can("call") else Action.check()


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_bid(state), b""
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), b""

        own = state.seats[state.seat]
        big_blind = max(1, state.blinds.big_blind)
        opponents = sum(seat.status in ("active", "all_in") and seat.seat != state.seat
                        for seat in state.seats)
        cost = min(state.to_call, own.stack)
        check = state.can("check")
        fold_or_check = Action.check() if check else Action.fold()
        raise_action = state.can("raise")

        if not state.board:
            quality = preflop_quality(state.cards)
            short = (own.stack + own.bet) / big_blind
            late = state.seat == state.button
            open_threshold = 0.60 - 0.025 * max(0, 5 - state.players_remaining) - 0.04 * late
            raised = any(event.kind == "raise" for event in state.history)
            if short <= 10 and quality >= 0.57 + 0.02 * min(opponents, 5):
                return raise_safely(state, own.stack + own.bet), b""
            if quality >= 0.82:
                size = max(big_blind * 3, own.bet + cost + max(big_blind * 2, state.pot * 0.75))
                return raise_safely(state, size), b""
            if not raised and quality >= open_threshold and raise_action:
                return raise_safely(state, big_blind * 2.5), b""
            call_threshold = 0.60 + min(0.20, cost / big_blind * 0.015)
            if state.can("call") and quality >= call_threshold and cost <= own.stack * 0.22:
                return Action.call(), b""
            return fold_or_check, b""

        quality, category = hand_quality(tuple(state.cards), tuple(state.board))
        quality *= 0.91 ** max(0, opponents - 1)
        pot_odds = cost / max(1, state.pot + cost)
        pressure = cost / max(1, own.stack)
        value_threshold = 0.71 + 0.025 * max(0, opponents - 1)
        if raise_action and quality >= value_threshold:
            target = own.bet + cost + max(big_blind, int((state.pot + cost) * 0.65))
            if quality >= 0.88 or own.stack <= state.pot * 0.8 and quality >= 0.73:
                target = own.bet + own.stack
            # Avoid repeated reraising weak made hands on the same street.
            if cost <= state.pot * 0.6 or category >= 3:
                return raise_safely(state, target), b""
        if check:
            return Action.check(), b""
        required = pot_odds + 0.09 + pressure * 0.12
        if state.can("call") and quality >= required:
            if pressure < 0.50 or quality >= 0.67 or category >= 2:
                return Action.call(), b""
        return fold_or_check, b""
