"""Boston agent with marginal-flop auction selection and conserved auction cash."""

from itertools import combinations

import poker_agent_sdk
from poker_agent_sdk import Action


RANKS = {rank: index + 2 for index, rank in enumerate("23456789TJQKA")}
FLOPS = tuple(combinations(range(5), 3))
STRAIGHTS = tuple(frozenset(range(high - 4, high + 1)) for high in range(14, 5, -1)) + (
    frozenset((14, 2, 3, 4, 5)),
)


def preflop_strength(cards):
    first, second = cards
    high, low = sorted((RANKS[first[0]], RANKS[second[0]]), reverse=True)
    if high == low:
        return 0.49 + high * 0.027
    strength = 0.29 + 0.014 * (high + low)
    strength += 0.035 if first[1] == second[1] else 0.0
    strength -= 0.018 * max(0, high - low - 1)
    strength += 0.025 if high == 14 else 0.0
    return max(0.31, min(0.75, strength))


def board_strength(hole, board):
    """Fast relative made-hand/draw estimate; deliberately no sampled rollouts."""
    cards = hole + tuple(board)
    counts = {}
    suits = {}
    for card in cards:
        rank = RANKS[card[0]]
        counts[rank] = counts.get(rank, 0) + 1
        suits[card[1]] = suits.get(card[1], 0) + 1
    private_ranks = {RANKS[card[0]] for card in hole}
    board_ranks = [RANKS[card[0]] for card in board]
    ranks = set(counts)
    straight = any(window <= ranks for window in STRAIGHTS)
    flush_suit = next((suit for suit, count in suits.items() if count >= 5), None)
    triples = sorted((rank for rank, count in counts.items() if count >= 3), reverse=True)
    pairs = sorted((rank for rank, count in counts.items() if count >= 2), reverse=True)
    four = any(count == 4 for count in counts.values())
    straight_flush = False
    if flush_suit:
        flush_ranks = {RANKS[card[0]] for card in cards if card[1] == flush_suit}
        straight_flush = any(window <= flush_ranks for window in STRAIGHTS)
    if straight_flush:
        value = 0.999
    elif four:
        value = 0.98
    elif triples and len(pairs) >= 2:
        value = 0.95
    elif flush_suit:
        private_flush = [RANKS[card[0]] for card in hole if card[1] == flush_suit]
        value = 0.82 + 0.007 * max(private_flush, default=0)
    elif straight:
        value = 0.89
    elif triples:
        value = 0.84 if triples[0] in private_ranks else 0.48
    elif len(pairs) >= 2:
        private_pairs = len(set(pairs[:2]) & private_ranks)
        value = 0.79 if private_pairs == 2 else 0.67 if private_pairs else 0.38
    elif pairs:
        rank = pairs[0]
        if rank in private_ranks:
            highest = max(board_ranks, default=0)
            value = 0.72 if rank > highest else 0.65 if rank == highest else 0.46
            value += 0.003 * max(private_ranks)
        else:
            value = 0.30 + 0.008 * max(private_ranks)
    else:
        value = 0.18 + 0.013 * max(private_ranks)
    if len(board) == 5:
        board_counts = {rank: board_ranks.count(rank) for rank in set(board_ranks)}
        if max(board_counts.values()) >= 3 and not (private_ranks & set(pairs)):
            value = min(value, 0.45)
        if len(set(card[1] for card in board)) == 1:
            own_flush = [RANKS[card[0]] for card in hole if card[1] == board[0][1]]
            if not own_flush or max(own_flush) <= min(board_ranks):
                value = min(value, 0.30)
        board_straight = next((index for index, window in enumerate(STRAIGHTS) if window <= set(board_ranks)), None)
        if board_straight is not None and not flush_suit:
            own_straight = next(index for index, window in enumerate(STRAIGHTS) if window <= ranks)
            if own_straight == board_straight:
                value = min(value, 0.30)
        return min(0.99, value)
    has_flush_draw = any(
        count == 4 and any(card[1] == suit for card in hole)
        for suit, count in suits.items()
    )
    straight_draw = max(len(window & ranks) for window in STRAIGHTS) == 4
    value += 0.10 if has_flush_draw else 0.0
    value += 0.055 if straight_draw else 0.0
    if len(board) == 3 and any(count == 3 for count in suits.values()):
        value += 0.018
    return min(0.99, value)


def auction_action(state):
    legal = state.can("bid")
    balance = max(0, int(legal.max_amount)) if legal else 0
    auction_cards = tuple(state.variant.get("auction_cards", ()))
    if balance == 0 or len(auction_cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))

    values = []
    for indices in FLOPS:
        board = tuple(auction_cards[index] for index in indices)
        value = board_strength(state.cards, board)
        # Shared high-card boards improve opponents too; paired low cards are less scary.
        unmatched_high = sum(RANKS[card[0]] >= 11 and card[0] not in (state.cards[0][0], state.cards[1][0]) for card in board)
        values.append(value - unmatched_high * 0.014)
    best_index = max(range(len(values)), key=values.__getitem__)
    best = FLOPS[best_index]
    gain = values[best_index] - sum(values) / len(values)
    if gain < 0.035 or values[best_index] < 0.46:
        return Action("bid", cards=(0, 0, 0, 0, 0))

    hero = next(seat for seat in state.seats if seat.seat == state.seat)
    big_blind = max(1, state.blinds.big_blind)
    # Spend faster in large pots, short stacks, and near the table cap.
    remaining_hands = max(1, state.config.hand_cap - state.hand_number)
    horizon = min(35, max(4, remaining_hands * 0.25), max(4, hero.stack / big_blind * 0.65))
    leverage = min(4.0, max(0.6, state.pot / max(big_blind * 8, hero.stack * 0.25)))
    budget = int(balance / horizon * min(2.4, gain * 6.0) * leverage)
    if hero.status == "all_in":
        budget = max(budget, int(balance * min(0.75, 0.15 + gain)))
    budget = max(1, min(balance, budget))

    # Each card's value is the drop if we remove it from our preferred flop.
    marginal = []
    for card_index in best:
        alternative = max(value for indices, value in zip(FLOPS, values) if card_index not in indices)
        marginal.append(max(0.01, values[best_index] - alternative))
    total = sum(marginal)
    bids = [0] * 5
    for card_index, weight in zip(best, marginal):
        bids[card_index] = int(budget * weight / total)
    critical = best[max(range(3), key=marginal.__getitem__)]
    bids[critical] += budget - sum(bids)
    return Action("bid", cards=tuple(bids))


def betting_action(state):
    hero = next(seat for seat in state.seats if seat.seat == state.seat)
    opponents = sum(seat.seat != state.seat and seat.status in ("active", "all_in") for seat in state.seats)
    can_raise = state.can("raise")
    can_check = state.can("check")
    can_call = state.can("call")
    big_blind = max(1, state.blinds.big_blind)
    cost = min(state.to_call, hero.stack)
    price = cost / max(1, state.pot + cost)
    preflop = not state.board
    strength = preflop_strength(state.cards) if preflop else board_strength(state.cards, state.board)
    equity = max(0.04, strength - 0.045 * max(0, opponents - 1))
    stack_blinds = hero.stack / big_blind

    if preflop:
        late = (state.seat - state.button) % len(state.seats) in (0, len(state.seats) - 1)
        threshold = 0.57 - (0.04 if late or opponents <= 2 else 0.0)
        if can_raise and strength > threshold:
            amount = hero.bet + cost + max(big_blind * 2, int(state.pot * 0.65))
            if (stack_blinds < 12 and strength > 0.55) or strength > 0.80:
                amount = can_raise.max_amount
            if cost <= big_blind * 4 or strength > 0.72:
                return Action.raise_to(max(can_raise.min_amount, min(can_raise.max_amount, amount)))
        call_threshold = 0.46 if cost <= big_blind else 0.57 if cost <= big_blind * 4 else 0.68
        if can_call and strength >= call_threshold and equity > price + 0.12:
            return Action.call()
    else:
        raise_threshold = 0.65 + 0.035 * max(0, opponents - 1)
        if can_raise and strength >= raise_threshold and (cost < state.pot * 0.65 or strength > 0.85):
            amount = hero.bet + cost + max(big_blind, int((state.pot + cost) * 0.65))
            if strength > 0.88 and hero.stack <= state.pot * 2:
                amount = can_raise.max_amount
            return Action.raise_to(max(can_raise.min_amount, min(can_raise.max_amount, amount)))
        margin = 0.10 + 0.025 * max(0, opponents - 1)
        if can_call and equity >= price + margin:
            return Action.call()
    if can_check:
        return Action.check()
    if state.can("fold"):
        return Action.fold()
    if can_call:
        return Action.call()
    first = state.legal_actions[0]
    return Action(first.kind, first.min_amount)


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_action(state), b""
        return betting_action(state), b""
