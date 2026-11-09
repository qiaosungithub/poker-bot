"""Boston pressure policy using short-stack push/fold and survival auction spending."""

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
    balance = legal.max_amount if legal else 0
    cards = tuple(state.variant.get("auction_cards", ()))
    if balance <= 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    values = [board_strength(state.cards, tuple(cards[index] for index in flop))
              for flop in FLOPS]
    best_index = max(range(10), key=values.__getitem__)
    best_value = values[best_index]
    best_flop = FLOPS[best_index]
    improvement = best_value - sum(values) / 10
    if improvement < 0.045 or best_value < 0.51:
        return Action("bid", cards=(0, 0, 0, 0, 0))

    hero = next(seat for seat in state.seats if seat.seat == state.seat)
    blind = max(1, state.blinds.big_blind)
    horizon = max(3, min(30, (state.config.hand_cap - state.hand_number) / 4,
                         (hero.stack + hero.bet) / blind))
    leverage = min(4.0, max(0.6, state.pot / max(blind * 6, hero.stack * 0.3)))
    spend = int(balance / horizon * min(2.5, improvement * 7) * leverage)
    if hero.status == "all_in":
        spend = max(spend, int(balance * min(0.85, 0.18 + improvement)))
    spend = min(balance, max(1, spend))

    weights = []
    for index in best_flop:
        without = max(value for flop, value in zip(FLOPS, values) if index not in flop)
        weights.append(max(0.04, best_value - without))
    bids = [0] * 5
    total_weight = sum(weights)
    for index, weight in zip(best_flop, weights):
        bids[index] = int(spend * weight / total_weight)
    bids[best_flop[weights.index(max(weights))]] += spend - sum(bids)
    return Action("bid", cards=tuple(bids))


def safe_raise(legal, target):
    return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(target))))


def betting_action(state):
    hero = next(seat for seat in state.seats if seat.seat == state.seat)
    active = [seat for seat in state.seats
              if seat.seat != state.seat and seat.status in ("active", "all_in")]
    opponents = len(active)
    blind = max(1, state.blinds.big_blind)
    cost = min(state.to_call, hero.stack)
    stack_bb = (hero.stack + hero.bet) / blind
    price = cost / max(1, state.pot + cost)
    legal_raise = state.can("raise")
    legal_call = state.can("call")
    legal_check = state.can("check")
    fallback = Action.check() if legal_check else Action.fold()
    paid_places = sum(payout > 0 for payout in state.config.payouts)
    shorter = sum(0 < seat.stack < hero.stack * 0.45 for seat in state.seats)
    bubble = state.players_remaining == paid_places + 1 and shorter > 0 and stack_bb > 6
    risk = 0.055 if bubble else 0.0

    if not state.board:
        strength = preflop_strength(state.cards)
        late = state.seat in (state.button, (state.button - 1) % len(state.seats))
        unopened = not any(event.kind == "raise" for event in state.history)
        jam_threshold = 0.54 + 0.018 * min(opponents, 6) + risk
        jam_threshold += 0.025 if stack_bb > 8 else 0.0
        jam_threshold -= 0.035 if late or opponents <= 2 else 0.0
        if not unopened:
            jam_threshold += 0.065
        if stack_bb <= 14 and strength >= jam_threshold:
            if legal_raise:
                return safe_raise(legal_raise, legal_raise.max_amount)
            if legal_call and strength > price + 0.15 + risk:
                return Action.call()
        if legal_raise:
            if strength >= 0.80 and (cost > blind * 8 or stack_bb < 25):
                return safe_raise(legal_raise, legal_raise.max_amount)
            threshold = 0.565 + risk - (0.045 if late or opponents <= 2 else 0.0)
            if unopened and strength >= threshold:
                return safe_raise(legal_raise, blind * (2.5 if opponents > 3 else 2.2))
            if strength >= 0.735 + risk and cost <= hero.stack * 0.20:
                return safe_raise(legal_raise, hero.bet + cost + max(3 * blind, state.pot))
        # Calling all-ins needs a tighter range than opening all-in.
        threshold = 0.66 + risk if cost > hero.stack * 0.35 else 0.58 + risk
        if cost > blind * 8:
            threshold += 0.065
        if legal_call and strength >= threshold and strength > price + 0.17 + risk:
            return Action.call()
        if legal_call and cost <= blind and price < 0.18 and strength >= 0.48:
            return Action.call()
        return fallback

    strength = board_strength(state.cards, state.board)
    equity = max(0.05, strength - 0.045 * max(0, opponents - 1))
    spr = hero.stack / max(1, state.pot)
    if legal_raise:
        jam_value = 0.73 + risk + 0.035 * max(0, opponents - 1)
        if equity > 0.92 or (spr <= 1.4 and equity >= jam_value):
            return safe_raise(legal_raise, legal_raise.max_amount)
        if equity >= 0.72 + risk and (cost < state.pot * 0.5 or equity >= 0.87):
            return safe_raise(legal_raise, hero.bet + cost + (state.pot + cost) * 0.7)
        # Cheap heads-up stabs capture uncontested pots with a modest made hand.
        if legal_check and opponents == 1 and equity > 0.56 and spr > 1.5:
            return safe_raise(legal_raise, hero.bet + max(blind, state.pot * 0.33))
    if legal_check:
        return Action.check()
    margin = 0.10 + risk + 0.025 * max(0, opponents - 1)
    large_call = cost >= hero.stack * 0.45
    if legal_call and equity > price + margin and (not large_call or equity >= 0.68 + risk):
        return Action.call()
    return fallback


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_action(state), b""
        return betting_action(state), b""


