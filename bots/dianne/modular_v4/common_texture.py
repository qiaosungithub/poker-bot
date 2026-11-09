"""Boston policy valuing tournament survival and payout-sensitive call thresholds."""

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


def hand_quality(hole, board, dead=()):
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
        best_pairs = sorted((rank for rank, count in rank_counts.items() if count >= 2), reverse=True)[:2]
        private_pairs = sum(rank in best_pairs and board_counts.get(rank, 0) < 2 for rank in set(hole_ranks))
        quality = 0.84 if private_pairs == 2 else 0.68 if private_pairs else 0.30
        if private_pairs and hole_ranks[0] == hole_ranks[1]:
            # A shared board pair does not promote a pocket underpair into a
            # strong private two-pair holding. Any higher board connection wins.
            higher = sum(rank > hole_ranks[0] for rank in set(board_ranks))
            quality = (0.76, 0.56, 0.42, 0.30, 0.24)[min(higher, 4)]
    elif category == 3:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        quality = 0.90 if hole_ranks.count(trips) == 2 else 0.83 if trips in hole_ranks else 0.35
    elif category == 5:
        flush_suit = next(suit for suit in "cdhs" if sum(card[1] == suit for card in all_cards) >= 5)
        if sum(card[1] == flush_suit for card in board) >= 4:
            private_rank = max((RANKS[card[0]] for card in hole if card[1] == flush_suit), default=0)
            exposed = {RANKS[card[0]] for card in all_cards if card[1] == flush_suit}
            higher = sum(rank not in exposed for rank in range(private_rank + 1, 15))
            quality = max(0.34, 0.94 - 0.07 * higher)
    elif category == 6:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        if board_counts.get(trips, 0) >= 3:
            pair = max(rank for rank, count in rank_counts.items() if rank != trips and count >= 2)
            higher_board = sum(rank > pair for rank in set(board_ranks) if rank != trips)
            quality = max(0.48, 0.91 - 0.18 * higher_board)
    elif category == 7:
        quads = next(rank for rank, count in rank_counts.items() if count == 4)
        if board_counts.get(quads, 0) == 4:
            # On board quads, only the best kicker distinguishes players.
            kicker = max(rank for rank in ranks if rank != quads)
            unavailable = set(all_cards + tuple(dead))
            remaining = [RANKS[card[0]] for card in CARDS if card not in unavailable]
            lower = sum(rank < kicker for rank in remaining)
            equal = sum(rank == kicker for rank in remaining)
            pairs = len(remaining) * (len(remaining) - 1)
            winning = lower * (lower - 1)
            tying = (lower + equal) * (lower + equal - 1) - winning
            quality = (winning + tying * 0.5) / max(1, pairs)

    if len(board) == 5:
        board_score = eval7.evaluate([CARDS[card] for card in board])
        if encoded == board_score:
            quality = 0.48 if category >= 4 else 0.22
    else:
        suits = [card[1] for card in all_cards]
        flush_draw = max((max(0.0, 1.0 - sum(card[1] == suit for card in dead) / 9)
                          for suit in "cdhs" if suits.count(suit) == 4
                          and any(card[1] == suit for card in hole)), default=0.0)
        distinct = set(ranks)
        if 14 in distinct:
            distinct.add(1)
        board_distinct = set(board_ranks)
        if 14 in board_distinct:
            board_distinct.add(1)
        live_straight_outs = 0
        for start in range(1, 11):
            window = set(range(start, start + 5))
            present = distinct.intersection(window)
            if len(present) != 4 or not present.difference(board_distinct):
                continue
            missing = next(iter(window - present))
            missing = 14 if missing == 1 else missing
            live_straight_outs = max(live_straight_outs, 4 - sum(RANKS[card[0]] == missing for card in dead))
        four_run = max(0, live_straight_outs) / 4
        # Draws add option value, while strong made hands retain their baseline.
        improvement = 0.14 * flush_draw + 0.075 * four_run
        quality += improvement * (1.0 - quality)
    return min(0.999, quality), category


def own_icm(stacks, hero, payouts):
    """Exact own top-three ICM; higher paid places receive a chip-share approximation."""
    own = stacks[hero]
    others = [stack for seat, stack in enumerate(stacks) if seat != hero and stack > 0]
    if own <= 0:
        return payouts[len(others)] if len(others) < len(payouts) else 0.0
    total = own + sum(others)
    first = own / total
    second = 0.0
    third = 0.0
    for index, stack in enumerate(others):
        after_first = total - stack
        second += stack / total * own / after_first
        for following, other in enumerate(others):
            if following != index:
                third += stack / total * other / after_first * own / (after_first - other)
    probabilities = (first, second, third)
    value = sum(probability * payout for probability, payout in zip(probabilities, payouts))
    if len(payouts) > 3:
        remaining_places = min(len(others) + 1, len(payouts)) - 3
        if remaining_places > 0:
            value += (1.0 - first - second - third) * sum(payouts[3:3 + remaining_places]) / remaining_places
    return value


def risk_premium(state):
    own = state.seats[state.seat]
    stacks = [seat.stack + seat.bet for seat in state.seats if seat.status != "busted"]
    live = len(stacks)
    paid = sum(payout > 0 for payout in state.config.payouts)
    if live <= 2 or paid <= 1:
        return 0.0
    own_total = own.stack + own.bet
    shortest = min(stacks)
    covered = any(stack > own_total for stack in stacks)
    bubble = 1.0 if live == paid + 1 else 0.55 if live <= paid else 0.18
    cushion = max(0.0, min(1.0, 1.0 - shortest / max(1, own_total)))
    flatness = sum(state.config.payouts[1:live]) / max(1, sum(state.config.payouts[:live]))
    return min(0.18, bubble * flatness * (0.07 + 0.20 * cushion) * (1.0 if covered else 0.5))


def contestable_pot(state, cost):
    """Exclude current-street contributions above our possible all-in total.

    An acting player has matched earlier streets, so earlier collected pots
    remain eligible. Other current bets beyond our total enter side pots.
    """
    cap = state.seats[state.seat].bet + cost
    excess = sum(max(0, seat.bet - cap) for seat in state.seats)
    return max(0, state.pot - excess)


def call_threshold(state, opponents, cost, premium):
    eligible_pot = contestable_pot(state, cost)
    required = cost / max(1, eligible_pot + cost) + 0.06 + premium
    own = state.seats[state.seat]
    if len(opponents) != 1 or cost < own.stack * 0.55:
        return required
    # All committed chips already belong to the pot. Estimate a terminal call
    # against the sole remaining opponent, holding folded stacks fixed.
    other = opponents[0].seat
    stacks = [seat.stack for seat in state.seats]
    folded = stacks.copy()
    folded[other] += state.pot
    winning = stacks.copy()
    winning[state.seat] += eligible_pot
    # The sole opponent receives the uncallable side pot even when hero wins.
    winning[other] += state.pot - eligible_pot
    losing = stacks.copy()
    losing[state.seat] -= cost
    losing[other] += state.pot + cost
    fold_value = own_icm(folded, state.seat, state.config.payouts)
    win_value = own_icm(winning, state.seat, state.config.payouts)
    lose_value = own_icm(losing, state.seat, state.config.payouts)
    if win_value > lose_value:
        required = max(required, (fold_value - lose_value) / (win_value - lose_value) + 0.04)
    return min(0.97, required)


def raise_safely(state, amount):
    legal = state.can("raise")
    if legal:
        return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(amount))))
    if state.can("call"):
        return Action.call()
    if state.can("check"):
        return Action.check()
    return Action.fold()


def cheap_preflop_continue(state, opponents, cost, pot, required):
    if not state.can("call") or len(opponents) != 1:
        return False
    # Preserve the blind-remnant fix even without an earlier voluntary raise.
    if cost <= pot * 0.04 and required <= 0.10:
        return True
    if opponents[0].status != "all_in" or cost / max(1, pot + cost) > 0.09:
        return False
    if not any(event.kind == "raise" and event.seat == state.seat for event in state.history):
        return False
    # A deliberately modest tight-range estimate protects cheap continuations
    # without allowing the sunk raise to justify calling a large rereraise.
    ranks = [RANKS[card[0]] for card in state.cards]
    equity_guard = 0.18
    equity_guard += 0.03 * (state.cards[0][1] == state.cards[1][1])
    equity_guard += 0.02 * (min(ranks) >= 8)
    equity_guard += 0.03 * (ranks[0] == ranks[1])
    return required <= equity_guard


def postflop_hand_quality(hole, board, dead=()):
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
        best_pairs = sorted((rank for rank, count in rank_counts.items() if count >= 2), reverse=True)[:2]
        private_pairs = sum(rank in best_pairs and board_counts.get(rank, 0) < 2 for rank in set(hole_ranks))
        quality = 0.84 if private_pairs == 2 else 0.68 if private_pairs else 0.30
        if private_pairs and hole_ranks[0] == hole_ranks[1]:
            # A shared board pair does not promote a pocket underpair into a
            # strong private two-pair holding. Any higher board connection wins.
            higher = sum(rank > hole_ranks[0] for rank in set(board_ranks))
            quality = (0.76, 0.56, 0.42, 0.30, 0.24)[min(higher, 4)]
    elif category == 3:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        quality = 0.90 if hole_ranks.count(trips) == 2 else 0.83 if trips in hole_ranks else 0.35
    elif category == 5:
        flush_suit = next(suit for suit in "cdhs" if sum(card[1] == suit for card in all_cards) >= 5)
        if sum(card[1] == flush_suit for card in board) >= 4:
            private_rank = max((RANKS[card[0]] for card in hole if card[1] == flush_suit), default=0)
            exposed = {RANKS[card[0]] for card in all_cards if card[1] == flush_suit}
            higher = sum(rank not in exposed for rank in range(private_rank + 1, 15))
            quality = max(0.34, 0.94 - 0.07 * higher)
    elif category == 6:
        trips = max(rank for rank, count in rank_counts.items() if count >= 3)
        if board_counts.get(trips, 0) >= 3:
            pair = max(rank for rank, count in rank_counts.items() if rank != trips and count >= 2)
            higher_board = sum(rank > pair for rank in set(board_ranks) if rank != trips)
            quality = max(0.48, 0.91 - 0.18 * higher_board)
    elif category == 7:
        quads = next(rank for rank, count in rank_counts.items() if count == 4)
        if board_counts.get(quads, 0) == 4:
            # On board quads, only the best kicker distinguishes players.
            kicker = max(rank for rank in ranks if rank != quads)
            unavailable = set(all_cards + tuple(dead))
            remaining = [RANKS[card[0]] for card in CARDS if card not in unavailable]
            lower = sum(rank < kicker for rank in remaining)
            equal = sum(rank == kicker for rank in remaining)
            pairs = len(remaining) * (len(remaining) - 1)
            winning = lower * (lower - 1)
            tying = (lower + equal) * (lower + equal - 1) - winning
            quality = (winning + tying * 0.5) / max(1, pairs)

    if len(board) == 5:
        board_score = eval7.evaluate([CARDS[card] for card in board])
        if encoded == board_score:
            quality = 0.48 if category >= 4 else 0.22
    else:
        suits = [card[1] for card in all_cards]
        flush_draw = max((max(0.0, 1.0 - sum(card[1] == suit for card in dead) / 9)
                          for suit in "cdhs" if suits.count(suit) == 4
                          and any(card[1] == suit for card in hole)), default=0.0)
        distinct = set(ranks)
        if 14 in distinct:
            distinct.add(1)
        board_distinct = set(board_ranks)
        if 14 in board_distinct:
            board_distinct.add(1)
        live_straight_outs = 0
        for start in range(1, 11):
            window = set(range(start, start + 5))
            present = distinct.intersection(window)
            if len(present) != 4 or not present.difference(board_distinct):
                continue
            missing = next(iter(window - present))
            missing = 14 if missing == 1 else missing
            live_straight_outs = max(live_straight_outs, 4 - sum(RANKS[card[0]] == missing for card in dead))
        four_run = max(0, live_straight_outs) / 4
        # Draws add option value, while strong made hands retain their baseline.
        improvement = 0.14 * flush_draw + 0.075 * four_run
        quality += improvement * (1.0 - quality)
    return min(0.999, quality), category
