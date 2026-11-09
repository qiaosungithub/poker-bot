"""Boston value policy with compact per-table opponent and auction adaptation."""

from itertools import combinations
import struct

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action


RANKS = {rank: index + 2 for index, rank in enumerate("23456789TJQKA")}
CARDS = {rank + suit: eval7.Card(rank + suit)
         for rank in RANKS for suit in "cdhs"}
FLOPS = tuple(combinations(range(5), 3))
MEMORY = struct.Struct("<4sII" + "HHHHHHI" * 10)
ACTION_COLUMNS = {"fold": 0, "call": 1, "check": 2, "raise": 3, "bid": 5}


def observe(state, memory):
    """Consume delivered events once, keeping all learned information in bytes."""
    rows = [[0, 0, 0, 0, 1000, 0, 0] for _ in range(10)]
    previous_hand, previous_length = 0, 0
    if len(memory) == MEMORY.size:
        values = MEMORY.unpack(memory)
        if values[0] == b"ADP1":
            previous_hand, previous_length = values[1:3]
            rows = [list(values[3 + index * 7:10 + index * 7]) for index in range(10)]
    cursor = (state.hand_number, len(state.history))
    if cursor > (previous_hand, previous_length):
        # Ordinary delivered batches are short. Bound recovery work after a
        # lost/truncated memory rather than rescanning an entire table.
        for event in state.events[-256:]:
            column = ACTION_COLUMNS.get(event.kind)
            if column is not None and event.seat is not None and 0 <= event.seat < 10:
                row = rows[event.seat]
                row[column] = min(65535, row[column] + 1)
    balances = state.variant.get("flop_chips_by_seat", ())
    for seat, balance in enumerate(balances[:10]):
        row = rows[seat]
        row[6] += max(0, row[4] - balance)
        row[4] = balance
    encoded = MEMORY.pack(b"ADP1", cursor[0], cursor[1], *(value for row in rows for value in row))
    return rows, encoded


def opponent_profile(state, rows):
    opponents = [seat.seat for seat in state.seats
                 if seat.seat != state.seat and seat.status in ("active", "all_in")]
    last_raiser = None
    for event in reversed(state.history):
        if event.kind in ("board", "hand"):
            break
        if event.kind == "raise" and event.seat != state.seat:
            last_raiser = event.seat
            break
    selected = [last_raiser] if last_raiser is not None else opponents
    aggression = sum((rows[seat][3] + 2) / (sum(rows[seat][:4]) + 10)
                     for seat in selected) / max(1, len(selected))
    folding = sum((rows[seat][0] + 3) / (sum(rows[seat][:4]) + 10)
                  for seat in opponents) / max(1, len(opponents))
    return aggression, folding


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


def auction_bid(state, rows):
    legal = state.can("bid")
    budget = legal.max_amount if legal else 0
    cards = tuple(state.variant.get("auction_cards", ()))
    if budget <= 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    hole = tuple(state.cards)
    options = [(hand_quality(hole, tuple(cards[index] for index in combo))[0], combo)
               for combo in FLOPS]
    best_quality, best = max(options)
    # The free default is a useful reference only against very quiet bidders.
    # Mix it with the mean board to account for opponents selecting other cards.
    opponents = [seat.seat for seat in state.seats
                 if seat.seat != state.seat and seat.status in ("active", "all_in")]
    expected_spending = sum((rows[seat][6] + 6) / (rows[seat][5] + 3)
                            for seat in opponents if rows[seat][4] > 0)
    contest = min(0.8, expected_spending / 25)
    reference = options[0][0] * (1 - contest) + sum(value for value, _ in options) / 10 * contest
    gain = best_quality - reference
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
    # Observed total spend bounds a plausible clearing price; it never reveals
    # how a rival split bids. Commit strongly only on high-gain boards.
    target_price = max(1, int(expected_spending / 3) + 1)
    if gain > 0.24:
        spend = max(spend, min(budget // 6, target_price * 3))
    if expected_spending < 2 and own.status != "all_in":
        spend = min(spend, 3)
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
        rows, next_memory = observe(state, memory)
        if state.phase == "auction":
            return auction_bid(state, rows), next_memory
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), next_memory

        own = state.seats[state.seat]
        big_blind = max(1, state.blinds.big_blind)
        opponents = sum(seat.status in ("active", "all_in") and seat.seat != state.seat
                        for seat in state.seats)
        cost = min(state.to_call, own.stack)
        check = state.can("check")
        fold_or_check = Action.check() if check else Action.fold()
        raise_action = state.can("raise")
        aggression, folding = opponent_profile(state, rows)

        if not state.board:
            quality = preflop_quality(state.cards)
            short = (own.stack + own.bet) / big_blind
            late = state.seat == state.button
            open_threshold = 0.60 - 0.025 * max(0, 5 - state.players_remaining) - 0.04 * late
            open_threshold -= min(0.08, max(0, folding - 0.38) * 0.25)
            raised = any(event.kind == "raise" for event in state.history)
            if short <= 10 and quality >= 0.57 + 0.02 * min(opponents, 5):
                return raise_safely(state, own.stack + own.bet), next_memory
            if quality >= 0.82:
                size = max(big_blind * 3, own.bet + cost + max(big_blind * 2, state.pot * 0.75))
                return raise_safely(state, size), next_memory
            if not raised and quality >= open_threshold and raise_action:
                return raise_safely(state, big_blind * 2.5), next_memory
            call_threshold = 0.60 + min(0.20, cost / big_blind * 0.015)
            call_threshold += max(-0.05, min(0.05, (0.20 - aggression) * 0.30))
            if state.can("call") and quality >= call_threshold and cost <= own.stack * 0.22:
                return Action.call(), next_memory
            return fold_or_check, next_memory

        quality, category = hand_quality(tuple(state.cards), tuple(state.board))
        quality *= 0.91 ** max(0, opponents - 1)
        pot_odds = cost / max(1, state.pot + cost)
        pressure = cost / max(1, own.stack)
        value_threshold = 0.71 + 0.025 * max(0, opponents - 1)
        value_threshold -= min(0.04, max(0, aggression - 0.30) * 0.15)
        if raise_action and quality >= value_threshold:
            target = own.bet + cost + max(big_blind, int((state.pot + cost) * 0.65))
            if quality >= 0.88 or own.stack <= state.pot * 0.8 and quality >= 0.73:
                target = own.bet + own.stack
            # Avoid repeated reraising weak made hands on the same street.
            if cost <= state.pot * 0.6 or category >= 3:
                return raise_safely(state, target), next_memory
        # Small steals only against an observed folding tendency, with a
        # deterministic board/seat mixture and at most one prior street bet.
        mixture = (state.hand_number * 13 + state.seat * 7 + RANKS[state.cards[0][0]]) % 11
        already_raised = any(event.kind == "raise" and event.seat == state.seat
                             for event in state.history)
        if check and raise_action and opponents == 1 and folding > 0.50 and mixture < 3 and not already_raised:
            return raise_safely(state, own.bet + max(big_blind, int(state.pot * 0.4))), next_memory
        if check:
            return Action.check(), next_memory
        required = pot_odds + 0.09 + pressure * 0.12
        required += max(-0.07, min(0.07, (0.20 - aggression) * 0.40))
        if state.can("call") and quality >= required:
            if pressure < 0.50 or quality >= 0.67 or category >= 2:
                return Action.call(), next_memory
        return fold_or_check, next_memory
