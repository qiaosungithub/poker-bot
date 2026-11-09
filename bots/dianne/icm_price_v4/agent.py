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


def auction_bid(state):
    budget = state.can("bid").max_amount
    cards = tuple(state.variant.get("auction_cards", ()))
    if budget == 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    options = [(hand_quality(tuple(state.cards), tuple(cards[index] for index in combo))[0], combo)
               for combo in FLOPS]
    quality, target = max(options)
    gain = quality - options[0][0]
    if gain < 0.06:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    own = state.seats[state.seat]
    committed = state.pot / max(1, state.pot + own.stack)
    future_hands = max(6, min(45, state.config.hand_cap - state.hand_number))
    opportunity = min(5.0, gain * 10) * (1.0 + 4.0 * committed)
    spend = int(budget / future_hands * opportunity)
    if own.status == "all_in":
        spend = max(spend, int(budget * min(0.70, gain * 1.8)))
    elif own.stack < 8 * state.blinds.big_blind and quality > 0.75:
        spend = max(spend, int(budget * min(0.30, gain)))
    spend = min(budget, max(3, spend))
    bids = [0] * 5
    quotient, remainder = divmod(spend, 3)
    for offset, index in enumerate(target):
        bids[index] = quotient + int(offset < remainder)
    return Action("bid", cards=tuple(bids))


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


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_bid(state), b""
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), b""
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

        if not state.board:
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


