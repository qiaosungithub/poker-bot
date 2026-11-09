"""Boston shadow-price auction policy with contestable-pot and hand-rank safeguards."""

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


def auction_bid(state):
    """Buy useful changes only when their current value beats future bidding value."""
    legal = state.can("bid")
    balance = legal.max_amount if legal else 0
    cards = tuple(state.variant.get("auction_cards", ()))
    if balance <= 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    values = {combo: hand_quality(tuple(state.cards), tuple(cards[i] for i in combo))[0]
              for combo in FLOPS}
    own = state.seats[state.seat]
    big_blind = max(1, state.blinds.big_blind)
    balances = state.variant.get("flop_chips_by_seat", ())
    opponents = [seat for seat in state.seats
                 if seat.seat != state.seat and seat.status in ("active", "all_in")]
    # An anticipated number of meaningful future auction opportunities.
    future = max(3.0, min(30.0, (state.config.hand_cap - state.hand_number) * 0.17,
                           (own.stack + own.bet) / big_blind * 0.8 + 3))
    leverage = min(8.0, max(0.4, state.pot / (big_blind * 10)))
    if own.status == "all_in":
        future = 1.5
        leverage = max(4.0, leverage)
    opponent_supply = sum(balances[seat.seat] if seat.seat < len(balances) else 1000
                          for seat in opponents)
    # All players can prefer a public high card. This is a broad prior, never
    # an inference about their hidden bids in the current auction.
    pressure = opponent_supply / max(1.0, len(opponents) * 70.0)
    pressure *= min(2.0, max(0.5, len(opponents) / 3))
    prior = [pressure * (0.50 + (RANKS[card[0]] - 2) / 18) for card in cards]
    scenarios = [[round(value * scale) for value in prior] for scale in (0.0, 0.7, 1.8)]
    weights = (0.25, 0.45, 0.30)
    def selected(scores):
        return tuple(sorted(sorted(range(5), key=lambda i: (-scores[i], i))[:3]))
    free_values = [values[selected(scores)] for scores in scenarios]
    max_spend = min(balance, max(3, int(balance / future * min(6.0, leverage + 0.8))))
    shadow_price = 0.18 * future / max(1, balance)
    best_utility = 0.0
    best_bids = (0, 0, 0, 0, 0)
    for target in FLOPS:
        outside = [i for i in range(5) if i not in target]
        for predicted in scenarios:
            strongest_outside = max(outside, key=lambda i: (predicted[i], -i))
            threshold = predicted[strongest_outside]
            bids = [0] * 5
            for i in target:
                # With equal totals the earlier auction card wins.
                bids[i] = max(0, threshold - predicted[i] + (i > strongest_outside))
            cost = sum(bids)
            if not cost or cost > max_spend:
                continue
            utility = 0.0
            for scores, weight, free_value in zip(scenarios, weights, free_values):
                chosen = selected([score + bid for score, bid in zip(scores, bids)])
                paid = sum(bids[i] for i in chosen)
                utility += weight * ((values[chosen] - free_value) * leverage
                                     - paid * shadow_price)
            if utility > best_utility:
                best_utility = utility
                best_bids = tuple(bids)
    return Action("bid", cards=best_bids)


def raise_safely(state, target):
    legal = state.can("raise")
    if legal:
        return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(target))))
    return Action.call() if state.can("call") else Action.check()


def contestable_pot(state, cost):
    """Exclude current-street contributions above our possible all-in total.

    An acting player has matched earlier streets, so earlier collected pots
    remain eligible. Other current bets beyond our total enter side pots.
    """
    cap = state.seats[state.seat].bet + cost
    excess = sum(max(0, seat.bet - cap) for seat in state.seats)
    return max(0, state.pot - excess)


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
        eligible_pot = contestable_pot(state, cost)
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
                size = max(big_blind * 3, own.bet + cost + max(big_blind * 2, eligible_pot * 0.75))
                return raise_safely(state, size), b""
            if not raised and quality >= open_threshold and raise_action:
                return raise_safely(state, big_blind * 2.5), b""
            call_threshold = 0.60 + min(0.20, cost / big_blind * 0.015)
            if state.can("call") and quality >= call_threshold and cost <= own.stack * 0.22:
                return Action.call(), b""
            return fold_or_check, b""

        quality, category = hand_quality(tuple(state.cards), tuple(state.board))
        quality *= 0.91 ** max(0, opponents - 1)
        pot_odds = cost / max(1, eligible_pot + cost)
        pressure = cost / max(1, own.stack)
        value_threshold = 0.71 + 0.025 * max(0, opponents - 1)
        if raise_action and quality >= value_threshold:
            target = own.bet + cost + max(big_blind, int((eligible_pot + cost) * 0.65))
            if quality >= 0.88 or own.stack <= eligible_pot * 0.8 and quality >= 0.73:
                target = own.bet + own.stack
            # Avoid repeated reraising weak made hands on the same street.
            if cost <= eligible_pot * 0.6 or category >= 3:
                return raise_safely(state, target), b""
        if check:
            return Action.check(), b""
        required = pot_odds + 0.12 + pressure * 0.15
        if state.can("call") and quality >= required:
            if pressure < 0.50 or quality >= 0.67 or category >= 2:
                return Action.call(), b""
        return fold_or_check, b""
