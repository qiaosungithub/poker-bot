"""Original range-weighted sampling and position-pressure policy for Boston."""

from itertools import combinations
import random

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action


NAMES = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
CARDS = {name: eval7.Card(name) for name in NAMES}
INDEX = {name: i for i, name in enumerate(NAMES)}
FLOPS = tuple(combinations(range(5), 3))


def quality(first, second):
    high, low = sorted((INDEX[first] // 4 + 2, INDEX[second] // 4 + 2), reverse=True)
    if high == low:
        return 0.49 + high * 0.027
    value = 0.24 + high * 0.019 + low * 0.010
    value += 0.045 * (first[1] == second[1])
    value += 0.035 * (high - low <= 2)
    value -= 0.025 * (high - low >= 5)
    return min(0.85, value)


def state_seed(state):
    value = state.hand_number * 10007 + state.seat * 13 + state.pot * 17
    for card in state.cards + state.board:
        value = (value * 53 + INDEX[card]) & 0xFFFFFFFF
    return value


def preflop_raises(state):
    count = 0
    for event in state.history:
        if event.kind in ("auction", "board"):
            break
        count += event.kind == "raise" and event.seat != state.seat
    return count


def current_raises(state):
    count = 0
    for event in reversed(state.history):
        if event.kind in ("board", "auction"):
            break
        count += event.kind == "raise" and event.seat != state.seat
    return count


def range_weight(first, second, raised):
    strength = quality(first, second)
    if raised:
        threshold = 0.62 + min(0.08, (raised - 1) * 0.04)
        return 1.0 if strength >= threshold else 0.12
    return 0.3 + strength


def equity(state):
    known = set(state.cards + state.board + tuple(state.variant.get("auction_cards", ())))
    deck = [name for name in NAMES if name not in known]
    hole = [CARDS[name] for name in state.cards]
    board = [CARDS[name] for name in state.board]
    missing = 5 - len(board)
    rng = random.Random(state_seed(state))
    pre_raises = preflop_raises(state)
    facing = current_raises(state)
    wins = total = 0.0
    for _ in range(28):
        draw = rng.sample(deck, missing + 2)
        other = draw[-2:]
        full_board = board + [CARDS[name] for name in draw[:missing]]
        other_cards = [CARDS[name] for name in other]
        own_rank = eval7.evaluate(hole + full_board)
        other_rank = eval7.evaluate(other_cards + full_board)
        weight = range_weight(other[0], other[1], pre_raises)
        if facing:
            current = eval7.evaluate(other_cards + board) >> 24
            weight *= 1.0 if current >= 1 else 0.25
        total += weight
        wins += weight * ((own_rank > other_rank) + 0.5 * (own_rank == other_rank))
    opponents = sum(seat.seat != state.seat and seat.status in ("active", "all_in")
                    for seat in state.seats)
    return (wins / total) ** max(1.0, opponents * 0.85)


def auction(state):
    legal = state.can("bid")
    balance = legal.max_amount if legal else 0
    cards = tuple(state.variant.get("auction_cards", ()))
    if balance <= 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    known = set(state.cards + cards)
    deck = [name for name in NAMES if name not in known]
    rng = random.Random(state_seed(state))
    hole = [CARDS[name] for name in state.cards]
    raised = preflop_raises(state)
    draws = [rng.sample(deck, 4) for _ in range(10)]
    samples = [([CARDS[name] for name in draw[:2]], [CARDS[name] for name in draw[2:]],
                range_weight(draw[2], draw[3], raised)) for draw in draws]
    total = sum(sample[2] for sample in samples)
    values = []
    for indexes in FLOPS:
        flop = [CARDS[cards[index]] for index in indexes]
        value = 0.0
        for runout, other, weight in samples:
            board = flop + runout
            own_rank = eval7.evaluate(hole + board)
            other_rank = eval7.evaluate(other + board)
            value += weight * ((own_rank > other_rank) + 0.5 * (own_rank == other_rank))
        values.append(value / total)
    best_index = max(range(10), key=values.__getitem__)
    gain = values[best_index] - values[0]
    if gain < 0.10:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    own = state.seats[state.seat]
    leverage = min(4.0, max(0.5, state.pot / max(1, state.blinds.big_blind * 6)))
    horizon = max(12, min(40, state.config.hand_cap - state.hand_number))
    spend = min(balance, max(3, int(balance / horizon * leverage * min(3.0, gain * 6))))
    if own.status == "all_in":
        spend = min(balance, max(spend, int(balance * min(0.7, gain * 1.3))))
    selected = FLOPS[best_index]
    contributions = []
    for index in selected:
        with_card = sum(value for combo, value in zip(FLOPS, values) if index in combo) / 6
        without = sum(value for combo, value in zip(FLOPS, values) if index not in combo) / 4
        contributions.append(max(0.03, with_card - without))
    bids = [0] * 5
    weight_sum = sum(contributions)
    for index, weight in zip(selected, contributions):
        bids[index] = int(spend * weight / weight_sum)
    bids[selected[contributions.index(max(contributions))]] += spend - sum(bids)
    return Action("bid", cards=tuple(bids))


def safe_raise(state, target):
    legal = state.can("raise")
    if legal:
        return Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(target))))
    return Action.check() if state.can("check") else Action.call()


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction(state), b""
        own = state.seats[state.seat]
        check = state.can("check")
        passive = Action.check() if check else Action.fold()
        cost = min(state.to_call, own.stack)
        blind = max(1, state.blinds.big_blind)
        alive = [seat.seat for seat in state.seats if seat.status != "busted"]
        after = sum(0 < (seat - state.seat) % len(state.seats) <=
                    (state.button - state.seat) % len(state.seats) for seat in alive)
        late = after <= 1
        mixed = (state_seed(state) * 2654435761 & 0xFFFFFFFF) / 2**32
        opponents = sum(seat.seat != state.seat and seat.status in ("active", "all_in")
                        for seat in state.seats)
        odds = cost / max(1, state.pot + cost)
        if not state.board:
            strength = quality(*state.cards)
            raises = preflop_raises(state)
            short = (own.stack + own.bet) / blind
            open_threshold = 0.65 - 0.055 * late - 0.035 * (len(alive) <= 3)
            if raises:
                threshold = 0.69 + min(0.12, cost / blind * 0.006)
                if strength >= 0.85 or short <= 10 and strength >= threshold:
                    return safe_raise(state, own.stack + own.bet), b""
                if state.can("call") and strength >= threshold and odds < 0.31:
                    return Action.call(), b""
            elif strength >= open_threshold or late and strength >= open_threshold - 0.045 and mixed < 0.25:
                target = own.stack + own.bet if short <= 12 else int(2.3 * blind)
                return safe_raise(state, target), b""
            return passive, b""

        chance = equity(state)
        payout_count = sum(payout > 0 for payout in state.config.payouts)
        bubble = state.players_remaining == payout_count + 1 and own.stack > 7 * blind
        margin = 0.055 + 0.055 * bubble + 0.07 * (cost / max(1, own.stack))
        value_threshold = 0.68 if opponents <= 1 else 0.57
        if state.can("raise") and chance >= value_threshold and chance > odds + margin + 0.12:
            fraction = 0.70 if chance > 0.86 else 0.48
            return safe_raise(state, own.bet + cost + (state.pot + cost) * fraction), b""
        if check:
            # Small, sparse pressure bets when last to act and checked to.
            if state.can("raise") and late and opponents <= 2 and current_raises(state) == 0:
                board_suits = [card[1] for card in state.board]
                dry = max(board_suits.count(suit) for suit in "cdhs") < 3
                if dry and 0.23 < chance < 0.50 and mixed < 0.20:
                    return safe_raise(state, own.bet + max(blind, state.pot * 0.33)), b""
            return Action.check(), b""
        if state.can("call") and chance > odds + margin:
            return Action.call(), b""
        return passive, b""
