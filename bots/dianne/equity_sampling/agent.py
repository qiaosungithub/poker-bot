"""Bounded equity sampling and marginal-value flop-auction bidding."""

import itertools
import random

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action


CARD_NAMES = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
CARDS = {name: eval7.Card(name) for name in CARD_NAMES}
CARD_INDEX = {name: index for index, name in enumerate(CARD_NAMES)}
FLOPS = tuple(itertools.combinations(range(5), 3))


def preflop_strength(cards):
    first, second = cards
    high, low = sorted((CARD_INDEX[first] // 4 + 2, CARD_INDEX[second] // 4 + 2), reverse=True)
    if high == low:
        return 0.48 + high * 0.027
    strength = 0.31 + high * 0.018 + low * 0.008
    if first[1] == second[1]:
        strength += 0.035
    gap = high - low
    if gap <= 2:
        strength += 0.025
    if high == 14:
        strength += 0.015
    return min(0.82, strength)


def state_random(state):
    seed = state.hand_number * 100003 + state.seat * 131 + state.pot * 17
    for card in state.cards + state.board:
        seed = seed * 53 + CARD_INDEX[card]
    return random.Random(seed)


def auction_bid(state):
    legal = state.can("bid")
    balance = legal.max_amount
    auction = tuple(state.variant.get("auction_cards", ()))
    if balance <= 0 or len(auction) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    known = set(state.cards + auction)
    deck = [CARDS[name] for name in CARD_NAMES if name not in known]
    hole = [CARDS[name] for name in state.cards]
    auction_objects = [CARDS[name] for name in auction]
    rng = state_random(state)
    # Common samples reduce noise when comparing the ten possible flops.
    samples = [rng.sample(deck, 4) for _ in range(12)]
    values = []
    for indexes in FLOPS:
        flop = [auction_objects[index] for index in indexes]
        score = 0.0
        for sample in samples:
            board = flop + sample[:2]
            own = eval7.evaluate(hole + board)
            opponent = eval7.evaluate(sample[2:] + board)
            score += (own > opponent) + 0.5 * (own == opponent)
        values.append(score / len(samples))
    best = max(range(10), key=lambda index: values[index])
    spread = values[best] - min(values)
    if spread < 0.08:
        return Action("bid", cards=(0, 0, 0, 0, 0))
    selected = FLOPS[best]
    remaining_hands = max(12, min(45, (state.config.hand_cap - state.hand_number) // 3))
    ordinary_budget = balance / remaining_hands
    pot_factor = min(4.0, max(0.4, state.pot / max(1, state.blinds.big_blind * 8)))
    budget = min(balance, max(3, int(ordinary_budget * pot_factor * min(2.0, spread * 4))))
    own_seat = next(seat for seat in state.seats if seat.seat == state.seat)
    if own_seat.stack == 0:
        budget = min(balance, max(budget, int(balance * min(0.6, spread))))
    # Average marginal value estimates which selected cards need protection.
    weights = []
    for card_index in selected:
        included = sum(value for flop, value in zip(FLOPS, values) if card_index in flop) / 6
        excluded = sum(value for flop, value in zip(FLOPS, values) if card_index not in flop) / 4
        weights.append(max(0.025, included - excluded))
    weight_sum = sum(weights)
    bids = [0] * 5
    for card_index, weight in zip(selected, weights):
        bids[card_index] = int(budget * weight / weight_sum)
    bids[selected[weights.index(max(weights))]] += budget - sum(bids)
    return Action("bid", cards=tuple(bids))


def showdown_equity(state, opponents):
    hole = [CARDS[name] for name in state.cards]
    board = [CARDS[name] for name in state.board]
    auction_cards = tuple(state.variant.get("auction_cards", ()))
    # All five auction cards are unavailable after the auction, including the
    # two rejected cards that are absent from state.board.
    known = set(state.cards + state.board + auction_cards)
    deck = [CARDS[name] for name in CARD_NAMES if name not in known]
    rng = state_random(state)
    missing = 5 - len(board)
    samples = 32
    wins = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, missing + 2 * opponents)
        final_board = board + draw[:missing]
        own = eval7.evaluate(hole + final_board)
        tied = 1
        beaten = False
        for index in range(opponents):
            offset = missing + index * 2
            other = eval7.evaluate(draw[offset:offset + 2] + final_board)
            if other > own:
                beaten = True
                break
            if other == own:
                tied += 1
        if not beaten:
            wins += 1.0 / tied
    return wins / samples


def betting_action(state):
    check = state.can("check")
    call = state.can("call")
    raise_action = state.can("raise")
    own_seat = next(seat for seat in state.seats if seat.seat == state.seat)
    opponents = sum(seat.seat != state.seat and seat.status in ("active", "all_in") for seat in state.seats)
    opponents = max(1, opponents)
    big_blind = max(1, state.blinds.big_blind)
    pot_odds = state.to_call / max(1, state.pot + state.to_call)
    if not state.board:
        strength = preflop_strength(state.cards)
        seats_to_button = (state.button - state.seat) % len(state.seats)
        late = seats_to_button <= 2
        opening_threshold = 0.60 if late or opponents <= 2 else 0.65
        stack_bb = own_seat.stack / big_blind
        facing_raise = state.to_call > big_blind
        call_threshold = 0.62 + min(0.16, state.to_call / max(big_blind * 30, own_seat.stack) * 0.20)
        if raise_action and strength >= opening_threshold + (0.08 if facing_raise else 0):
            if (stack_bb <= 11 and strength >= 0.64) or strength >= 0.83:
                target = raise_action.max_amount
            else:
                target = max(big_blind * 3, own_seat.bet + state.to_call + int(state.pot * 0.7))
            return Action.raise_to(min(raise_action.max_amount, max(raise_action.min_amount, target)))
        if check:
            return Action.check()
        if call and strength >= call_threshold and pot_odds < strength - 0.18:
            return Action.call()
        if call and not facing_raise and strength >= opening_threshold - 0.06 and pot_odds < 0.20:
            return Action.call()
        return Action.fold() if state.can("fold") else Action.call()

    equity = showdown_equity(state, opponents)
    # Uniform-card equity is optimistic against opponents who make large bets.
    risk_margin = 0.055 + min(0.10, state.to_call / max(1, state.pot) * 0.12)
    risk_margin += 0.015 * min(4, opponents - 1)
    positive_payouts = sum(payout > 0 for payout in state.config.payouts)
    if state.players_remaining <= positive_payouts + 1 and own_seat.stack > big_blind * 8:
        risk_margin += 0.045
    raise_threshold = 0.64 if opponents == 1 else 0.52
    if raise_action and equity >= raise_threshold and equity > pot_odds + risk_margin + 0.13:
        fraction = 0.70 if equity > 0.83 else 0.48
        target = own_seat.bet + state.to_call + int((state.pot + state.to_call) * fraction)
        if equity > 0.91 or (own_seat.stack < state.pot * 0.60 and equity > 0.69):
            target = raise_action.max_amount
        return Action.raise_to(min(raise_action.max_amount, max(raise_action.min_amount, target)))
    if check:
        return Action.check()
    if call and equity > pot_odds + risk_margin:
        return Action.call()
    return Action.fold() if state.can("fold") else Action.call()


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_bid(state), b""
        return betting_action(state), b""
