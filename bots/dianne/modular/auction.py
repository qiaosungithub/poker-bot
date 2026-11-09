"""Boston flop-chips allocation phase."""

from .common import Action, FLOPS, hand_quality


def auction_decision(state, memory):
    budget = state.can("bid").max_amount
    cards = tuple(state.variant.get("auction_cards", ()))
    if budget == 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0)), b""
    options = [(hand_quality(tuple(state.cards), tuple(cards[index] for index in combo))[0], combo)
               for combo in FLOPS]
    quality, target = max(options)
    gain = quality - options[0][0]
    if gain < 0.06:
        return Action("bid", cards=(0, 0, 0, 0, 0)), b""
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
    return Action("bid", cards=tuple(bids)), b""
