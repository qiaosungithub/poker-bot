"""Defend valuable default flops while preserving existing positive bids."""

from .common import Action, FLOPS, postflop_hand_quality


def auction_decision(state, memory):
    budget = state.can("bid").max_amount
    cards = tuple(state.variant.get("auction_cards", ()))
    if budget == 0 or len(cards) != 5:
        return Action("bid", cards=(0, 0, 0, 0, 0)), b""
    options = [(postflop_hand_quality(tuple(state.cards), tuple(cards[index] for index in combo),
                            tuple(cards[index] for index in range(5) if index not in combo))[0], combo)
               for combo in FLOPS]
    quality, target = max(options)
    gain = quality - options[0][0]
    if gain < 0.06:
        balances = state.variant.get("flop_chips_by_seat", ())
        rivals = [seat for seat in state.seats
                  if seat.seat != state.seat and seat.status in ("active", "all_in")]
        rival_budget = sum(balances[seat.seat] if seat.seat < len(balances) else 1000
                           for seat in rivals)
        if rival_budget > 0:
            # Defend a good free flop when rivals can displace its cards.
            mean_quality = sum(value for value, _ in options) / len(options)
            gain = max(gain, 0.6 * (quality - mean_quality))
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
