"""Independent spending phase from icm_spend_v3."""

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
    big_blind = max(1, state.blinds.big_blind)
    stack_blinds = (own.stack + own.bet) / big_blind
    # Remaining bidding opportunities shrink with blind pressure and the cap.
    horizon = max(8.0, min(35.0, (state.config.hand_cap - state.hand_number) * 0.3,
                           10.0 + stack_blinds * 0.6))
    urgency = min(1.0, max(0.0, future_hands / horizon - 1.0))
    balances = state.variant.get("flop_chips_by_seat", ())
    opponents = [seat for seat in state.seats
                 if seat.seat != state.seat and seat.status in ("active", "all_in")]
    rival_budget = sum(balances[seat.seat] if seat.seat < len(balances) else 1000
                       for seat in opponents)
    # Public balances indicate remaining competition, never the hidden current bids.
    competition = min(1.0, rival_budget / max(1.0, len(opponents) * 120.0))
    strength_weight = min(1.0, max(0.0, (gain - 0.06) / 0.25))
    multiplier = 1.0 + competition * urgency * (0.35 + 0.55 * strength_weight)
    spend = int(budget / future_hands * opportunity * multiplier)
    if own.status == "all_in":
        spend = max(spend, int(budget * min(0.70, gain * 1.8)))
    elif own.stack < 8 * state.blinds.big_blind and quality > 0.75:
        spend = max(spend, int(budget * min(0.30, gain)))
    spend = min(budget, max(3, spend))
    if rival_budget == 0:
        # Three positive bids guarantee the preferred three when nobody can bid.
        spend = min(budget, 3)
    bids = [0] * 5
    quotient, remainder = divmod(spend, 3)
    for offset, index in enumerate(target):
        bids[index] = quotient + int(offset < remainder)
    return Action("bid", cards=tuple(bids)), b""
