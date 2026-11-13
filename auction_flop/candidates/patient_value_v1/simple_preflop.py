"""A small tight-aggressive Hold'em chart. No auction-specific preflop rules."""

from cards import passive, call, raise_to

from config import OPEN_LOOSER, CALL_SCALE, SHOVE_BB

RANKS = "23456789TJQKA"

# Pair minimum, followed by an explicit list of non-pair hands.
OPEN = {
    "early": (7, "AJs AQs AKs KQs AQo AKo"),
    "middle": (5, "ATs AJs AQs AKs KJs KQs QJs AJo AQo AKo KQo"),
    "late": (2, "A2s A3s A4s A5s A6s A7s A8s A9s ATs AJs AQs AKs "
                "K9s KTs KJs KQs QTs QJs JTs T9s 98s ATo AJo AQo AKo KJo KQo QJo"),
}
OPEN = {position: (pair, frozenset(hands.split()))
        for position, (pair, hands) in OPEN.items()}
CALL = frozenset("ATs AJs AQs AKs KQs AJo AQo AKo KQo".split())


def hand_key(cards):
    a, b = sorted(cards, key=lambda c: RANKS.index(c[0]), reverse=True)
    return a[0] + b[0] + ("" if a[0] == b[0] else "s" if a[1] == b[1] else "o")


def position(state):
    # Clockwise after the button: SB, BB, UTG, ..., cutoff, button.
    order = sorted((s.seat for s in state.seats if s.status != "busted"),
                   key=lambda seat: (seat - state.button - 1) % len(state.seats))
    if len(order) <= 3 or state.seat in order[-2:]:
        return "late"
    index = order.index(state.seat)
    if index < 2 or index >= 2 + (len(order) - 4) // 2:
        return "middle"
    return "early"


def preflop(state):
    key = hand_key(state.cards)
    pair = RANKS.index(key[0]) + 2 if len(key) == 2 else 0
    pos = position(state)
    chart = {"early": "middle", "middle": "late", "late": "late"}[pos] if OPEN_LOOSER else pos
    pair_min, hands = OPEN[chart]
    opening = pair >= pair_min or key in hands
    high, low = (RANKS.index(r) + 2 for r in key[:2])
    short_handed = state.players_remaining <= 3
    short_range = (pair >= 2 or high == 14 or high == 13 and low >= 8
                   or high >= 11 and low >= 10
                   or key.endswith('s') and high - low == 1 and low >= 7)
    if short_handed:
        opening = opening or short_range
    premium = pair >= 12 or key in ("AKs", "AKo")
    strong = pair >= 10 or key in ("AQs", "AQo", "AKs", "AKo")
    me = state.seats[state.seat]
    bb = max(1, state.blinds.big_blind)
    stack = me.stack + me.bet
    cost = min(me.stack, state.to_call)
    raises = sum(event.kind == "raise" for event in state.history)

    if raises == 0:
        if not opening:
            return passive(state)
        if stack <= SHOVE_BB * bb:
            shove = pair >= 6 or key in ("ATs", "AJs", "AQs", "AKs", "AJo", "AQo", "AKo", "KQs")
            if short_handed:
                shove = short_range
            return raise_to(state, stack) if shove else passive(state)
        limpers = sum(event.kind == "call" for event in state.history)
        return raise_to(state, ((2.5 if pos == "late" else 3.0) + limpers) * bb)

    # With major preflop pressure, require KK+ deep; allow TT+/AQ at <=20 BB.
    if raises >= 2 or cost >= .30 * stack:
        commit = pair >= 13 or (stack <= 20 * bb and strong)
        return raise_to(state, stack) if commit else passive(state)
    if premium:
        return raise_to(state, 3 * (me.bet + state.to_call))
    # One ordinary raise: medium pairs and good broadways may take a cheap flop.
    playable = pair >= 7 or key in CALL
    cheap = cost <= 4 * CALL_SCALE * bb and cost <= .15 * CALL_SCALE * stack
    return call(state) if playable and cheap else passive(state)
