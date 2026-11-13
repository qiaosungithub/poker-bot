"""No pure bluffs. Bet private strength on every street, with bounded equity."""
from collections import Counter
import eval7
from poker_agent_sdk import Action
from cards import CARDS, RANKS, equity, random_stream, passive, call, raise_to
from hand_contribution import shared_nut_action, public_quads_share
from pot_accounting import eligible_call


def made_hand(hole, board):
    """Return category, genuine two-pair-or-better, and top-pair/overpair."""
    value = eval7.evaluate([CARDS[c] for c in hole + board])
    category = value >> 24
    hr = [RANKS.index(c[0]) for c in hole]
    br = [RANKS.index(c[0]) for c in board]
    counts = Counter(hr + br)
    if len(board) == 5 and value == eval7.evaluate([CARDS[c] for c in board]):
        return category, False, False
    if category >= 4:
        genuine = True
    elif category == 3:
        genuine = any(counts[r] == 3 for r in hr)
    elif category == 2:
        genuine = len(set(hr)) == 2 and set(hr).issubset(br) and len(set(br)) == len(br)
    else:
        genuine = False
    top_pair = category in (1, 2) and (
        hr[0] == hr[1] and hr[0] > max(br) or
        hr[0] != hr[1] and max(br) in hr and counts[max(br)] == 2)
    return category, genuine, top_pair


def value_raise(state, monster):
    """Build a pot across remaining streets; use an effective-stack-sized raise."""
    me = state.seats[state.seat]
    cost = min(me.stack, state.to_call)
    pot = max(1, state.pot + cost)
    opponents = [s.stack for s in state.seats if s.seat != state.seat and s.status == "active"]
    effective = min(me.stack - cost, max(opponents, default=0))
    if effective <= 0:
        return call(state)
    current = me.bet + cost
    if monster and effective <= 1.5 * pot:
        return raise_to(state, current + effective)
    streets = 6 - len(state.board)  # Flop=3, turn=2, river=1.
    fraction = ((1 + 2 * effective / pot) ** (1 / streets) - 1) / 2
    fraction = min(1.0 if monster else .75, max(.5, fraction))
    return raise_to(state, current + min(effective, round(fraction * pot)))


def postflop(state):
    shared = shared_nut_action(state)
    if shared is not None:
        return shared
    price = eligible_call(state)
    opponents = sum(s.status in ("active", "all_in") for s in state.seats) - 1
    quad_share = public_quads_share(state.cards, state.board, tuple(state.variant.get("auction_cards", ())))
    if quad_share is not None:
        return call(state) if quad_share ** max(1, opponents) >= price.threshold + .10 else passive(state)
    eq = equity(state, random_stream(state))
    category, genuine, top_pair = made_hand(state.cards, state.board)
    strong = genuine and eq >= (.72 if opponents <= 2 else .60)
    monster = genuine and (eq >= .90 or category >= 6 and eq >= .82)
    me = state.seats[state.seat]
    if state.can("check"):
        if strong:
            return value_raise(state, monster)
        if top_pair and opponents <= 2 and eq >= .65:
            return raise_to(state, me.bet + round(.4 * state.pot))
        return Action.check()
    fraction = price.call_cost / max(1, me.stack + me.bet)
    large = fraction >= .30 or price.call_cost >= .75 * price.eligible_pot_before_call
    if monster:
        return value_raise(state, True)
    if strong:
        return call(state) if large else value_raise(state, False)
    # Marginal hands and draws continue only for a modest price with a margin.
    if not large and fraction <= .20 and eq >= price.threshold + .18:
        return call(state)
    return passive(state)
