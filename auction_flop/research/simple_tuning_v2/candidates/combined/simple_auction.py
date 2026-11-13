"""Rank ten flops; split one budget equally over the best three cards.

Only our hole cards, five public auction cards, number of live hands and the
legal balance are used. No history, opponent model, pot or stack weighting.
"""

from collections import Counter
from itertools import combinations
import eval7
from poker_agent_sdk import Action
from config import BASE_BID, AUCTION_HORIZON

RANK = {r: i + 2 for i, r in enumerate("23456789TJQKA")}
CARDS = {r + s: eval7.Card(r + s) for r in RANK for s in "shdc"}
FLOPS = tuple(combinations(range(5), 3))


def flop_quality(hole, board):
    """Priority units, not a probability: pair 1/2, two pair 3, trips 4, etc."""
    hr = {RANK[c[0]] for c in hole}
    br = {RANK[c[0]] for c in board}
    counts = Counter(RANK[c[0]] for c in hole + board)
    made = eval7.evaluate([CARDS[c] for c in hole + board]) >> 24
    private_pairs = [r for r in hr if counts[r] >= 2]
    if made >= 4:
        score = float(made + 1)  # Straight 5, flush 6, full house 7, quads 8, SF 9.
    elif made == 3:
        score = 4.0 if any(counts[r] == 3 for r in hr) else 0.0
    elif made == 2 and len(private_pairs) == 2:
        score = 3.0
    elif private_pairs:
        score = 2.0 if max(private_pairs) >= max(br) else 1.0
    else:
        score = 0.0  # A pair/trips entirely on the board earns no private value.

    suits = Counter(c[1] for c in hole + board)
    flush_draw = any(suits[c[1]] == 4 for c in hole)
    ranks = hr | br
    if 14 in ranks:
        ranks.add(1)
    private_ranks = hr - br
    if 14 in private_ranks:
        private_ranks.add(1)
    straight_draw = any(len(ranks & window) == 4 and private_ranks & window
                        for window in (set(range(lo, lo + 5)) for lo in range(1, 11)))
    return score + float(flush_draw) + .5 * bool(straight_draw)


def auction(state):
    cards = tuple(state.variant["auction_cards"])
    values = [flop_quality(state.cards, tuple(cards[i] for i in flop)) for flop in FLOPS]
    best = max(range(len(FLOPS)), key=values.__getitem__)
    improvement = values[best] - min(values)
    players = max(2, sum(s.status in ("active", "all_in") for s in state.seats))
    balance = state.can("bid").max_amount
    base = balance / AUCTION_HORIZON if AUCTION_HORIZON > 0 else BASE_BID
    budget = min(balance, max(0, round(base * improvement * 2 / players)))
    each, remainder = divmod(budget, 3)
    bids = [0] * 5
    for order, index in enumerate(FLOPS[best]):
        bids[index] = each + int(order < remainder)
    return Action("bid", cards=tuple(bids))
