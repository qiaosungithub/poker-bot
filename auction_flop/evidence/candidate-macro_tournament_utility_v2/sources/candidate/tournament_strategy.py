"""Fresh event-time deterministic tournament kernels; no SDK assumptions."""
from itertools import combinations

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
TRIPLES = tuple(combinations(range(5), 3))


def card_code(card):
    """Internal cards are rank*4+suit, rank 0..12. External adapter owns parsing."""
    if isinstance(card, int):
        return card
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def _straight(ranks):
    bits = 0
    for r in ranks:
        bits |= 1 << r
    if bits & (1 << 14):
        bits |= 1 << 1
    for high in range(14, 4, -1):
        if bits & (31 << (high - 4)) == 31 << (high - 4):
            return high
    return 0


def hand_rank(cards):
    """Exact standard-high five-to-seven-card rank; larger tuples are stronger."""
    ranks = [c // 4 + 2 for c in cards]
    counts = [0] * 15
    suits = [[], [], [], []]
    for c, rank in zip(cards, ranks):
        counts[rank] += 1
        suits[c % 4].append(rank)
    flush = next((r for r in suits if len(r) >= 5), None)
    if flush:
        sf = _straight(flush)
        if sf:
            return (8, sf)
    groups = sorted(((counts[r], r) for r in range(2, 15) if counts[r]), reverse=True)
    if groups[0][0] == 4:
        q = groups[0][1]
        return (7, q, max(r for r in ranks if r != q))
    triples = [r for count, r in groups if count >= 3]
    if triples:
        t = triples[0]
        pairs = [r for count, r in groups if count >= 2 and r != t]
        if pairs:
            return (6, t, max(pairs))
    if flush:
        return (5, *sorted(flush, reverse=True)[:5])
    straight = _straight(ranks)
    if straight:
        return (4, straight)
    if triples:
        t = triples[0]
        return (3, t, *sorted((r for r in set(ranks) if r != t), reverse=True)[:2])
    pairs = sorted((r for count, r in groups if count >= 2), reverse=True)
    if len(pairs) >= 2:
        a, b = pairs[:2]
        return (2, a, b, max(r for r in ranks if r != a and r != b))
    if pairs:
        p = pairs[0]
        return (1, p, *sorted((r for r in set(ranks) if r != p), reverse=True)[:3])
    return (0, *sorted(set(ranks), reverse=True)[:5])


def preflop_quality(hole):
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    if a == b:
        return 0.51 + (a - 2) * 0.037
    suited = hole[0] % 4 == hole[1] % 4
    gap = a - b - 1
    q = 0.16 + 0.022 * (a - 2) + 0.012 * (b - 2)
    if suited:
        q += 0.055
    q += max(0.0, 0.055 - gap * 0.025)
    if a == 14:
        q += 0.035
    if a >= 12 and b >= 10:
        q += 0.045
    return min(0.88, q)


def board_quality(hole, board):
    """A heuristic hand-quality number, not claimed to be showdown equity."""
    all_cards = tuple(hole) + tuple(board)
    rank = hand_rank(all_cards)
    category = rank[0]
    hr = [c // 4 + 2 for c in hole]
    br = [c // 4 + 2 for c in board]
    own = set(hr)
    counts = {r: br.count(r) for r in set(br)}
    quality = (0.16, 0.52, 0.74, 0.84, 0.88, 0.93, 0.97, 0.995, 1.0)[category]
    if category == 0:
        quality += 0.015 * sum(r > max(br) for r in hr)
        quality += 0.025 if max(hr) == 14 else 0
    elif category == 1:
        pair = rank[1]
        if pair not in own:
            quality = 0.20 + 0.025 * sum(r > pair for r in hr)
        elif hr[0] == hr[1]:
            quality = 0.76 if pair > max(br) else 0.43 + 0.025 * sum(r < pair for r in set(br))
        else:
            higher = sum(r > pair for r in set(br))
            kicker = max(r for r in hr if r != pair)
            quality = max(0.38, 0.70 - 0.13 * higher + 0.006 * (kicker - 8))
    elif category == 2:
        private_pairs = sum(counts.get(r, 0) < 2 and r in own for r in rank[1:3])
        quality = 0.29 if private_pairs == 0 else (0.65 if private_pairs == 1 else 0.78)
    elif category == 3 and counts.get(rank[1], 0) >= 3:
        quality = 0.29 + 0.012 * (max(hr) - 2)
    elif category == 7 and counts.get(rank[1], 0) == 4:
        quality = 0.30 + 0.035 * (max(hr) - 2)
    if len(board) == 5 and hand_rank(board) == rank:
        quality = 0.27 if category < 4 else 0.48
    draw = 0.0
    if len(board) < 5:
        for suit in range(4):
            count = sum(c % 4 == suit for c in all_cards)
            own_suited = [c // 4 + 2 for c in hole if c % 4 == suit]
            if count == 4 and own_suited:
                draw = max(draw, 0.13 + 0.005 * max(own_suited))
            elif count == 3 and len(board) == 3 and own_suited:
                draw = max(draw, 0.025)
        values = set(hr + br)
        if 14 in values:
            values.add(1)
        for low in range(1, 11):
            window = set(range(low, low + 5))
            if len(window & values) == 4 and (window & own):
                draw = max(draw, 0.10)
    # Draws contribute less as made hands approach the top of the scale.
    return min(1.0, quality + draw * (1.0 - quality) * 1.4)


def risk_premium(stack, biggest_opponent, stacks, payouts, to_call):
    """Bounded payout-risk heuristic, not exact ICM; independent auction currency."""
    if not payouts or len([p for p in payouts if p > 0]) <= 1 or stack <= 0:
        return 0.0
    live = sorted(s for s in stacks if s > 0)
    paid = sum(p > 0 for p in payouts)
    if len(live) <= 1 or biggest_opponent < 0.85 * stack:
        return 0.0
    near_bubble = max(0.0, 1.0 - abs(len(live) - paid - 1) / 4.0)
    shortest = stack <= live[0] * 1.20
    commitment = min(1.0, to_call / max(stack, 1))
    return (0.035 + 0.09 * near_bubble) * commitment * (0.35 if shortest else 1.0)


def auction_bids(hole, candidates, budget, *, hand_number, hand_cap,
                 stack, big_blind, pot, opponents, all_in=False):
    """Score all ten boards and bid a bounded, tournament-aware total.

    Inputs are explicit primitives until the released SDK fixes field meanings.
    Tie ordering and dead-card rules are deliberately absent from this kernel.
    """
    if budget <= 0:
        return (0, 0, 0, 0, 0)
    qualities = {combo: board_quality(hole, tuple(candidates[i] for i in combo))
                 for combo in TRIPLES}
    target = max(TRIPLES, key=lambda combo: qualities[combo])
    best = qualities[target]
    spread = best - min(qualities.values())
    # When no board changes our quality, saving chips dominates a blind bid.
    if spread < 0.025:
        return (0, 0, 0, 0, 0)
    remaining = max(1, hand_cap - hand_number + 1)
    stack_bb = stack / max(big_blind, 1)
    # Approximate future auctions, declining with shallow-stack bust risk.
    horizon = min(remaining * 0.32, 8.0 + stack_bb * 0.55)
    horizon = max(1.0, horizon)
    importance = pot / max(pot + stack, big_blind)
    pressure = 1.0 + 3.0 * importance + 3.0 * spread
    if all_in:
        pressure *= 2.0
        horizon = min(horizon, 5.0)
    fraction_cap = 1.0 if remaining == 1 else (0.65 if all_in or remaining < 5 else 0.28)
    spend = min(budget, max(1, int(min(budget * fraction_cap, budget * pressure / horizon))))
    # Interaction-aware weights: compare the best board with all one-card
    # replacements, so pair/draw synergies influence each preferred card's bid.
    outside = tuple(i for i in range(5) if i not in target)
    weights = []
    for i in target:
        alternatives = [tuple(sorted((set(target) - {i}) | {j})) for j in outside]
        loss = best - sum(qualities[c] for c in alternatives) / len(alternatives)
        weights.append(max(0.025, loss))
    total = sum(weights)
    bids = [0] * 5
    for i, weight in zip(target, weights):
        bids[i] = int(spend * weight / total)
    spare = spend - sum(bids)
    order = sorted(range(3), key=lambda k: weights[k], reverse=True)
    for k in range(spare):
        bids[target[order[k % 3]]] += 1
    return tuple(bids)
