"""Fresh Boston auction kernels, written after live reveal on 2026-11-09.

The API adapter must supply five ordered candidates and public balances. No
engine imports, opponent-private state, or hidden bids are used here.
"""
from itertools import combinations

TRIPLES = tuple(combinations(range(5), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(TRIPLES)}
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
STRAIGHT_MASKS = tuple(sum(1 << r for r in range(high - 4, high + 1)) for high in range(5, 15))


def parse_card(card):
    """Return (rank 2..14, suit 0..3) for a normal two-character card."""
    if isinstance(card, tuple):
        return card
    s = str(card)
    return RANKS.index(s[0]) + 2, SUITS.index(s[1].lower())


def board_utility(hole, board):
    """Bounded showdown-potential proxy; meaningful only for normal Hold'em.

    Exact five-card category plus private contribution and draw potential avoids
    treating a dangerous public pair as the same thing as a private set.
    """
    ranks = [c[0] for c in hole + board]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    group = sorted(((n, r) for r, n in counts.items()), reverse=True)
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    rank_mask = sum(1 << r for r in unique)
    matched = tuple((rank_mask & mask).bit_count() for mask in STRAIGHT_MASKS)
    straight_high = next((i + 5 for i in range(9, -1, -1) if matched[i] == 5), 0)
    suits = [c[1] for c in hole + board]
    flush = len(set(suits)) == 1
    private_pair = max((r for r in (hole[0][0], hole[1][0]) if counts[r] >= 2), default=0)
    pocket = hole[0][0] == hole[1][0]
    bcounts = {}
    for r, _ in board:
        bcounts[r] = bcounts.get(r, 0) + 1
    board_high = max(c[0] for c in board)
    if flush and straight_high:
        value = .995
    elif group[0][0] == 4:
        value = .985 if hole[0][0] == group[0][1] or hole[1][0] == group[0][1] else .76
    elif group[0][0] == 3 and group[1][0] == 2:
        value = .96
    elif flush:
        value = .93 + max(c[0] for c in hole) / 700
    elif straight_high:
        value = .89 + straight_high / 1000
    elif group[0][0] == 3:
        value = .84 if group[0][1] in (hole[0][0], hole[1][0]) else .32 + max(ranks[:2]) / 100
    elif group[0][0] == 2 and group[1][0] == 2:
        # On a flop, board-only two pair is impossible; one public pair with
        # a single private match remains weaker than using both hole ranks.
        private_groups = sum(1 for r in counts if counts[r] == 2 and r in ranks[:2])
        value = .78 if private_groups == 2 else .61
    elif private_pair:
        if pocket and private_pair > board_high:
            value = .73 + private_pair / 1000
        elif private_pair == board_high:
            value = .63 + min(ranks[:2]) / 180
        else:
            value = .39 + private_pair / 90
    elif group[0][0] == 2:
        value = .18 + max(ranks[:2]) / 100
    else:
        value = .13 + max(ranks[:2]) / 110 + min(ranks[:2]) / 250
    # Potential must involve a private card: otherwise a public texture also
    # supplies these draws to opponents and should not count as private value.
    flush_count = max(suits.count(s) for s in set(suits))
    if flush_count == 4 and any(suits.count(c[1]) == 4 for c in hole):
        value += .13 + max(c[0] for c in hole if suits.count(c[1]) == 4) / 500
    elif flush_count == 3 and hole[0][1] == hole[1][1]:
        value += .035
    draw_count = max(matched)
    if not straight_high and draw_count == 4:
        value += .075
    return min(.999, max(.01, value))


def combination_utilities(hole_cards, candidates):
    hole = tuple(parse_card(c) for c in hole_cards)
    cards = tuple(parse_card(c) for c in candidates)
    return tuple(board_utility(hole, tuple(cards[i] for i in t)) for t in TRIPLES)


def chosen_indices(totals, tie_order=(0, 1, 2, 3, 4)):
    """Ordered-index tie convention is supplied explicitly by the adapter."""
    priority = {card: place for place, card in enumerate(tie_order)}
    selected = sorted(range(5), key=lambda i: (-totals[i], priority[i]))[:3]
    return tuple(sorted(selected))


def settle(bids_by_player, balances, tie_order=(0, 1, 2, 3, 4)):
    """Pure rule model for tests; not a replacement for official engine tests."""
    for bids, balance in zip(bids_by_player, balances):
        if len(bids) != 5 or any(type(v) is not int or v < 0 for v in bids) or sum(bids) > balance:
            raise ValueError('invalid bid vector')
    totals = [sum(b[i] for b in bids_by_player) for i in range(5)]
    selected = chosen_indices(totals, tie_order)
    remaining = tuple(balance - sum(b[i] for i in selected) for b, balance in zip(bids_by_player, balances))
    return selected, remaining


def forcing_bid(target, opposition, tie_order=(0, 1, 2, 3, 4)):
    """Smallest componentwise nonnegative bid ensuring target under one model."""
    priority = {card: place for place, card in enumerate(tie_order)}
    excluded = tuple(i for i in range(5) if i not in target)
    result = [0] * 5
    for i in target:
        need = max(opposition[j] + (priority[i] > priority[j]) for j in excluded)
        result[i] = max(0, need - opposition[i])
    return tuple(result)


def opponent_scenarios(balances, horizon):
    """Seven anonymous public-budget scenarios, with zero-bid support."""
    spend = sum(min(b, max(0, int(b / max(2., horizon)))) for b in balances)
    if spend <= 0:
        return (((0, 0, 0, 0, 0), 1.),)
    scale = max(1, spend // 5)
    cases = [((0, 0, 0, 0, 0), .12), ((scale,) * 5, .18)]
    pattern = (0, 1, 1, 2, 4)
    for shift in range(5):
        cases.append((tuple(pattern[(i + shift) % 5] * scale for i in range(5)), .14))
    return tuple(cases)


def choose_bids_rich(utilities, budget, opponent_balances, hands_left=80,
                stack_bb=30., pot_bb=5., all_in=False, tie_order=(0, 1, 2, 3, 4)):
    """Optimize board utility over bounded opponent scenarios and opportunity cost.

    Actual cost is only own bids on selected cards. Flop chips are not normal
    chips; their opportunity cost shrinks as useful remaining auctions shrink.
    """
    budget = max(0, int(budget))
    if not budget:
        return (0, 0, 0, 0, 0)
    horizon = min(30., max(1., hands_left * .25), max(2., stack_bb * .7))
    if all_in:
        horizon = min(horizon, 3.)
    scenarios = opponent_scenarios(opponent_balances, horizon)
    candidates = {(0, 0, 0, 0, 0)}
    spread = max(utilities) - min(utilities)
    if spread < .005:
        return (0, 0, 0, 0, 0)
    # Score all ten boards, but optimize only the three strongest targets.
    # This caps search at 3 * 7 + 1 vectors independent of chip magnitude.
    targets = tuple(TRIPLES[i] for i in sorted(range(10), key=utilities.__getitem__, reverse=True)[:3])
    for opposition, _ in scenarios:
        for target in targets:
            proposal = forcing_bid(target, opposition, tie_order)
            if sum(proposal) <= budget:
                candidates.add(proposal)
    # Larger normal-chip pots justify more auction spending, without literally
    # subtracting currency from normal-chip EV. Near the cap saving is worthless.
    cost = .025 * horizon / max(1, budget) / max(1., (pot_bb / 5.) ** .5)
    if hands_left <= 1:
        cost = 0.
    best = (0, 0, 0, 0, 0)
    best_score = -1e100
    best_cost = budget + 1
    for bids in sorted(candidates):
        value = paid = 0.
        for opposition, weight in scenarios:
            selected = chosen_indices(tuple(opposition[i] + bids[i] for i in range(5)), tie_order)
            value += weight * utilities[TRIPLE_INDEX[selected]]
            paid += weight * sum(bids[i] for i in selected)
        score = value - cost * paid
        if score > best_score + 1e-12 or abs(score - best_score) <= 1e-12 and paid < best_cost:
            best, best_score, best_cost = bids, score, paid
    return best


def choose_bids(utilities, budget, opponent_balances, hands_left=80,
                stack_bb=30., pot_bb=5., all_in=False, tie_order=(0, 1, 2, 3, 4)):
    """Fast analytic candidate construction, at most seven vectors/three cases.

    A low-spend case represents abstainers, a public-budget case represents
    ordinary competition, and a higher-spend case checks sensitivity. We form
    minimum-forcing vectors for only the two best complete boards.
    """
    budget = max(0, int(budget))
    zero = (0, 0, 0, 0, 0)
    if not budget or max(utilities) - min(utilities) < .005:
        return zero
    horizon = min(30., max(1., hands_left * .25), max(2., stack_bb * .7))
    if all_in:
        horizon = min(horizon, 3.)
    spend = sum(min(b, max(0, int(b / max(2., horizon)))) for b in opponent_balances)
    targets = tuple(TRIPLES[i] for i in sorted(range(10), key=utilities.__getitem__, reverse=True)[:2])
    if spend <= 0:
        # Exact best attainable board under zero opposition, minimum payment.
        proposals = [(zero, utilities[TRIPLE_INDEX[chosen_indices(zero, tie_order)]]) ]
        for target in TRIPLES:
            bid = forcing_bid(target, zero, tie_order)
            if sum(bid) <= budget:
                proposals.append((bid, utilities[TRIPLE_INDEX[target]]))
        return max(proposals, key=lambda x: (x[1], -sum(x[0])))[0]
    scale = max(1, spend // 5)
    # No preference for physical index: rotate the uncertain profile using a
    # deterministic score-derived offset. Hidden bids are never observations.
    shift = int(sum((i + 1) * u for i, u in enumerate(utilities)) * 31) % 5
    pattern = (0, 1, 1, 2, 4)
    typical = tuple(pattern[(i + shift) % 5] * scale for i in range(5))
    high = tuple(pattern[(i + shift + 2) % 5] * scale * 2 for i in range(5))
    scenarios = ((zero, .3), (typical, .45), (high, .25))
    proposals = {zero}
    for opposition, _ in scenarios:
        for target in targets:
            bid = forcing_bid(target, opposition, tie_order)
            if sum(bid) <= budget:
                proposals.add(bid)
    cost = .025 * horizon / max(1, budget) / max(1., (pot_bb / 5.) ** .5) if hands_left > 1 else 0.
    priority = [0] * 5
    for place, card in enumerate(tie_order):
        priority[card] = 5 - place
    best, best_score, best_paid = zero, -1e100, budget + 1
    for bid in sorted(proposals):
        value = paid = 0.
        for opposition, weight in scenarios:
            # Integer priorities encode exact ties without per-case dictionaries.
            ranks = tuple(8 * (opposition[i] + bid[i]) + priority[i] for i in range(5))
            selected = tuple(sorted(sorted(range(5), key=ranks.__getitem__, reverse=True)[:3]))
            value += weight * utilities[TRIPLE_INDEX[selected]]
            paid += weight * sum(bid[i] for i in selected)
        score = value - cost * paid
        if score > best_score + 1e-12 or abs(score - best_score) <= 1e-12 and paid < best_paid:
            best, best_score, best_paid = bid, score, paid
    return best
