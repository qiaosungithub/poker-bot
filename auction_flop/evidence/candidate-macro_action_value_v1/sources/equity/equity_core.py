"""Event-written bounded equity kernels; no SDK or game-engine dependency."""
from itertools import combinations
from math import sqrt
from random import Random

import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARDS = tuple(eval7.Card(r + s) for r in RANKS for s in SUITS)
CARD_ID = {str(card): i for i, card in enumerate(CARDS)}
TRIPLES = tuple(combinations(range(5), 3))


def ids(cards):
    return tuple(CARD_ID[str(card)] for card in cards)


def stable_seed(hole, board, salt=0):
    value = 0x9E3779B9 ^ int(salt)
    for card in tuple(hole) + tuple(board):
        value = ((value ^ (card + 1)) * 0x45D9F3B) & 0xFFFFFFFF
    return value


def preflop_strength(hole):
    """Ordering score in [0,1], deliberately not labeled exact equity."""
    a, b = hole
    ra, rb = a // 4 + 2, b // 4 + 2
    high, low = max(ra, rb), min(ra, rb)
    if ra == rb:
        return min(.99, .49 + .039 * (ra - 2))
    gap = high - low
    score = .20 + .035 * (high - 2) + .018 * (low - 2)
    score += .055 if a % 4 == b % 4 else 0
    score += .035 if gap == 1 else .014 if gap == 2 else 0
    score -= min(.11, max(0, gap - 3) * .02)
    score += .035 if high == 14 else 0
    return min(.97, max(.05, score))


def board_texture(hole, board):
    """Cheap hand contribution/draw features, independent of evaluator labels."""
    ranks = [c // 4 + 2 for c in board]
    hole_ranks = [c // 4 + 2 for c in hole]
    all_cards = tuple(hole) + tuple(board)
    suits = [sum(c % 4 == s for c in all_cards) for s in range(4)]
    own_pair = max((r for r in hole_ranks if r in ranks), default=0)
    pocket = hole_ranks[0] == hole_ranks[1]
    top_pair = bool(ranks and own_pair == max(ranks))
    overpair = bool(pocket and ranks and hole_ranks[0] > max(ranks))
    flush_draw = any(suits[s] == 4 and any(c % 4 == s for c in hole) for s in range(4))
    unique = {c // 4 + 2 for c in all_cards}
    if 14 in unique:
        unique.add(1)
    straight_draw = max((len(unique.intersection(range(lo, lo + 5))) for lo in range(1, 11)), default=0) >= 4
    paired_board = len(set(ranks)) < len(ranks)
    wet = max((sum(c % 4 == s for c in board) for s in range(4)), default=0) >= 3
    return top_pair, overpair, bool(flush_draw), bool(straight_draw), paired_board, wet


def showdown_share(hole, board, opponents, seed, samples=20, dead=()):
    """Fixed-size complete-deal Monte Carlo with an explicit evaluator budget."""
    opponents = max(1, min(9, int(opponents)))
    samples = max(4, min(int(samples), 96 // (opponents + 1)))
    excluded = set(hole) | set(board) | set(dead)
    deck = [c for i, c in enumerate(CARDS) if i not in excluded]
    needed_board = 5 - len(board)
    need = needed_board + 2 * opponents
    if need > len(deck) or needed_board < 0:
        raise ValueError("incompatible card/deck state")
    rng = Random(seed)
    hero = [CARDS[i] for i in hole]
    public = [CARDS[i] for i in board]
    evaluate = eval7.evaluate
    total = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, need)
        full_board = public + draw[:needed_board]
        hero_rank = evaluate(hero + full_board)
        ties = 0
        lost = False
        for j in range(opponents):
            start = needed_board + j * 2
            other_rank = evaluate(draw[start:start + 2] + full_board)
            if other_rank > hero_rank:
                lost = True
                break
            if other_rank == hero_rank:
                ties += 1
        if not lost:
            total += 1.0 / (1 + ties)
    return total / samples


def auction_equities(hole, candidates, opponents, seed, samples=8,
                     dead_candidates=True):
    """Compare all ten flops using common runouts to reduce ranking variance.

    Modeling at most three opponents bounds runtime; an explicit exponent
    approximates extra opponents. Dead-candidate semantics are caller-selected.
    """
    if len(candidates) != 5 or len(set(hole + candidates)) != 7:
        raise ValueError("auction requires two hole cards and five unique candidates")
    opponents = max(1, min(9, int(opponents)))
    modeled = min(3, opponents)
    samples = max(4, min(int(samples), 32 // (modeled + 1)))
    rng = Random(seed)
    excluded = set(hole) | set(candidates)
    outside = [c for i, c in enumerate(CARDS) if i not in excluded]
    hero = [CARDS[i] for i in hole]
    pools = [[CARDS[candidates[i]] for i in triple] for triple in TRIPLES]
    sums = [0.0] * 10
    evaluate = eval7.evaluate
    full_deck = outside + [CARDS[i] for i in candidates]
    outside_count = len(outside)
    for _ in range(samples):
        common = rng.sample(outside, 2 + 2 * modeled)
        ordering = None if dead_candidates else rng.sample(range(len(full_deck)), 5 + 2 * modeled)
        for k, triple in enumerate(TRIPLES):
            draw = common
            if not dead_candidates:
                # A shared random permutation of the full available deck preserves
                # the correct marginal law for each alternative candidate board.
                forbidden = {outside_count + i for i in triple}
                draw = [full_deck[i] for i in ordering if i not in forbidden][:2 + 2 * modeled]
            board = pools[k] + draw[:2]
            rank = evaluate(hero + board)
            ties = 0
            lost = False
            for j in range(modeled):
                other = evaluate(draw[2 + 2*j:4 + 2*j] + board)
                if other > rank:
                    lost = True
                    break
                ties += other == rank
            if not lost:
                sums[k] += 1.0 / (1 + ties)
    # Small prior stabilizes tiny Monte Carlo budgets without washing out a
    # clear structural advantage between candidate boards.
    prior = 1.0 / (modeled + 1)
    exponent = 1.0 + .70 * (opponents - modeled)
    return tuple(((value + 1.5 * prior) / (samples + 1.5)) ** exponent for value in sums)


def auction_bids(equities, budget, stack_bb, pot_bb, hands_remaining,
                 all_in=False, tie_order=(0, 1, 2, 3, 4)):
    """Spend a separate currency on a jointly favorable three-card board."""
    budget = max(0, int(budget))
    if budget == 0:
        return (0, 0, 0, 0, 0)
    best = max(range(10), key=lambda k: equities[k])
    target = TRIPLES[best]
    baseline = tuple(sorted(tie_order[:3]))
    default_value = equities[TRIPLES.index(baseline)]
    average = sum(equities) / 10
    gain = max(0.0, equities[best] - min(default_value, average))
    if gain < .018:
        return (0, 0, 0, 0, 0)
    opportunities = max(1.0, min(float(hands_remaining), 5.0 + .18 * stack_bb))
    if all_in:
        opportunities = 1.0
    leverage = min(2.5, max(.5, sqrt(max(1.0, pot_bb) / 5.0)))
    fraction = min(1.0 if all_in else .32, (0.8 + 7.0 * gain) * leverage / opportunities)
    spend = min(budget, max(1, round(budget * fraction)))
    weights = []
    for i in target:
        present = sum(v for v, triple in zip(equities, TRIPLES) if i in triple) / 6
        absent = sum(v for v, triple in zip(equities, TRIPLES) if i not in triple) / 4
        weights.append(max(.02, present - absent + gain * .15))
    total_weight = sum(weights)
    bids = [0] * 5
    allocated = 0
    for i, weight in zip(target, weights):
        bids[i] = int(spend * weight / total_weight)
        allocated += bids[i]
    # Remainder goes to the highest-influence card, keeping the exact budget.
    bids[target[max(range(3), key=lambda j: weights[j])]] += spend - allocated
    return tuple(bids)


def auction_competitive(equities, budget, stack_bb, pot_bb, hands_remaining,
                        opponent_balances, seed, all_in=False):
    """Select bids by estimated outcome changes, net of future-currency value."""
    budget = max(0, int(budget))
    if not budget or max(equities) - min(equities) < .025:
        return (0,) * 5
    zero = (0,) * 5
    preferred = TRIPLES[max(range(10), key=equities.__getitem__)]
    opportunities = 1.0 if all_in else max(3., min(hands_remaining, 8. + .16 * stack_bb))
    unit = budget / opportunities
    cap = budget if all_in else min(budget, max(3, int(.30 * budget)))
    rng = Random(seed ^ 0xA018)
    scenarios = [(0, 0, 0, 0, 0)]
    typical = sum(opponent_balances) / max(1, len(opponent_balances)) / 24.
    for k in range(7):
        scores = [0.] * 5
        for balance in opponent_balances:
            # A broad public-information prior; never infer hidden per-card bids
            # from public balance changes or bidder order.
            scale = min(balance, typical * (.3 + 1.5 * rng.random()))
            for card in rng.sample(range(5), 3):
                scores[card] += scale * (.2 + rng.random())
        scenarios.append(tuple(scores))
    lookup = {triple: value for triple, value in zip(TRIPLES, equities)}
    def outcome(scores, bids):
        selected = tuple(sorted(sorted(range(5), key=lambda i: (-scores[i]-bids[i], i))[:3]))
        return lookup[selected], sum(bids[i] for i in selected)
    baseline = sum(outcome(s, zero)[0] for s in scenarios) / len(scenarios)
    proposals = {zero}
    for scale in (.25, .6, 1.2, 2.5):
        spend = min(cap, max(1, round(unit * scale)))
        bids = [0] * 5
        for j, i in enumerate(preferred):
            bids[i] = spend // 3 + (j < spend % 3)
        proposals.add(tuple(bids))
        # A single-card intervention can change an unwanted default flop much
        # more cheaply than attempting to buy all three cards.
        for i in preferred:
            proposal = [0] * 5
            proposal[i] = spend
            proposals.add(tuple(proposal))
    if all_in:
        proposals.add(tuple(budget // 3 + (preferred.index(i) < budget % 3) if i in preferred else 0 for i in range(5)))
    shadow = (.002 if all_in else .035) / max(1., unit)
    shadow /= min(2.5, max(.7, sqrt(max(1., pot_bb) / 5.)))
    best, best_gain = zero, 0.005
    for proposal in sorted(proposals):
        values = [outcome(s, proposal) for s in scenarios]
        mean_equity = sum(v for v, _ in values) / len(values)
        mean_cost = sum(c for _, c in values) / len(values)
        gain = mean_equity - baseline - shadow * mean_cost
        if gain > best_gain:
            best, best_gain = proposal, gain
    return best
