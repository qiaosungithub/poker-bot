"""Bounded relative-equity auction, using only public observations.

Common random numbers compare all ten flops; every one of the five candidates
is unavailable to opponents and future runouts. Opponent sealed bids remain
latent scenarios, calibrated solely by observed total payment fractions.
"""
from itertools import combinations
import random
import eval7
from poker_agent_sdk import Action
from preflop_table import EQUITY

NAMES = tuple(r + s for r in '23456789TJQKA' for s in 'cdhs')
CODE = {c: i for i, c in enumerate(NAMES)}
CARDS = tuple(eval7.Card(c) for c in NAMES)
TRIPLES = tuple(combinations(range(5), 3))
MASK_INDEX = {sum(1 << i for i in t): n for n, t in enumerate(TRIPLES)}
ZERO = (0, 0, 0, 0, 0)
SAMPLES = 24


def _equity(a, b):
    x, y = NAMES[max(a, b)], NAMES[min(a, b)]
    key = x[0] + y[0] + ('' if x[0] == y[0] else 's' if x[1] == y[1] else 'o')
    return EQUITY[key]


# Immutable range tables, never table-specific memory. Soft support preserves
# weak hands and avoids treating a public raise as proof of a premium holding.
PAIRS = tuple(combinations(range(52), 2))
PAIR_EQ = tuple(_equity(a, b) for a, b in PAIRS)
RANGES = tuple(tuple(pair for pair, eq in zip(PAIRS, PAIR_EQ)
                     for _ in range(1 if eq < threshold else 3 if eq < threshold + .08 else 6))
               for threshold in (.40, .49, .55, .61))
AFFINITY = tuple(tuple(.25 * (a // 4 == c // 4) + .055 * (a % 4 == c % 4)
                       + .015 * (abs(a // 4 - c // 4) <= 2) for c in range(52))
                 for a in range(52))
CARD_PRIOR = tuple(.02 + .004 * (c // 4) for c in range(52))


def select(totals):
    """Integer score embeds the official lower-index tie priority exactly."""
    ranked = sorted(range(5), key=lambda i: 8 * totals[i] - i, reverse=True)
    return tuple(ranked[:3])


def forced(target, opposition):
    """Componentwise minimum nonnegative bid forcing this exact three-set."""
    outside = tuple(i for i in range(5) if i not in target)
    a, b = outside
    bid = [0] * 5
    for i in target:
        bid[i] = max(0, opposition[a] + (i > a) - opposition[i],
                     opposition[b] + (i > b) - opposition[i])
    return tuple(bid)


def range_classes(state, opponents):
    raises = {}
    calls = set()
    for e in state.history:
        if e.kind == 'raise':
            raises[e.seat] = raises.get(e.seat, 0) + 1
        elif e.kind == 'call':
            calls.add(e.seat)
    return tuple(3 if raises.get(s.seat, 0) > 1 else 2 if s.seat in raises
                 else 1 if s.seat in calls else 0 for s in opponents)


def relative_values(hole, candidates, classes, seed):
    """24 shared heads-up rollouts, then a conservative multiway transform.

This is a range-aware continuation proxy, not exact multiway equity. Shared
opponent cards and runouts reduce the variance of differences between flops.
All sampled ranges retain weak-hand support, with blocker-correct rejection.
"""
    rng = random.Random(seed)
    known = set(hole + candidates)
    available = tuple(c for c in range(52) if c not in known)
    hero = [CARDS[c] for c in hole]
    boards = [[CARDS[candidates[i]] for i in t] for t in TRIPLES]
    wins = [0.] * 10
    sampled_holes = []
    evaluate = eval7.evaluate
    for n in range(SAMPLES):
        population = RANGES[classes[n % len(classes)]]
        for _ in range(12):
            pair = population[rng.randrange(len(population))]
            if pair[0] not in known and pair[1] not in known:
                break
        else:
            pair = tuple(rng.sample(available, 2))
        sampled_holes.append(pair)
        # No dead auction candidate can reappear as a river or opponent card.
        for _ in range(12):
            runout = rng.sample(available, 2)
            if pair[0] not in runout and pair[1] not in runout:
                break
        else:
            runout = rng.sample(tuple(c for c in available if c not in pair), 2)
        tail = [CARDS[runout[0]], CARDS[runout[1]]]
        villain = [CARDS[pair[0]], CARDS[pair[1]]]
        for i, flop in enumerate(boards):
            full = flop + tail
            ours = evaluate(hero + full)
            theirs = evaluate(villain + full)
            wins[i] += 1. if ours > theirs else .5 if ours == theirs else 0.
    exponent = 1. + .65 * (len(classes) - 1)
    return tuple(((w + .5) / (SAMPLES + 1.)) ** exponent for w in wins), tuple(sampled_holes)


def scenarios(state, features, opponents, candidates, samples):
    """Six public-budget scenarios; no per-card bids are inferred as facts."""
    if not any(features.balances[s.seat] for s in opponents):
        return ((ZERO, 1.),)
    bb = max(1, state.blinds.big_blind)
    remaining = max(1, state.config.hand_cap - state.hand_number + 1)
    offered = []
    for s in opponents:
        budget = features.balances[s.seat]
        row = features.history[s.seat]
        horizon = max(2., min(remaining * .32, 8. + (s.stack + s.bet) / bb * .55))
        prior = min(.30, (3.0 + 2.0 * state.pot / max(bb, state.pot + s.stack)) / horizon)
        count = min(12, row.fraction_observations)
        # A total payment is a lower bound on total offers, not a card signal.
        fraction = (4. * prior + count * min(.75, row.fraction_mean / .8)) / (4. + count)
        offered.append(min(budget, max(0, budget * fraction)))
    result = [(ZERO, .08)]
    for scenario, scale in enumerate((.45, .75, 1., 1.4, 2.)):
        totals = [0] * 5
        for j, s in enumerate(opponents):
            pair = samples[(scenario * len(opponents) + j) % len(samples)]
            a, b = AFFINITY[pair[0]], AFFINITY[pair[1]]
            weights = tuple(CARD_PRIOR[c] + a[c] + b[c] for c in candidates)
            target = sorted(range(5), key=weights.__getitem__, reverse=True)[:3]
            total = sum(weights[i] for i in target)
            cap = min(features.balances[s.seat], int(offered[j] * scale))
            # Floors preserve sum-of-five affordability for each hypothetical player.
            for i in target:
                totals[i] += int(cap * weights[i] / total)
        result.append((tuple(totals), .184))
    return tuple(result)


def optimize(values, budget, cases, shadow, downside=.20):
    """At most seven threshold vectors, judged against zero in every scenario.

Each scenario nominates its best net-value forcing bid. This bounds work
independently of balances without reducing the number of equity samples.
"""
    proposals = {ZERO}
    targets = sorted(range(10), key=values.__getitem__, reverse=True)[:3]
    baselines = []
    for opposition, weight in cases:
        before = select(opposition)
        baseline = values[MASK_INDEX[sum(1 << i for i in before)]]
        baselines.append(baseline)
        local_best, local_net = ZERO, 0.
        for index in targets:
            if values[index] <= baseline:
                continue
            bid = forced(TRIPLES[index], opposition)
            paid = sum(bid)
            net = values[index] - baseline - shadow * paid
            if paid <= budget and net > local_net:
                local_best, local_net = bid, net
        proposals.add(local_best)
    best, best_score, best_paid = ZERO, 0., 0.
    for bid in sorted(proposals):
        if bid == ZERO:
            continue
        gain = paid = loss = 0.
        payment = tuple(sum(bid[i] for i in t) for t in TRIPLES)
        for (opposition, weight), baseline in zip(cases, baselines):
            # Sort five encoded totals once; two excluded indexes determine
            # both board value and selected-card payment from tiny lookups.
            ranked = [8 * (opposition[0] + bid[0]) + 4,
                      8 * (opposition[1] + bid[1]) + 3,
                      8 * (opposition[2] + bid[2]) + 2,
                      8 * (opposition[3] + bid[3]) + 1,
                      8 * (opposition[4] + bid[4])]
            ranked.sort()
            index = MASK_INDEX[31 ^ (1 << (4 - ranked[0] % 8)) ^ (1 << (4 - ranked[1] % 8))]
            value = values[index]
            delta = value - baseline
            gain += weight * delta
            paid += weight * payment[index]
            loss += weight * max(0., -delta)
        net = gain - shadow * paid - downside * loss
        if net > best_score + 1e-12 or abs(net-best_score) <= 1e-12 and paid < best_paid:
            best, best_score, best_paid = bid, net, paid
    return best


def auction_action(state, features):
    legal = state.can('bid')
    budget = max(0, int(legal.max_amount)) if legal else 0
    if budget == 0:
        return Action('bid', cards=ZERO)
    hole = tuple(CODE[c] for c in state.cards)
    candidates = tuple(CODE[c] for c in state.variant['auction_cards'])
    opponents = tuple(s for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in'))
    if not opponents:
        return Action('bid', cards=ZERO)
    seed = state.hand_number * 1009 + state.seat * 7919
    for card in hole + candidates:
        seed = ((seed * 16777619) ^ card) & 0xffffffff
    values, samples = relative_values(hole, candidates, range_classes(state, opponents), seed)
    cases = scenarios(state, features, opponents, candidates, samples)
    remaining = max(1, state.config.hand_cap - state.hand_number + 1)
    me = state.seats[state.seat]
    bb = max(1, state.blinds.big_blind)
    horizon = min(.24 * max(0, remaining - 1), 4. + .35 * (me.stack + me.bet) / bb)
    if me.status == 'all_in':
        horizon = min(horizon, 2.)
    importance = max(1., (state.pot / (5. * bb)) ** .5)
    shadow = .025 * horizon / max(1, budget) / importance
    return Action('bid', cards=optimize(values, budget, cases, shadow))
