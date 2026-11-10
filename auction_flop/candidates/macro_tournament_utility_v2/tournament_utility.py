"""Public tournament utility and side-pot settlement; no engine imports.

ICM is exact at a chip-based hand cap, but only a continuation model before it.
All currency here is normal chips; auction balances never enter chip totals.
"""
from poker_agent_sdk import icm


def icm_values(stacks, payouts):
    """Exact live-player ICM. Previously busted players have value zero here."""
    return tuple(icm(stacks, payouts))


def _bust_value(stacks, hero, payouts, start_stacks):
    if start_stacks is None or start_stacks[hero] <= 0:
        return 0.0
    live = sum(x > 0 for x in stacks)
    # Larger starting stacks finish ahead among simultaneous eliminations.
    better = sum(stacks[i] <= 0 and start_stacks[i] > start_stacks[hero]
                 for i in range(len(stacks)))
    tied = sum(stacks[i] <= 0 and start_stacks[i] == start_stacks[hero]
               for i in range(len(stacks)))
    first = live + better
    return sum(payouts[j] if j < len(payouts) else 0.0
               for j in range(first, first + tied)) / max(1, tied)


def hero_utility(stacks, hero, payouts, start_stacks=None):
    """One player's exact ICM, with newly busted payouts if starts are supplied.

    Closed top-three probabilities cost O(n^2), versus SDK all-player subset DP.
    Arbitrary longer payout schedules deliberately fall back to the SDK oracle.
    start_stacks is the stack at hand start (zero for historical eliminations).
    """
    h = stacks[hero]
    if h <= 0:
        return _bust_value(stacks, hero, payouts, start_stacks)
    if len(payouts) > 3:
        return icm(stacks, payouts)[hero]
    if not payouts:
        return 0.0
    total = sum(stacks)
    p1 = h / total
    value = payouts[0] * p1
    # Equal stacks are exchangeable; grouping is especially useful early on.
    groups = {}
    for i, x in enumerate(stacks):
        if i != hero and x > 0:
            groups[x] = groups.get(x, 0) + 1
    opponents = tuple(groups.items())
    if len(payouts) > 1:
        value += payouts[1] * p1 * sum(count*x/(total-x) for x,count in opponents)
    if len(payouts) > 2:
        p3 = 0.0
        for i, (x, count) in enumerate(opponents):
            remaining = total-x
            if count > 1:
                p3 += count*(count-1)*x*x / (remaining*(total-2*x))
            for y, other_count in opponents[i+1:]:
                p3 += count*other_count*x*y/(remaining-y) * (1/remaining + 1/(total-y))
        value += payouts[2] * p1 * p3
    return value


def settled_utilities(stacks, payouts, start_stacks=None):
    """Live ICM plus this hand's bust payouts; historical busts stay zero."""
    values = list(icm(stacks, payouts))
    if start_stacks is not None:
        for i, x in enumerate(stacks):
            if x <= 0:
                values[i] = _bust_value(stacks, i, payouts, start_stacks)
    return tuple(values)


def showdown_stacks(stacks, contributions, active, ranking):
    """Settle all side pots from a complete rank order, including exact ties.

    stacks: chips BEHIND after proposed actions; contributions: total this hand.
    active: nonfolded seats; ranking: strongest first, e.g. ((1, 3), (0,), (2,)).
    Returned tie splits are fractional expectations, not button-ordered odd chips.
    An unmatched layer returns to its sole contributor, including after a fold.
    """
    result = list(stacks)
    active = set(active)
    last = 0
    for level in sorted(set(x for x in contributions if x > 0)):
        participants = [i for i, x in enumerate(contributions) if x >= level]
        pot = (level - last) * len(participants)
        last = level
        if len(participants) == 1:
            result[participants[0]] += pot
            continue
        eligible = active.intersection(participants)
        if not eligible:
            raise ValueError('Contested side pot has no nonfolded eligible seat')
        winners = next((tuple(i for i in group if i in eligible)
                        for group in ranking if any(i in eligible for i in group)), ())
        if not winners:
            raise ValueError('Ranking omits a side-pot winner')
        share = pot / len(winners)
        for i in winners:
            result[i] += share
    return tuple(result)


def expected_utility(outcomes, hero, payouts, start_stacks=None):
    """Average terminal utilities, not utility of the expected chip vector."""
    weighted = total = 0.0
    for probability, stacks in outcomes:
        if probability < 0:
            raise ValueError('Negative outcome probability')
        weighted += probability * hero_utility(stacks, hero, payouts, start_stacks)
        total += probability
    if total <= 0:
        raise ValueError('No positive outcome mass')
    return weighted / total
