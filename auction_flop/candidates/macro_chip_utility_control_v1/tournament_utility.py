"""Objective-only control: expected normal-chip share instead of prize ICM.

Every terminal outcome uses 100 * hero_stack / total_normal_chips. Normal-chip
conservation makes the denominator identical across actions at a given table.
Payout schedules, prior prizes and starting-stack elimination ranks are ignored.
Auction currency never enters this value. Settlement code is unchanged from v2.
"""


def icm_values(stacks, payouts):
    """Compatibility name: linear normal-chip values, not ICM in this control."""
    total = sum(stacks)
    return tuple(100.0 * value / total if total else 0.0 for value in stacks)


def hero_utility(stacks, hero, payouts, start_stacks=None):
    """The same linear chip-share formula applies to survivors and busts."""
    total = sum(stacks)
    return 100.0 * stacks[hero] / total if total else 0.0


def settled_utilities(stacks, payouts, start_stacks=None):
    """Normal-chip values only; no secured prize or elimination-rank additions."""
    return icm_values(stacks, payouts)


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
