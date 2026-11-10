"""Bounded preflop action search in expected tournament points.

This is a continuation approximation: range-conditioned Hold'em runout priors,
public action frequencies, and postflop realization, not an equilibrium solver.
"""
import math
from poker_agent_sdk import Action
from pot_accounting import eligible_call
from independent_preflop import choose_preflop_loose
from tournament_utility import hero_utility, showdown_stacks
from range_equity import WIDTHS, EQUITIES

RANKS = '23456789TJQKA'
QUADRATURE = 12
PRIMES = (2, 3, 5, 7, 11, 13, 17)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _key(cards):
    a, b = sorted(cards, key=lambda c: RANKS.index(c[0]), reverse=True)
    return a[0] + b[0] + ('' if a[0] == b[0] else 's' if a[1] == b[1] else 'o')


def _equity(key, width):
    row = EQUITIES[key]
    if width <= WIDTHS[0]:
        return row[0]
    for j in range(1, len(WIDTHS)):
        if width <= WIDTHS[j]:
            f = (math.log(width) - math.log(WIDTHS[j-1])) / (math.log(WIDTHS[j]) - math.log(WIDTHS[j-1]))
            return row[j-1] * (1-f) + row[j] * f
    return row[-1]


def _radical(index, base):
    value, divisor = 0.0, float(base)
    while index:
        value += (index % base) / divisor
        index //= base
        divisor *= base
    return value


def _caller_sets(probabilities):
    """Exact at <=3 opponents, fixed low-discrepancy 12-deal quadrature otherwise."""
    n = len(probabilities)
    if n <= 3:
        result = []
        for mask in range(1 << n):
            p = 1.0
            callers = []
            for j, q in enumerate(probabilities):
                p *= q if mask >> j & 1 else 1-q
                if mask >> j & 1:
                    callers.append(j)
            if p > 1e-9:
                result.append((p, tuple(callers)))
        return result
    counts = {}
    for sample in range(1, QUADRATURE + 1):
        chosen = tuple(j for j, p in enumerate(probabilities)
                       if (_radical(sample, PRIMES[j]) + .38196601125 * j) % 1 < p)
        counts[chosen] = counts.get(chosen, 0) + 1
    return [(count / QUADRATURE, chosen) for chosen, count in counts.items()]


def _range_models(state, features, opponents, target, is_raise):
    bb = max(1, state.blinds.big_blind)
    hero = state.seats[state.seat]
    alive = state.players_remaining
    short = min((s.stack + s.bet for s in state.seats if s.status != 'busted'), default=0)
    paid = sum(p > 0 for p in state.config.payouts)
    models = []
    for seat in opponents:
        row = features.history[seat.seat]
        vpip = (row.vpip_hands + 8) / (row.hands_seen + 24)
        pfr = (row.preflop_raises + 4) / (row.hands_seen + 24)
        action = next((e.kind for e in reversed(state.history)
                       if e.seat == seat.seat and e.kind in ('raise', 'call', 'check')), None)
        if action == 'raise':
            prior = _clip(pfr * 1.35, .10, .55)
        elif action == 'call':
            prior = _clip(vpip * 1.5, .22, .75)
        else:
            prior = 1.0
        if alive <= 3 and action is not None:
            prior = min(1., prior * (1.8 if alive == 2 else 1.35))
        price = min(seat.stack, max(0, target-seat.bet))
        if seat.status == 'all_in' or price == 0:
            models.append((1., prior))
            continue
        # Observed participation shrinks a population prior, never to zero.
        base = .31 + .11 * (alive <= 3) + .16 * (alive == 2)
        base *= _clip(vpip / (1/3), .65, 1.5)
        width = base / (1 + .060 * price / bb)
        if not is_raise:
            width *= 1.15
        commitment = price / max(1, seat.stack + seat.bet)
        width *= 1 - .22 * commitment
        # A covered middle stack's payout risk creates leader fold equity.
        if alive == paid+1 and hero.stack + hero.bet > seat.stack + seat.bet > short * 1.4:
            width *= .88
        if price <= bb and not is_raise:
            width = max(width, .45 + .20 * (alive <= 3))
        continuation = min(prior, _clip(width, .055, .85))
        models.append((_clip(continuation/prior, .05, 1.), continuation))
    return models


def _showdown_value(behind, contributions, live, hero, payouts, start,
                    hero_win, strength):
    """Enumerate each possible winner, retaining side-pot eligibility and busts."""
    if len(live) == 1:
        final = showdown_stacks(behind, contributions, live, (tuple(live),))
        return hero_utility(final, hero, payouts, start)
    opponents = [i for i in live if i != hero]
    if hero not in live:
        hero_win = 0.
    ordered = sorted(opponents, key=lambda i: -strength.get(i, 1.))
    value = 0.
    if hero in live:
        ranks = ((hero,),) + tuple((i,) for i in ordered)
        final = showdown_stacks(behind, contributions, live, ranks)
        value += hero_win * hero_utility(final, hero, payouts, start)
    total_strength = sum(strength.get(i, 1.) for i in opponents)
    for winner in opponents:
        # Hero second in loss outcomes approximates side-pot recovery; when
        # multiway, average second and last rather than gifting every side pot.
        middle = [i for i in ordered if i != winner]
        ranks = ((winner,),) + tuple((i,) for i in middle)
        if hero in live:
            ranks += ((hero,),)
        final = showdown_stacks(behind, contributions, live, ranks)
        u = hero_utility(final, hero, payouts, start)
        if hero in live and len(live) > 2:
            ranks = ((winner,), (hero,)) + tuple((i,) for i in middle)
            alternative = showdown_stacks(behind, contributions, live, ranks)
            u = .5 * (u + hero_utility(alternative, hero, payouts, start))
        value += (1-hero_win) * strength.get(winner, 1.) / total_strength * u
    return value


def action_values(state, features):
    """Return legal (Action, expected_points) candidates, for review/testing."""
    accounting = eligible_call(state)
    if not accounting.exact:
        return ()
    hero = state.seat
    me = state.seats[hero]
    payouts = state.config.payouts
    contributions = accounting.contributions
    stacks = tuple(s.stack for s in state.seats)
    start = tuple(a+b for a,b in zip(stacks, contributions))
    opponents = [s for s in state.seats if s.seat != hero and s.status in ('active', 'all_in')]
    if not opponents:
        return ()
    key = _key(state.cards)
    bb = max(1, state.blinds.big_blind)
    pot = state.pot
    target_call = me.bet + min(me.stack, state.to_call)
    legal_raise = state.can('raise')
    actions = []
    if state.can('fold'):
        live = [s.seat for s in opponents]
        v = _showdown_value(stacks, contributions, live, hero, payouts, start, 0., {})
        actions.append((Action.fold(), v))
    if state.can('check'):
        actions.append((Action.check(), None))
    elif state.can('call'):
        actions.append((Action.call(), None))
    if legal_raise is not None:
        current = me.bet + state.to_call
        limpers = sum(e.kind == 'call' for e in state.history)
        raised = any(e.kind == 'raise' for e in state.history)
        small = (current * 2.6 if raised else bb*(2.2+.65*min(limpers, 3)))
        medium = max(small, current + .8*(pot + state.to_call))
        candidates = (small, medium, legal_raise.max_amount)
        seen = set()
        for size in candidates:
            amount = min(legal_raise.max_amount, max(legal_raise.min_amount, round(size)))
            if amount not in seen:
                actions.append((Action.raise_to(amount), None))
                seen.add(amount)
    result = []
    for action, fixed in actions:
        if fixed is not None:
            result.append((action, fixed))
            continue
        raised = action.kind == 'raise'
        target = action.amount if raised else target_call
        models = _range_models(state, features, opponents, target, raised)
        value = 0.
        for probability, chosen in _caller_sets([m[0] for m in models]):
            behind = list(stacks)
            paid = list(contributions)
            own_cost = min(me.stack, max(0, target-me.bet))
            behind[hero] -= own_cost
            paid[hero] += own_cost
            live = [hero]
            raw_equities = []
            strength = {}
            for j in chosen:
                seat = opponents[j]
                added = min(seat.stack, max(0, target-seat.bet))
                behind[seat.seat] -= added
                paid[seat.seat] += added
                live.append(seat.seat)
                q = _equity(key, models[j][1])
                # Auction selection changes the runout distribution; preserve
                # uncertainty and allow only a bounded bankroll advantage.
                balance_edge = (features.balances[hero]-features.balances[seat.seat]) / 1000
                q = _clip(.5 + .94*(q-.5) + .018*balance_edge, .05, .95)
                raw_equities.append(q)
                strength[seat.seat] = 1.0 / (.25+models[j][1])
            equity = math.prod(raw_equities) ** .92 if raw_equities else 1.
            if raw_equities and behind[hero] > 0:
                # Equity realization accounts for later betting; an all-in
                # realizes its entire runout equity. Position and aggression
                # improve realization, without adding a global risk tax.
                late = (hero-state.button-1) % len(state.seats) >= len(state.seats)-3
                realization = .84 + .07*late + .06*raised
                realization -= .025*max(0,len(live)-2)
                equity *= max(.70, realization)
            value += probability * _showdown_value(behind, paid, live, hero, payouts, start,
                                                   equity, strength)
        result.append((action, value))
    return tuple(result)


def choose_preflop(state, features):
    candidates = action_values(state, features)
    if not candidates:
        return choose_preflop_loose(state, features)
    # Deterministic tie handling favors the smaller commitment already ordered.
    return max(candidates, key=lambda pair: pair[1])[0]
