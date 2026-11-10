"""Public-history auction policy for the Boston event.

History records selected-card PAYMENT, not hidden bids. Scenario profiles are
explicit uncertain hypotheses, shrunk toward a public-budget prior. There is no
mutable state, private opponent input, or cross-table adaptation in this module.
"""
from auction_value import combination_utilities
from auction_selector import choose_with_scenarios

ZERO = (0, 0, 0, 0, 0)
PROFILE = (0, 1, 1, 2, 4)
PRIOR_STRENGTH = 8.0


def _clamp(value, low, high):
    return min(high, max(low, value))


def _split(total, shift):
    """Whole-number feasible allocation; five components sum exactly to total."""
    weights = tuple(PROFILE[(i + shift) % 5] for i in range(5))
    bids = [total * w // 8 for w in weights]
    left = total - sum(bids)
    order = sorted(range(5), key=lambda i: (-(total * weights[i] % 8), i))
    for i in order[:left]:
        bids[i] += 1
    return tuple(bids)


def pressure_estimates(state, features, history_enabled=True):
    """Per-opponent (seat, balance, modeled total bid, paid-positive proxy).

    Multiplying mean paid by 1.35 is a deliberately coarse model for unseen bids
    on losing cards. It is NOT an unbiased estimator. A pressure floor and a
    high-pressure case remain even after many observed zero payments.
    """
    bb = max(1, state.blinds.big_blind)
    own = next(s for s in state.seats if s.seat == state.seat)
    hands_left = max(1, state.config.hand_cap - state.hand_number + 1)
    stack_bb = (own.stack + own.bet) / bb
    horizon = min(30.0, max(2.0, hands_left * .25), max(3.0, stack_bb * .7))
    balances = features.balances
    estimates = []
    for seat in state.seats:
        if seat.seat == state.seat or seat.status not in ('active', 'all_in'):
            continue
        balance = max(0, int(balances[seat.seat]))
        if balance == 0:
            continue
        opponent_horizon = min(30.0, max(2.0, hands_left * .25),
                               max(3.0, (seat.stack + seat.bet) / bb * .7))
        prior_fraction = 1.0 / max(12.0, opponent_horizon)
        if seat.status == 'all_in':
            # Elimination can destroy future opportunities, not their current balance.
            prior_fraction = max(.20, prior_fraction)
        fraction = prior_fraction
        positive = .55
        if history_enabled:
            stats = features.history[seat.seat]
            n = max(0, stats.fraction_observations)
            # Observed fractions include zero payments, but not unknown values.
            if n:
                empirical = _clamp(stats.fraction_mean, 0.0, 1.0)
                fraction = (n * empirical + PRIOR_STRENGTH * prior_fraction) / (n + PRIOR_STRENGTH)
            m = max(0, stats.positive_observations)
            if m:
                positive = (stats.positive_auctions + PRIOR_STRENGTH * .55) / (m + PRIOR_STRENGTH)
        # Never infer willingness=0 from payment=0: losing-card bids are censored.
        fraction = _clamp(fraction, prior_fraction * .25, .40)
        total = min(balance, max(1, int(balance * fraction * 1.35 + .5)))
        estimates.append((seat.seat, balance, total, _clamp(positive, .08, .95)))
    return tuple(estimates), horizon


def build_scenarios(state, features, history_enabled=True):
    """Four feasible aggregate profiles: quiet, low, typical and high pressure.

    Candidate preference is unobserved. Rotating spread profiles across seats
    and cases keeps this uncertainty explicit; public cards seed only the
    deterministic rotation, not an invented inference of hidden hands.
    """
    estimates, horizon = pressure_estimates(state, features, history_enabled)
    if not estimates:
        return ((ZERO, 1.0),), horizon
    cards = tuple(state.variant.get('auction_cards', ()))
    seed = state.hand_number * 17 + sum((i + 1) * sum(map(ord, c)) for i, c in enumerate(cards))
    quiet = 1.0
    for _, _, _, positive in estimates:
        quiet *= 1.0 - .75 * positive
    quiet = _clamp(quiet, .08, .40)
    # Low and high cases include different allocations, not just a scale change.
    profiles = []
    for case, factor in enumerate((.5, 1.0, 2.0)):
        aggregate = [0] * 5
        for seat, balance, typical, _ in estimates:
            total = min(balance, max(1, int(typical * factor + .5)))
            allocation = _split(total, (seed + seat * 2 + case * 2) % 5)
            for i in range(5):
                aggregate[i] += allocation[i]
        profiles.append(tuple(aggregate))
    remain = 1.0 - quiet
    return ((ZERO, quiet), (profiles[0], remain * .25),
            (profiles[1], remain * .50), (profiles[2], remain * .25)), horizon


def eligible_auction_pot(state):
    """Use public preflop contributions to exclude side pots above our all-in.

    Auction follows preflop, so there has been only one betting street. If a
    synthetic caller supplies no history, retain pot as an explicitly coarse
    scale; the normal SDK supplies the current hand's public betting history.
    """
    own = next(s for s in state.seats if s.seat == state.seat)
    if own.status != 'all_in':
        return state.pot
    contributions = [0] * state.config.players
    street = [0] * state.config.players
    for event in state.history:
        i = event.seat
        if i is None or not 0 <= i < len(contributions):
            continue
        if event.kind == 'ante':
            contributions[i] += event.amount
        elif event.kind in ('blind', 'call'):
            contributions[i] += event.amount
            street[i] += event.amount
        elif event.kind == 'raise':
            contributions[i] += max(0, event.amount - street[i])
            street[i] = event.amount
    own_contribution = contributions[state.seat]
    if own_contribution <= 0:
        return state.pot
    return min(state.pot, sum(min(c, own_contribution) for c in contributions))


def choose_bid(state, features, history_enabled=True):
    """Pure modular contract: official state + immutable public features -> bids."""
    legal = state.can('bid')
    if state.phase != 'auction' or legal is None:
        return ZERO
    budget = max(0, min(int(features.balances[state.seat]), legal.max_amount))
    candidates = tuple(state.variant.get('auction_cards', ()))
    if budget <= 0 or len(candidates) != 5 or len(state.cards) != 2:
        return ZERO
    utilities = combination_utilities(state.cards, candidates)
    scenarios, horizon = build_scenarios(state, features, history_enabled)
    own = next(s for s in state.seats if s.seat == state.seat)
    if own.status == 'all_in':
        horizon = min(horizon, 3.0)
    bb = max(1, state.blinds.big_blind)
    # Pot importance changes opportunity cost, never the normal-chip pot itself.
    accessible = eligible_auction_pot(state)
    importance = max(1.0, (accessible / (5.0 * bb)) ** .5)
    shadow = .025 * horizon / max(1, budget) / importance
    if state.hand_number >= state.config.hand_cap:
        shadow = 0.0
    return choose_with_scenarios(utilities, budget, scenarios, shadow)


def choose_bid_no_history(state, features):
    """Ablation: identical valuation, optimizer and budget; public prior only."""
    return choose_bid(state, features, history_enabled=False)
