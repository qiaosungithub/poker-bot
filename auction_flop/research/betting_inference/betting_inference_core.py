"""Independent betting policy using public auction-payment evidence only.

Fresh 2026-11-09 event implementation. The baseline is frozen variant-v1's
betting. This module adds a conservative bounded strength-proxy penalty;
it is a heuristic range tilt, not Bayesian recovery of hidden bids/cards.
"""
from dataclasses import dataclass
from betting_baseline import choose_baseline_bet, hand_quality, RANKS, SUITS

ALL_CARDS = tuple(r + s for r in RANKS for s in SUITS)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    enabled: bool = True
    use_history: bool = True
    maximum_penalty: float = .055
    prior_auctions: float = 8.0
    minimum_paid: int = 4


@dataclass(frozen=True, slots=True)
class Inference:
    penalty: float = 0.0
    largest_signal: float = 0.0
    informative_opponents: int = 0
    made_weight: float = 1.0
    draw_weight: float = 0.0


DEFAULT_CONFIG = InferenceConfig()
OFF_CONFIG = InferenceConfig(enabled=False)
NO_HISTORY_CONFIG = InferenceConfig(use_history=False)


def public_unseen_cards(state, features):
    """Known discarded candidates are dead, never potential opponent cards."""
    known = set(state.cards)
    known.update(state.board)
    known.update(getattr(features, 'auction_cards', ()))
    known.update(getattr(features, 'dead_cards', ()))
    # The native variant field is also public and prevents a missing feature
    # adapter from accidentally admitting known dead candidates.
    known.update(state.variant.get('auction_cards', ()))
    return tuple(c for c in ALL_CARDS if c not in known)


def _texture_weights(state, features):
    flop = tuple(state.board[:3])
    ranks = [RANKS.index(c[0]) + 2 for c in flop]
    suits = [c[1] for c in flop]
    unseen = public_unseen_cards(state, features)
    available_ranks = {r: 0 for r in range(2, 15)}
    for card in unseen:
        available_ranks[RANKS.index(card[0]) + 2] += 1
    pair_rank = next((r for r in ranks if ranks.count(r) >= 2), None)
    # If both remaining copies of a paired rank are known dead/ours, a paid
    # preference cannot imply an opponent privately matching that rank.
    paired_made = pair_rank is not None and available_ranks[pair_rank] > 0
    suit = max(SUITS, key=suits.count)
    suit_count = suits.count(suit)
    flush_draw = .80 if suit_count == 2 else .25 if suit_count == 3 else 0.
    rank_set = set(ranks)
    if 14 in rank_set:
        rank_set.add(1)
    connected = max(sum(r in rank_set for r in range(high - 4, high + 1)) for high in range(5, 15))
    straight_draw = .65 if connected >= 3 else .15
    draw_weight = max(flush_draw, straight_draw)
    if paired_made:
        draw_weight *= .45
    made_weight = 1.0 - .60 * draw_weight
    if pair_rank is not None and not paired_made:
        made_weight *= .70
    if len(state.board) > 3:
        current_suits = [c[1] for c in state.board]
        current_ranks = {RANKS.index(c[0]) + 2 for c in state.board}
        if 14 in current_ranks:
            current_ranks.add(1)
        current_connected = max(sum(r in current_ranks for r in range(high - 4, high + 1)) for high in range(5, 15))
        completed_draw = (suit_count >= 2 and current_suits.count(suit) >= 3) or (connected >= 3 and current_connected >= 4)
        if completed_draw:
            made_weight = max(made_weight, .92)
        elif len(state.board) == 5:
            # High flop spending can have bought a draw which missed. Keep
            # residual made-hand support; do not infer a bluff with certainty.
            made_weight *= 1.0 - .65 * draw_weight
    return made_weight, draw_weight


def infer_pressure(state, features, config=DEFAULT_CONFIG):
    if (not config.enabled or features is None or state.phase != 'betting'
            or len(state.board) < 3 or not getattr(features, 'auction_settled', False)):
        return Inference()
    payments = getattr(features, 'paid_this_hand', ())
    before = getattr(features, 'balance_before_hand', ())
    stats = getattr(features, 'history', ())
    preflop_raisers = set()
    for event in state.history:
        if event.kind in ('auction', 'board'):
            break
        if event.kind == 'raise' and event.seat is not None:
            preflop_raisers.add(event.seat)
    signals = []
    for seat in state.seats:
        i = seat.seat
        if i == state.seat or seat.status not in ('active', 'all_in') or i >= len(payments):
            continue
        paid = payments[i]
        # Zero is censored by selection; no negative range inference follows.
        if paid is None or paid < config.minimum_paid or i >= len(before) or before[i] is None:
            continue
        budget = max(1, int(before[i]))
        if paid > budget:
            continue  # Inconsistent observations provide no strategic signal.
        record = stats[i] if i < len(stats) else None
        n = max(0, int(getattr(record, 'auctions', 0))) if config.use_history else 0
        prior_mean = max(4., min(30., budget * .03))
        historical_mean = max(0., float(getattr(record, 'paid_mean', prior_mean)))
        current_in_history = n > 0 and getattr(record, 'last_paid', None) == paid
        if current_in_history:
            total = max(0., float(getattr(record, 'paid_total', historical_mean * n)) - paid)
            n -= 1
            historical_mean = total / n if n else prior_mean
        support = n / (n + config.prior_auctions)
        expected = (1. - support) * prior_mean + support * historical_mean
        frac_n = max(0, int(getattr(record, 'fraction_observations', 0))) if config.use_history else 0
        frac_prior = .03
        historical_fraction = max(0., float(getattr(record, 'fraction_mean', frac_prior)))
        if current_in_history and frac_n > 0:
            total_fraction = max(0., historical_fraction * frac_n - paid / budget)
            frac_n -= 1
            historical_fraction = total_fraction / frac_n if frac_n else frac_prior
        frac_support = frac_n / (frac_n + config.prior_auctions)
        fraction_mean = (1. - frac_support) * frac_prior + frac_support * historical_fraction
        fraction = paid / budget
        excess = max(0., (paid - expected) / (paid + expected + 1.))
        fraction_excess = max(0., (fraction - fraction_mean) / (fraction + fraction_mean + .05))
        signal = (.65 * excess + .35 * fraction_excess) * paid / (paid + 8.)
        # Uncalibrated high spending is weak evidence; repeated personally
        # unusual spending has greater but still bounded influence.
        signal *= .25 + .75 * support
        if i in preflop_raisers:
            signal *= 1.12
        signals.append(min(1., signal))
    if not signals:
        return Inference()
    made_weight, draw_weight = _texture_weights(state, features)
    largest = max(signals)
    combined = min(1., largest + .20 * (sum(signals) - largest))
    quality, category, draw = hand_quality(state.cards, state.board)
    # Keep genuinely strong private holdings robust against this noisy cue.
    strength_scale = 0. if category >= 6 and quality > .9 else .35 if category >= 3 and quality > .7 else 1.
    penalty = min(config.maximum_penalty, config.maximum_penalty * combined * made_weight * strength_scale)
    return Inference(penalty, largest, len(signals), made_weight, draw_weight)


def choose_preflop(state, features):
    """Stage one: unchanged frozen baseline; no future auction information."""
    return choose_baseline_bet(state)


def choose_postflop(state, features):
    """Stage three: baseline plus bounded public-payment range evidence."""
    return choose_baseline_bet(state, infer_pressure(state, features).penalty)


def choose_bet(state, features):
    return choose_postflop(state, features) if state.board else choose_preflop(state, features)


def choose_bet_without_inference(state, features):
    return choose_baseline_bet(state)


def choose_bet_without_history(state, features):
    return choose_baseline_bet(state, infer_pressure(state, features, NO_HISTORY_CONFIG).penalty)
