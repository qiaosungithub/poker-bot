"""Bounded five-stage coordination over the frozen event-time hybrid.

All functions are pure. COUPLING_OFF is exact variant betting / tournament
auction. Auction currency never changes a poker pot, call price, or chip stack.
"""
from poker_agent_sdk import Action
from joint_betting_base import Agent as BettingBase, preflop_quality, hand_quality, legal_raise, safe_pass, RANK
from joint_auction_base import card_code, auction_bids, board_quality, TRIPLES


def _own(state):
    return next(s for s in state.seats if s.seat == state.seat)


def _opponents(state):
    return tuple(s for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in'))


def _baseline_bet(state):
    return BettingBase().act(state, b'')[0]


def _optionality(cards):
    """Shape of a speculative option, not a claim of exact auction equity."""
    a, b = sorted((RANK[c[0]] for c in cards), reverse=True)
    if a == b and a <= 9:
        return .8
    if cards[0][1] == cards[1][1] and a >= 8 and a - b <= 3:
        return 1.0
    return 0.0


def _balances(state, features):
    values = getattr(features, 'balances', ()) if features is not None else ()
    if not values:
        values = state.variant.get('flop_chips_by_seat', ())
    return values


def preflop_adjustment(state, features):
    """At most 0.025 in the baseline's heuristic quality scale."""
    if len(state.cards) != 2 or state.board or state.phase != 'betting':
        return 0.0
    own = _own(state)
    bb = max(1, state.blinds.big_blind)
    option = _optionality(state.cards)
    price = min(own.stack, state.to_call)
    # The option is not used to justify large calls or short-stack speculation.
    if not option or own.stack < 20 * bb or price > 2.5 * bb or price > own.stack * .08:
        return 0.0
    opponents = _opponents(state)
    balances = _balances(state, features)
    if state.seat >= len(balances) or not opponents or any(s.seat >= len(balances) for s in opponents):
        return 0.0
    ours = max(0, balances[state.seat])
    other = sum(max(0, balances[s.seat]) for s in opponents) / len(opponents)
    advantage = (ours - other) / max(200., ours, other)
    return .025 * option * advantage


def _preflop_with_adjustment(state, adjustment):
    """Same frozen preflop rules, with only a bounded quality adjustment."""
    own = _own(state)
    bb = max(1, state.blinds.big_blind)
    stack_bb = own.stack / bb
    to_call = min(own.stack, state.to_call)
    active = [s.seat for s in state.seats if s.status != 'busted']
    ordered = sorted(active, key=lambda seat: (seat - state.button - 1) % len(state.seats))
    late = state.seat in ordered[-3:]
    quality = preflop_quality(state.cards) + adjustment
    raises = 0
    for event in reversed(state.history):
        if event.kind in ('board', 'auction'):
            break
        if event.kind == 'raise':
            raises += 1
    open_threshold = .59 if late else .66
    if state.players_remaining <= 3:
        open_threshold -= .075
    if stack_bb <= 10 and quality >= open_threshold - .02 and raises < 2:
        return legal_raise(state, own.stack + own.bet)
    if raises == 0 and quality >= open_threshold:
        limpers = sum(e.kind == 'call' for e in state.history)
        return legal_raise(state, bb * (2.4 + min(3, limpers) * .8))
    if quality >= .89 or quality >= .83 and raises == 1 and state.players_remaining <= 4:
        return legal_raise(state, max(bb * 7, (own.bet + state.to_call) * 2.7))
    if state.can('check'):
        return Action.check()
    required = .56 if to_call <= bb else .65 if to_call <= 2.5 * bb else .73 if to_call <= 6 * bb else .81
    if raises > 1:
        required += .04
    if state.players_remaining <= 3:
        required -= .055
    if to_call > own.stack * .45 and stack_bb > 10:
        required = max(required, .82)
    if quality >= required and state.can('call'):
        return Action.call()
    return safe_pass(state)


def choose_preflop(state, features, coupling=True):
    if not coupling:
        return _baseline_bet(state)
    adjustment = preflop_adjustment(state, features)
    return _preflop_with_adjustment(state, adjustment) if adjustment else _baseline_bet(state)


def _auction_args(state):
    own = _own(state)
    legal = state.can('bid')
    return (tuple(card_code(c) for c in state.cards),
            tuple(card_code(c) for c in state.variant['auction_cards']),
            max(0, legal.max_amount if legal else 0), own)


def _base_auction(state):
    hole, candidates, budget, own = _auction_args(state)
    return auction_bids(hole, candidates, budget,
                        hand_number=state.hand_number, hand_cap=state.config.hand_cap,
                        stack=own.stack, big_blind=max(1, state.blinds.big_blind),
                        pot=state.pot, opponents=max(1, len(_opponents(state))),
                        all_in=own.status == 'all_in')


def _speculative_entry(state, features):
    if features is None or getattr(features, 'memory_reset', False):
        return False
    own = _own(state)
    bb = max(1, state.blinds.big_blind)
    added = getattr(features, 'own_preflop_added', 0)
    return bool(_optionality(state.cards) and own.status != 'all_in'
                and own.stack >= 15 * bb and 0 < added <= 4 * bb)


def choose_bid(state, features, coupling=True):
    if not coupling or not _speculative_entry(state, features):
        return _base_auction(state)
    hole, candidates, budget, own = _auction_args(state)
    if budget <= 0:
        return (0, 0, 0, 0, 0)
    qualities = {combo: board_quality(hole, tuple(candidates[i] for i in combo))
                 for combo in TRIPLES}
    target = max(TRIPLES, key=lambda combo: qualities[combo])
    best = qualities[target]
    spread = best - min(qualities.values())
    if spread < .025:
        return (0, 0, 0, 0, 0)
    # Same auction component; this one multiplier is its coordination term.
    factor = 1.12 if best >= .68 and spread >= .20 else .88 if best < .38 or spread < .10 else 1.0
    bb = max(1, state.blinds.big_blind)
    remaining = max(1, state.config.hand_cap - state.hand_number + 1)
    horizon = max(1., min(remaining * .32, 8. + own.stack / bb * .55))
    importance = state.pot / max(state.pot + own.stack, bb)
    pressure = 1. + 3. * importance + 3. * spread
    fraction_cap = 1. if remaining == 1 else (.65 if remaining < 5 else .28)
    spend = min(budget, max(1, int(min(budget * fraction_cap, budget * pressure / horizon) * factor)))
    # Preserve the original cap; extra confidence never creates a bigger risk limit.
    spend = min(spend, max(1, int(budget * fraction_cap)))
    outside = tuple(i for i in range(5) if i not in target)
    weights = []
    for i in target:
        alternatives = [tuple(sorted((set(target) - {i}) | {j})) for j in outside]
        loss = best - sum(qualities[c] for c in alternatives) / len(alternatives)
        weights.append(max(.025, loss))
    total = sum(weights)
    bids = [0] * 5
    for i, weight in zip(target, weights):
        bids[i] = int(spend * weight / total)
    spare = spend - sum(bids)
    order = sorted(range(3), key=lambda k: weights[k], reverse=True)
    for k in range(spare):
        bids[target[order[k % 3]]] += 1
    return tuple(bids)


def _action_kind(action):
    return getattr(action, 'kind', None) if action is not None else None


def _preflop_initiative(features):
    if features is None:
        return False
    actions = getattr(features, 'own_preflop_actions', ())
    return bool(actions and any(_action_kind(a) == 'raise' for a in actions))


def _plan_missed(state, features, current_quality):
    """Compare the earlier preferred flop with the realized cards, not bid cost."""
    if features is None or getattr(features, 'own_bid_reconciled', None) is not True:
        return False
    bids = getattr(features, 'own_bid', None)
    candidates = tuple(getattr(features, 'auction_cards', ()))
    if not bids or len(bids) != 5 or len(candidates) != 5 or len(state.board) < 3:
        return False
    target = tuple(i for i, amount in enumerate(bids) if amount > 0)
    if len(target) != 3:
        return False
    desired = tuple(candidates[i] for i in target)
    if set(desired) == set(state.board[:3]):
        return False
    target_quality = hand_quality(state.cards, desired)[0]
    actual_flop_quality = hand_quality(state.cards, tuple(state.board[:3]))[0]
    # A useful turn/river can overturn the earlier disappointment.
    return target_quality - actual_flop_quality >= .15 and current_quality < target_quality - .08


def _heavy_competing_spend(state, features):
    if features is None:
        return False
    payments = getattr(features, 'paid_this_hand', ())
    history = getattr(features, 'history', ())
    for opponent in _opponents(state):
        seat = opponent.seat
        if seat >= len(payments) or payments[seat] is None:
            continue
        average = getattr(history[seat], 'paid_mean', 0.) if seat < len(history) else 0.
        if payments[seat] >= max(60., 1.8 * average + 15.):
            return True
    return False


def _draw_completion(board):
    """Public texture transition, not proof of an opponent's actual draw."""
    flop = board[:3]
    if len(board) < 4:
        return False
    for suit in 'cdhs':
        if sum(c[1] == suit for c in flop) == 2 and sum(c[1] == suit for c in board) >= 3:
            return True
    before = {RANK[c[0]] for c in flop}
    after = {RANK[c[0]] for c in board}
    if 14 in before:
        before.add(1)
    if 14 in after:
        after.add(1)
    return any(len(before.intersection(range(low, low + 5))) >= 2
               and len(after.intersection(range(low, low + 5))) >= 4 for low in range(1, 11))


def _marginal_defend(state, quality, category, draw):
    players = len(_opponents(state)) + 1
    q = quality - min(.16, max(0, players - 2) * .035)
    paid_places = sum(p > 0 for p in state.config.payouts)
    risk = .03 if state.players_remaining == paid_places + 1 else 0.
    raises = 0
    for event in reversed(state.history):
        if event.kind in ('board', 'auction'):
            break
        raises += event.kind == 'raise'
    own = _own(state)
    defend = q - .22 - risk - min(.10, max(0, raises - 1) * .05)
    if category >= 3:
        defend += .10
    call = min(own.stack, state.to_call)
    if call > own.stack * .5 and category < 2 and not draw:
        defend -= .12
    odds = call / max(1, max(state.blinds.big_blind, state.pot) + call)
    return defend - odds


def choose_postflop(state, features, coupling=True):
    base = _baseline_bet(state)
    if not coupling or features is None or getattr(features, 'memory_reset', False):
        return base
    own = _own(state)
    quality, category, draw = hand_quality(state.cards, state.board)
    players = len(_opponents(state)) + 1
    missed = _plan_missed(state, features, quality)
    heavy = _heavy_competing_spend(state, features)
    # Lost speculative option: do not turn the old bid into a commitment to bluff.
    if missed and base.kind == 'raise' and state.can('check') and category < 2 and quality < .68:
        return Action.check()
    if missed and heavy and base.kind == 'call' and category < 3 and state.can('fold'):
        # Current texture and betting must corroborate; missed rivers attenuate it.
        margin = .025 if len(state.board) == 3 else .035 if _draw_completion(state.board) else .008
        if 0 <= _marginal_defend(state, quality, category, draw) < margin:
            return Action.fold()
    if base.kind != 'check' or players != 2 or missed or heavy or not state.can('raise'):
        return base
    bb = max(1, state.blinds.big_blind)
    pot = max(bb, state.pot)
    mix = (state.hand_number * 43 + state.seat * 31 + RANK[state.cards[0][0]] * 19 + len(state.history) * 7) % 100
    if len(state.board) == 3 and _preflop_initiative(features):
        dry = max(sum(c[1] == suit for c in state.board) for suit in 'cdhs') <= 2
        if dry and .40 <= quality < .68 and mix < 16:
            return legal_raise(state, own.bet + max(bb, int(.30 * pot)))
    if len(state.board) == 4 and _action_kind(getattr(features, 'own_flop_action', None)) == 'raise':
        # Re-evaluate today's draw; a previous bet alone is not enough to barrel.
        if draw and .43 <= quality < .68 and not _draw_completion(state.board) and mix < 12:
            return legal_raise(state, own.bet + max(bb, int(.40 * pot)))
    # No automatic river follow-through on a failed draw. Ordinary value rules win.
    return base


def choose_flop(state, features, coupling=True):
    return choose_postflop(state, features, coupling)


def choose_turn(state, features, coupling=True):
    return choose_postflop(state, features, coupling)


def choose_river(state, features, coupling=True):
    return choose_postflop(state, features, coupling)


def choose_bet(state, features, coupling=True):
    return choose_preflop(state, features, coupling) if not state.board else choose_postflop(state, features, coupling)
