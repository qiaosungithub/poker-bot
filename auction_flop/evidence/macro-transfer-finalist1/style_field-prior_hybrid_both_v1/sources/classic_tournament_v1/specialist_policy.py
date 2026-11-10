"""Fresh transcription of TournamentSpecialist's executable rules.

Only public state is consumed. Numerical ranges, sizes and mixing probabilities
are source-matched; semantic fixes are enumerated in SOURCE_TO_RULE.md.
"""
import hashlib
from collections import Counter
import eval7
from poker_agent_sdk import Action
from pot_accounting import eligible_call
from hand_contribution import shared_nut_action, public_quads_share

RANKS = '23456789TJQKA'
RANK = {r: i for i, r in enumerate(RANKS)}
DECK = tuple(r + s for r in RANKS for s in 'cdhs')
CARD = {c: eval7.Card(c) for c in DECK}


def range_set(text):
    return frozenset(text.split())


JAM10 = range_set('AA KK QQ JJ TT 99 88 77 66 55 44 33 22 AKs AKo AQs AQo AJs AJo ATs ATo A9s A8s A7s A5s A4s A3s A2s KQs KQo KJs KJo KTs K9s QJs QTs JTs T9s 98s')
JAM15 = range_set('AA KK QQ JJ TT 99 88 77 66 55 AKs AKo AQs AQo AJs AJo ATs KQs KQo KJs QJs JTs')
JAMCALL = range_set('AA KK QQ JJ TT 99 AKs AKo AQs AQo AJs KQs')
MIDOPEN = range_set('AA KK QQ JJ TT 99 88 77 66 55 44 AKs AKo AQs AQo AJs AJo ATs ATo A9s A5s A4s KQs KQo KJs KTs QJs QTs JTs T9s 98s')
DEEPOPEN = range_set('AA KK QQ JJ TT 99 88 77 66 55 AKs AKo AQs AQo AJs AJo ATs KQs KQo KJs KTs QJs QTs JTs T9s 98s 87s 76s')
LEADEREXTRA = range_set('44 33 22 A9s A8s A7s A5s A4s A3s A2s K9s Q9s J9s T8s 97s 86s 65s 54s ATo KJo QJo')
THREEBET = range_set('AA KK QQ JJ AKs AKo AQs AQo AJs A5s A4s')
SHORTTHREEBET = range_set('AA KK QQ AKs AKo')
CALL3 = range_set('TT 99 88 77 AJs ATs AQo KQs KJs QJs JTs T9s')


def passive(state):
    if state.can('check'):
        return Action.check()
    if state.can('fold'):
        return Action.fold()
    return Action.call()


def call(state):
    return Action.call() if state.can('call') else passive(state)


def raise_to(state, target):
    legal = state.can('raise')
    if legal is None:
        return call(state)
    return Action.raise_to(min(legal.max_amount, max(legal.min_amount, int(target))))


def jam(state):
    me = state.seats[state.seat]
    return raise_to(state, me.stack + me.bet)


def hand_class(cards):
    a, b = sorted(cards, key=lambda c: RANK[c[0]], reverse=True)
    return a[0] + b[0] + ('' if a[0] == b[0] else 's' if a[1] == b[1] else 'o')


def context(state):
    me = state.seats[state.seat]
    bb = max(1, state.blinds.big_blind)
    opponents = [s.stack for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in')]
    # The original implementation uses the maximum opponent, despite its
    # docstring saying minimum. Preserve the executable rule.
    depth = min(me.stack, max(opponents)) / bb if opponents else me.stack / bb
    leader = me.stack > 1.3 * max((s.stack for s in state.seats if s.seat != state.seat), default=0)
    short = bool(opponents) and min(opponents) / bb < 15
    raises = sum(e.kind == 'raise' for e in state.history)
    last = next((e.seat for e in reversed(state.history) if e.kind == 'raise'), None)
    return depth, leader, short, raises, last == state.seat


def preflop(state):
    key = hand_class(state.cards)
    depth, leader, short, raises, _ = context(state)
    me = state.seats[state.seat]
    cost = min(me.stack, state.to_call)
    bb = max(1, state.blinds.big_blind)
    if depth <= 15:
        opening = JAM10 if depth <= 10 else JAM15
        allowed = (key in opening if raises == 0 else
                   key in JAMCALL or depth <= 12 and key in opening and cost <= .30 * me.stack if raises == 1 else
                   key in SHORTTHREEBET)
        return jam(state) if allowed else passive(state)
    if depth <= 30:
        if raises == 0:
            return raise_to(state, int(2.2 * bb)) if key in MIDOPEN else passive(state)
        if raises == 1:
            if key in JAMCALL:
                return jam(state)
            return call(state) if key in MIDOPEN and cost <= .12 * me.stack else passive(state)
        return jam(state) if key in ('AA', 'KK', 'QQ', 'AKs') else passive(state)
    opening = DEEPOPEN | LEADEREXTRA if leader and not short else DEEPOPEN
    if raises == 0:
        return raise_to(state, int(2.5 * bb)) if key in opening else passive(state)
    if raises == 1:
        if key in (SHORTTHREEBET if short else THREEBET):
            return raise_to(state, int(3.0 * (me.bet + state.to_call)))
        return call(state) if key in opening and key in CALL3 and cost <= .12 * me.stack else passive(state)
    if key in ('AA', 'KK', 'AKs'):
        return jam(state)
    if raises == 2 and (key in ('QQ', 'JJ', 'AKo') and cost <= .50 * me.stack or key in CALL3 and cost <= .20 * me.stack):
        return call(state)
    return passive(state)


def known_unavailable(state):
    """Only public cards, including both auction losers and actual shown hands."""
    return set(state.variant.get('auction_cards', ())) | {
        card for seat in state.seats for card in seat.shown_cards}


def hand_strength(cards, board, unavailable=()):
    """Original five labels, with numeric evaluation and private contribution.

    Public-made hands do not inherit a strong label merely from their board.
    Real private trips/quads use eval7 categories 3/7 (labels are Trips/Quads).
    """
    score = eval7.evaluate([CARD[c] for c in tuple(cards) + tuple(board)])
    category = score >> 24
    hr = [RANK[c[0]] for c in cards]
    br = [RANK[c[0]] for c in board]
    counts = Counter(br)
    combined = Counter(hr + br)
    board_only = len(board) == 5 and score == eval7.evaluate([CARD[c] for c in board])
    own_pair = hr[0] == hr[1] or any(r in counts for r in hr)
    overpair = hr[0] == hr[1] and hr[0] > max(br)
    if not board_only:
        if category in (4, 5, 6, 8):
            return 'strong'
        if category in (3, 7):
            made_rank = next(r for r, n in combined.items() if n == (3 if category == 3 else 4))
            if made_rank in hr:
                return 'strong'
            # Public trips with a private kicker are a marginal made hand.
            if category == 3:
                return 'medium'
        if category == 2:
            own_pairs = {r for r in hr if counts[r] and combined[r] >= 2}
            if len(own_pairs) == 2 or overpair:
                return 'strong'
            if own_pair:
                return 'medium'
        if category == 1 and own_pair:
            return 'strong' if overpair else 'medium'
    # Draw logic retains the source's suited-hole / four-in-five detector,
    # then requires a live private-contributing completion; no river draws.
    if len(board) < 5:
        known = set(cards) | set(board) | set(unavailable)
        if cards[0][1] == cards[1][1] and sum(c[1] == cards[0][1] for c in board) >= 2:
            if any(c[1] == cards[0][1] and c not in known for c in DECK):
                return 'draw'
        ranks = set(hr + br)
        for low in range(11):
            window = set(range(low, low + 5))
            missing = window - ranks
            if len(missing) == 1 and window.intersection(set(hr) - set(br)):
                rank = next(iter(missing))
                if rank < 13 and any(RANKS[rank] + suit not in known for suit in 'cdhs'):
                    return 'draw'
    return 'overcard' if max(hr) >= 12 else 'air'


def mix(state):
    street = {3: 'flop', 4: 'turn', 5: 'river'}[len(state.board)]
    key = (state.config.table_id, state.hand_number, state.seat,
           tuple(state.cards), tuple(state.board), street, 'ts')
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), 'big') / (1 << 64)


def postflop(state):
    shared = shared_nut_action(state)
    if shared is not None:
        return shared
    price = eligible_call(state)
    unavailable = known_unavailable(state)
    quads = public_quads_share(state.cards, state.board, unavailable)
    if quads is not None:
        if state.can('check'):
            return Action.check()
        opponents = [s for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in')]
        if len(state.board) == 5 and opponents and all(len(s.shown_cards) == 2 for s in opponents):
            hero = eval7.evaluate([CARD[c] for c in state.cards + state.board])
            values = [eval7.evaluate([CARD[c] for c in s.shown_cards + state.board]) for s in opponents]
            best = max(values)
            quads = 0. if best > hero else 1. / (1 + values.count(hero)) if best == hero else 1.
        else:
            quads **= max(1, len(opponents))
        return call(state) if quads >= price.threshold else passive(state)
    strength = hand_strength(state.cards, state.board, unavailable)
    depth, leader, short, _, aggressor = context(state)
    roll = mix(state)
    me = state.seats[state.seat]
    cost = price.call_cost
    count = len(state.board)

    def bet(fraction):
        # Eligible normal chips replace an inaccessible side pot. Preserve
        # the original pot sizing whenever the whole pot is reachable.
        pot = max(1, price.eligible_pot_before_call)
        return raise_to(state, me.bet + cost + int((pot + cost) * fraction))

    if depth < 8:
        if not state.can('check') and cost > 0:
            return jam(state) if strength == 'strong' or strength == 'medium' and cost <= .30 * me.stack else passive(state)
        return jam(state) if strength in ('strong', 'medium') else passive(state)
    if not state.can('check') and cost > 0:
        if strength == 'strong':
            return bet(.7) if count in (3, 4) and roll < .30 else call(state)
        if strength == 'medium':
            return call(state) if price.threshold <= .30 else passive(state)
        if strength == 'draw':
            return call(state) if price.threshold <= .28 else passive(state)
        return passive(state)
    frequency = .50 if short else .80 if leader else .65
    if aggressor and count == 3:
        if strength == 'strong':
            return bet(.6)
        if strength in ('medium', 'draw') and roll < frequency:
            return bet(.5)
        if strength in ('overcard', 'air') and roll < frequency * .7:
            return bet(.4)
    elif aggressor and count == 4:
        if strength == 'strong':
            return bet(.7)
        if strength == 'draw' and roll < .50:
            return bet(.6)
    elif aggressor and count == 5:
        if strength == 'strong':
            return bet(.75)
        if strength == 'overcard' and roll < .20 and leader:
            return bet(.6)
    elif not aggressor and strength == 'strong' and roll < .50:
        return bet(.5)
    return passive(state)
