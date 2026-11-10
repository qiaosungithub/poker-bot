"""Public-only table observations for independent auction and betting policies.

Written during Boston 2026. All mutable state is serialized in returned bytes.
No per-card opponent bids or simulator-private information are consumed.
"""
from dataclasses import dataclass
import json
import math

VERSION = 2
MAX_MEMORY = 8192


@dataclass(frozen=True, slots=True)
class OwnDecision:
    kind: str
    amount: int
    added_chips: int
    history_length: int


@dataclass(frozen=True, slots=True)
class OpponentStats:
    auctions: int
    paid_total: int
    paid_mean: float
    positive_auctions: int
    positive_observations: int
    positive_rate: float
    fraction_mean: float
    fraction_observations: int
    last_paid: int | None
    hands_seen: int
    vpip_hands: int
    preflop_raises: int


@dataclass(frozen=True, slots=True)
class Features:
    balances: tuple[int, ...]
    paid_this_hand: tuple[int | None, ...]
    balance_before_hand: tuple[int | None, ...]
    auction_cards: tuple[str, ...]
    selected_cards: tuple[str, ...]
    dead_cards: tuple[str, ...]
    auction_settled: bool
    history: tuple[OpponentStats, ...]
    hand_number: int
    memory_reset: bool
    auction_participants: tuple[int, ...] = ()
    own_preflop_actions: tuple[OwnDecision, ...] = ()
    own_preflop_added: int = 0
    own_bid: tuple[int, ...] | None = None
    own_bid_reconciled: bool | None = None
    own_flop_action: OwnDecision | None = None
    own_turn_action: OwnDecision | None = None
    own_river_action: OwnDecision | None = None


def _new_own(hand):
    return {'h': hand, 'pre': [], 'added': 0, 'bid': None, 'ok': None,
            'post': [None, None, None], 'pending': None, 'recorded': None}


def _valid_key(key):
    return (isinstance(key, list) and len(key) == 3
            and type(key[0]) is int and key[0] >= 0
            and type(key[1]) is int and key[1] >= 0
            and key[2] in ('betting', 'auction'))


def _fresh(state, balances):
    n = state.config.players
    before = list(balances) if len(state.board) < 3 else [None] * n
    return {'v': VERSION, 't': state.config.table_id, 's': state.seat, 'n': n,
            'h': state.hand_number, 'k': None, 'b': list(balances), 'before': before,
            'eh': 0, 'street': 0, 'pending': 0, 'in_auction': False,
            'ah': 0, 'am': 0, 'seen': 0, 'vpip': 0, 'raises': 0,
            'own': _new_own(state.hand_number),
            # auction count, paid total, positive count, positive known count,
            # fraction sum, fraction known count, last paid, hands, VPIP, raises
            'rows': [[0, 0, 0, 0, 0.0, 0, -1, 0, 0, 0] for _ in range(n)]}


def _decode(state, memory, balances):
    try:
        if not memory or len(memory) > MAX_MEMORY:
            raise ValueError
        m = json.loads(memory)
        n = state.config.players
        if (m['v'] != VERSION or m['t'] != state.config.table_id or m['s'] != state.seat
                or m['n'] != n or m['h'] > state.hand_number):
            raise ValueError
        if len(m['b']) != n or len(m['before']) != n or len(m['rows']) != n:
            raise ValueError
        if any(type(x) is not int or not 0 <= x <= 1000 for x in m['b']):
            raise ValueError
        if any(x is not None and (type(x) is not int or not 0 <= x <= 1000) for x in m['before']):
            raise ValueError
        if any(new > old for new, old in zip(balances, m['b'])):
            raise ValueError  # Currency cannot replenish inside a table.
        for row in m['rows']:
            if len(row) != 10 or any(type(row[j]) is not int or row[j] < 0 for j in (0, 1, 2, 3, 5, 7, 8, 9)):
                raise ValueError
            if not isinstance(row[4], (int, float)) or not math.isfinite(row[4]) or row[4] < 0:
                raise ValueError
            if type(row[6]) is not int or not -1 <= row[6] <= 1000:
                raise ValueError
        if m['k'] is not None and not _valid_key(m['k']):
            raise ValueError
        # Access all cursor fields now so malformed memory resets before mutation.
        for key in ('eh', 'street', 'pending', 'ah', 'am', 'seen', 'vpip', 'raises'):
            if type(m[key]) is not int or m[key] < 0:
                raise ValueError
        if type(m['in_auction']) is not bool:
            raise ValueError
        own = m['own']
        if (own['h'] != m['h'] or not isinstance(own['pre'], list)
                or len(own['pre']) > 16 or type(own['added']) is not int or own['added'] < 0):
            raise ValueError
        for row in own['pre']:
            if (len(row) != 4 or row[0] not in ('fold', 'check', 'call', 'raise')
                    or any(type(v) is not int or v < 0 for v in row[1:])):
                raise ValueError
        if own['added'] < sum(row[2] for row in own['pre']) or len(own['post']) != 3:
            raise ValueError
        for row in own['post']:
            if row is not None and (len(row) != 4 or row[0] not in ('fold', 'check', 'call', 'raise')
                    or any(type(v) is not int or v < 0 for v in row[1:])):
                raise ValueError
        if own['bid'] is not None and (len(own['bid']) != 5
                or any(type(v) is not int or v < 0 for v in own['bid'])
                or sum(own['bid']) > 1000):
            raise ValueError
        if own['ok'] is not None and type(own['ok']) is not bool:
            raise ValueError
        if own['recorded'] is not None and not _valid_key(own['recorded']):
            raise ValueError
        if own['pending'] is not None:
            pending = own['pending']
            if (not _valid_key(pending['key']) or pending['kind'] not in ('fold', 'check', 'call', 'raise', 'bid')
                    or any(type(pending[k]) is not int or pending[k] < 0 for k in ('amount', 'added', 'bet'))
                    or type(pending['index']) is not int or not -4 <= pending['index'] <= 15):
                raise ValueError
        return m, False
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError):
        return _fresh(state, balances), True


def _features(state, m, balances, reset):
    n = state.config.players
    cards = tuple(state.variant.get('auction_cards', ()))
    settled = len(cards) == 5 and len(state.board) >= 3 and state.phase != 'auction'
    selected = tuple(state.board[:3]) if settled else ()
    paid = [None] * n
    if settled:
        for i, previous in enumerate(m['before']):
            if previous is not None and previous >= balances[i]:
                paid[i] = previous - balances[i]
        own_paid = state.variant.get('flop_chips_paid')
        if type(own_paid) is int and 0 <= own_paid <= 1000:
            paid[state.seat] = own_paid
    history = tuple(OpponentStats(r[0], r[1], r[1] / r[0] if r[0] else 0.0,
                    r[2], r[3], (r[2] + 1) / (r[3] + 2),
                    r[4] / r[5] if r[5] else 0.0, r[5],
                    None if r[6] < 0 else r[6], r[7], r[8], r[9]) for r in m['rows'])
    if state.phase == 'auction':
        participants = tuple(s.seat for s in state.seats if s.status in ('active', 'all_in'))
    else:
        mask = m['am'] if m['ah'] == state.hand_number else 0
        participants = tuple(i for i in range(n) if mask & (1 << i))
    own = m['own']
    bid = tuple(own['bid']) if own['bid'] is not None and own['ok'] is not False else None
    return Features(tuple(balances), tuple(paid), tuple(m['before']), cards,
                    selected, tuple(c for c in cards if c not in selected) if settled else (),
                    settled, history, state.hand_number, reset, participants,
                    tuple(OwnDecision(*row) for row in own['pre']), own['added'], bid, own['ok'],
                    *(OwnDecision(*row) if row is not None else None for row in own['post']))


def _reconcile_own(state, m):
    own = m['own']
    pending = own['pending']
    if pending is not None and pending['key'][0] == state.hand_number:
        offset = pending['key'][1]
        for event in state.history[offset:]:
            if event.seat != state.seat or event.kind not in ('fold', 'check', 'call', 'raise', 'bid'):
                continue
            index = pending['index']
            amount = event.amount if event.kind == 'raise' else 0
            added = (max(0, event.amount - pending['bet']) if event.kind == 'raise'
                     else event.amount if event.kind == 'call' else 0)
            if index >= 0 and index < len(own['pre']):
                own['added'] += added - own['pre'][index][2]
                own['pre'][index] = [event.kind, amount, added, offset]
            elif -4 <= index <= -2:
                own['post'][-index - 2] = [event.kind, amount, added, offset]
            own['pending'] = None
            break
    if own['bid'] is not None and len(state.board) >= 3 and state.phase != 'auction':
        candidates = state.variant.get('auction_cards', ())
        actual = state.variant.get('flop_chips_paid')
        if len(candidates) == 5 and type(actual) is int:
            expected = sum(bid for card, bid in zip(candidates, own['bid']) if card in state.board[:3])
            own['ok'] = expected == actual


def prepare(state, memory):
    """Update once from official incremental events and public balance snapshots.

    Timeout replay is deterministic because old memory and the replayed event
    stream are processed together. Feeding the exact same state with its newly
    returned memory is also idempotent, though production does not require it.
    """
    n = state.config.players
    raw = state.variant.get('flop_chips_by_seat', ())
    valid_balances = len(raw) == n and all(type(x) is int and 0 <= x <= 1000 for x in raw)
    balances = tuple(raw) if valid_balances else tuple(
        int(state.variant.get('flop_chips', 0)) if i == state.seat else 0 for i in range(n))
    m, reset = _decode(state, memory, balances)
    key = [state.hand_number, len(state.history), state.phase]
    if m['k'] == key and m['b'] == list(balances):
        return _features(state, m, balances, reset), m
    if m['h'] != state.hand_number:
        m['before'] = list(balances) if len(state.board) < 3 else [None] * n
        m['h'] = state.hand_number
        m['own'] = _new_own(state.hand_number)
    _reconcile_own(state, m)
    if state.phase == 'auction':
        # The platform delays every payment until all sealed bids are in.
        m['before'] = list(balances)
    old_balances = m['b']
    counts = [0] * n
    rows = m['rows']
    # Fresh or malformed memory may receive a long historical event prefix.
    # Rebuild current-hand cursors, but do not count unavailable past payments.
    events = state.events
    if reset:
        # history contains only the current hand and remains available even
        # when returned memory was truncated but the delivery cursor advanced.
        m['eh'] = state.hand_number
        events = state.history
    for event in events:
        kind = event.kind
        if kind == 'hand':
            m['eh'] = event.amount
            m['street'] = m['pending'] = m['seen'] = m['vpip'] = m['raises'] = 0
            m['in_auction'] = False
            continue
        seat = event.seat
        if seat is not None and 0 <= seat < n:
            bit = 1 << seat
            if not m['seen'] & bit:
                rows[seat][7] += 1
                m['seen'] |= bit
            if m['street'] == 0 and not m['in_auction'] and kind in ('call', 'raise'):
                if not m['vpip'] & bit:
                    rows[seat][8] += 1
                    m['vpip'] |= bit
                if kind == 'raise' and not m['raises'] & bit:
                    rows[seat][9] += 1
                    m['raises'] |= bit
        if kind == 'auction':
            m['pending'] = 0
            m['in_auction'] = True
        elif kind == 'bid' and seat is not None and 0 <= seat < n:
            m['pending'] |= 1 << seat
        elif kind == 'board':
            if m['in_auction']:
                mask = m['pending']
                m['ah'], m['am'] = m['eh'], mask
                for i in range(n):
                    if mask & (1 << i):
                        counts[i] += 1
                        rows[i][0] += 1
                m['in_auction'] = False
                m['pending'] = 0
            m['street'] += len(event.cards)
    if valid_balances and not reset:
        for i, count in enumerate(counts):
            if not count:
                continue
            delta = max(0, old_balances[i] - balances[i])
            row = rows[i]
            row[1] += delta
            if count == 1:
                row[2] += delta > 0
                row[3] += 1
                row[6] = delta
                if old_balances[i] > 0:
                    row[4] += delta / old_balances[i]
                    row[5] += 1
            else:
                row[6] = -1
    elif reset:
        # Current postflop history can prove participation but its starting
        # balances are unknown: do not mix these into a paid/auction mean.
        for i, count in enumerate(counts):
            rows[i][0] -= count
    m['b'] = list(balances)
    m['k'] = key
    features = _features(state, m, balances, reset)
    return features, m


def _encode(prepared):
    return json.dumps(prepared, separators=(',', ':'), allow_nan=False).encode()


def observe(state, memory):
    """Public compatible API; pure policies do not need the mutable prepared state."""
    features, prepared = prepare(state, memory)
    return features, _encode(prepared)


def finish(state, action, prepared):
    """Record the final legal returned action and serialize once per decision."""
    own = prepared['own']
    key = [state.hand_number, len(state.history), state.phase]
    if own['recorded'] == key:
        return _encode(prepared)
    index, added = -1, 0
    me = state.seats[state.seat]
    if state.phase == 'auction':
        own['bid'] = list(action.cards)
        own['ok'] = None
    elif not state.board:
        added = (max(0, action.amount - me.bet) if action.kind == 'raise'
                 else min(me.stack, state.to_call) if action.kind == 'call' else 0)
        if len(own['pre']) == 16:
            own['pre'].pop(0)
        index = len(own['pre'])
        own['pre'].append([action.kind, action.amount if action.kind == 'raise' else 0,
                           added, len(state.history)])
        own['added'] += added
    elif len(state.board) in (3, 4, 5):
        added = (max(0, action.amount - me.bet) if action.kind == 'raise'
                 else min(me.stack, state.to_call) if action.kind == 'call' else 0)
        street = len(state.board) - 3
        own['post'][street] = [action.kind, action.amount if action.kind == 'raise' else 0,
                               added, len(state.history)]
        index = -street - 2
    own['pending'] = {'key': key, 'kind': action.kind, 'amount': action.amount,
                      'added': added, 'bet': me.bet, 'index': index}
    own['recorded'] = key
    return _encode(prepared)


def record_own_decision(state, action, observed_memory):
    """Compatibility helper for callers using observe rather than prepare/finish."""
    _, prepared = prepare(state, observed_memory)
    return finish(state, action, prepared)
