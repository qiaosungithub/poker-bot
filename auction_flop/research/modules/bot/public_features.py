"""Public-only table observations for independent auction and betting policies.

Written during Boston 2026. All mutable state is serialized in returned bytes.
No per-card opponent bids or simulator-private information are consumed.
"""
from dataclasses import dataclass
import json
import math

VERSION = 1
MAX_MEMORY = 8192


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


def _fresh(state, balances):
    n = state.config.players
    before = list(balances) if len(state.board) < 3 else [None] * n
    return {'v': VERSION, 't': state.config.table_id, 's': state.seat, 'n': n,
            'h': state.hand_number, 'k': None, 'b': list(balances), 'before': before,
            'eh': 0, 'street': 0, 'pending': 0, 'in_auction': False,
            'ah': 0, 'am': 0, 'seen': 0, 'vpip': 0, 'raises': 0,
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
        if m['k'] is not None and (not isinstance(m['k'], list) or len(m['k']) != 3):
            raise ValueError
        # Access all cursor fields now so malformed memory resets before mutation.
        for key in ('eh', 'street', 'pending', 'ah', 'am', 'seen', 'vpip', 'raises'):
            if type(m[key]) is not int or m[key] < 0:
                raise ValueError
        if type(m['in_auction']) is not bool:
            raise ValueError
        return m, False
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError):
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
    return Features(tuple(balances), tuple(paid), tuple(m['before']), cards,
                    selected, tuple(c for c in cards if c not in selected) if settled else (),
                    settled, history, state.hand_number, reset, participants)


def observe(state, memory):
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
        return _features(state, m, balances, reset), memory
    if m['h'] != state.hand_number:
        m['before'] = list(balances) if len(state.board) < 3 else [None] * n
        m['h'] = state.hand_number
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
    encoded = json.dumps(m, separators=(',', ':'), allow_nan=False).encode()
    return features, encoded
