"""Read-only observation hooks; strategy inputs and official transitions are unchanged."""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random

from ..benchmark import run

arena = run.arena
BaseTable = arena.SitAndGo
CHECKPOINTS = (0, 20, 40, 80, 120, 250)
_CONTEXT = None
_STREAM = None
_POLICY = None
_PREFLOP = None
_TABLE = None


def emit(kind, **data):
    _STREAM.write(json.dumps({'kind': kind, **_CONTEXT, **data}, separators=(',', ':')) + '\n')


def ranking(stacks, hero):
    value = stacks[hero]
    return {'stack': value, 'share': value / max(1, sum(stacks)),
            'relative_equal_share': value * sum(s > 0 for s in stacks) / max(1, sum(stacks)),
            'rank': 1 + sum(s > value for s in stacks),
            'tied_leaders': sum(s == max(stacks) for s in stacks),
            'players': sum(s > 0 for s in stacks)}


def public_features(state, action):
    """Deterministic helpers from the exact candidate snapshot; no future/opponent holes."""
    own = state.seats[state.seat]
    bb = state.blinds.big_blind
    totals = [s.stack + s.bet for s in state.seats]
    liquid = [s.stack for s in state.seats]
    opponents = [s for s in state.seats if s.seat != state.seat and s.status in ('active', 'all_in')]
    price = _POLICY['eligible_call'](state)
    context = _PREFLOP['_context'](state)
    _, _, _, small, big, position, _, _, effective_bb = context
    position_name = ('SB' if state.seat == small else 'BB' if state.seat == big else
                     'BTN' if state.seat == state.button else 'late' if position >= .65 else
                     'middle' if position >= .35 else 'early')
    street = 'auction' if state.phase == 'auction' else {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}[len(state.board)]
    current_events = []
    for event in reversed(state.history):
        if event.kind in ('board', 'auction'):
            break
        current_events.append(event)
    action_kind = action.kind if action is not None else 'missing'
    cost = (price.call_cost if action_kind == 'call' else
            min(own.stack, max(0, action.amount - own.bet)) if action_kind == 'raise' else 0)
    flags = []
    result = {'hand': state.hand_number, 'street': street, 'position': position_name,
              'players_remaining': state.players_remaining, 'opponents_in_hand': len(opponents),
              'pre_action_stack': ranking(totals, state.seat),
              'liquid_stack': ranking(liquid, state.seat),
              'stack_bb': own.stack / bb, 'effective_bb': effective_bb,
              'pot': state.pot, 'call_cost': price.call_cost, 'eligible_price': price.threshold,
              'eligible_pot': price.eligible_pot_before_call, 'price_exact': price.exact,
              'price_reason': price.reason, 'call_fraction': price.call_cost / max(1, own.stack + own.bet),
              'cost': cost, 'cost_bb': cost / bb, 'cost_fraction': cost / max(1, own.stack + own.bet),
              'can_check': state.can('check') is not None, 'can_raise': state.can('raise') is not None,
              'raises_this_street': sum(e.kind == 'raise' for e in current_events),
              'action': asdict(action) if action is not None else None,
              'cards': state.cards, 'board': state.board,
              'auction_cards': state.variant.get('auction_cards', ()), 'flags': flags}
    if street == 'auction':
        return result
    if action_kind == 'call' and result['call_fraction'] >= .25:
        flags.append('expensive_call')
    if action_kind == 'raise' and result['cost_fraction'] >= .25:
        flags.append('expensive_raise')
    closed = bool(opponents) and all(s.status == 'all_in' for s in opponents)
    if action_kind == 'fold' and closed and len(opponents) == 1 and price.exact and price.threshold <= .05:
        flags.append('cheap_closed_allin_fold')
    if street == 'preflop':
        result['preflop_quality'] = _PREFLOP['preflop_quality'](state.cards)
        unopened = not any(e.kind in ('call', 'raise') for e in state.history)
        result['unopened'] = unopened
        if unopened and action_kind == 'fold' and position_name in ('SB', 'BTN', 'late') and state.can('raise'):
            flags.append('unopened_late_fold')
        return result
    eq = _POLICY['equity'](state, _POLICY['random_stream'](state))
    category, commitment, fd, sd = _POLICY['hand_features'](state.cards, state.board, contribution_veto=True)
    result.update(sampled_equity=eq, equity_samples=_POLICY['SAMPLES'], category=category,
                  commitment=commitment, flush_draw=fd, straight_draw=sd,
                  sample_equity_minus_price=eq-price.threshold)
    board_suits = Counter(c[1] for c in state.board)
    board_only = fd and any(n == 4 and all(c[1] != suit for c in state.cards)
                            for suit, n in board_suits.items())
    result['board_only_flush_draw'] = board_only
    if action_kind == 'call' and board_only:
        flags.append('board_only_flush_draw_call')
    if action_kind == 'call' and eq < price.threshold:
        flags.append('call_below_sampled_price')
        if fd or sd:
            flags.append('draw_call_below_sampled_price')
    if action_kind == 'check' and eq >= .70 and category >= 1 and state.can('raise') and any(s.stack > 0 for s in opponents):
        flags.append('high_equity_free_check')
    if action_kind == 'raise' and state.can('check') and eq < .25:
        flags.append('low_sampled_equity_free_bet')
    return result


class ObservedTable(BaseTable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global _TABLE
        _TABLE = self
        self.hero = _CONTEXT['hero']
        self.observed = None
        self.memory_before = b''
        self.examples = set()
        self.checkpoints = {}
        self.hu_entry = None
        self.entries = {}
        self.digest = hashlib.sha256()
        self.save_checkpoint(0)

    def save_checkpoint(self, hand, absorbed=False):
        from poker_engine.scoring import equity
        self.checkpoints[hand] = {'hand': hand, 'absorbed': absorbed,
                                 **ranking(self.stacks, self.hero),
                                 'icm': equity(self.stacks, self.bust_order, self.payouts, self.ties)[self.hero]}

    def _start_hand(self):
        super()._start_hand()
        if self.hu_entry is None and len(self._positions) == 2 and self.hero in self._positions:
            from poker_engine.scoring import equity
            self.hu_entry = {'hand': self.hands_played + 1, **ranking(self.stacks, self.hero),
                             'icm': equity(self.stacks, self.bust_order, self.payouts, self.ties)[self.hero]}

    def decision(self, **kwargs):
        state = super().decision(**kwargs)
        self.observed = state
        return state

    def apply(self, action, **kwargs):
        state = self.observed
        payload = [state.hand_number, state.seat, state.phase, len(state.history),
                   asdict(action) if action is not None else None]
        self.digest.update(json.dumps(payload, sort_keys=True).encode())
        if state.seat == self.hero:
            fields = public_features(state, action)
            fields['hand_start_stack'] = ranking(self.stacks, self.hero)
            fields['hand_start_icm'] = self._equity[self.hero]
            new_flags = set(fields['flags']) - self.examples
            if new_flags:
                # Only public state plus hero memory. No engine-private cards are exposed.
                fields['example_for'] = sorted(new_flags)
                fields['public_state'] = asdict(state)
                fields['memory_before_hex'] = self.memory_before.hex()
                self.examples.update(new_flags)
            valid = super().apply(action, **kwargs)
            emit('decision', valid=valid, **fields)
            return valid
        return super().apply(action, **kwargs)

    def _end_hand(self):
        super()._end_hand()
        record = self.log[-1]
        before, after = record['stacks_before'], record['stacks_after']
        emit('hand', hand=self.hands_played, before=ranking(before, self.hero),
             after=ranking(after, self.hero), chip_delta=after[self.hero]-before[self.hero],
             icm_before=record['equity_before'][self.hero],
             icm_after=record['equity_after'][self.hero],
             icm_delta=record['equity_after'][self.hero]-record['equity_before'][self.hero],
             normal_bb=record['blinds'][1],
             auction_spent=record.get('flop_chips_before', [1000]*8)[self.hero]
                - record.get('flop_chips_after', [1000]*8)[self.hero])
        if self.hands_played in CHECKPOINTS:
            self.save_checkpoint(self.hands_played)
        for threshold in (4, 3, 2):
            if threshold not in self.entries and sum(s > 0 for s in after) <= threshold:
                from poker_engine.scoring import equity
                self.entries[threshold] = {'threshold': threshold, 'hand': self.hands_played,
                    **ranking(after, self.hero), 'hero_alive': after[self.hero] > 0,
                    'icm': equity(after, self.bust_order, self.payouts, self.ties)[self.hero]}

    def result(self, failures, decisions=()):
        result = super().result(failures, decisions)
        points = arena.table_points(result, self.payouts)
        for hand in CHECKPOINTS:
            if hand not in self.checkpoints:
                self.save_checkpoint(hand, absorbed=True)
        alive = sum(s > 0 for s in self.stacks)
        emit('table', final_points=points[self.hero], hands=self.hands_played,
             outright_win=alive == 1 and self.stacks[self.hero] > 0,
             cap_reached=self.hands_played == self.rules.hand_cap and alive > 1,
             final_rank=ranking(self.stacks, self.hero), hu_entry=self.hu_entry,
             survivor_entries=list(self.entries.values()),
             checkpoints=list(self.checkpoints.values()), action_sha256=self.digest.hexdigest(),
             candidate_decisions=decisions[self.hero], candidate_failures=failures[self.hero])
        return result


class DigestTable(BaseTable):
    """Independent lightweight action capture around the unmodified engine."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.digest = hashlib.sha256()
    def decision(self, **kwargs):
        self.observed = super().decision(**kwargs)
        return self.observed
    def apply(self, action, **kwargs):
        state = self.observed
        self.digest.update(json.dumps([state.hand_number, state.seat, state.phase, len(state.history),
            asdict(action) if action is not None else None], sort_keys=True).encode())
        return super().apply(action, **kwargs)
    def result(self, failures, decisions=()):
        result = super().result(failures, decisions)
        emit('table', action_sha256=self.digest.hexdigest())
        return result


def initialize(config, directory, digest_only=False):
    global _POLICY, _PREFLOP, _DIRECTORY
    arena.initialize(config)
    cls = arena._WORKER[1]['candidate']
    _POLICY = cls.act.__globals__['postflop'].__globals__
    _PREFLOP = cls.act.__globals__['choose_preflop_loose'].__globals__
    _DIRECTORY = Path(directory)
    class WithMemory:
        def __init__(self):
            self.inner = cls()
        def act(self, state, memory):
            _TABLE.memory_before = memory
            return self.inner.act(state, memory)
    if not digest_only:
        arena._WORKER[1]['candidate'] = WithMemory
    arena.SitAndGo = DigestTable if digest_only else ObservedTable


def block(task):
    global _CONTEXT, _STREAM
    panel, index, seed, names = task
    names = names[:]
    random.Random(seed + ':seats').shuffle(names)
    rows = []
    target = _DIRECTORY / f'panel-{panel}-block-{index:04d}.jsonl.gz'
    with gzip.open(target, 'wt', compresslevel=1) as stream:
        _STREAM = stream
        for rotation in range(len(names)):
            seats = names[rotation:] + names[:rotation]
            _CONTEXT = {'panel': panel, 'block': index, 'rotation': rotation,
                        'hero': seats.index('candidate'), 'seed': seed}
            row = arena.run_table(seed, seats)
            row.update(panel=panel, block=index, rotation=rotation)
            rows.append(row)
    return rows, target.name, run.digest(target)
