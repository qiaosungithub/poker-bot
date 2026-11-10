"""Validate data/runtime context parity against actual official-engine states.

Also measures isolated prior lookup cost (not SDK or complete-bot latency).
"""
import argparse
from bisect import bisect_left
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ('agent-protocol', 'engine', 'agent-sdk'):
    sys.path.insert(0, str(ROOT / 'vendor' / 'sdk' / package / 'src'))
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant
from .extract import context_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.candidate.resolve()))
    from agent import Agent
    from independent_preflop import _context
    from distilled_prior import runtime_context, opening_threshold, OPEN_PROBABILITY, QUALITIES
    from variant_agent import preflop_quality
    deck = [r+s for r in '23456789TJQKA' for s in 'cdhs']
    assert QUALITIES == tuple(sorted(preflop_quality(c) for c in combinations(deck, 2)))
    counts = Counter()
    times = []
    examples = {}
    for i in range(8):
        rules = TableRules(variant='flop_auction_holdem', players=8)
        table = SitAndGo(rules=rules, variant=load_variant(rules.variant), table_id=f'prior-parity-{i}', seed=f'prior-parity-{i}')
        memories = [b''] * 8
        while (state := table.decision()) is not None:
            if state.phase == 'betting' and not state.board:
                own, opponents, bb, small, big, position, raises, limpers, effective = _context(state)
                alive = [s.seat for s in state.seats if s.status != 'busted']
                expected = context_key(alive, state.button, state.seat, len(state.seats), effective, 0, 0)
                assert runtime_context(state, effective) == expected
                baseline = .69 - .12 * position
                if state.seat == small: baseline = .63
                elif state.seat == big: baseline = .64
                if state.players_remaining == 2: baseline = .50 if state.seat == small else .56
                elif state.players_remaining == 3: baseline -= .045
                baseline += min(.03, .01 * limpers)
                if raises == 0 and limpers == 0 and position >= .65 and state.seat not in (small,big) and state.players_remaining >= 3: baseline -= .035
                start = time.perf_counter_ns()
                projected = opening_threshold(state, baseline, effective)
                times.append(time.perf_counter_ns() - start)
                if effective <= 10 or raises or limpers:
                    assert projected == baseline
                    counts['unchanged_guard_cases'] += 1
                elif expected in OPEN_PROBABILITY:
                    counts['supported_first_open_cases'] += 1
                    examples[expected] = {'old_threshold': baseline, 'new_threshold': projected,
                        'old_open_rate': (1326-bisect_left(QUALITIES, baseline))/1326,
                        'new_open_rate': (1326-bisect_left(QUALITIES, projected))/1326,
                        'teacher_open_rate': OPEN_PROBABILITY[expected]}
                counts['preflop_states'] += 1
            action, memories[state.seat] = Agent().act(state, memories[state.seat])
            assert table.apply(action), (state, action)
            counts['legal_decisions'] += 1
    assert counts['supported_first_open_cases'] > 100
    assert counts['unchanged_guard_cases'] > 100
    out = {'tables': 8, 'counts': counts, 'lookup_p50_us': statistics.median(times)/1000,
           'lookup_p99_us': sorted(times)[int(.99*(len(times)-1))]/1000,
           'lookup_max_us': max(times)/1000, 'scope': 'Direct in-process prior lookup only; shared host, excludes SDK/IPC and other bot stages',
           'projection_examples': examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k,v in out.items() if k != 'projection_examples'}, indent=2))


if __name__ == '__main__':
    main()
