"""Extract first-action priors from official PUBLIC history, without hole cards.

Run from repository root. Private records are only used to validate pairing and
identify the team's scored tables. Never read their cards or future public cards.
All decisions from an iteration stay in the same train/test partition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

ACTIONS = ('fold', 'check', 'call', 'raise')


def position_key(alive, button, seat, seats):
    clockwise = sorted(alive, key=lambda i: (i - button - 1) % seats)
    small = button if len(alive) == 2 else clockwise[0]
    big = clockwise[0] if len(alive) == 2 else clockwise[1]
    if seat == small:
        return 'SB'
    if seat == big:
        return 'BB'
    if seat == button:
        return 'BTN'
    if seat == clockwise[-2]:
        return 'CO'
    return 'EARLY' if clockwise.index(seat) / (len(alive) - 1) < .5 else 'MIDDLE'


def context_key(alive, button, seat, seats, effective_bb, raises, limpers):
    players = '2' if len(alive) == 2 else '3' if len(alive) == 3 else '4-6' if len(alive) <= 6 else '7-10'
    depth = '0-10' if effective_bb <= 10 else '10-25' if effective_bb <= 25 else '25-60' if effective_bb <= 60 else '60+'
    situation = 'reraise' if raises >= 2 else 'raised' if raises else 'limped' if limpers else 'unopened'
    return '|'.join((players, position_key(alive, button, seat, seats), depth, situation))


def decisions(hand):
    """Yield causally reconstructed decisions, before applying each action."""
    total = list(hand['stacks_before'])
    alive = [i for i, stack in enumerate(total) if stack > 0]
    committed = [0] * len(total)
    folded, acted = set(), set()
    raises = limpers = 0
    bb = max(1, hand['blinds'][1])
    for index, event in enumerate(hand['events']):
        kind, seat = event['kind'], event.get('seat')
        if kind in ('auction', 'board', 'win', 'show'):
            break
        amount = event.get('amount', 0)
        if kind == 'ante':
            total[seat] -= amount
        elif kind == 'blind':
            committed[seat] += amount
        elif kind in ACTIONS:
            if seat not in acted:
                covered = max((total[i] for i in alive if i != seat and i not in folded), default=total[seat])
                effective = min(total[seat], covered) / bb
                key = context_key(alive, hand['button'], seat, len(total), effective, raises, limpers)
                yield {'seat': seat, 'event': index, 'context': key, 'action': kind,
                       'to_call_bb': (max(committed) - committed[seat]) / bb,
                       'effective_bb': effective,
                       'raise_to_bb': amount / bb if kind == 'raise' else None}
            acted.add(seat)
            if kind == 'fold':
                folded.add(seat)
            elif kind == 'call':
                committed[seat] += amount
                limpers += raises == 0
            elif kind == 'raise':
                assert amount > committed[seat]
                committed[seat] = amount
                raises += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit-tables', type=int)
    args = ap.parse_args()
    start = time.monotonic()
    results = json.loads(args.results.read_text())
    leaders = sorted((r for r in results['items'] if not r['forfeited']), key=lambda r: -r['average_points'])[:10]
    counts = defaultdict(Counter)
    sizes = defaultdict(Counter)
    coverage = Counter()
    team_tables = Counter()
    examples = []
    with gzip.open(args.history, 'rt') as stream:
        header = json.loads(next(stream))
        assert header['tournament_id'] == results['tournament_id']
        names = {x['entry_id']: x['team_name'] for x in header['entries']}
        assert len(set(names.values())) == len(names), 'Need team ID mapping for duplicate names'
        teacher_names = {r['team_name'] for r in leaders}
        own_name = names[header['you']]
        for table_line in stream:
            table = json.loads(table_line)
            own = json.loads(next(stream))['you']
            assert (table['iteration'], table['table']) == (own['iteration'], own['table'])
            assert table['seats'][own['seat']] == header['you']
            assert table['variant'] == 'flop_auction_holdem'
            coverage['downloaded_tables'] += 1
            if table['fillers'][own['seat']]:
                coverage['own_filler_tables_skipped'] += 1
                continue
            split = 'test' if table['iteration'] % 5 == 4 else 'train'
            coverage[split + '_tables'] += 1
            table_names = [names[e] for e in table['seats']]
            team_tables.update(set(n for i, n in enumerate(table_names) if not table['fillers'][i]))
            automatic = set(map(tuple, own.get('automatic', [])))
            for hand in table['hands']:
                coverage['hands'] += 1
                for row in decisions(hand):
                    seat = row['seat']
                    if table['fillers'][seat]:
                        coverage['filler_decisions_skipped'] += 1
                        continue
                    if seat == own['seat'] and (hand['hand'], row['event']) in automatic:
                        coverage['own_automatic_skipped'] += 1
                        continue
                    name = table_names[seat]
                    key = row['context']
                    for group in ('FIELD', name) if name in teacher_names or name == own_name else ('FIELD',):
                        counts[(split, group, key)][row['action']] += 1
                    coverage['first_decisions'] += 1
                    if name in teacher_names:
                        coverage['teacher_first_decisions'] += 1
                        if row['action'] == 'raise':
                            sizes[(split, name, key)][str(round(row['raise_to_bb'], 2))] += 1
                        if len(examples) < 20:
                            examples.append(dict(iteration=table['iteration'], table=table['table'], hand=hand['hand'], team=name, **row))
            if coverage['downloaded_tables'] % 1000 == 0:
                print(dict(coverage), flush=True)
            if args.limit_tables and coverage['downloaded_tables'] >= args.limit_tables:
                break
    output = {'schema': 1, 'tournament_id': header['tournament_id'], 'you': header['you'],
              'own_name': own_name, 'history_sha256': hashlib.file_digest(args.history.open('rb'), 'sha256').hexdigest(),
              'results_sha256': hashlib.file_digest(args.results.open('rb'), 'sha256').hexdigest(),
              'split': 'test iff iteration % 5 == 4; all other iterations train',
              'selection': 'Top 10 nonforfeited teams by full-round average_points; ranking is not independently held out',
              'teachers': leaders, 'coverage': coverage, 'team_tables': team_tables,
              'counts': [{'split': s, 'team': t, 'context': c, 'actions': dict(v)} for (s,t,c),v in sorted(counts.items())],
              'raise_sizes': [{'split': s, 'team': t, 'context': c, 'histogram': dict(v)} for (s,t,c),v in sorted(sizes.items())],
              'examples': examples, 'seconds': time.monotonic() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'coverage': coverage, 'seconds': output['seconds']}), flush=True)


if __name__ == '__main__':
    main()
