"""Generate isolated candidates; never change the baseline or fixed opponents."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'candidates' / 'corrected_preflop_loose_v2'

RUNTIME = '''"""Public-action opening prior distilled from Practice 4 top-ten teams.

Only the first, unopened decision above 10 effective BB is changed. No runtime
learning, identities, future cards, or cross-table state. Donor frequency is
projected through the baseline's existing monotone hand-strength ranking.
"""
from bisect import bisect_left

ALPHA = {alpha!r}
OPEN_PROBABILITY = {probabilities!r}
QUALITIES = {qualities!r}


def runtime_context(state, effective_bb):
    alive = [s.seat for s in state.seats if s.status != 'busted']
    clockwise = sorted(alive, key=lambda i: (i-state.button-1) % len(state.seats))
    small = state.button if len(alive) == 2 else clockwise[0]
    big = clockwise[0] if len(alive) == 2 else clockwise[1]
    if state.seat == small:
        position = 'SB'
    elif state.seat == big:
        position = 'BB'
    elif state.seat == state.button:
        position = 'BTN'
    elif state.seat == clockwise[-2]:
        position = 'CO'
    else:
        position = 'EARLY' if clockwise.index(state.seat)/(len(alive)-1) < .5 else 'MIDDLE'
    players = '2' if len(alive) == 2 else '3' if len(alive) == 3 else '4-6' if len(alive) <= 6 else '7-10'
    depth = '0-10' if effective_bb <= 10 else '10-25' if effective_bb <= 25 else '25-60' if effective_bb <= 60 else '60+'
    return '|'.join((players, position, depth, 'unopened'))


def opening_threshold(state, baseline, effective_bb):
    if effective_bb <= 10 or any(e.kind in ('raise', 'call') or
            (e.seat == state.seat and e.kind in ('check', 'fold')) for e in state.history):
        return baseline
    target = OPEN_PROBABILITY.get(runtime_context(state, effective_bb))
    if target is None:
        return baseline
    original = (len(QUALITIES)-bisect_left(QUALITIES, baseline))/len(QUALITIES)
    probability = (1-ALPHA)*original + ALPHA*target
    index = min(len(QUALITIES)-1, max(0, round((1-probability)*len(QUALITIES))))
    return QUALITIES[index]
'''


def build(model_path, target, alpha):
    model = json.loads(model_path.read_text())
    for package in ('agent-protocol', 'agent-sdk'):
        sys.path.insert(0, str(ROOT / 'vendor' / 'sdk' / package / 'src'))
    sys.path.insert(0, str(BASE))
    from variant_agent import preflop_quality
    deck = [r+s for r in '23456789TJQKA' for s in 'cdhs']
    qualities = tuple(sorted(preflop_quality(cards) for cards in combinations(deck, 2)))
    probabilities = {key: value['probabilities']['raise'] for key, value in model['priors'].items()
                     if key.endswith('|unopened') and '|0-10|' not in key and value['train_n'] >= 300}
    assert 0 < alpha <= 1 and len(qualities) == 1326
    shutil.copytree(BASE, target, ignore=shutil.ignore_patterns('__pycache__'))
    (target / 'distilled_prior.py').write_text(RUNTIME.format(alpha=alpha, probabilities=probabilities, qualities=qualities))
    path = target / 'independent_preflop.py'
    source = path.read_text()
    marker = '    if raises==0 and quality>=open_threshold:\n'
    assert source.count(marker) == 1
    source = source.replace('from poker_agent_sdk import Action\n', 'from poker_agent_sdk import Action\nfrom distilled_prior import opening_threshold\n')
    source = source.replace(marker, '    if raises==0 and limpers==0 and effective_bb>10:\n        open_threshold = opening_threshold(state, open_threshold, effective_bb)\n' + marker)
    path.write_text(source)
    print(json.dumps({'target': str(target), 'alpha': alpha, 'supported_contexts': len(probabilities)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--alpha', type=float, required=True)
    args = ap.parse_args()
    build(args.model, args.target, args.alpha)
