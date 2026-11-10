"""Fit equal-teacher, hierarchically smoothed public-action priors.

This is frequency estimation, not a recovered private-card strategy. The policy
projection assumes stronger hands should open first under the existing ranking.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

ACTIONS = ('fold', 'check', 'call', 'raise')


def normalized(counts, parent=None, mass=0):
    parent = parent or dict.fromkeys(ACTIONS, .25)
    n = sum(counts.values())
    return {a: (counts[a] + mass * parent[a]) / (n + mass) for a in ACTIONS}


def fit(data):
    teachers = [r['team_name'] for r in data['teachers']]
    rows = {(r['split'], r['team'], r['context']): Counter(r['actions']) for r in data['counts']}
    contexts = sorted({r['context'] for r in data['counts']})
    global_counts = Counter()
    parent_counts = defaultdict(Counter)
    for (split, team, context), counts in rows.items():
        if split != 'train' or team not in teachers:
            continue
        global_counts.update(counts)
        players, position, depth, situation = context.split('|')
        parent_counts[(players, position, situation)].update(counts)
    global_prior = normalized(global_counts, mass=4)
    priors = {}
    for context in contexts:
        players, position, depth, situation = context.split('|')
        parent = normalized(parent_counts[(players, position, situation)], global_prior, 100)
        # Each donor gets one vote. Sparse donor cells shrink to the pooled
        # same-position/situation parent with 50 pseudo-observations.
        ps = [normalized(rows.get(('train', t, context), Counter()), parent, 50) for t in teachers]
        priors[context] = {'probabilities': {a: sum(p[a] for p in ps) / len(ps) for a in ACTIONS},
                           'train_n': sum(sum(rows.get(('train', t, context), {}).values()) for t in teachers),
                           'test_n': sum(sum(rows.get(('test', t, context), {}).values()) for t in teachers)}
    metrics = {}
    for subset in ('all_first_actions', 'unopened_deep'):
        n = 0
        losses = Counter()
        for (split, team, context), counts in rows.items():
            if split != 'test' or team not in teachers:
                continue
            if subset == 'unopened_deep' and not (context.endswith('|unopened') and '|0-10|' not in context):
                continue
            field = normalized(rows.get(('train', 'FIELD', context), Counter()), global_prior, 100)
            predictions = {'context_prior': priors[context]['probabilities'], 'global_teacher_prior': global_prior,
                           'field_context_prior': field}
            n += sum(counts.values())
            for method, prediction in predictions.items():
                losses[method] += sum(-count * math.log(max(1e-12, prediction[a])) for a, count in counts.items())
        metrics[subset] = {'test_decisions': n, 'cross_entropy_nats': {k: v / n for k, v in losses.items()}}
    return {'schema': 1, 'tournament_id': data['tournament_id'], 'history_sha256': data['history_sha256'],
            'teachers': data['teachers'], 'split': data['split'], 'coverage': data['coverage'],
            'method': 'Equal weight across top 10 teachers; cell smoothing mass 50; parent smoothing mass 100',
            'selection_caveat': data['selection'], 'metrics': metrics, 'priors': priors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    model = fit(json.loads(args.counts.read_text()))
    args.out.write_text(json.dumps(model, indent=2) + '\n')
    print(json.dumps(model['metrics'], indent=2))
    for key, value in model['priors'].items():
        if key.startswith('7-10|') and '|60+|unopened' in key:
            print(key, value)


if __name__ == '__main__':
    main()
