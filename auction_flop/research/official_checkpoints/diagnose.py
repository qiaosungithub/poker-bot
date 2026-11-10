"""Public-state, pre-result checkpoints and exact ICM-change accounting.

No hole cards are read. All strata are fixed before inspecting outcomes.
One non-filler hero table per iteration is required for reported intervals.
ICM and chip proportionality are descriptive baselines, not calibrated truths.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

HAND_CHECKPOINTS = (1, 21, 41, 61, 81, 101, 121, 161, 201)
ENTRY_CHECKPOINTS = (6, 4, 3, 2)
ACTIONS = ('fold', 'check', 'call', 'raise')


def bucket(value, boundaries, labels):
    return next((label for bound, label in zip(boundaries, labels) if value < bound), labels[-1])


def icm_features(stacks, hero, payouts=(50, 30, 20)):
    """Top-three Plackett-Luce probabilities and exact expected rank identity."""
    s = stacks[hero]
    assert s > 0
    other = [v for i,v in enumerate(stacks) if i != hero and v > 0]
    total = s + sum(other)
    p1 = s / total
    p2 = sum(v / total * s / (total-v) for v in other)
    p3 = sum(v/total * w/(total-v) * s/(total-v-w)
             for i,v in enumerate(other) for j,w in enumerate(other) if i != j)
    return {'chip_share': p1, 'expected_top3': min(1., p1+p2+p3),
            'expected_rank': 1 + sum(v/(s+v) for v in other),
            'computed_icm_points': sum(p*q for p,q in zip(payouts, (p1,p2,p3)))}


def checkpoint(hand, hero):
    stacks = hand['stacks_before']
    live = [v for v in stacks if v > 0]
    value = icm_features(stacks, hero)
    value.update(hand=hand['hand'], players_left=len(live), hero_stack=stacks[hero],
                 hero_rank=1+sum(v > stacks[hero] for v in live),
                 own_bb=stacks[hero]/hand['blinds'][1],
                 effective_bb=min(stacks[hero],max(v for i,v in enumerate(stacks) if i != hero))/hand['blinds'][1],
                 auction_chips=hand['flop_chips_before'][hero],
                 icm_points=hand['equity_before'][hero])
    assert abs(value['computed_icm_points']-value['icm_points']) < .00011, value
    return value


def final_result(table, hero):
    final = table['hands'][-1]
    rank = None
    for hand in table['hands']:
        before, after = hand['stacks_before'], hand['stacks_after']
        if before[hero] > 0 and after[hero] == 0:
            busted = [i for i in range(len(before)) if before[i] > 0 and after[i] == 0]
            rank = sum(v > 0 for v in after) + 1 + sum(before[i] > before[hero] for i in busted)
            break
    if rank is None:
        rank = 1 + sum(v > final['stacks_after'][hero] for v in final['stacks_after'])
    capped = sum(v > 0 for v in final['stacks_after']) > 1
    return {'final_points': final['equity_after'][hero], 'final_rank': rank,
            'win': int(rank == 1), 'top3': int(rank <= 3), 'capped': int(capped),
            'final_stack': final['stacks_after'][hero],
            'auction_chips_left': final['flop_chips_after'][hero]}


def action_counts(hand, hero, automatic):
    out = Counter()
    remaining = list(hand['stacks_before'])
    bets = [0] * len(remaining)
    board = 0
    raises = calls = 0
    acted = False
    for index, event in enumerate(hand['events']):
        kind, seat, amount = event['kind'], event.get('seat'), event.get('amount', 0)
        if kind == 'board':
            board += len(event['cards']);bets = [0]*len(bets)
        elif kind in ('ante', 'blind'):
            remaining[seat] -= amount
            if kind == 'blind': bets[seat] += amount
            if seat == hero: out['forced_chips'] += amount
        elif kind in ACTIONS:
            if seat == hero and (hand['hand'], index) not in automatic:
                street = 'preflop' if board == 0 else 'postflop'
                out[street+'_decisions'] += 1
                out[street+'_'+kind] += 1
                if kind == 'call' and amount >= remaining[seat] * .25:
                    out[street+'_calls_ge25pct_remaining_stack'] += 1
                if board == 0 and not acted and raises == 0 and calls == 0:
                    out['unopened_first_opportunities'] += 1
                    out['unopened_first_raises'] += kind == 'raise'
                acted = True
            if kind == 'call':
                remaining[seat] -= amount;bets[seat] += amount;calls += 1
            elif kind == 'raise':
                remaining[seat] -= amount-bets[seat];bets[seat] = amount;raises += 1
    return out


def reduce_table(table, own):
    hero = own['seat']
    outcome = final_result(table, hero)
    cps = {}
    accounting = defaultdict(Counter)
    automatic = set(map(tuple, own.get('automatic', [])))
    for hand in table['hands']:
        before = hand['stacks_before']
        if before[hero] <= 0:
            continue
        players = sum(v > 0 for v in before)
        assert players >= 2
        stage = accounting[str(players)]
        stage['hands'] += 1
        stage['points_change'] += hand['equity_after'][hero] - hand['equity_before'][hero]
        stage['chips_change'] += hand['stacks_after'][hero] - before[hero]
        stage['auction_chips_spent'] += hand['flop_chips_before'][hero] - hand['flop_chips_after'][hero]
        stage.update(action_counts(hand, hero, automatic))
        if hand['hand'] in HAND_CHECKPOINTS:
            cps['hand_'+str(hand['hand'])] = checkpoint(hand, hero)
        for n in ENTRY_CHECKPOINTS:
            key = 'first_'+str(n)+'_or_fewer'
            if players <= n and key not in cps:
                cps[key] = checkpoint(hand, hero)
    initial = table['hands'][0]['equity_before'][hero]
    assert abs(sum(v['points_change'] for v in accounting.values()) - (outcome['final_points']-initial)) < 1e-7
    return {'iteration': table['iteration'], 'table': table['table'], 'hero': hero,
            'last_hand': table['hands'][-1]['hand'], 'initial_points': initial,
            'outcome': outcome, 'checkpoints': cps, 'accounting_by_players_before_hand': dict(accounting)}


class Moments:
    def __init__(self):
        self.n = 0;self.sums = Counter();self.squares = Counter()

    def add(self, values):
        self.n += 1
        for k,v in values.items():
            self.sums[k] += v;self.squares[k] += v*v

    def report(self, total=None):
        n = self.n if total is None else total
        result = {'n': self.n, 'denominator': n, 'metrics': {}}
        for k,s in sorted(self.sums.items()):
            mean = s/n
            se = math.sqrt(max(0., (self.squares[k]-s*s/n)/(n-1)/n)) if n > 1 else None
            result['metrics'][k] = {'mean': mean, 'ci95': [mean-1.96*se, mean+1.96*se] if se is not None else None}
        return result


def strata(cp):
    share = bucket(cp['chip_share'], (.1,.2,.35,.5,.65,.8), ('<10%','10-20%','20-35%','35-50%','50-65%','65-80%','80%+'))
    depth = bucket(cp['own_bb'], (5,10,20,40), ('<5','5-10','10-20','20-40','40+'))
    effective = bucket(cp['effective_bb'], (5,10,20,40), ('<5','5-10','10-20','20-40','40+'))
    auction = bucket(cp['auction_chips'], (1,50,200,500), ('0','1-49','50-199','200-499','500+'))
    p, rank = str(cp['players_left']), str(cp['hero_rank'])
    return ('all', 'players='+p, 'share='+share, 'rank='+rank, 'bb='+depth,
            'players='+p+';share='+share, 'players='+p+';rank='+rank,
            'players='+p+';bb='+depth, 'players='+p+';effective_bb='+effective,
            'players='+p+';auction_chips='+auction)


def summarize(rows):
    groups = defaultdict(Moments)
    accounting = defaultdict(Moments)
    totals = Moments()
    place_counts = Counter()
    coverage = Counter()
    for row in rows:
        outcome = row['outcome']
        totals.add(outcome)
        place_counts[str(outcome['final_rank'])] += 1
        for n in HAND_CHECKPOINTS:
            key = 'hand_'+str(n)
            if key in row['checkpoints']: coverage[key+':hero_alive'] += 1
            elif row['last_hand'] < n: coverage[key+':table_finished_before'] += 1
            else: coverage[key+':hero_already_busted'] += 1
        for name, cp in row['checkpoints'].items():
            values = {**outcome, **{k:cp[k] for k in ('icm_points','chip_share','expected_top3','expected_rank','own_bb','effective_bb','auction_chips')}}
            values.update(points_minus_icm=outcome['final_points']-cp['icm_points'],
                          win_minus_chip_share=outcome['win']-cp['chip_share'],
                          top3_minus_icm=outcome['top3']-cp['expected_top3'],
                          rank_minus_icm=outcome['final_rank']-cp['expected_rank'])
            for stratum in strata(cp): groups[(name,stratum)].add(values)
        for stage, values in row['accounting_by_players_before_hand'].items():
            accounting[stage].add(values)
    return {'tables': len(rows), 'overall': totals.report(), 'finish_rank_counts': place_counts,
            'checkpoint_coverage': coverage,
            'stage_accounting_per_initial_table': {k:v.report(total=len(rows)) for k,v in sorted(accounting.items())},
            'stage_accounting_per_reached_table': {k:v.report() for k,v in sorted(accounting.items())},
            'checkpoint_groups': [{'checkpoint': cp, 'stratum': st, **v.report()} for (cp,st),v in sorted(groups.items())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--health', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = json.loads(args.results.read_text())
    health = next(r for r in json.loads(args.health.read_text())['items'] if r['tournament_id'] == results['tournament_id'])
    official = next(r for r in results['items'] if r['team_id'] == 'f570222b-b5b0-4dce-8499-a32476665215')
    rows = [];iterations = set();filler_tables = 0;hands = 0
    started = time.monotonic()
    with gzip.open(args.history, 'rt') as stream, gzip.open(args.out/'table-checkpoints.jsonl.gz', 'wt') as out:
        header = json.loads(next(stream))
        assert header['tournament_id'] == results['tournament_id']
        assert next(e['team_name'] for e in header['entries'] if e['entry_id'] == header['you']) == 'Timothy'
        for line in stream:
            table=json.loads(line);own=json.loads(next(stream))['you']
            assert table['iteration']==own['iteration'] and table['table']==own['table']
            assert table['seats'][own['seat']]==header['you']
            assert table['payouts']==[50,30,20]
            if table['fillers'][own['seat']]: filler_tables += 1;continue
            assert table['iteration'] not in iterations
            iterations.add(table['iteration']);hands += len(table['hands'])
            row = reduce_table(table, own);rows.append(row);out.write(json.dumps(row,separators=(',',':'))+'\n')
            if len(rows)%2000==0:print(args.name,len(rows),'tables',round(time.monotonic()-started,1),'seconds',flush=True)
    assert len(rows)==official['tables']
    summary = summarize(rows)
    actual = summary['overall']['metrics']['final_points']['mean']
    assert abs(actual-official['average_points']) <= .00051, (actual,official['average_points'])
    summary.update(name=args.name,tournament_id=header['tournament_id'],entry_id=header['you'],
                   submission_id=health['submission_id'],official_result=official,official_health=health,
                   input_history=str(args.history.resolve()),
                   history_sha256=hashlib.file_digest(args.history.open('rb'),'sha256').hexdigest(),
                   checkpoint_rows_sha256=hashlib.file_digest((args.out/'table-checkpoints.jsonl.gz').open('rb'),'sha256').hexdigest(),
                   filler_tables_excluded=filler_tables,public_hands=hands,
                   independent_iteration_clusters=len(iterations),seconds=time.monotonic()-started,
                   definitions={
                       'checkpoints':'Start of fixed hand numbers and first live hand with <=6/4/3/2 players; hero must be alive. No outcome-based inclusion.',
                       'finish_rank':'Official place rule: simultaneous busts ordered by start-of-hand stack, ties share better place; survivors at cap ranked by stack. Win/top3 refer to these ranks.',
                       'points':'Official equity_after at final hand, rounded to 4 decimals in archive; independently matches public round mean to reporting precision.',
                       'intervals':'Unadjusted pointwise normal 95% intervals over independent iterations; exactly one non-filler hero table per iteration verified. Many exploratory strata: intervals are descriptive, not selection-adjusted.',
                       'ICM':'Malmuth-Harville descriptive null, not calibrated probability or counterfactual action value. Exact top3 probabilities, E[rank]=1+sum(other/(hero+other)).',
                       'selection':'Later checkpoints condition on survival and opponents/table state; round/version/field are confounded. No causal policy comparison.',
                       'accounting':'Sum of per-hand ICM point changes grouped by players BEFORE hand; telescopes to final points minus initial points. Includes equity changes caused by other players.',
                       'currencies':'own_bb and stack_share use ordinary chips only; auction chips reported separately.'})
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'name':args.name,'tables':len(rows),'points':actual,'win_rate':summary['overall']['metrics']['win']['mean'],'top3':summary['overall']['metrics']['top3']['mean'],'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':main()
