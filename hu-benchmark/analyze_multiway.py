"""Audit all six-player records and replace the comparison with table scores."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from collections import Counter

HERE=Path(__file__).resolve().parent


def brute_icm(stacks,payouts):
    n=len(stacks);out=[0.]*n
    for order in itertools.permutations(range(n)):
        prob=1.;remaining=sum(stacks)
        for i in order:prob*=stacks[i]/remaining;remaining-=stacks[i]
        for rank,i in enumerate(order):out[i]+=prob*payouts[rank]
    return out


def main():
    plan=json.loads((HERE/'multiway_plan.json').read_text())
    for name,digest in {**plan['inherited_source_hashes'],**plan['source_hashes']}.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    result=json.loads((HERE/'multiway_arena.json').read_text())
    latency=json.loads((HERE/'multiway_latency.json').read_text())
    raw=HERE/'multiway_arena.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
    cfg=plan['confirmation'];names=plan['names'];payouts=plan['payouts']
    assert len(rows)==cfg['blocks'] and result['errors']==0
    assert sorted(r['seed'] for r in rows)==list(range(cfg['seed_start'],cfg['seed_start']+cfg['blocks']))
    corrections=Counter();decisions=Counter();survivors=Counter();hands=0
    for row in rows:
        assert 'error' not in row and sorted(row['roster'])==sorted(names) and len(row['legs'])==6
        for leg in row['legs']:
            assert sum(leg['stacks'].values())==60000 and min(leg['stacks'].values())>=0
            assert abs(sum(leg['scores'].values())-sum(payouts))<1e-8
            assert all(0<=v<=100 for v in leg['scores'].values())
            alive=[name for name in names if leg['stacks'][name]>0]
            independent=brute_icm([leg['stacks'][name] for name in alive],payouts[:len(alive)])
            for name,value in zip(alive,independent):assert math.isclose(value,leg['scores'][name],abs_tol=1e-8)
            corrections.update(leg['normalizations']);decisions.update(leg['decisions'])
            survivors[len(alive)]+=1;hands+=leg['hands']
        for name in names:assert math.isclose(row['scores'][name],statistics.mean(leg['scores'][name] for leg in row['legs']),abs_tol=1e-10)
    table={}
    for name in names:
        values=[r['scores'][name] for r in rows];mean=statistics.mean(values)
        se=statistics.stdev(values)/math.sqrt(len(values));interval=[mean-1.96*se,mean+1.96*se]
        assert math.isclose(mean,result['results'][name]['mean'],abs_tol=1e-10)
        assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(interval,result['results'][name]['ci95']))
        timed=latency['results'][name]
        table[name]=dict(multiplayer_support='Six-player table; remaining table shrinks to 2-5 after eliminations.',
                         mean_score=mean,ci95=interval,latency_p99_ms=timed['wall']['p99_ms'],
                         latency_max_ms=timed['wall']['max_ms'],over_1ms=timed['wall']['over_1ms'],
                         latency_samples=timed['wall']['n'],decisions=decisions[name])
    assert math.isclose(sum(v['mean_score'] for v in table.values()),sum(payouts),abs_tol=1e-10)
    report=dict(table=table,protocol=plan,matches=len(rows)*6,hands=hands,
                environment=json.loads((HERE/'multiway_environment.json').read_text()),
                elapsed_seconds=result['elapsed_seconds'],independent_blocks=len(rows),
                normalizations=dict(corrections),total_decisions=sum(decisions.values()),
                survivors_at_end=dict(survivors),
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                audit='All raw match totals and block means checked; every surviving-player ICM score independently recomputed by enumerating all finish permutations.',
                limitations=[
                    'Custom six-place payout table, not the official unpublished payout table.',
                    'Only six-seat support is established; Fullhouse is a six-max model, not an eight-player model.',
                    'Warm single-core CPU latency, no process startup, JSON or IPC; P99 is not a hard worst-case guarantee.',
                    'Strength matches have no time penalties. Policy versions and weights unchanged from the low-latency comparison.',
                    'PokerKit adapter corrects unavailable requested actions consistently for every policy; counts are reported.',
                    'All ranked policies derive from the same public repository. Scores depend on this lineup and payout table.',
                ])
    with (HERE/'multiway_report.json').open('x') as f:json.dump(report,f,indent=2)
    with (HERE/'multiway_comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['bot','players','p99_ms','score','ci95_low','ci95_high','over_1ms','latency_samples'])
        for name,v in sorted(table.items(),key=lambda x:-x[1]['mean_score']):
            w.writerow([name,6,v['latency_p99_ms'],v['mean_score'],*v['ci95'],v['over_1ms'],v['latency_samples']])
    print(json.dumps({k:report[k] for k in ['table','matches','hands','elapsed_seconds','normalizations','total_decisions','survivors_at_end']},indent=2))


if __name__=='__main__':main()
