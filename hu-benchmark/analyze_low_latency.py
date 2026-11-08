"""Audit raw paired results; report matrix and seed-clustered pool scores."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def ci(values,z=1.96):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values))
    return dict(mean=mean,ci=[mean-z*se,mean+z*se],se=se)


def main():
    plan=json.loads((HERE/'low_latency_plan.json').read_text())
    for name,digest in plan['hashes'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    summary=json.loads((HERE/'low_latency_arena.json').read_text())
    timing=json.loads((HERE/'latency_confirm.json').read_text())
    raw=HERE/'low_latency_arena.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    names=plan['candidates'];n=plan['arena']['pairs_per_matchup'];start=plan['arena']['seed_start']
    assert len(rows)==len(names)*(len(names)-1)//2*n
    values={a:{b:[] for b in names if b!=a} for a in names}
    clusters={a:{seed:[] for seed in range(start,start+n)} for a in names}
    seen=set();total_hands=0
    for r in rows:
        assert 'error' not in r,r
        key=(r['a'],r['b'],r['seed']);assert key not in seen;seen.add(key)
        assert len(r['legs'])==2
        assert r['score']==sum(g['score'] for g in r['legs'])/2
        assert all(0<=g['score']<=100 and 1<=g['hands']<=200 for g in r['legs'])
        for a,b,s in ((r['a'],r['b'],r['score']),(r['b'],r['a'],100-r['score'])):
            values[a][b].append(s);clusters[a][r['seed']].append(s)
        total_hands+=sum(g['hands'] for g in r['legs'])
    matrix={a:{b:ci(v) for b,v in row.items()} for a,row in values.items()}
    for a in names:
        assert all(len(v)==n for v in values[a].values())
        assert all(len(v)==len(names)-1 for v in clusters[a].values())
    for v in summary['results'].values():
        assert v['errors']==0
        assert math.isclose(v['score_a']['mean'],matrix[v['a']][v['b']]['mean'],abs_tol=1e-10)
    ranks={a:ci([statistics.mean(v) for v in clusters[a].values()]) for a in names}
    assert math.isclose(sum(v['mean'] for v in ranks.values()),50*len(names),abs_tol=1e-10)
    primary={a:ci(values[a]['fullhouse'],2.2414027276) for a in ('cfr_lite64','hydra32')}
    sources={}
    for name in ('latency_screen.json','latency_screen2.json'):
        data=json.loads((HERE/name).read_text())
        sources.update({k:dict(p99_ms=v['wall']['p99_ms'],over_1ms=v['wall']['over_1ms'],n=v['wall']['n']) for k,v in data['results'].items()})
    report=dict(protocol=plan['arena'],variants=plan['modifications'],
                pool_scores=ranks,matrix=matrix,primary_comparisons_97_5pct_ci=primary,
                latency={a:timing['results'][a] for a in names},screened_original_candidates=sources,
                total_matches=2*len(rows),total_hands=total_hands,
                elapsed_seconds=summary['elapsed_seconds'],workers=summary['config']['workers'],
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                limitations=[
                    'Pool scores depend on this fixed opponent pool; no universal poker rating is claimed.',
                    'Only heads-up play was tested. This does not establish strength in an eight-player tournament.',
                    'Latency is measured separately on one AMD EPYC 7B12 core, with models warm and GC disabled. P99 below 1ms is not a hard worst-case guarantee; all observed exceedances are reported.',
                    'Timing includes preprocessing and action selection; excludes process startup and IPC. JSON processing measured separately.',
                    'No action timeout penalties in strength matches.',
                    'CFR Lite 64 and Hydra 32 are local sampling-budget modifications of existing public bots, not unchanged upstream releases.',
                    'All six ranked policies derive from the same public Fullhouse repository. The independent txnujk submission failed the latency screen.',
                    'New HoldemBot PPO checkpoint was located but not ported or strength-tested; Talibus remains excluded pending cross-engine adapter validation.',
                ],validation=plan['validation'])
    with (HERE/'low_latency_report.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ('pool_scores','matrix','primary_comparisons_97_5pct_ci','total_matches','total_hands','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
