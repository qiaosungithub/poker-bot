"""Independently audit tournament scores and compute paired uncertainty."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist,mean,stdev

PAYOUTS=[100.,60.,40.,25.,10.,0.]


def brute_icm(stacks,prizes):
    result=[0.]*len(stacks)
    for order in itertools.permutations(range(len(stacks))):
        p=1.;remaining=sum(stacks)
        for i in order:p*=stacks[i]/remaining;remaining-=stacks[i]
        for place,i in enumerate(order):result[i]+=p*prizes[place]
    return result


def audit_leg(leg):
    assert len(leg['scores'])==len(leg['stacks'])==len(leg['roster'])==6
    assert sum(leg['stacks'])==60000 and min(leg['stacks'])>=0
    assert abs(sum(leg['scores'])-235)<1e-8
    awards={}
    for event in leg['eliminations']:
        assert sum(event['before'])==sum(event['after'])
        dead=[i for i,c in enumerate(event['after']) if c==0]
        sizes=sorted({event['before'][i] for i in dead})
        bottom=len(event['alive'])
        expected={}
        for size in sizes:
            tied=[i for i in dead if event['before'][i]==size]
            prize=sum(PAYOUTS[bottom-len(tied):bottom])/len(tied)
            for i in tied:expected[event['alive'][i]]=prize
            bottom-=len(tied)
        assert expected==event['awarded']
        assert not awards.keys() & expected.keys()
        awards.update(expected)
    living=[i for i,c in enumerate(leg['stacks']) if c>0]
    expected=brute_icm([leg['stacks'][i] for i in living],PAYOUTS[:len(living)])
    for i,value in zip(living,expected):assert math.isclose(value,leg['scores'][i],abs_tol=1e-8)
    for i,c in enumerate(leg['stacks']):
        if c==0:assert math.isclose(awards['seat_'+str(i)],leg['scores'][i],abs_tol=1e-8)


def summarize(xs,z=1.96):
    m=mean(xs);se=stdev(xs)/len(xs)**.5 if len(xs)>1 else None
    return {'mean':m,'se':se,'ci95':[m-z*se,m+z*se] if se is not None else None}


def analyze(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    original=json.loads((folder/'summary.json').read_text())
    raw=folder/'blocks.jsonl'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==original['raw_sha256']
    cfg=manifest['config'];groups=defaultdict(list)
    counts=Counter();exceptions=Counter();normalizations=Counter();decisions=Counter();chips=defaultdict(list)
    first=None
    with raw.open() as f:
        for line in f:
            row=json.loads(line);groups[row['label']].append(row)
            assert len(row['legs'])==6
            assert sorted(row['legs'][0]['slot_order'])==list(range(6))
            seat_counts=Counter()
            for leg in row['legs']:
                if first is None:first=leg
                audit_leg(leg)
                for physical,logical in enumerate(leg['slot_order']):
                    assert leg['roster'][physical]==row['names'][logical]
                    seat_counts[(logical,physical)]+=1
                    name=leg['roster'][physical]
                    decisions[name]+=leg['decisions'].get('seat_'+str(physical),0)
                    for ex,n in leg['exceptions'].get('seat_'+str(physical),{}).items():exceptions[name+' '+ex]+=n
                for key,n in leg['normalizations'].items():
                    physical=int(key.split('_')[1]);name=leg['roster'][physical]
                    normalizations[name+'_'+'_'.join(key.split('_')[2:])]+=n
                counts['tables']+=1;counts['hands']+=leg['hands']
            assert set(seat_counts.values())=={1} and len(seat_counts)==36
            for i in range(6):
                values=[leg['scores'][leg['slot_order'].index(i)] for leg in row['legs']]
                assert math.isclose(mean(values),row['scores'][i],abs_tol=1e-9)
    # Known-bad score must not pass the same auditor.
    import copy
    broken=copy.deepcopy(first);broken['scores'][0]+=1
    try:audit_leg(broken)
    except AssertionError:pass
    else:raise AssertionError('Negative score control did not fail')
    reports={}
    for label,rows in groups.items():
        assert len(rows)==cfg['blocks']
        assert sorted(r['seed'] for r in rows)==list(range(cfg['seed'],cfg['seed']+cfg['blocks']))
        names=rows[0]['names'];result={};paired={}
        for i,name in enumerate(names):
            xs=[r['scores'][i] for r in rows]
            item=summarize(xs)
            item['mean_final_chips']=mean(mean(leg['stacks'][leg['slot_order'].index(i)] for leg in r['legs']) for r in rows)
            assert math.isclose(item['mean'],original['results'][label][f'{i}:{name}']['mean'],abs_tol=1e-9)
            result[f'{i}:{name}']=item
        z=NormalDist().inv_cdf(1-.05/(2*15))
        for i,j in itertools.combinations(range(6),2):
            xs=[r['scores'][i]-r['scores'][j] for r in rows]
            paired[f'{i}:{names[i]} minus {j}:{names[j]}']=dict(summarize(xs),simultaneous_ci95=summarize(xs,z)['ci95'])
        reports[label]={'results':result,'paired':paired}
    report={'groups':reports,'counts':dict(counts),'decisions':dict(decisions),'caught_exceptions':dict(exceptions),
            'normalizations':dict(normalizations),'negative_score_control_passed':True,'audit':'pass',
            'raw_sha256':original['raw_sha256'],'elapsed_seconds':original['elapsed_seconds']}
    (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');a=p.parse_args()
    report=analyze(a.folder)
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
