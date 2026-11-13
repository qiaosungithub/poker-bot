"""Full-score audits and the two declared paired comparisons."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import zipfile

from auction_flop.benchmark import run


def main():
    here = Path(__file__).resolve().parent
    control = run.ROOT/'evidence/candidate-simple_tuned_v2'
    _, old_rows = run.audit(control)
    old_blocks = run.candidate_blocks(old_rows)
    z = NormalDist().inv_cdf(1 - .05 / (2 * 2))
    results = {}
    for name in ('patient_value_v1', 'simple_keycard_v1'):
        directory = run.ROOT / ('evidence/candidate-' + name)
        manifest, rows = run.audit(directory)
        summary = run.read_json(directory/'summary.json')
        blocks = run.candidate_blocks(rows)
        assert blocks.keys() == old_blocks.keys()
        effect = run.estimate({key:blocks[key]-old_blocks[key] for key in blocks})
        mean = effect.pop('mean_points')
        se = (effect['ci95'][1]-effect['ci95'][0]) / (2*1.96)
        effect.update(difference=mean, bonferroni_ci97_5=[mean-z*se, mean+z*se],
                      family='Two declared comparisons versus simple_tuned_v2')
        hero = [next(e for e in row['entries'] if e['name']=='candidate') for row in rows]
        assert abs(sum(e['points'] for e in hero)/len(hero)-summary['mean_points']) < 1e-12
        expected = manifest['source_hashes']['candidate']
        assert run.source_hashes(run.ROOT/'candidates'/name) == expected
        with zipfile.ZipFile(here/(name+'.zip')) as archive:
            assert {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()} == expected
        elapsed = (datetime.fromisoformat(manifest['completed'])-
                   datetime.fromisoformat(manifest['started'])).total_seconds()
        results[name] = dict(summary=summary, comparison=effect, seconds=elapsed,
            sole_chip_wins=sum(e['stack']==16000 for e in hero),
            first_place_rate=sum(e['stack']==16000 for e in hero)/len(hero),
            mean_auctions=sum(e['auction_calls'] for e in hero)/len(hero),
            package_equals_measured_source=True, source_hashes=expected)
    old_source = run.source_hashes(run.ROOT/'candidates/simple_tuned_v2')
    key_source = results['simple_keycard_v1']['source_hashes']
    assert {k:v for k,v in old_source.items() if k!='simple_auction.py'} == {
        k:v for k,v in key_source.items() if k!='simple_auction.py'}
    result = dict(control='simple_tuned_v2', control_score=10.67138671875,
                  keycard_only_changes_auction=True, results=results)
    run.write_json(here/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
