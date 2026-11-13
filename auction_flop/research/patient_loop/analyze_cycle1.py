"""Audit the full result, retained ZIP, screening and fresh paired holdout."""
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from auction_flop.benchmark import run


def load_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def paired(left, right):
    a, b = run.candidate_blocks(left), run.candidate_blocks(right)
    assert a.keys() == b.keys()
    result = run.estimate({k: a[k] - b[k] for k in a})
    result['difference'] = result.pop('mean_points')
    return result


def main():
    here = Path(__file__).resolve().parent
    evidence = run.ROOT / 'evidence'
    old_manifest, old_rows = run.audit(evidence / 'candidate-patient_value_v1')
    manifest, rows = run.audit(evidence / 'candidate-patient_defend_v2')
    expected = manifest['source_hashes']['candidate']
    assert run.source_hashes(run.ROOT/'candidates/patient_defend_v2') == expected
    with zipfile.ZipFile(here/'patient_defend_v2.zip') as archive:
        assert {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()} == expected
    hero = [next(e for e in r['entries'] if e['name']=='candidate') for r in rows]
    old_hero = [next(e for e in r['entries'] if e['name']=='candidate') for r in old_rows]
    screen = evidence/'patient-loop-c1-screen'
    control = load_rows(screen/'patient_value_v1.jsonl.gz')
    screening = {}
    for item in run.read_json(screen/'summary.json')['scores']:
        name = item['name']
        screening[name] = {'mean_points':item['mean_points'],
                           'paired':paired(load_rows(screen/(name+'.jsonl.gz')),control)}
    combo = evidence/'patient-loop-c1-combined'
    screening['patient_defend_v2'] = {
        'mean_points':run.read_json(combo/'summary.json')['scores'][0]['mean_points'],
        'paired':paired(load_rows(combo/'patient_defend_v2.jsonl.gz'),control)}
    holdout = evidence/'patient-loop-c1-holdout'
    held_old = load_rows(holdout/'patient_value_v1.jsonl.gz')
    held_new = load_rows(holdout/'patient_defend_v2.jsonl.gz')
    sdk = run.read_json(here/'sdk-wiring.json')
    assert all(bot['failed_decisions']==0 for bot in sdk['bots'].values())
    result = dict(
        baseline='patient_value_v1', candidate='patient_defend_v2',
        full_summary=run.read_json(evidence/'candidate-patient_defend_v2/summary.json'),
        full_paired=paired(rows,old_rows),screening=screening,
        holdout={'tables_per_arm':len(held_new),
                 'baseline':run.estimate(run.candidate_blocks(held_old)),
                 'candidate':run.estimate(run.candidate_blocks(held_new)),
                 'paired':paired(held_new,held_old)},
        full_seconds=(datetime.fromisoformat(manifest['completed'])-
                      datetime.fromisoformat(manifest['started'])).total_seconds(),
        full_candidate_failures=sum(e['failures'] for e in hero),
        full_candidate_decisions=sum(e['decisions'] for e in hero),
        full_first_place_count=sum(e['stack']==16000 for e in hero),
        baseline_first_place_count=sum(e['stack']==16000 for e in old_hero),
        source_hashes=expected,package_equals_measured_source=True,
        sdk_wiring='Two 20-hand tables, eight seats, 1000 ms deadline, zero failures; not 1 ms qualification',
        scope='One final full comparison after fresh-seed screening; independent held-out confirmation. Untimed strength.')
    run.write_json(here/'cycle1-comparison.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
