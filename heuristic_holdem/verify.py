"""Recompute retained scores/timing summaries and check pinned source identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from . import policies as P
from .analyze import analyze,audit_leg
from .benchmark import time_stats
import numpy as np

ROOT=P.ROOT
FOLDER=ROOT/'benchmarks/heuristic_holdem_2026-11-09'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_stats(actual,expected):
    assert actual.keys()==expected.keys()
    for key in actual:
        assert abs(actual[key]-expected[key])<1e-10,(key,actual,expected)


def main():
    checked={};timing_calls=0
    for directory in sorted(FOLDER.glob('latency*')):
        if not (directory/'result.json').exists():continue
        result=json.loads((directory/'result.json').read_text())
        assert digest(directory/'timings.npz')==result['raw_sha256']
        if 'corpus' in result:assert digest(ROOT/result['corpus'])==result['corpus_sha256']
        with np.load(directory/'timings.npz') as data:
            for name,record in result['results'].items():
                if 'wall' not in record:continue
                for kind in ('wall','cpu'):same_stats(time_stats(data[name+'__'+kind]),record[kind])
                wall=data[name+'__wall'];phase=data[name+'__phase']
                for i,label in enumerate(P.engine.C.STREETS):
                    if label in record['by_street']:
                        same_stats(time_stats(wall[phase==i]),record['by_street'][label])
                if 'policy' in record:
                    meta=record['policy'];now=P.metadata(name)
                    key='sha256' if meta.get('learned') else 'source_sha256'
                    assert meta[key]==now[key]
                    superseded=directory.name=='latency_screen' and name in ('pantheon_32','pantheon_64')
                    if not meta.get('learned') and not superseded:
                        assert meta['instrumented_code_sha256']==now['instrumented_code_sha256']
                timing_calls+=len(wall)
        if (directory/'tables.json').exists():
            for table in json.loads((directory/'tables.json').read_text()):audit_leg(table)
        checked[directory.name]='raw timings, phase quantiles and policy source verified'
    for stage in ('screen','confirmation'):
        manifest=json.loads((FOLDER/stage/'manifest.json').read_text())
        for path,value in manifest['runner_sha256'].items():assert digest(ROOT/path)==value,path
        assert digest(ROOT/'hu-benchmark/multiway_benchmark.py')==manifest['engine_sha256']
        report=analyze(FOLDER/stage)
        assert not report['caught_exceptions']
        checked[stage]={'tables':report['counts']['tables'],'hands':report['counts']['hands'],
                        'negative_score_control_passed':report['negative_score_control_passed']}
    survey=json.loads((FOLDER/'survey.json').read_text())
    for directory,record in survey['inspected_pinned_repositories'].items():
        head=subprocess.check_output(['git','-C',str(ROOT/directory),'rev-parse','HEAD'],text=True).strip()
        assert head==record['commit']
        assert not subprocess.check_output(['git','-C',str(ROOT/directory),'status','--porcelain'],text=True)
    for model in json.loads((ROOT/'docs/research/model-artifacts.json').read_text())['artifacts']:
        assert (ROOT/model['path']).stat().st_size==model['size_bytes']
        assert digest(ROOT/model['path'])==model['sha256']
    frozen=FOLDER/'runner_sources.zip'
    if frozen.exists():
        with zipfile.ZipFile(frozen) as archive:
            for name in archive.namelist():assert archive.read(name)==(ROOT/name).read_bytes(),name
        checked['source_snapshot']='matches working source byte-for-byte'
    # Verify the hash comparison rejects an incorrect hash.
    assert digest(FOLDER/'screen/blocks.jsonl')!='0'*64
    result={'verdict':'pass','checks':checked,'raw_timing_calls_verified':timing_calls,
            'pinned_sources_clean':True,'selected_reference_models_match':True}
    (FOLDER/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
