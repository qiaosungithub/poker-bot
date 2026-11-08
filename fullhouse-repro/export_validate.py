import os
import runtime_paths
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, json, gzip, argparse
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'source'), str(ROOT.parent/'latency-survey')]
from fullhouse_onnx import export, ONNXLookup
from bot.deep_cfr_lookup import DeepCFRLookup


def main():
    p = argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('output')
    a = p.parse_args()
    export(a.checkpoint, a.output)
    reference, converted = DeepCFRLookup(a.checkpoint), ONNXLookup(a.output)
    errors = []
    corpus = ROOT.parent/'hu-benchmark/multiway_latency_states.jsonl.gz'
    with gzip.open(corpus, 'rt') as f:
        for i, line in enumerate(f):
            if i >= 1024:
                break
            state = json.loads(line)
            p, q = reference.get_strategy(state), converted.get_strategy(state)
            assert np.isfinite(q).all() and abs(q.sum()-1) < 1e-5
            errors.append(float(np.max(np.abs(p-q))))
    assert max(errors) < .001, max(errors)
    result = dict(states=len(errors), max_abs_error=max(errors), mean_abs_error=float(np.mean(errors)))
    Path(a.output+'.validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
