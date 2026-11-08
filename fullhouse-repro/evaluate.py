"""Replace only the Fullhouse checkpoint in the frozen six-player benchmark."""
import os
import runtime_paths
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, argparse, json, hashlib
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'hu-benchmark'))
import multiway_benchmark as M
ORIGINAL_INITIALIZE = M.C.R.initialize


def initialize():
    ORIGINAL_INITIALIZE()
    if os.environ.get('FULLHOUSE_CANDIDATE_ONNX'):
        from fullhouse_onnx import ONNXLookup
        M.C.R.HERO._DEEP_CFR = ONNXLookup(os.environ['FULLHOUSE_CANDIDATE_ONNX'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx')
    p.add_argument('--output', required=True)
    p.add_argument('--blocks', type=int, default=256)
    p.add_argument('--seed', type=int, default=11300000)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--hands', type=int, default=200)
    args = p.parse_args()
    args.output = str(Path(args.output).resolve())
    if args.onnx:
        os.environ['FULLHOUSE_CANDIDATE_ONNX'] = str(Path(args.onnx).resolve())
    else:
        os.environ.pop('FULLHOUSE_CANDIDATE_ONNX', None)
    M.C.R.initialize = initialize
    out = M.arena(args)
    if out['errors']:
        raise RuntimeError(f"Benchmark had {out['errors']} errors")
    out['model_sha256'] = hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest() if args.onnx else 'frozen_reference'
    with open(args.output, 'x') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
