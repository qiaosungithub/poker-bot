"""Run from a frozen JSON configuration. Imported by spawn workers safely."""
import os
import runtime_paths
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '1'
os.environ.setdefault('TPU_VISIBLE_CHIPS', '0')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def main():
    import json, random, numpy as np, torch
    from training import train
    config = json.loads(Path(sys.argv[1]).read_text())
    backend = config.pop('backend')
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if backend == 'tpu':
        import jax_backend as jb
        print('JAX devices:', jb.jax.devices(), flush=True)
        train.train_network = jb.train_network
        train.train_strategy_network = jb.train_strategy_network
    config['checkpoint_dir'] = str(ROOT / config['checkpoint_dir'])
    print('CONFIG', json.dumps(config, sort_keys=True), flush=True)
    train.train_deep_cfr(**config)


if __name__ == '__main__':
    main()
