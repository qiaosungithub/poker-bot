"""Full-length fits on a realistic buffer ($FITOPT_BUFFER: a checkpoint with upstream-format
replay buffers, e.g. the ``latest.pt`` written by train_fast.py with save_final_buffers=true;
the recorded validation used 8 iterations of the original upstream pipeline, generated on
``main`` with fit_opt/datagen.py): GPU fitter (bf16, CUDA graph) vs the fp32 JAX reference
``jax_reference.fit`` (fp32 'highest'; the fitting code the original TPU run used). Both nets are scored with the fp32 reference objective on the same
held-out-style evaluation minibatches (drawn with the training sampling protocol).

usage: CUDA_VISIBLE_DEVICES=0 python validate_fit.py [adv|strat] [n_seeds] [backend]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
HERE = Path(__file__).resolve().parent              # fullhouse-gpu/tests/fit
PACKAGE = HERE.parents[1]                          # fullhouse-gpu
# upstream source: $FULLHOUSE_SOURCE, default <repo>/fullhouse-repro/source
SOURCE = Path(os.environ.get('FULLHOUSE_SOURCE', str(PACKAGE.parent / 'fullhouse-repro' / 'source')))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))                      # jax_reference (fp32 JAX oracle)
sys.path.insert(0, str(PACKAGE / 'fit'))           # gpu_fit, fast_step, rowk.cu

import numpy as np
import torch

import gpu_fit as gf

DEV = torch.device('cuda')
CAP = 5_000_000


def load_buffers():
    t0 = time.time()
    path = os.environ.get('FITOPT_BUFFER')
    if not path:
        raise SystemExit('set FITOPT_BUFFER to a checkpoint with upstream-format replay buffers')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    print(f'loaded checkpoint in {time.time()-t0:.1f}s; iteration {ck["iteration"]}', flush=True)
    return ck


def eval_loss(sd, gset, strategy, n_batches=48, seed=123):
    """fp32 reference objective of state_dict `sd` averaged over fixed minibatches."""
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    sizes = torch.tensor(gset.sizes, device=DEV)
    offs = torch.arange(gset.n, device=DEV) * gset.capacity
    idx = gf.sample_indices(sizes, offs, n_batches, 8192, g)
    sdt = {k: v.to(DEV) for k, v in sd.items()}
    with torch.no_grad():
        tot = 0.0
        for k in range(n_batches):
            tot += float(gf.reference_objective(sdt, gset.data[idx[k]], strategy))
    return tot / n_batches


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'adv'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    backend = sys.argv[3] if len(sys.argv) > 3 else gf.DEFAULT_BACKEND
    strategy = kind == 'strat'
    steps = 3000 if strategy else 10000
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = load_buffers()
    it = ck['iteration'] + 1
    lr = gf.lr_schedule(it)
    if strategy:
        states = [ck['strategy_buffer']]
    else:
        states = ck['buffers']
    gset = gf.GPUReservoirSet(len(states), CAP, DEV)
    for b, st in enumerate(states):
        gset.load_state_dict(b, st)
    print('sizes', gset.sizes, 'lr', lr, flush=True)
    results = {'kind': kind, 'steps': steps, 'lr': lr, 'sizes': gset.sizes, 'gpu': [], 'jax': []}
    fitter = gf.make_fitter(kind, gset, backend=backend, seed=1)
    for s in range(n_seeds):
        torch.cuda.synchronize()
        r = fitter.fit(lr, steps, init_generator=torch.Generator(device=DEV).manual_seed(100 + s))
        ev = eval_loss(r.state_dict, gset, strategy)
        print(f'[gpu-{backend}] seed {s}: {r.seconds:.2f}s ({r.seconds/steps*1e3:.4f} ms/step incl. '
              f'capture on first) train avg loss {r.avg_loss:.5f} eval {ev:.5f}', flush=True)
        results['gpu'].append(dict(seconds=r.seconds, avg_loss=r.avg_loss, eval=ev))
    # reference: jax_reference.fit on upstream ReservoirBuffers (fp32 'highest')
    import jax_reference as jb
    from training.buffer import ReservoirBuffer
    from training.network import DuelingAdvantageNet, AverageStrategyNet
    bufs = []
    for st in states:
        rb = ReservoirBuffer(st['count'] + 1, 51, 5, store_sample_weights=strategy)
        rb.load_state_dict(st)
        bufs.append(rb)
    for s in range(n_seeds):
        torch.manual_seed(200 + s); np.random.seed(200 + s)
        net = AverageStrategyNet() if strategy else DuelingAdvantageNet()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        t0 = time.time()
        avg = jb.fit(net, bufs, opt, steps, 8192, strategy)
        dt = time.time() - t0
        ev = eval_loss({k: v for k, v in net.state_dict().items()}, gset, strategy)
        print(f'[jax-ref] seed {s}: {dt:.2f}s train avg loss {avg:.5f} eval {ev:.5f}', flush=True)
        results['jax'].append(dict(seconds=dt, avg_loss=avg, eval=ev))
    out = HERE / f'validate_{kind}_{backend}.json'
    out.write_text(json.dumps(results, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
