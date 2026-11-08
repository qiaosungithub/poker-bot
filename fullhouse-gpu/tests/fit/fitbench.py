"""End-to-end timing of the public API on full-size (5M-row) GPU reservoir buffers:
add_batch cost (host numpy -> GPU, fill and reservoir-replacement paths) and
fit_advantage / fit_strategy wall time (10k / 3k steps, B=8192).
usage: fitbench.py [backend] [--no-add]"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fit'))  # fullhouse-gpu/fit
import numpy as np, torch
import gpu_fit as gf

DEV = torch.device('cuda')
backend = next((a for a in sys.argv[1:] if not a.startswith('--')), gf.DEFAULT_BACKEND)
CAP = 5_000_000
SEAT_SIZES = [1873267, 2369553, 1289185, 1417271, 1324847, 1376458]  # realistic (iter 8)
out = {'backend': backend}


def gpu_rows(n, it_max, strategy=False, g=None):
    r = torch.zeros(n, gf.ROW_W, device=DEV)
    r[:, :gf.FEATURE_DIM] = torch.randn(n, gf.FEATURE_DIM, device=DEV, generator=g)
    m = (torch.rand(n, 5, device=DEV, generator=g) < 0.7).float(); m[:, 1] = 1
    r[:, gf.MASK_OFF:gf.MASK_OFF + 5] = m
    y = torch.randn(n, 5, device=DEV, generator=g) * m
    if strategy:
        y = y.abs(); y = y / y.sum(1, keepdim=True).clamp_min(1e-12)
        r[:, gf.COL_W] = torch.rand(n, device=DEV, generator=g) ** 3
    else:
        r[:, gf.COL_W] = 1
    r[:, gf.COL_Y:gf.COL_Y + 5] = y
    r[:, gf.COL_IT] = torch.randint(1, it_max + 1, (n,), device=DEV, generator=g).float()
    return r


def sync_time(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return time.perf_counter() - t


g = torch.Generator(device=DEV); g.manual_seed(0)
adv = gf.GPUReservoirSet(6, CAP, DEV, seed=1)
for b, n in enumerate(SEAT_SIZES):
    adv.add_rows(b, gpu_rows(n, 8, g=g))
strat = gf.GPUReservoirSet(1, CAP, DEV, seed=2)
strat.add_rows(0, gpu_rows(sum(SEAT_SIZES) // 6 * 3, 8, strategy=True, g=g))

if '--no-add' not in sys.argv:
    rng = np.random.default_rng(0)
    N = 1_300_000
    feats = rng.normal(size=(N, gf.FEATURE_DIM)).astype(np.float32)
    targs = rng.normal(size=(N, 5)).astype(np.float32)
    sw = rng.random(N).astype(np.float32)
    tmp = gf.GPUReservoirSet(2, CAP, DEV, seed=3)
    tmp.add_batch(0, feats[:1000], targs[:1000], 1)  # warm-up
    tmp.sizes[0] = 0; tmp.total_seen[0] = 0
    out['add_fill_host_1.3M_s'] = sync_time(lambda: tmp.add_batch(0, feats, targs, 9))
    tmp.add_rows(1, gpu_rows(CAP, 8, g=g))  # full buffer, total_seen = 5M
    tmp.total_seen[1] = 25_000_000  # late in training: ~20% acceptance
    out['add_replace_host_1.3M_s'] = sync_time(lambda: tmp.add_batch(1, feats, targs, 30))
    ft, tt = torch.as_tensor(feats, device=DEV), torch.as_tensor(targs, device=DEV)
    out['add_replace_gpu_1.3M_s'] = sync_time(lambda: tmp.add_batch(1, ft, tt, 31))
    tmp.total_seen[1] = 5_000_000  # just after filling: ~80% acceptance
    out['add_replace_gpu_1.3M_seen5M_s'] = sync_time(lambda: tmp.add_batch(1, ft, tt, 32))
    out['add_strat_host_1.3M_s'] = sync_time(lambda: tmp.add_batch(0, feats, targs, 9, sw))
    del tmp, ft, tt
    torch.cuda.empty_cache()

lr = gf.lr_schedule(9)
fa = gf.make_fitter('adv', adv, backend=backend)
r0 = gf.fit_advantage(fa, lr)
r1 = gf.fit_advantage(fa, lr)
out['adv_first_fit_s'] = r0.seconds
out['adv_fit_s'] = r1.seconds
out['adv_ms_per_step'] = r1.seconds / 10000 * 1e3
out['adv_avg_loss'] = r1.avg_loss
fs = gf.make_fitter('strat', strat, backend=backend)
s0 = gf.fit_strategy(fs, lr)
s1 = gf.fit_strategy(fs, lr)
out['strat_first_fit_s'] = s0.seconds
out['strat_fit_s'] = s1.seconds
out['strat_ms_per_step'] = s1.seconds / 3000 * 1e3
out['strat_avg_loss'] = s1.avg_loss
print(json.dumps(out, indent=1))
