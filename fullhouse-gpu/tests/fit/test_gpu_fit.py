"""Correctness tests for gpu_fit (run on the VM: CUDA_VISIBLE_DEVICES=0).

T1 init distribution + state_dict round trip vs torch modules
T2 loss / grad vs jax_reference.objective (fp32 'highest') on identical batches
T3 clip + Adam vs torch.nn.utils.clip_grad_norm_ + torch.optim.Adam
T4 GPU reservoir add_batch == upstream ReservoirBuffer.add_batch given the same uniforms
T5 fused-kernel training step (every backend) vs jax_reference.objective + clip/Adam update
T6 GraphFitter end-to-end smoke test (graph fast path, eager fallback path, fresh init)
"""
import os
import sys
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
torch.backends.cuda.matmul.allow_tf32 = False
FAILS = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def synth_rows(n, rng, n_iter=30, strategy=False):
    """Synthetic rows in buffer layout with realistic legal masks."""
    rows = np.zeros((n, gf.ROW_W), np.float32)
    rows[:, :gf.FEATURE_DIM] = rng.normal(size=(n, gf.FEATURE_DIM)).astype(np.float32)
    mask = (rng.random((n, 5)) < 0.7).astype(np.float32)
    mask[:, 1] = 1.0  # call/check always legal
    mask[rng.random(n) < 0.01] = 0.0  # a few degenerate all-illegal rows
    rows[:, gf.MASK_OFF:gf.MASK_OFF + 5] = mask
    if strategy:
        s = rng.random((n, 5)).astype(np.float32) * mask
        rows[:, gf.COL_Y:gf.COL_Y + 5] = s / np.maximum(s.sum(1, keepdims=True), 1e-12)
        rows[:, gf.COL_W] = rng.random(n).astype(np.float32) ** 3
    else:
        rows[:, gf.COL_Y:gf.COL_Y + 5] = (rng.normal(size=(n, 5)) * 2).astype(np.float32) * mask
        rows[:, gf.COL_W] = 1.0
    rows[:, gf.COL_IT] = rng.integers(1, n_iter + 1, n).astype(np.float32)
    return rows


def t1_init():
    from training.network import DuelingAdvantageNet, AverageStrategyNet
    for kind, cls in (('adv', DuelingAdvantageNet), ('strat', AverageStrategyNet)):
        L = gf.ADV if kind == 'adv' else gf.STRAT
        net = cls()
        flat = gf.init_flat(L, DEV)
        sd = gf.export_state_dict(L, flat)
        ref = net.state_dict()
        check(f'T1 {kind} keys/shapes', set(sd) == set(ref) and all(sd[k].shape == ref[k].shape for k in ref))
        net.load_state_dict(sd, strict=True)
        worst = 0.0
        for k, r in ref.items():
            a = sd[k]
            # same support and moments as torch default init (uniform bound 1/sqrt(fan_in))
            if r.numel() < 2:  # scalar bias: support check only (bound 1/sqrt(fan_in))
                fan_in = 128 if k == 'value_head.2.bias' else 256
                ok = float(a.abs().max()) <= 1 / fan_in ** 0.5
            elif r.std() == 0:
                ok = torch.equal(a, r)
            else:
                bound = float(r.abs().max())
                ok = float(a.abs().max()) <= bound * 1.02 + 1e-6 and abs(float(a.std()) / float(r.std()) - 1) < 0.25
                worst = max(worst, abs(float(a.std()) / float(r.std()) - 1))
            if not ok:
                check(f'T1 {kind} init {k}', False, f'max {float(a.abs().max()):.4g} vs {float(r.abs().max()):.4g}')
        check(f'T1 {kind} init distribution', True, f'worst std rel err {worst:.3f}')
        back = gf.import_state_dict(L, sd, DEV)
        check(f'T1 {kind} round trip', torch.equal(back, flat))
        # structural zeros
        check(f'T1 {kind} padding zero', float((flat * (1 - torch.as_tensor(L.mask_np, device=DEV))).abs().sum()) == 0)
    # negative control: a wrong bound must fail the support check
    r = DuelingAdvantageNet().state_dict()['trunk.3.weight']
    bad = torch.empty_like(r).uniform_(-2 / 16, 2 / 16)
    check('T1 negative control (wrong bound detected)', float(bad.abs().max()) > float(r.abs().max()) * 1.02)


def t2_objective():
    import jax
    import jax.numpy as jnp
    import jax_reference as jb
    rng = np.random.default_rng(0)
    for strategy in (False, True):
        kind = 'strat' if strategy else 'adv'
        L = gf.STRAT if strategy else gf.ADV
        for trial in range(3):
            rows = synth_rows(8192, rng, strategy=strategy)
            flat = gf.init_flat(L, DEV)
            if trial == 2:  # perturb away from init (non-trivial LN gains, etc.)
                flat = flat + 0.05 * torch.randn_like(flat) * torch.as_tensor(L.mask_np, device=DEV)
            sd = gf.export_state_dict(L, flat)
            # JAX reference
            p = {k: jnp.asarray(v.numpy()) for k, v in sd.items()}
            x = jnp.asarray(rows[:, :51]); y = jnp.asarray(rows[:, 51:56])
            it = jnp.asarray(rows[:, 56]); reach = jnp.asarray(rows[:, 57])
            w = (it / jnp.maximum(it.max(), 1.)) ** 1.5
            if strategy:
                w = w * reach
            jl, jg = jax.value_and_grad(jb.objective)(p, x, y, w, strategy)
            jl = float(jl)
            jg = {k: np.asarray(v) for k, v in jg.items()}
            jflat = gf.import_state_dict(L, jg, DEV)
            # torch reference autograd
            sdt = {k: v.to(DEV).requires_grad_(True) for k, v in sd.items()}
            rows_t = torch.as_tensor(rows, device=DEV)
            tl = gf.reference_objective(sdt, rows_t, strategy)
            tl.backward()
            tflat = gf.import_state_dict(L, {k: v.grad.cpu() for k, v in sdt.items()}, DEV)
            mflat = torch.as_tensor(L.mask_np, device=DEV)
            for cdt, ltol, gtol in ((torch.float32, 1e-5, 1e-4), (torch.bfloat16, 2e-2, 6e-2)):
                ml, mg = gf.fwd_bwd_flat(L, flat, rows_t, mflat, cdt)
                ml = float(ml)
                rel_l = abs(ml - jl) / abs(jl)
                rel_g = float((mg - jflat).norm() / jflat.norm())
                cos = float((mg * jflat).sum() / (mg.norm() * jflat.norm()))
                check(f'T2 {kind} trial{trial} manual-{str(cdt)[6:]} vs JAX', rel_l < ltol and rel_g < gtol,
                      f'loss {ml:.6f} vs {jl:.6f} (rel {rel_l:.2e}) grad rel-err {rel_g:.2e} cos {cos:.6f}')
            rel_l = abs(float(tl) - jl) / abs(jl)
            rel_g = float((tflat - jflat).norm() / jflat.norm())
            check(f'T2 {kind} trial{trial} torch-autograd-ref vs JAX', rel_l < 1e-5 and rel_g < 1e-4,
                  f'rel loss {rel_l:.2e} grad {rel_g:.2e}')
    # negative control: a wrong LN eps must be caught by the same thresholds
    L = gf.ADV
    rows = synth_rows(8192, rng)
    flat = gf.init_flat(L, DEV)
    sd = gf.export_state_dict(L, flat)
    p = {k: jnp.asarray(v.numpy()) for k, v in sd.items()}
    it = jnp.asarray(rows[:, 56]); w = (it / jnp.maximum(it.max(), 1.)) ** 1.5
    jl, jg = jax.value_and_grad(jb.objective)(p, jnp.asarray(rows[:, :51]), jnp.asarray(rows[:, 51:56]), w, False)
    jflat = gf.import_state_dict(L, {k: np.asarray(v) for k, v in jg.items()}, DEV)
    old = gf.SLOPE
    gf.SLOPE = 0.02
    ml, mg = gf.fwd_bwd_flat(L, flat, torch.as_tensor(rows, device=DEV), torch.as_tensor(L.mask_np, device=DEV))
    gf.SLOPE = old
    rel_g = float((mg - jflat).norm() / jflat.norm())
    check('T2 negative control (slope 0.02 detected)', rel_g > 1e-4, f'grad rel-err {rel_g:.2e}')


def t3_adam():
    from training.network import DuelingAdvantageNet
    L = gf.ADV
    net = DuelingAdvantageNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    flat = gf.import_state_dict(L, net.state_dict(), DEV)
    m = torch.zeros_like(flat); v = torch.zeros_like(flat)
    t = torch.zeros((), device=DEV); lr = torch.tensor(3e-4, device=DEV)
    for s in range(20):
        scale = 0.1 if s % 2 else 10.0  # alternate clipped / unclipped
        grads = {k: torch.randn_like(p) * scale for k, p in net.named_parameters()}
        for k, p in net.named_parameters():
            p.grad = grads[k].clone()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        G = gf.import_state_dict(L, {k: g.cpu() for k, g in grads.items()}, DEV)
        gf.adam_clip_update(flat, G, m, v, t, lr)
    ref = gf.import_state_dict(L, {k: p.detach().cpu() for k, p in net.state_dict().items()}, DEV)
    err = float((flat - ref).abs().max())
    check('T3 clip+Adam vs torch.optim.Adam (20 steps)', err < 1e-6, f'max abs diff {err:.2e}')


def t4_buffer():
    from training.buffer import ReservoirBuffer
    rng = np.random.default_rng(1)
    for store_w in (False, True):
        cap = 1000
        ref = ReservoirBuffer(cap, 51, 5, store_sample_weights=store_w)
        gb = gf.GPUReservoirSet(2, cap, DEV)
        sizes = [300, 500, 150, 700, 999, 2500, 1, 4000]
        for i, n in enumerate(sizes):
            f = rng.normal(size=(n, 51)).astype(np.float32)
            a = rng.normal(size=(n, 5)).astype(np.float32)
            sw = rng.random(n).astype(np.float32) if store_w else None
            free = cap - ref.size
            rem = n - max(free, 0)
            u = rng.random(max(rem, 0))
            orig = np.random.random
            np.random.random = lambda k, _u=u: _u[:k] if k == len(_u) else (_ for _ in ()).throw(AssertionError)
            try:
                ref.add_batch(f, a, i + 1, sw)
            finally:
                np.random.random = orig
            gb.add_batch(1, f, a, i + 1, sw, _u=torch.as_tensor(u) if rem > 0 else None)
        v = gb.buffer_view(1).cpu().numpy()
        ok = (gb.sizes[1] == ref.size and gb.total_seen[1] == ref.total_seen
              and np.array_equal(v[:, :51], ref.features[:ref.size])
              and np.array_equal(v[:, 51:56], ref.advantages[:ref.size])
              and np.array_equal(v[:, 56], ref.iterations[:ref.size])
              and np.array_equal(v[:, 57], ref.sample_weights[:ref.size]))
        check(f'T4 reservoir add_batch exact (store_w={store_w})', ok,
              f'size {gb.sizes[1]} seen {gb.total_seen[1]} (ref {ref.size}/{ref.total_seen})')
        check('T4 other buffer untouched', gb.sizes[0] == 0 and float(gb.data[:cap].abs().sum()) == 0)
    # duplicate-slot handling: force many collisions, last writer must win
    cap = 10
    ref = ReservoirBuffer(cap, 51, 5)
    gb = gf.GPUReservoirSet(1, cap, DEV)
    f0 = np.zeros((cap, 51), np.float32); ref.add_batch(f0, np.zeros((cap, 5), np.float32), 1)
    gb.add_batch(0, f0, np.zeros((cap, 5), np.float32), 1)
    n = 200
    f = np.arange(n, dtype=np.float32)[:, None].repeat(51, 1)
    u = rng.random(n) * 0.05  # all land in low slots -> many duplicates
    orig = np.random.random
    np.random.random = lambda k: u
    try:
        ref.add_batch(f, np.zeros((n, 5), np.float32), 2)
    finally:
        np.random.random = orig
    gb.add_batch(0, f, np.zeros((n, 5), np.float32), 2, _u=torch.as_tensor(u))
    check('T4 duplicate slots: last writer wins (== numpy)',
          np.array_equal(gb.buffer_view(0).cpu().numpy()[:, :51], ref.features))


def _jax_ref(L, flat, rows, strategy):
    import jax
    import jax.numpy as jnp
    import jax_reference as jb
    sd = gf.export_state_dict(L, flat)
    p = {k: jnp.asarray(v.numpy()) for k, v in sd.items()}
    it = jnp.asarray(rows[:, 56])
    w = (it / jnp.maximum(it.max(), 1.)) ** 1.5
    if strategy:
        w = w * jnp.asarray(rows[:, 57])
    jl, jg = jax.value_and_grad(jb.objective)(p, jnp.asarray(rows[:, :51]),
                                              jnp.asarray(rows[:, 51:56]), w, strategy)
    return float(jl), gf.import_state_dict(L, {k: np.asarray(v) for k, v in jg.items()}, DEV)


def t5_fast_backend():
    """Fused-kernel step (each backend) vs jax_reference.objective directly; tolerance is set
    relative to the error of an honest bf16-autocast torch implementation."""
    rng = np.random.default_rng(5)
    for strategy in (False, True):
        kind = 'strat' if strategy else 'adv'
        L = gf.STRAT if strategy else gf.ADV
        mflat = torch.as_tensor(L.mask_np, device=DEV)
        rows = synth_rows(8192, rng, strategy=strategy)
        rows_t = torch.as_tensor(rows, device=DEV)
        flat = gf.init_flat(L, DEV) + 0.05 * torch.randn(L.size, device=DEV) * mflat
        jl, jg = _jax_ref(L, flat, rows, strategy)
        _, g16 = gf.fwd_bwd_flat(L, flat, rows_t, mflat, torch.bfloat16)
        e16 = float((g16 - jg).norm() / jg.norm())
        idx = torch.arange(8192, device=DEV)
        itmax = rows_t[:, gf.COL_IT].max().reshape(1)
        for backend in gf.BACKENDS:
            st = gf._make_step(kind, 8192, DEV, backend)
            st.reset(flat, 1e-3)
            st.fwd_bwd(rows_t, idx, itmax)
            torch.cuda.synchronize()
            G = (st.full_grad() if hasattr(st, 'full_grad') else st.G) * mflat
            e = float((G - jg).norm() / jg.norm())
            le = abs(float(st.loss_acc) - jl) / jl
            check(f'T5 {kind} backend={backend} grad/loss vs JAX', e < 2.5 * e16 + 1e-3 and le < 2e-3,
                  f'grad rel-err {e:.2e} (torch bf16 {e16:.2e}) loss rel-err {le:.1e}')
            P2 = flat.clone(); m = torch.zeros_like(P2); v = torch.zeros_like(P2)
            gf.adam_clip_update(P2, G.clone(), m, v, torch.zeros((), device=DEV),
                                torch.tensor(1e-3, device=DEV))
            st.update(); torch.cuda.synchronize()
            d = float((st.P - P2).abs().max())
            check(f'T5 {kind} backend={backend} clip+Adam update', d < 1e-6, f'max abs {d:.1e}')


def t6_fitter():
    """GraphFitter smoke test on learnable synthetic data: graph fast path and the eager
    fallback (smallest occupied seat < batch) both reduce the reference loss; empty seats are
    skipped; repeated fits restart from a fresh init."""
    rng = np.random.default_rng(6)
    for strategy in (False, True):
        kind = 'strat' if strategy else 'adv'
        L = gf.STRAT if strategy else gf.ADV
        nb = 1 if strategy else 6
        bufs = gf.GPUReservoirSet(nb, 60000, DEV)
        sizes = [50000] if strategy else [50000, 30000, 0, 20000, 40000, 25000]
        for b, n in enumerate(sizes):
            if n:
                r = synth_rows(n, rng, strategy=strategy)
                m = r[:, 46:51]
                if strategy:  # learnable target: masked softmax of the first features
                    e = np.exp(2 * r[:, :5]) * m
                    r[:, 51:56] = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
                else:  # learnable target: linear in the features
                    r[:, 51:56] = r[:, :5] * m
                bufs.add_rows(b, torch.as_tensor(r, device=DEV))
        ev = torch.cat([bufs.buffer_view(b)[:4000] for b in range(nb) if sizes[b]])
        fitter = gf.make_fitter(kind, bufs, chunk=100)
        lr = 1e-3
        fit = gf.fit_strategy if strategy else gf.fit_advantage
        l0 = float(gf.reference_objective(
            {k: v.to(DEV) for k, v in gf.export_state_dict(L, gf.init_flat(L, DEV)).items()},
            ev, strategy))
        res = [fit(fitter, lr, steps=300) for _ in range(2)]
        evl = [float(gf.reference_objective({k: v.to(DEV) for k, v in r.state_dict.items()}, ev,
                                            strategy)) for r in res]
        check(f'T6 {kind} graph fit reduces loss', max(evl) < 0.9 * l0 and all(np.isfinite(evl)),
              f'init {l0:.4f} -> {evl} avg {[round(r.avg_loss, 4) for r in res]}')
        check(f'T6 {kind} fresh init per fit', float((res[0].flat - res[1].flat).abs().max()) > 0)
        bufs.sizes[0] = 3000  # smallest occupied seat < batch -> eager fallback path
        r2 = fit(fitter, lr, steps=150)
        e2 = float(gf.reference_objective({k: v.to(DEV) for k, v in r2.state_dict.items()}, ev,
                                          strategy))
        check(f'T6 {kind} fallback path (batch cap 3000, 150 steps)', e2 < 0.95 * l0,
              f'eval {e2:.4f} avg {r2.avg_loss:.4f}')


if __name__ == '__main__':
    which = sys.argv[1:] or ['t1', 't2', 't3', 't4', 't5', 't6']
    names = {'t1': 't1_init', 't2': 't2_objective', 't3': 't3_adam', 't4': 't4_buffer',
             't5': 't5_fast_backend', 't6': 't6_fitter'}
    for name in which:
        globals()[names[name]]()
    print('FAILED:' if FAILS else 'ALL PASS', FAILS)
    sys.exit(1 if FAILS else 0)
