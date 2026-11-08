"""GPU-resident Deep CFR reservoir buffers and fast advantage / strategy fitting.

Faithful to upstream fullhouse-bot `training/train.py::train_network` /
`train_strategy_network` as reimplemented by `fullhouse-repro/jax_backend.py`:

* fresh torch-default init of DuelingAdvantageNet / AverageStrategyNet per fit;
* Adam(b1=.9, b2=.999, eps=1e-8), torch `clip_grad_norm_(1.0)` semantics;
* each minibatch picks ONE seat buffer with probability proportional to its
  size, then samples rows uniformly (with replacement) inside it;
* advantage targets normalised per sample over legal actions; weight
  (iter / max_iter_in_batch) ** 1.5 (times stored reach for the strategy net);
* MSE (advantage) / masked cross-entropy (strategy).

Rows live in one fp32 [N, 64] tensor per buffer family:
cols 0:51 features (46:51 = legal mask), 51:56 target, 56 iteration,
57 sample weight (1 for advantage rows), 58:64 zero padding.

Parameters live in one flat fp32 vector (see `Layout`) so Adam, clipping and
the CUDA-graph replay touch a single static buffer. `export_state_dict` maps
it back to the torch module state_dict naming.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

FEATURE_DIM = 51
N_ACTIONS = 5
MASK_OFF = 46
ROW_W = 64
COL_Y = 51
COL_IT = 56
COL_W = 57
TRUNK = 256
HEAD = 128
LN_EPS = 1e-5
SLOPE = 0.01


# ----------------------------------------------------------------------------
# Parameter layout
# ----------------------------------------------------------------------------
class Layout:
    """Flat parameter layout.

    Advantage net ('adv'): W1[256,64] (cols 51: zero), b1, g1, be1, W2[256,256],
    b2, g2, be2, W3[256,256] (= concat(value_head.0, advantage_head.0)), b3[256],
    Wo[8,256] (row 0 = value_head.2 on cols 0:128, rows 1:6 = advantage_head.2 on
    cols 128:256, everything else structurally zero), bo[8].
    Strategy net ('strat'): W1, b1, g1, be1, W2, b2, g2, be2, Wo[8,256] (rows 0:5 =
    policy_head), bo[8].
    """

    def __init__(self, kind: str):
        assert kind in ('adv', 'strat')
        self.kind = kind
        # big GEMM weights first, then every 'small' parameter (biases, LN params, the
        # output layer) contiguously from `small_off` on: kernels accumulate the small
        # gradients into replicated buffers to avoid contended atomics.
        shapes = [('W1', (TRUNK, ROW_W)), ('W2', (TRUNK, TRUNK))]
        if kind == 'adv':
            shapes += [('W3', (2 * HEAD, TRUNK))]
        shapes += [('b1', (TRUNK,)), ('g1', (TRUNK,)), ('be1', (TRUNK,)),
                   ('b2', (TRUNK,)), ('g2', (TRUNK,)), ('be2', (TRUNK,))]
        if kind == 'adv':
            shapes += [('b3', (2 * HEAD,))]
        shapes += [('Wo', (8, TRUNK)), ('bo', (8,))]
        self.shapes = dict(shapes)
        self.numel = {k: int(np.prod(v)) for k, v in shapes}
        self.offsets = {}
        off = 0
        for name, shp in shapes:
            self.offsets[name] = off
            off += int(np.prod(shp))
        self.size = off
        self.small_off = self.offsets['b1']
        self.n_small = off - self.small_off
        # structural mask: 1 where a real parameter lives, 0 for padding
        mask = np.zeros(off, np.float32)
        def put(name, sub):
            o = self.offsets[name]
            n = int(np.prod(self.shapes[name]))
            full = np.zeros(self.shapes[name], np.float32)
            full[sub] = 1
            mask[o:o + n] = full.ravel()
        put('W1', (slice(None), slice(0, FEATURE_DIM)))
        for nm in ('b1', 'g1', 'be1', 'W2', 'b2', 'g2', 'be2'):
            put(nm, (Ellipsis,))
        if kind == 'adv':
            put('W3', (Ellipsis,)); put('b3', (Ellipsis,))
            m = np.zeros((8, TRUNK), np.float32); m[0, :HEAD] = 1; m[1:6, HEAD:] = 1
            o = self.offsets['Wo']; mask[o:o + 8 * TRUNK] = m.ravel()
            o = self.offsets['bo']; mask[o:o + 6] = 1
        else:
            m = np.zeros((8, TRUNK), np.float32); m[:5] = 1
            o = self.offsets['Wo']; mask[o:o + 8 * TRUNK] = m.ravel()
            o = self.offsets['bo']; mask[o:o + 5] = 1
        self.mask_np = mask
        self.n_real = int(mask.sum())

    def views(self, flat: torch.Tensor) -> dict:
        return {k: flat[o:o + self.numel[k]].view(self.shapes[k])
                for k, o in self.offsets.items()}


ADV = Layout('adv')
STRAT = Layout('strat')
assert ADV.n_real == 146694, ADV.n_real


def init_flat(layout: Layout, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """torch default init: nn.Linear weight & bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    (kaiming_uniform(a=sqrt(5)) == that bound), LayerNorm weight=1, bias=0."""
    flat = torch.zeros(layout.size, device=device, dtype=torch.float32)
    v = layout.views(flat)
    def u(t, fan_in):
        b = 1.0 / math.sqrt(fan_in)
        t.uniform_(-b, b, generator=generator)
    u(v['W1'][:, :FEATURE_DIM], FEATURE_DIM); u(v['b1'], FEATURE_DIM)
    v['g1'].fill_(1.0)
    u(v['W2'], TRUNK); u(v['b2'], TRUNK)
    v['g2'].fill_(1.0)
    if layout.kind == 'adv':
        u(v['W3'], TRUNK); u(v['b3'], TRUNK)
        u(v['Wo'][0, :HEAD], HEAD); u(v['bo'][0:1], HEAD)
        u(v['Wo'][1:6, HEAD:], HEAD); u(v['bo'][1:6], HEAD)
    else:
        u(v['Wo'][:5], TRUNK); u(v['bo'][:5], TRUNK)
    return flat


def export_state_dict(layout: Layout, flat: torch.Tensor) -> dict:
    """Flat params -> torch module state_dict (CPU float32 tensors)."""
    v = {k: t.detach().float().cpu() for k, t in layout.views(flat).items()}
    sd = {'trunk.0.weight': v['W1'][:, :FEATURE_DIM], 'trunk.0.bias': v['b1'],
          'trunk.1.weight': v['g1'], 'trunk.1.bias': v['be1'],
          'trunk.3.weight': v['W2'], 'trunk.3.bias': v['b2'],
          'trunk.4.weight': v['g2'], 'trunk.4.bias': v['be2']}
    if layout.kind == 'adv':
        sd.update({'value_head.0.weight': v['W3'][:HEAD], 'value_head.0.bias': v['b3'][:HEAD],
                   'value_head.2.weight': v['Wo'][0:1, :HEAD], 'value_head.2.bias': v['bo'][0:1],
                   'advantage_head.0.weight': v['W3'][HEAD:],
                   'advantage_head.0.bias': v['b3'][HEAD:],
                   'advantage_head.2.weight': v['Wo'][1:6, HEAD:],
                   'advantage_head.2.bias': v['bo'][1:6]})
    else:
        sd.update({'policy_head.weight': v['Wo'][:5], 'policy_head.bias': v['bo'][:5]})
    return {k: t.contiguous().clone() for k, t in sd.items()}


def import_state_dict(layout: Layout, sd: dict, device) -> torch.Tensor:
    """torch module state_dict -> flat params (inverse of export_state_dict)."""
    flat = torch.zeros(layout.size, device=device, dtype=torch.float32)
    v = layout.views(flat)
    def g(k):
        t = sd[k]
        t = t.detach() if isinstance(t, torch.Tensor) else torch.as_tensor(np.asarray(t))
        return t.to(device=device, dtype=torch.float32)
    v['W1'][:, :FEATURE_DIM] = g('trunk.0.weight'); v['b1'][:] = g('trunk.0.bias')
    v['g1'][:] = g('trunk.1.weight'); v['be1'][:] = g('trunk.1.bias')
    v['W2'][:] = g('trunk.3.weight'); v['b2'][:] = g('trunk.3.bias')
    v['g2'][:] = g('trunk.4.weight'); v['be2'][:] = g('trunk.4.bias')
    if layout.kind == 'adv':
        v['W3'][:HEAD] = g('value_head.0.weight'); v['b3'][:HEAD] = g('value_head.0.bias')
        v['W3'][HEAD:] = g('advantage_head.0.weight'); v['b3'][HEAD:] = g('advantage_head.0.bias')
        v['Wo'][0:1, :HEAD] = g('value_head.2.weight'); v['bo'][0:1] = g('value_head.2.bias')
        v['Wo'][1:6, HEAD:] = g('advantage_head.2.weight'); v['bo'][1:6] = g('advantage_head.2.bias')
    else:
        v['Wo'][:5] = g('policy_head.weight'); v['bo'][:5] = g('policy_head.bias')
    return flat


# ----------------------------------------------------------------------------
# GPU reservoir buffers
# ----------------------------------------------------------------------------
class GPUReservoirSet:
    """`n_buffers` reservoir buffers of `capacity` rows each, stored contiguously in
    one fp32 [n_buffers * capacity, 64] device tensor (buffer b owns rows
    [b*capacity, (b+1)*capacity)).

    `add_batch` reproduces upstream `ReservoirBuffer.add_batch`: fill free slots in
    order; for the remaining rows the k-th incoming row (global count
    total_seen+k, 1-based) draws r = floor(U * (total_seen+k)) and replaces slot r
    if r < capacity. When several incoming rows of one batch draw the same slot the
    LAST one wins (numpy fancy-assignment order = sequential reservoir order).
    Random numbers come from a torch CUDA generator instead of numpy's global RNG.
    """

    def __init__(self, n_buffers: int, capacity: int, device, seed: int = 0):
        self.n = n_buffers
        self.capacity = capacity
        self.device = torch.device(device)
        self.data = torch.zeros(n_buffers * capacity, ROW_W, dtype=torch.float32, device=self.device)
        self.sizes = [0] * n_buffers
        self.total_seen = [0] * n_buffers
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self._winner = None  # lazily allocated int64 [capacity] scratch, -1 = empty

    def __len__(self):
        return sum(self.sizes)

    def buffer_view(self, b: int) -> torch.Tensor:
        return self.data[b * self.capacity: b * self.capacity + self.sizes[b]]

    def _rows(self, features, targets, iteration, sample_weights) -> torch.Tensor:
        dev = self.device
        f = torch.as_tensor(features)
        n = f.shape[0]
        rows = torch.zeros(n, ROW_W, dtype=torch.float32, device=dev)
        rows[:, :FEATURE_DIM] = f.to(dev, torch.float32, non_blocking=True)
        rows[:, COL_Y:COL_Y + N_ACTIONS] = torch.as_tensor(targets).to(dev, torch.float32,
                                                                       non_blocking=True)
        rows[:, COL_IT] = float(iteration)
        if sample_weights is None:
            rows[:, COL_W] = 1.0
        else:
            sw = torch.as_tensor(sample_weights)
            if sw.shape[0] != n:
                raise ValueError(f'sample_weights length {sw.shape[0]} does not match batch size {n}')
            rows[:, COL_W] = sw.to(dev, torch.float32, non_blocking=True)
        return rows

    def add_batch(self, b: int, features, targets, iteration: int, sample_weights=None,
                  _u=None) -> None:
        """features [n,51], targets [n,5] (numpy or torch, any device)."""
        n = int(features.shape[0])
        if n == 0:
            return
        rows = self._rows(features, targets, iteration, sample_weights)
        self.add_rows(b, rows, _u)

    def add_rows(self, b: int, rows: torch.Tensor, _u: torch.Tensor | None = None) -> None:
        """rows: fp32 [n, 64] on self.device, already in row layout.
        `_u` (tests only): float64 uniforms to use instead of the generator."""
        n = rows.shape[0]
        cap = self.capacity
        base = b * cap
        size = self.sizes[b]
        free = cap - size
        if free >= n:
            self.data[base + size: base + size + n] = rows
            self.sizes[b] = size + n
            self.total_seen[b] += n
            return
        if free > 0:
            self.data[base + size: base + cap] = rows[:free]
            self.sizes[b] = cap
            self.total_seen[b] += free
        free = max(free, 0)
        remaining = n - free
        start_seen = self.total_seen[b]
        totals = torch.arange(start_seen + 1, start_seen + remaining + 1, dtype=torch.float64,
                              device=self.device)
        u = (torch.rand(remaining, dtype=torch.float64, device=self.device, generator=self.gen)
             if _u is None else _u.to(self.device, torch.float64))
        r = (u * totals).to(torch.int64)
        accept = r < cap
        pos = r[accept]
        src = torch.nonzero(accept).squeeze(1) + free
        if self._winner is None:
            self._winner = torch.full((cap,), -1, dtype=torch.int64, device=self.device)
        win = self._winner
        win.scatter_reduce_(0, pos, src, reduce='amax')  # last incoming row wins
        keep = win[pos] == src
        pos_k, src_k = pos[keep], src[keep]
        self.data[base + pos_k] = rows[src_k]
        win[pos] = -1
        self.total_seen[b] += remaining

    # state for checkpointing (host copies)
    def state_dict(self, b: int) -> dict:
        v = self.buffer_view(b).cpu().numpy()
        return {'features': v[:, :FEATURE_DIM].copy(), 'advantages': v[:, COL_Y:COL_Y + 5].copy(),
                'iterations': v[:, COL_IT].copy(), 'sample_weights': v[:, COL_W].copy(),
                'count': self.sizes[b], 'total_seen': self.total_seen[b]}

    def load_state_dict(self, b: int, d: dict) -> None:
        cnt = int(d['count'])
        base = b * self.capacity
        sw = d.get('sample_weights')
        self.sizes[b] = 0
        self.total_seen[b] = 0
        rows = self._rows(d['features'][:cnt], d['advantages'][:cnt], 0,
                          None if sw is None else sw[:cnt])
        rows[:, COL_IT] = torch.as_tensor(np.asarray(d['iterations'][:cnt])).to(self.device)
        self.data[base: base + cnt] = rows
        self.sizes[b] = cnt
        self.total_seen[b] = int(d['total_seen'])


# ----------------------------------------------------------------------------
# Reference objective (fp32, autograd) == jax_backend.objective
# ----------------------------------------------------------------------------
def reference_objective(sd: dict, rows: torch.Tensor, strategy: bool) -> torch.Tensor:
    """Line-by-line port of jax_backend.objective + weight computation from fit_chunk.
    `sd` holds torch-state_dict-named tensors (requires_grad for gradient checks)."""
    x = rows[:, :FEATURE_DIM]
    y = rows[:, COL_Y:COL_Y + 5]
    it = rows[:, COL_IT]
    w = (it / torch.clamp(it.max(), min=1.0)) ** 1.5
    if strategy:
        w = w * rows[:, COL_W]
    mask = x[:, MASK_OFF:MASK_OFF + 5]
    lin = lambda h, p: h @ sd[p + '.weight'].T + sd[p + '.bias']
    h = x
    for i in (0, 3):
        h = lin(h, f'trunk.{i}')
        mean = h.mean(-1, keepdim=True)
        var = ((h - mean) ** 2).mean(-1, keepdim=True)
        h = (h - mean) * torch.rsqrt(var + LN_EPS)
        h = h * sd[f'trunk.{i+1}.weight'] + sd[f'trunk.{i+1}.bias']
        h = torch.nn.functional.leaky_relu(h, SLOPE)
    if strategy:
        pred = torch.where(mask > 0, lin(h, 'policy_head'), torch.full_like(mask, -1e9))
        yy = y * mask
        yy = yy / torch.clamp(yy.sum(-1, keepdim=True), min=1e-12)
        loss = -(yy * torch.log_softmax(pred, -1)).sum(-1)
    else:
        lr_ = lambda t: torch.nn.functional.leaky_relu(t, SLOPE)
        v = lin(lr_(lin(h, 'value_head.0')), 'value_head.2')
        a = lin(lr_(lin(h, 'advantage_head.0')), 'advantage_head.2') * mask
        mean = a.sum(-1, keepdim=True) / torch.clamp(mask.sum(-1, keepdim=True), min=1)
        pred = (v + a - mean * mask) * mask
        n = torch.clamp(mask.sum(-1, keepdim=True), min=1)
        mean = (y * mask).sum(-1, keepdim=True) / n
        var = (((y * mask - mean * mask) ** 2) * mask).sum(-1, keepdim=True) / n
        yn = (y - mean) / torch.sqrt(var + 1e-6) * mask
        loss = ((pred - yn) ** 2).mean(-1)
    return (loss * w).mean()


# ----------------------------------------------------------------------------
# Manual forward + backward on the flat layout (torch ops; compiled by inductor)
# ----------------------------------------------------------------------------
def _lrelu(t):
    return torch.where(t >= 0, t, t * SLOPE)


def _dlrelu(pre, g):
    return torch.where(pre >= 0, g, g * SLOPE)


def _ln_fwd(h, g, be):
    mu = h.mean(-1, keepdim=True)
    xc = h - mu
    var = (xc * xc).mean(-1, keepdim=True)
    r = torch.rsqrt(var + LN_EPS)
    xh = xc * r
    return xh, r, xh * g + be


def _ln_bwd(dz, xh, r, g):
    dxh = dz * g
    return r * (dxh - dxh.mean(-1, keepdim=True) - xh * (dxh * xh).mean(-1, keepdim=True))


def fwd_bwd_flat(layout: Layout, P: torch.Tensor, rows: torch.Tensor, mask_flat: torch.Tensor,
                 cdt: torch.dtype = torch.float32):
    """Returns (loss, flat grad). Matmul inputs in `cdt` (fp32 or bf16), fp32 accumulate
    semantics of torch.mm, LayerNorm / losses in fp32."""
    V = layout.views(P)
    B = rows.shape[0]
    strategy = layout.kind == 'strat'
    x = rows.clone()
    x[:, FEATURE_DIM:] = 0
    m = rows[:, MASK_OFF:MASK_OFF + 5]
    y = rows[:, COL_Y:COL_Y + 5]
    it = rows[:, COL_IT]
    w = (it / torch.clamp(it.max(), min=1.0)) ** 1.5
    if strategy:
        w = w * rows[:, COL_W]
    mm = lambda a, b: torch.mm(a.to(cdt), b.to(cdt)).float()

    h1 = mm(x, V['W1'].t()) + V['b1']
    xh1, r1, z1 = _ln_fwd(h1, V['g1'], V['be1'])
    a1 = _lrelu(z1)
    h2 = mm(a1, V['W2'].t()) + V['b2']
    xh2, r2, z2 = _ln_fwd(h2, V['g2'], V['be2'])
    a2 = _lrelu(z2)
    if strategy:
        top = a2
        o8 = mm(top, V['Wo'].t()) + V['bo']
        logits = o8[:, :5]
        logits = torch.where(m > 0, logits, torch.full_like(logits, -1e9))
        yy = y * m
        ys = torch.clamp(yy.sum(-1, keepdim=True), min=1e-12)
        yy = yy / ys
        lse = torch.logsumexp(logits, -1, keepdim=True)
        logp = logits - lse
        loss_i = -(yy * logp).sum(-1)
        loss = (loss_i * w).mean()
        p = torch.exp(logp)
        dlog = (p * yy.sum(-1, keepdim=True) - yy) * (w / B)[:, None]
        dlog = torch.where(m > 0, dlog, torch.zeros_like(dlog))
        do8 = torch.cat([dlog, torch.zeros(B, 3, device=rows.device)], 1)
    else:
        h3 = mm(a2, V['W3'].t()) + V['b3']
        a3 = _lrelu(h3)
        top = a3
        o8 = mm(a3, V['Wo'].t()) + V['bo']
        v = o8[:, 0:1]
        A = o8[:, 1:6] * m
        n = torch.clamp(m.sum(-1, keepdim=True), min=1)
        amean = A.sum(-1, keepdim=True) / n
        pred = (v + A - amean * m) * m
        ym = y * m
        ymean = ym.sum(-1, keepdim=True) / n
        yvar = (((ym - ymean * m) ** 2) * m).sum(-1, keepdim=True) / n
        yn = (y - ymean) / torch.sqrt(yvar + 1e-6) * m
        diff = pred - yn
        loss = ((diff * diff).mean(-1) * w).mean()
        dpred = diff * (2.0 / (5 * B)) * w[:, None]
        g = dpred * m                       # d wrt (v + A - amean*m)
        dv = g.sum(-1, keepdim=True)
        dA = g - (g * m).sum(-1, keepdim=True) / n   # through amean
        dA = dA * m                          # through A = raw * m
        do8 = torch.cat([dv, dA, torch.zeros(B, 2, device=rows.device)], 1)
    dWo = mm(do8.t(), top)
    dbo = do8.sum(0)
    dtop = mm(do8, V['Wo'])
    if not strategy:
        dh3 = _dlrelu(h3, dtop)
        dW3 = mm(dh3.t(), a2)
        db3 = dh3.sum(0)
        da2 = mm(dh3, V['W3'])
    else:
        da2 = dtop
    dz2 = _dlrelu(z2, da2)
    dg2 = (dz2 * xh2).sum(0)
    dbe2 = dz2.sum(0)
    dh2 = _ln_bwd(dz2, xh2, r2, V['g2'])
    dW2 = mm(dh2.t(), a1)
    db2 = dh2.sum(0)
    da1 = mm(dh2, V['W2'])
    dz1 = _dlrelu(z1, da1)
    dg1 = (dz1 * xh1).sum(0)
    dbe1 = dz1.sum(0)
    dh1 = _ln_bwd(dz1, xh1, r1, V['g1'])
    dW1 = mm(dh1.t(), x)
    db1 = dh1.sum(0)
    parts = dict(W1=dW1, b1=db1, g1=dg1, be1=dbe1, W2=dW2, b2=db2, g2=dg2, be2=dbe2,
                 Wo=dWo, bo=dbo)
    if not strategy:
        parts.update(W3=dW3, b3=db3)
    grad = torch.cat([parts[k].reshape(-1) for k in layout.offsets]) * mask_flat
    return loss, grad


def adam_clip_update(P, G, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, max_norm=1.0):
    """torch clip_grad_norm_(max_norm) followed by torch.optim.Adam step (in place).
    t: fp32 device scalar step count BEFORE this update; lr: fp32 device scalar."""
    norm = torch.sqrt((G * G).sum())
    coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    g = G * coef
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    t.add_(1.0)
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    denom = v.sqrt() / bc2.sqrt() + eps
    P.sub_(lr / bc1 * m / denom)


# ----------------------------------------------------------------------------
# Sampling (graph-safe)
# ----------------------------------------------------------------------------
def sample_indices(sizes: torch.Tensor, offsets: torch.Tensor, n_steps: int, batch: int,
                   gen: torch.Generator | None = None) -> torch.Tensor:
    """[n_steps, batch] int64 global row ids. Per step: seat b ~ sizes/sum(sizes), then
    rows uniform with replacement in [0, sizes[b]). Device-only (CUDA-graph safe)."""
    cs = torch.cumsum(sizes, 0)
    total = cs[-1]
    u = torch.randint(0, 2 ** 62, (n_steps,), device=sizes.device, generator=gen) % total
    seat = torch.searchsorted(cs, u, right=True)
    r = torch.randint(0, 2 ** 62, (n_steps, batch), device=sizes.device, generator=gen)
    r = r % sizes[seat][:, None]
    return offsets[seat][:, None] + r


# ----------------------------------------------------------------------------
# Fitter: static buffers + CUDA graph of `chunk` steps
# ----------------------------------------------------------------------------
@dataclass
class FitResult:
    state_dict: dict
    avg_loss: float
    seconds: float
    flat: torch.Tensor


class CompiledFitter:
    """(benchmark variant: torch.compile of fwd_bwd_flat) Owns static params / optimizer state and a captured CUDA graph of `chunk`
    optimizer steps (sampling included). One instance per (net kind, data tensor,
    batch); reuse it every iteration (params are re-initialised in place)."""

    def __init__(self, kind: str, data: torch.Tensor, n_buffers: int, capacity: int,
                 batch: int = 8192, chunk: int = 250, cdt: torch.dtype = torch.bfloat16,
                 impl: str = 'compiled', seed: int = 0):
        self.layout = ADV if kind == 'adv' else STRAT
        self.kind = kind
        self.data = data
        self.dev = data.device
        self.n_buffers = n_buffers
        self.capacity = capacity
        self.batch = batch
        self.chunk = chunk
        self.cdt = cdt
        self.impl = impl
        L, dev = self.layout, self.dev
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self.P = torch.zeros(L.size, device=dev)
        self.m = torch.zeros_like(self.P)
        self.v = torch.zeros_like(self.P)
        self.t = torch.zeros((), device=dev)
        self.lr = torch.zeros((), device=dev)
        self.loss_acc = torch.zeros((), device=dev)
        self.mask_flat = torch.as_tensor(L.mask_np, device=dev)
        self.sizes = torch.zeros(n_buffers, dtype=torch.int64, device=dev)
        self.offsets = torch.arange(n_buffers, dtype=torch.int64, device=dev) * capacity
        self.idx = torch.zeros(chunk, batch, dtype=torch.int64, device=dev)
        self._graph = None
        self._step_fn = self._make_step_fn()

    def _make_step_fn(self):
        L, cdt = self.layout, self.cdt

        def step(P, m, v, t, lr, loss_acc, data, idx, mask_flat):
            rows = data[idx]
            loss, G = fwd_bwd_flat(L, P, rows, mask_flat, cdt)
            adam_clip_update(P, G, m, v, t, lr)
            loss_acc.add_(loss)

        if self.impl == 'compiled':
            return torch.compile(step, mode='max-autotune-no-cudagraphs', fullgraph=True,
                                 dynamic=False)
        if self.impl == 'eager':
            return step
        raise ValueError(self.impl)

    def _chunk_body(self):
        self.idx.copy_(sample_indices(self.sizes, self.offsets, self.chunk, self.batch, self.gen))
        for k in range(self.chunk):
            self._step_fn(self.P, self.m, self.v, self.t, self.lr, self.loss_acc, self.data,
                          self.idx[k], self.mask_flat)

    def _capture(self):
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):  # warm up / autotune outside capture
                self._step_fn(self.P, self.m, self.v, self.t, self.lr, self.loss_acc, self.data,
                              self.idx[0], self.mask_flat)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        torch.cuda.synchronize(self.dev)
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.gen)  # custom generator: philox offset per replay
        with torch.cuda.graph(g, stream=s):
            self._chunk_body()
        torch.cuda.synchronize(self.dev)
        self._graph = g

    def reset(self, lr: float, flat_init: torch.Tensor | None = None):
        if flat_init is None:
            flat_init = init_flat(self.layout, self.dev, self.gen)
        self.P.copy_(flat_init)
        self.m.zero_(); self.v.zero_(); self.t.zero_(); self.loss_acc.zero_()
        self.lr.fill_(lr)

    def fit(self, sizes: list[int], lr: float, steps: int, flat_init: torch.Tensor | None = None,
            use_graph: bool = True) -> FitResult:
        t0 = time.monotonic()
        sizes_np = np.asarray(sizes, np.int64)
        occupied = sizes_np[sizes_np > 0]
        if occupied.size == 0:
            raise ValueError('all buffers empty')
        self.sizes.copy_(torch.as_tensor(sizes_np))
        if self._graph is None and use_graph:
            self.reset(lr, flat_init)
            self.idx.copy_(sample_indices(self.sizes, self.offsets, self.chunk, self.batch, self.gen))
            self._capture()
        self.reset(lr, flat_init)
        batch_cap = int(occupied.min())
        if batch_cap < self.batch or steps % self.chunk != 0 or not use_graph:
            # rare path (tiny buffers / odd step counts): eager loop, upstream batch cap
            b = min(self.batch, batch_cap)
            done = 0
            while done < steps:
                n = min(self.chunk, steps - done)
                idx = sample_indices(self.sizes, self.offsets, n, b, self.gen)
                for k in range(n):
                    rows = self.data[idx[k]]
                    loss, G = fwd_bwd_flat(self.layout, self.P, rows, self.mask_flat, self.cdt)
                    adam_clip_update(self.P, G, self.m, self.v, self.t, self.lr)
                    self.loss_acc.add_(loss)
                done += n
        else:
            for _ in range(steps // self.chunk):
                self._graph.replay()
        avg = float(self.loss_acc) / steps
        return FitResult(export_state_dict(self.layout, self.P), avg, time.monotonic() - t0,
                         self.P.clone())


def lr_schedule(iteration: int, lr: float = 1e-3) -> float:
    """Upstream train_deep_cfr learning rate: linear warm-up 1e-4 -> lr over iters 1..20."""
    return 1e-4 + (lr - 1e-4) * (iteration / 20) if iteration <= 20 else lr


# ----------------------------------------------------------------------------
# Public fitting API
# ----------------------------------------------------------------------------
BACKENDS = ('cuda', 'cublas_triton')
DEFAULT_BACKEND = 'cuda'


def _make_step(kind: str, batch: int, device, backend: str):
    layout = ADV if kind == 'adv' else STRAT
    if backend == 'cuda':
        import fast_step
        return fast_step.CudaRowStep(layout, batch, device)
    if backend == 'cublas_triton':
        import triton_fit3
        return triton_fit3.CublasStep(layout, batch, device)
    raise ValueError(f'unknown backend {backend!r}; choose from {BACKENDS}')


class GraphFitter:
    """Fits a freshly initialised advantage ('adv') or strategy ('strat') net on the rows
    of a GPUReservoirSet, faithfully to upstream train_network / train_strategy_network
    (see module docstring), with the whole optimizer step + minibatch sampling captured
    in one CUDA graph of `chunk` steps that is reused for every fit.

    Create ONE instance per buffer set and keep it (graph capture costs ~1-3 s once).
    The buffer tensor `buffers.data` must not be reallocated afterwards.
    """

    def __init__(self, kind: str, buffers: 'GPUReservoirSet', batch: int = 8192,
                 chunk: int = 250, backend: str = DEFAULT_BACKEND, seed: int = 0):
        assert kind in ('adv', 'strat')
        self.kind = kind
        self.layout = ADV if kind == 'adv' else STRAT
        self.buffers = buffers
        self.data = buffers.data
        self.dev = self.data.device
        self.batch = batch
        self.chunk = chunk
        self.backend = backend
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        with torch.cuda.device(self.dev):
            self.step = _make_step(kind, batch, self.dev, backend)
        n = buffers.n
        self.sizes = torch.zeros(n, dtype=torch.int64, device=self.dev)
        self.offsets = torch.arange(n, dtype=torch.int64, device=self.dev) * buffers.capacity
        self.idx = torch.zeros(chunk, batch, dtype=torch.int64, device=self.dev)
        self.itmax = torch.zeros(chunk, dtype=torch.float32, device=self.dev)
        self._graph = None

    def _sample_into(self, n_steps, batch, idx, itmax):
        i = sample_indices(self.sizes, self.offsets, n_steps, batch, self.gen)
        idx.copy_(i)
        itmax.copy_(self.data[:, COL_IT][i].amax(dim=1))

    def _chunk_body(self):
        self._sample_into(self.chunk, self.batch, self.idx, self.itmax)
        for k in range(self.chunk):
            self.step.step(self.data, self.idx[k], self.itmax[k:k + 1])

    def _capture(self):
        dev = self.dev
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            self._sample_into(self.chunk, self.batch, self.idx, self.itmax)
            for k in range(3):  # warm-up: JIT / cuBLAS heuristics outside capture
                self.step.step(self.data, self.idx[k], self.itmax[k:k + 1])
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.gen)
        with torch.cuda.graph(g, stream=s):
            self._chunk_body()
        torch.cuda.synchronize(dev)
        self._graph = g

    def fit(self, lr: float, steps: int, flat_init: torch.Tensor | None = None,
            init_generator: torch.Generator | None = None) -> FitResult:
        """Returns FitResult(state_dict (CPU fp32, torch module naming), avg_loss over all
        steps (upstream semantics), seconds, flat fp32 params on device)."""
        t0 = time.monotonic()
        dev = self.dev
        with torch.cuda.device(dev):
            sizes_np = np.asarray(self.buffers.sizes, np.int64)
            occupied = sizes_np[sizes_np > 0]
            if occupied.size == 0:
                raise ValueError('all buffers empty')
            self.sizes.copy_(torch.as_tensor(sizes_np))
            if flat_init is None:
                flat_init = init_flat(self.layout, dev, init_generator if init_generator is not None
                                      else self.gen)
            batch_cap = int(occupied.min())
            fast = batch_cap >= self.batch and steps % self.chunk == 0
            if fast and self._graph is None:
                self.step.reset(flat_init, lr)
                self._capture()
            if fast:
                step = self.step
                step.reset(flat_init, lr)
                for _ in range(steps // self.chunk):
                    self._graph.replay()
            else:
                # rare path: some occupied seat has < batch rows (upstream caps the batch at
                # the smallest occupied buffer, as jax_backend does) or odd step counts.
                b = min(self.batch, batch_cap)
                step = self.step if b == self.batch else _make_step(self.kind, b, dev, self.backend)
                step.reset(flat_init, lr)
                done = 0
                while done < steps:
                    n = min(self.chunk, steps - done)
                    idx = torch.empty(n, b, dtype=torch.int64, device=dev)
                    itm = torch.empty(n, dtype=torch.float32, device=dev)
                    self._sample_into(n, b, idx, itm)
                    for k in range(n):
                        step.step(self.data, idx[k], itm[k:k + 1])
                    done += n
            avg = float(step.loss_acc) / steps
            return FitResult(export_state_dict(self.layout, step.P), avg,
                             time.monotonic() - t0, step.P.clone())


def make_fitter(kind: str, buffers: 'GPUReservoirSet', **kw) -> GraphFitter:
    """kind 'adv' (6-seat advantage buffers) or 'strat' (1 strategy buffer)."""
    return GraphFitter(kind, buffers, **kw)


def fit_advantage(fitter: GraphFitter, lr: float, steps: int = 10000, **kw) -> FitResult:
    assert fitter.kind == 'adv'
    return fitter.fit(lr, steps, **kw)


def fit_strategy(fitter: GraphFitter, lr: float, steps: int = 3000, **kw) -> FitResult:
    assert fitter.kind == 'strat'
    return fitter.fit(lr, steps, **kw)
