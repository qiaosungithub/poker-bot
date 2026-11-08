"""Fused Triton kernels for one Deep CFR fitting step (advantage or strategy net).

One optimizer step = 8 kernels (advantage) / 6 kernels (strategy):
  K1 fwd1   : gather rows by index, target normalisation, x@W1^T, LN1, LeakyReLU
  K2 fwd2   : a1@W2^T, LN2, LeakyReLU
  K3 head   : adv:   a2@W3^T, LeakyReLU, output layer, dueling combine, MSE, and the
                     backward of the output layer -> dh3 (+ dWo, dbo, db3)
              strat: output layer, masked CE, backward -> da2, then LN2 backward
  K4 bwd2   : (adv) dh3@W3 -> da2, LN2 backward -> dh2 (+ dg2, dbe2, db2)
  K5 bwd1   : dh2@W2 -> da1, LN1 backward -> dh1 (+ dg1, dbe1, db1)
  K6 wgrad  : split-K dW3 = dh3^T a2, dW2 = dh2^T a1, dW1 = dh1^T x (atomics into G)
  K7 norm   : ||G * mask||^2
  K8 adam   : clip coef, Adam update of fp32 master params, bf16 weight copy, G := 0

Matmul operands of the three big layers are bf16 with fp32 accumulation (the
numerics of upstream's bf16 autocast path: linear outputs are rounded to bf16
before LayerNorm, LN statistics and all losses in fp32). The tiny output layer
(256 -> 6) and its backward run in fp32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import gpu_fit as gf

LN_EPS = 1e-5
SLOPE = 0.01


@triton.jit
def _lrelu(x):
    return tl.where(x >= 0, x, x * 0.01)


@triton.jit
def _ln_fwd_epilogue(acc, rows, b_ptr, g_ptr, be_ptr, h_ptr, st_ptr, out_ptr, N: tl.constexpr):
    """acc: [BM, N] fp32 matmul result (bias not yet added). Stores h (bf16),
    (mu, rstd) and lrelu(LN(h)) (bf16)."""
    rn = tl.arange(0, N)
    h = acc + tl.load(b_ptr + rn)[None, :]
    hb = h.to(tl.bfloat16)
    tl.store(h_ptr + rows[:, None] * N + rn[None, :], hb)
    hf = hb.to(tl.float32)
    mu = tl.sum(hf, axis=1) / N
    xc = hf - mu[:, None]
    var = tl.sum(xc * xc, axis=1) / N
    r = 1.0 / tl.sqrt(var + 1e-5)
    z = xc * r[:, None] * tl.load(g_ptr + rn)[None, :] + tl.load(be_ptr + rn)[None, :]
    a = _lrelu(z)
    tl.store(out_ptr + rows[:, None] * N + rn[None, :], a.to(tl.bfloat16))
    tl.store(st_ptr + rows * 2, mu)
    tl.store(st_ptr + rows * 2 + 1, r)


@triton.jit
def _ln_bwd_epilogue(da, rows, h_ptr, st_ptr, g_ptr, be_ptr, dh_ptr,
                     gb_ptr, gg_ptr, gbe_ptr, N: tl.constexpr):
    """da: [BM, N] fp32 grad wrt lrelu(LN(h)). Stores dh (bf16) and atomically adds
    column sums into G (bias, LN gain, LN bias)."""
    rn = tl.arange(0, N)
    hf = tl.load(h_ptr + rows[:, None] * N + rn[None, :]).to(tl.float32)
    mu = tl.load(st_ptr + rows * 2)
    r = tl.load(st_ptr + rows * 2 + 1)
    g = tl.load(g_ptr + rn)
    xh = (hf - mu[:, None]) * r[:, None]
    z = xh * g[None, :] + tl.load(be_ptr + rn)[None, :]
    dz = tl.where(z >= 0, da, da * 0.01)
    tl.atomic_add(gg_ptr + rn, tl.sum(dz * xh, axis=0))
    tl.atomic_add(gbe_ptr + rn, tl.sum(dz, axis=0))
    dxh = dz * g[None, :]
    m1 = tl.sum(dxh, axis=1) / N
    m2 = tl.sum(dxh * xh, axis=1) / N
    dh = r[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
    tl.atomic_add(gb_ptr + rn, tl.sum(dh, axis=0))
    tl.store(dh_ptr + rows[:, None] * N + rn[None, :], dh.to(tl.bfloat16))


@triton.jit
def k_fwd1(data_ptr, idx_ptr, w1_ptr, b1_ptr, g1_ptr, be1_ptr,
           xb_ptr, aux_ptr, h1_ptr, st1_ptr, a1_ptr, itmax_ptr, ctrl_ptr,
           STRAT: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:  # per-step control: norm2 := 0, t += 1
        tl.store(ctrl_ptr, 0.0)
        tl.store(ctrl_ptr + 1, tl.load(ctrl_ptr + 1) + 1.0)
    rows = pid * BM + tl.arange(0, BM)
    ids = tl.load(idx_ptr + rows).to(tl.int64)
    c64 = tl.arange(0, 64)
    base = data_ptr + ids[:, None] * 64
    x = tl.load(base + c64[None, :], mask=c64[None, :] < 51, other=0.0)
    xb = x.to(tl.bfloat16)
    tl.store(xb_ptr + rows[:, None] * 64 + c64[None, :], xb)
    # targets / mask / weights
    j = tl.arange(0, 16)
    jm = j[None, :] < 5
    mk = tl.load(base + 46 + j[None, :], mask=jm, other=0.0)
    yv = tl.load(base + 51 + j[None, :], mask=jm, other=0.0)
    it = tl.load(data_ptr + ids * 64 + 56)
    sw = tl.load(data_ptr + ids * 64 + 57)
    if STRAT:
        yy = yv * mk
        yn = yy / tl.maximum(tl.sum(yy, axis=1), 1e-12)[:, None]
    else:
        n = tl.maximum(tl.sum(mk, axis=1), 1.0)
        ym = yv * mk
        mean = tl.sum(ym, axis=1) / n
        d = ym - mean[:, None] * mk
        var = tl.sum(d * d * mk, axis=1) / n
        yn = (yv - mean[:, None]) / tl.sqrt(var + 1e-6)[:, None] * mk
    tl.store(aux_ptr + rows[:, None] * 48 + j[None, :], mk)
    tl.store(aux_ptr + rows[:, None] * 48 + 16 + j[None, :], yn)
    tl.store(aux_ptr + rows * 48 + 32, it)
    tl.store(aux_ptr + rows * 48 + 33, sw)
    tl.atomic_max(itmax_ptr, tl.max(it, axis=0).to(tl.int32))
    # layer 1: W1 is [256, 64] (out, in) bf16 -> B operand W1^T [64, 256]
    rn = tl.arange(0, 256)
    w1t = tl.load(w1_ptr + rn[None, :] * 64 + c64[:, None])
    acc = tl.dot(xb, w1t)
    _ln_fwd_epilogue(acc, rows, b1_ptr, g1_ptr, be1_ptr, h1_ptr, st1_ptr, a1_ptr, 256)


@triton.jit
def _mm_rows_wT(a_ptr, w_ptr, rows, BM: tl.constexpr, BK: tl.constexpr):
    """acc[BM,256] = A[rows, :256] @ W^T with W [256 out, 256 in] row-major (bf16)."""
    rn = tl.arange(0, 256)
    acc = tl.zeros((BM, 256), tl.float32)
    for k in range(0, 256, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rows[:, None] * 256 + rk[None, :])
        w = tl.load(w_ptr + rn[None, :] * 256 + rk[:, None])
        acc = tl.dot(a, w, acc)
    return acc


@triton.jit
def _mm_rows_w(a_ptr, w_ptr, rows, BM: tl.constexpr, BK: tl.constexpr):
    """acc[BM,256] = A[rows, :256] @ W with W [256 out(K), 256 in(N)] row-major (bf16)."""
    rn = tl.arange(0, 256)
    acc = tl.zeros((BM, 256), tl.float32)
    for k in range(0, 256, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rows[:, None] * 256 + rk[None, :])
        w = tl.load(w_ptr + rk[:, None] * 256 + rn[None, :])
        acc = tl.dot(a, w, acc)
    return acc


@triton.jit
def k_fwd2(a1_ptr, w2_ptr, b2_ptr, g2_ptr, be2_ptr, h2_ptr, st2_ptr, a2_ptr,
           BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    acc = _mm_rows_wT(a1_ptr, w2_ptr, rows, BM, BK)
    _ln_fwd_epilogue(acc, rows, b2_ptr, g2_ptr, be2_ptr, h2_ptr, st2_ptr, a2_ptr, 256)


@triton.jit
def k_head_adv(a2_ptr, w3_ptr, b3_ptr, wo_ptr, bo_ptr, aux_ptr, itmax_ptr,
               dh3_ptr, gb3_ptr, gwo_ptr, gbo_ptr, loss_ptr, inv_5b,
               BM: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    j = tl.arange(0, 16)
    jm = j < 5
    h3 = _mm_rows_wT(a2_ptr, w3_ptr, rows, BM, BK) + tl.load(b3_ptr + rn)[None, :]
    a3 = _lrelu(h3)
    # output layer in fp32. Wo is [8, 256]: row 0 value (cols :128), rows 1..5 advantage.
    wv_t = tl.load(wo_ptr + rn[:, None] + j[None, :] * 0)                       # [256,16] all = Wo[0]
    wa_t = tl.load(wo_ptr + (1 + j[None, :]) * 256 + rn[:, None], mask=jm[None, :], other=0.0)
    ov = tl.dot(a3, wv_t, input_precision=PREC)
    oa = tl.dot(a3, wa_t, input_precision=PREC)
    bo0 = tl.load(bo_ptr)
    boa = tl.load(bo_ptr + 1 + j, mask=jm, other=0.0)
    mk = tl.load(aux_ptr + rows[:, None] * 48 + j[None, :])
    yn = tl.load(aux_ptr + rows[:, None] * 48 + 16 + j[None, :])
    it = tl.load(aux_ptr + rows * 48 + 32)
    itmax = tl.maximum(tl.load(itmax_ptr).to(tl.float32), 1.0)
    q = it / itmax
    w = q * tl.sqrt(q)
    v = ov + bo0
    A = (oa + boa[None, :]) * mk
    n = tl.maximum(tl.sum(mk, axis=1), 1.0)
    amean = tl.sum(A, axis=1) / n
    pred = (v + A - amean[:, None] * mk) * mk
    diff = pred - yn
    tl.atomic_add(loss_ptr, tl.sum(tl.sum(diff * diff, axis=1) * w, axis=0) * inv_5b)
    g = diff * (2.0 * inv_5b) * w[:, None] * mk
    dv = tl.sum(g, axis=1)
    dA = (g - (tl.sum(g * mk, axis=1) / n)[:, None]) * mk
    # backward of the output layer
    wa = tl.load(wo_ptr + (1 + j[:, None]) * 256 + rn[None, :], mask=jm[:, None], other=0.0)
    wv = tl.load(wo_ptr + rn)
    dtop = tl.dot(dA, wa, input_precision=PREC) + dv[:, None] * wv[None, :]
    gwa = tl.dot(tl.trans(dA), a3, input_precision=PREC)                    # [16, 256]
    tl.atomic_add(gwo_ptr + (1 + j[:, None]) * 256 + rn[None, :], gwa, mask=jm[:, None])
    tl.atomic_add(gwo_ptr + rn, tl.sum(dv[:, None] * a3, axis=0))
    tl.atomic_add(gbo_ptr, tl.sum(dv, axis=0))
    tl.atomic_add(gbo_ptr + 1 + j, tl.sum(dA, axis=0), mask=jm)
    dh3 = tl.where(h3 >= 0, dtop, dtop * 0.01)
    tl.atomic_add(gb3_ptr + rn, tl.sum(dh3, axis=0))
    tl.store(dh3_ptr + rows[:, None] * 256 + rn[None, :], dh3.to(tl.bfloat16))


@triton.jit
def k_head_strat(a2_ptr, wo_ptr, bo_ptr, aux_ptr, itmax_ptr,
                 h2_ptr, st2_ptr, g2_ptr, be2_ptr, dh2_ptr, gb2_ptr, gg2_ptr, gbe2_ptr,
                 gwo_ptr, gbo_ptr, loss_ptr, inv_b,
                 BM: tl.constexpr, PREC: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    j = tl.arange(0, 16)
    jm = j < 5
    a2 = tl.load(a2_ptr + rows[:, None] * 256 + rn[None, :]).to(tl.float32)
    wo_t = tl.load(wo_ptr + j[None, :] * 256 + rn[:, None], mask=jm[None, :], other=0.0)
    logits = tl.dot(a2, wo_t, input_precision=PREC) + tl.load(bo_ptr + j, mask=jm, other=0.0)[None, :]
    mk = tl.load(aux_ptr + rows[:, None] * 48 + j[None, :])
    yn = tl.load(aux_ptr + rows[:, None] * 48 + 16 + j[None, :])
    it = tl.load(aux_ptr + rows * 48 + 32)
    sw = tl.load(aux_ptr + rows * 48 + 33)
    itmax = tl.maximum(tl.load(itmax_ptr).to(tl.float32), 1.0)
    q = it / itmax
    w = q * tl.sqrt(q) * sw
    logits = tl.where(mk > 0, logits, -1e9)
    mx = tl.max(logits, axis=1)
    e = tl.exp(logits - mx[:, None])
    se = tl.sum(e, axis=1)
    logp = logits - mx[:, None] - tl.log(se)[:, None]
    loss_i = -tl.sum(yn * logp, axis=1)
    tl.atomic_add(loss_ptr, tl.sum(loss_i * w, axis=0) * inv_b)
    p = e / se[:, None]
    dlog = (p * tl.sum(yn, axis=1)[:, None] - yn) * (w * inv_b)[:, None]
    dlog = tl.where(mk > 0, dlog, 0.0)
    wo = tl.load(wo_ptr + j[:, None] * 256 + rn[None, :], mask=jm[:, None], other=0.0)
    da2 = tl.dot(dlog, wo, input_precision=PREC)
    gwo = tl.dot(tl.trans(dlog), a2, input_precision=PREC)
    tl.atomic_add(gwo_ptr + j[:, None] * 256 + rn[None, :], gwo, mask=jm[:, None])
    tl.atomic_add(gbo_ptr + j, tl.sum(dlog, axis=0), mask=jm)
    _ln_bwd_epilogue(da2, rows, h2_ptr, st2_ptr, g2_ptr, be2_ptr, dh2_ptr,
                     gb2_ptr, gg2_ptr, gbe2_ptr, 256)


@triton.jit
def k_bwd_ln(dh_ptr, w_ptr, h_ptr, st_ptr, g_ptr, be_ptr, out_ptr, gb_ptr, gg_ptr, gbe_ptr,
             BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    da = _mm_rows_w(dh_ptr, w_ptr, rows, BM, BK)
    _ln_bwd_epilogue(da, rows, h_ptr, st_ptr, g_ptr, be_ptr, out_ptr, gb_ptr, gg_ptr, gbe_ptr, 256)


@triton.jit
def k_wgrad(dh3_ptr, a2_ptr, gw3_ptr, dh2_ptr, a1_ptr, gw2_ptr, dh1_ptr, xb_ptr, gw1_ptr,
            B, N_W3_TILES: tl.constexpr, KS: tl.constexpr,
            TM: tl.constexpr, TN: tl.constexpr, BK: tl.constexpr):
    """G[W] += dh^T @ a over the batch rows [s*KS, (s+1)*KS). Tiles TM x TN.
    tile ids: [0, N_W3_TILES) -> W3, next 16 -> W2 (256x256), next 4 -> W1 (256x64)."""
    t = tl.program_id(0)
    s = tl.program_id(1)
    n_big = 256 // TN  # tiles per row of a 256x256 matrix
    if t < N_W3_TILES:
        dh_p = dh3_ptr
        a_p = a2_ptr
        g_p = gw3_ptr
        tt = t
        NA = 256
    elif t < N_W3_TILES + (256 // TM) * n_big:
        dh_p = dh2_ptr
        a_p = a1_ptr
        g_p = gw2_ptr
        tt = t - N_W3_TILES
        NA = 256
    else:
        dh_p = dh1_ptr
        a_p = xb_ptr
        g_p = gw1_ptr
        tt = t - N_W3_TILES - (256 // TM) * n_big
        NA = 64
    n_tiles_n = NA // TN
    m0 = (tt // n_tiles_n) * TM
    n0 = (tt % n_tiles_n) * TN
    rm = m0 + tl.arange(0, TM)
    rn = n0 + tl.arange(0, TN)
    acc = tl.zeros((TM, TN), tl.float32)
    for k in range(s * KS, (s + 1) * KS, BK):
        rk = k + tl.arange(0, BK)
        d = tl.load(dh_p + rk[:, None] * 256 + rm[None, :])      # [BK, TM]
        a = tl.load(a_p + rk[:, None] * NA + rn[None, :])         # [BK, TN]
        acc = tl.dot(tl.trans(d), a, acc)
    tl.atomic_add(g_p + rm[:, None] * NA + rn[None, :], acc)


@triton.jit
def k_norm(g_ptr, mask_ptr, ctrl_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    g = tl.load(g_ptr + o, mask=m, other=0.0) * tl.load(mask_ptr + o, mask=m, other=0.0)
    tl.atomic_add(ctrl_ptr, tl.sum(g * g, axis=0))


@triton.jit
def k_adam(p_ptr, pb_ptr, g_ptr, m_ptr, v_ptr, mask_ptr, ctrl_ptr, lr_ptr, itmax_ptr, n,
           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(itmax_ptr, 0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    msk = o < n
    norm = tl.sqrt(tl.load(ctrl_ptr))
    t = tl.load(ctrl_ptr + 1)
    lr = tl.load(lr_ptr)
    coef = tl.minimum(1.0 / (norm + 1e-6), 1.0)
    g = tl.load(g_ptr + o, mask=msk, other=0.0) * tl.load(mask_ptr + o, mask=msk, other=0.0) * coef
    m = tl.load(m_ptr + o, mask=msk, other=0.0)
    v = tl.load(v_ptr + o, mask=msk, other=0.0)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    bc1 = 1.0 - tl.exp(t * -0.10536051565782628)    # 1 - 0.9 ** t
    bc2 = 1.0 - tl.exp(t * -0.0010005003335835335)  # 1 - 0.999 ** t
    denom = tl.sqrt(v) / tl.sqrt(bc2) + 1e-8
    p = tl.load(p_ptr + o, mask=msk, other=0.0) - (lr / bc1) * m / denom
    tl.store(p_ptr + o, p, mask=msk)
    tl.store(pb_ptr + o, p.to(tl.bfloat16), mask=msk)
    tl.store(m_ptr + o, m, mask=msk)
    tl.store(v_ptr + o, v, mask=msk)
    tl.store(g_ptr + o, tl.zeros_like(g), mask=msk)


class TritonStep:
    """Static buffers + kernel launches for one optimizer step on batch B.

    State tensors (all on `device`): P fp32 master, Pb bf16 copy, G fp32 grad
    accumulator (kept zeroed between steps), m, v Adam moments, ctrl = [norm2, t],
    lr, itmax (int32), loss_acc.
    """

    def __init__(self, layout: gf.Layout, batch: int, device, BM=64, BK=64, NW=8,
                 WTM=64, WTN=64, WBK=64, WSPLIT=4, NW_W=4, STAGES=2, PREC='tf32',
                 EBLOCK=1024, W_STAGES=3):
        self.L = layout
        self.B = batch
        self.dev = device
        self.cfg = dict(BM=BM, BK=BK, NW=NW, WTM=WTM, WTN=WTN, WBK=WBK, WSPLIT=WSPLIT, NW_W=NW_W,
                        STAGES=STAGES, PREC=PREC, EBLOCK=EBLOCK, W_STAGES=W_STAGES)
        assert batch % BM == 0 and batch % (WSPLIT * WBK) == 0
        n = layout.size
        f32 = dict(device=device, dtype=torch.float32)
        bf = dict(device=device, dtype=torch.bfloat16)
        self.P = torch.zeros(n, **f32)
        self.Pb = torch.zeros(n, **bf)
        self.G = torch.zeros(n, **f32)
        self.m = torch.zeros(n, **f32)
        self.v = torch.zeros(n, **f32)
        self.mask = torch.as_tensor(layout.mask_np, **f32)
        self.ctrl = torch.zeros(2, **f32)
        self.lr = torch.zeros((), **f32)
        self.itmax = torch.zeros(1, device=device, dtype=torch.int32)
        self.loss_acc = torch.zeros((), **f32)
        B = batch
        self.xb = torch.empty(B, 64, **bf)
        self.aux = torch.empty(B, 48, **f32)
        self.h1 = torch.empty(B, 256, **bf); self.st1 = torch.empty(B, 2, **f32)
        self.a1 = torch.empty(B, 256, **bf)
        self.h2 = torch.empty(B, 256, **bf); self.st2 = torch.empty(B, 2, **f32)
        self.a2 = torch.empty(B, 256, **bf)
        self.dh3 = torch.empty(B, 256, **bf) if layout.kind == 'adv' else None
        self.dh2 = torch.empty(B, 256, **bf)
        self.dh1 = torch.empty(B, 256, **bf)
        self.Pv = layout.views(self.P)
        self.Pbv = layout.views(self.Pb)
        self.Gv = layout.views(self.G)

    def reset(self, flat_init: torch.Tensor, lr: float):
        self.P.copy_(flat_init)
        self.Pb.copy_(flat_init)
        self.G.zero_(); self.m.zero_(); self.v.zero_(); self.ctrl.zero_(); self.itmax.zero_()
        self.loss_acc.zero_()
        self.lr.fill_(lr)

    def step(self, data: torch.Tensor, idx: torch.Tensor):
        for _, fn in self.launches(data, idx):
            fn()

    def fwd_bwd(self, data: torch.Tensor, idx: torch.Tensor):
        """Accumulate the minibatch gradient into G (must be zero) and loss into loss_acc."""
        for name, fn in self.launches(data, idx):
            if name not in ('norm', 'adam'):
                fn()

    def update(self):
        for name, fn in self.launches(None, None):
            if name in ('norm', 'adam'):
                fn()

    def launches(self, data, idx):
        """[(name, zero-arg launch closure)] for one optimizer step."""
        c = self.cfg
        B, BM, BK, NW, ST = self.B, c['BM'], c['BK'], c['NW'], c['STAGES']
        P, Pb, G = self.Pv, self.Pbv, self.Gv
        strat = self.L.kind == 'strat'
        grid = (B // BM,)
        out = []
        out.append(('fwd1', lambda: k_fwd1[grid](
            data, idx, Pb['W1'], P['b1'], P['g1'], P['be1'],
            self.xb, self.aux, self.h1, self.st1, self.a1, self.itmax, self.ctrl,
            STRAT=strat, BM=BM, num_warps=NW, num_stages=ST)))
        out.append(('fwd2', lambda: k_fwd2[grid](
            self.a1, Pb['W2'], P['b2'], P['g2'], P['be2'], self.h2, self.st2, self.a2,
            BM=BM, BK=BK, num_warps=NW, num_stages=ST)))
        if strat:
            out.append(('head', lambda: k_head_strat[grid](
                self.a2, P['Wo'], P['bo'], self.aux, self.itmax,
                self.h2, self.st2, P['g2'], P['be2'], self.dh2,
                G['b2'], G['g2'], G['be2'], G['Wo'], G['bo'], self.loss_acc,
                1.0 / B, BM=BM, PREC=c['PREC'], num_warps=NW, num_stages=ST)))
        else:
            out.append(('head', lambda: k_head_adv[grid](
                self.a2, Pb['W3'], P['b3'], P['Wo'], P['bo'], self.aux, self.itmax,
                self.dh3, G['b3'], G['Wo'], G['bo'], self.loss_acc, 1.0 / (5 * B),
                BM=BM, BK=BK, PREC=c['PREC'], num_warps=NW, num_stages=ST)))
            out.append(('bwd2', lambda: k_bwd_ln[grid](
                self.dh3, Pb['W3'], self.h2, self.st2, P['g2'], P['be2'], self.dh2,
                G['b2'], G['g2'], G['be2'], BM=BM, BK=BK, num_warps=NW, num_stages=ST)))
        out.append(('bwd1', lambda: k_bwd_ln[grid](
            self.dh2, Pb['W2'], self.h1, self.st1, P['g1'], P['be1'], self.dh1,
            G['b1'], G['g1'], G['be1'], BM=BM, BK=BK, num_warps=NW, num_stages=ST)))
        TM, TN = c['WTM'], c['WTN']
        n_w3 = 0 if strat else (256 // TM) * (256 // TN)
        n_tiles = n_w3 + (256 // TM) * (256 // TN) + (256 // TM) * (64 // TN)
        out.append(('wgrad', lambda: k_wgrad[(n_tiles, c['WSPLIT'])](
            self.dh2 if strat else self.dh3, self.a2, G['W2'] if strat else G['W3'],
            self.dh2, self.a1, G['W2'], self.dh1, self.xb, G['W1'],
            B, N_W3_TILES=n_w3, KS=B // c['WSPLIT'], TM=TM, TN=TN, BK=c['WBK'],
            num_warps=c['NW_W'], num_stages=c['W_STAGES'])))
        n = self.L.size
        BLOCK = c['EBLOCK']
        out.append(('norm', lambda: k_norm[(triton.cdiv(n, BLOCK),)](
            self.G, self.mask, self.ctrl, n, BLOCK=BLOCK, num_warps=4)))
        out.append(('adam', lambda: k_adam[(triton.cdiv(n, BLOCK),)](
            self.P, self.Pb, self.G, self.m, self.v, self.mask,
            self.ctrl, self.lr, self.itmax, n, BLOCK=BLOCK, num_warps=4)))
        return out
