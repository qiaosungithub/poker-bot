"""cuBLAS GEMMs + high-occupancy Triton row kernels, wgrad GEMMs on a side stream.

Per advantage step (critical path):
  prep(gather, targets) -> addmm L1 -> LN1 -> addmm L2 -> LN2 -> addmm L3 -> head(+bwd of
  output layer) -> mm da2 -> LN2 bwd -> mm da1 -> LN1 bwd -> mm dW1 -> norm -> adam
with dW3 / dW2 GEMMs (fp32 output written straight into G) overlapped on a side stream.
Strategy step: prep -> L1 -> LN1 -> L2 -> LN2 -> head_strat(+LN2 bwd) -> mm da1 -> LN1 bwd.

Numerics = upstream bf16 autocast: GEMM operands bf16, linear outputs (incl. bias)
rounded to bf16, LayerNorm / losses / output layer in fp32, weight grads fp32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import gpu_fit as gf
from triton_fit import k_adam


@triton.jit
def k_norm_rep(g_ptr, rep_ptr, mask_ptr, ctrl_ptr, n, small_off, R: tl.constexpr,
               NS: tl.constexpr, BLOCK: tl.constexpr):
    """G[small] += sum of the R gradient replicas (replicas := 0); ctrl[0] += ||G*mask||^2."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    g = tl.load(g_ptr + o, mask=m, other=0.0)
    if (pid + 1) * BLOCK > small_off:
        sm = m & (o >= small_off)
        so = o - small_off
        for r in tl.static_range(R):
            g += tl.load(rep_ptr + r * NS + so, mask=sm, other=0.0)
            tl.store(rep_ptr + r * NS + so, tl.zeros_like(g), mask=sm)
        tl.store(g_ptr + o, g, mask=sm)
    g = g * tl.load(mask_ptr + o, mask=m, other=0.0)
    tl.atomic_add(ctrl_ptr, tl.sum(g * g, axis=0))


@triton.jit
def k_prep(data_ptr, idx_ptr, itmax_ptr, xb_ptr, aux_ptr, ctrl_ptr,
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
    tl.store(xb_ptr + rows[:, None] * 64 + c64[None, :], x.to(tl.bfloat16))
    j = tl.arange(0, 16)
    jm = j[None, :] < 5
    mk = tl.load(base + 46 + j[None, :], mask=jm, other=0.0)
    yv = tl.load(base + 51 + j[None, :], mask=jm, other=0.0)
    it = tl.load(data_ptr + ids * 64 + 56)
    itmax = tl.maximum(tl.load(itmax_ptr), 1.0)
    q = it / itmax
    w = q * tl.sqrt(q)
    if STRAT:
        w = w * tl.load(data_ptr + ids * 64 + 57)
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
    tl.store(aux_ptr + rows * 48 + 32, w)


@triton.jit
def k_ln_fwd(h_ptr, g_ptr, be_ptr, st_ptr, a_ptr, BM: tl.constexpr):
    """a = lrelu(LN(h)) with h bf16 [B,256] (bias already included)."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    hf = tl.load(h_ptr + rows[:, None] * 256 + rn[None, :]).to(tl.float32)
    mu = tl.sum(hf, axis=1) / 256
    xc = hf - mu[:, None]
    var = tl.sum(xc * xc, axis=1) / 256
    r = 1.0 / tl.sqrt(var + 1e-5)
    z = xc * r[:, None] * tl.load(g_ptr + rn)[None, :] + tl.load(be_ptr + rn)[None, :]
    a = tl.where(z >= 0, z, z * 0.01)
    tl.store(a_ptr + rows[:, None] * 256 + rn[None, :], a.to(tl.bfloat16))
    tl.store(st_ptr + rows * 2, mu)
    tl.store(st_ptr + rows * 2 + 1, r)


@triton.jit
def _ln_bwd_rows(da, rows, h_ptr, st_ptr, g_ptr, be_ptr, dh_ptr, gb_ptr, gg_ptr, gbe_ptr):
    """gb/gg/gbe point into this CTA's gradient replica."""
    rn = tl.arange(0, 256)
    hf = tl.load(h_ptr + rows[:, None] * 256 + rn[None, :]).to(tl.float32)
    mu = tl.load(st_ptr + rows * 2)
    r = tl.load(st_ptr + rows * 2 + 1)
    g = tl.load(g_ptr + rn)
    xh = (hf - mu[:, None]) * r[:, None]
    z = xh * g[None, :] + tl.load(be_ptr + rn)[None, :]
    dz = tl.where(z >= 0, da, da * 0.01)
    tl.atomic_add(gg_ptr + rn, tl.sum(dz * xh, axis=0))
    tl.atomic_add(gbe_ptr + rn, tl.sum(dz, axis=0))
    dxh = dz * g[None, :]
    m1 = tl.sum(dxh, axis=1) / 256
    m2 = tl.sum(dxh * xh, axis=1) / 256
    dh = r[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
    tl.atomic_add(gb_ptr + rn, tl.sum(dh, axis=0))
    tl.store(dh_ptr + rows[:, None] * 256 + rn[None, :], dh.to(tl.bfloat16))


@triton.jit
def k_ln_bwd(da_ptr, h_ptr, st_ptr, g_ptr, be_ptr, dh_ptr, gb_ptr, gg_ptr, gbe_ptr,
             BM: tl.constexpr, R: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    rep = (pid % R) * NS
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    da = tl.load(da_ptr + rows[:, None] * 256 + rn[None, :]).to(tl.float32)
    _ln_bwd_rows(da, rows, h_ptr, st_ptr, g_ptr, be_ptr, dh_ptr, gb_ptr + rep, gg_ptr + rep,
                 gbe_ptr + rep)


@triton.jit
def k_head_adv(h3_ptr, wo_ptr, bo_ptr, aux_ptr, dh3_ptr, gb3_ptr, gwo_ptr, gbo_ptr, loss_ptr,
               inv_5b, BM: tl.constexpr, R: tl.constexpr, NS: tl.constexpr):
    """Output layer + dueling combine + weighted MSE + its backward -> dh3."""
    pid = tl.program_id(0)
    rep = (pid % R) * NS
    gb3_ptr += rep
    gwo_ptr += rep
    gbo_ptr += rep
    loss_ptr += pid % R
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    j = tl.arange(0, 16)
    jm = j < 5
    h3 = tl.load(h3_ptr + rows[:, None] * 256 + rn[None, :]).to(tl.float32)
    a3 = tl.where(h3 >= 0, h3, h3 * 0.01)
    wv = tl.load(wo_ptr + rn)
    v = tl.sum(a3 * wv[None, :], axis=1) + tl.load(bo_ptr)
    wa_t = tl.load(wo_ptr + (1 + j[None, :]) * 256 + rn[:, None], mask=jm[None, :], other=0.0)
    oa = tl.dot(a3.to(tl.bfloat16), wa_t.to(tl.bfloat16))
    mk = tl.load(aux_ptr + rows[:, None] * 48 + j[None, :])
    yn = tl.load(aux_ptr + rows[:, None] * 48 + 16 + j[None, :])
    w = tl.load(aux_ptr + rows * 48 + 32)
    boa = tl.load(bo_ptr + 1 + j, mask=jm, other=0.0)
    A = (oa + boa[None, :]) * mk
    n = tl.maximum(tl.sum(mk, axis=1), 1.0)
    amean = tl.sum(A, axis=1) / n
    pred = (v[:, None] + A - amean[:, None] * mk) * mk
    diff = pred - yn
    tl.atomic_add(loss_ptr, tl.sum(tl.sum(diff * diff, axis=1) * w, axis=0) * inv_5b)
    g = diff * (2.0 * inv_5b) * w[:, None] * mk
    dv = tl.sum(g, axis=1)
    dA = (g - (tl.sum(g * mk, axis=1) / n)[:, None]) * mk
    wa = tl.load(wo_ptr + (1 + j[:, None]) * 256 + rn[None, :], mask=jm[:, None], other=0.0)
    dAb = dA.to(tl.bfloat16)
    dtop = tl.dot(dAb, wa.to(tl.bfloat16)) + dv[:, None] * wv[None, :]
    gwa = tl.dot(tl.trans(dAb), a3.to(tl.bfloat16))
    tl.atomic_add(gwo_ptr + (1 + j[:, None]) * 256 + rn[None, :], gwa, mask=jm[:, None])
    tl.atomic_add(gwo_ptr + rn, tl.sum(dv[:, None] * a3, axis=0))
    tl.atomic_add(gbo_ptr, tl.sum(dv, axis=0))
    tl.atomic_add(gbo_ptr + 1 + j, tl.sum(dA, axis=0), mask=jm)
    dh3 = tl.where(h3 >= 0, dtop, dtop * 0.01)
    tl.atomic_add(gb3_ptr + rn, tl.sum(dh3, axis=0))
    tl.store(dh3_ptr + rows[:, None] * 256 + rn[None, :], dh3.to(tl.bfloat16))


@triton.jit
def k_head_strat(a2_ptr, wo_ptr, bo_ptr, aux_ptr, h2_ptr, st2_ptr, g2_ptr, be2_ptr, dh2_ptr,
                 gb2_ptr, gg2_ptr, gbe2_ptr, gwo_ptr, gbo_ptr, loss_ptr, inv_b,
                 BM: tl.constexpr, R: tl.constexpr, NS: tl.constexpr):
    """policy head (fp32 accumulate, bf16 operands) + masked CE + backward + LN2 bwd."""
    pid = tl.program_id(0)
    rep = (pid % R) * NS
    gb2_ptr += rep
    gg2_ptr += rep
    gbe2_ptr += rep
    gwo_ptr += rep
    gbo_ptr += rep
    loss_ptr += pid % R
    rows = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, 256)
    j = tl.arange(0, 16)
    jm = j < 5
    a2b = tl.load(a2_ptr + rows[:, None] * 256 + rn[None, :])
    wo_t = tl.load(wo_ptr + j[None, :] * 256 + rn[:, None], mask=jm[None, :], other=0.0)
    logits = tl.dot(a2b, wo_t.to(tl.bfloat16)) + tl.load(bo_ptr + j, mask=jm, other=0.0)[None, :]
    mk = tl.load(aux_ptr + rows[:, None] * 48 + j[None, :])
    yn = tl.load(aux_ptr + rows[:, None] * 48 + 16 + j[None, :])
    w = tl.load(aux_ptr + rows * 48 + 32)
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
    dlb = dlog.to(tl.bfloat16)
    da2 = tl.dot(dlb, wo.to(tl.bfloat16))
    gwo = tl.dot(tl.trans(dlb), a2b)
    tl.atomic_add(gwo_ptr + j[:, None] * 256 + rn[None, :], gwo, mask=jm[:, None])
    tl.atomic_add(gbo_ptr + j, tl.sum(dlog, axis=0), mask=jm)
    _ln_bwd_rows(da2, rows, h2_ptr, st2_ptr, g2_ptr, be2_ptr, dh2_ptr, gb2_ptr, gg2_ptr, gbe2_ptr)


class CublasStep:
    """Same state layout / interface as triton_fit2.RowFusedStep."""

    def __init__(self, layout: gf.Layout, batch: int, device, BM=32, NW=4, BM_LN=32,
                 NW_LN=4, BM_B=64, NW_B=4, R=16, EBLOCK=1024, side_stream=True):
        self.L = layout
        self.B = batch
        self.dev = device
        self.cfg = dict(BM=BM, NW=NW, BM_LN=BM_LN, NW_LN=NW_LN, BM_B=BM_B, NW_B=NW_B, R=R,
                        EBLOCK=EBLOCK)
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
        self.itmax_dummy = torch.zeros(1, device=device, dtype=torch.int32)
        self.loss_rep = torch.zeros(R, **f32)
        self.Grep = torch.zeros(R, layout.n_small, **f32)
        so = layout.small_off
        self.Gs = {k: self.Grep[0, o - so: o - so + layout.numel[k]].view(layout.shapes[k])
                   for k, o in layout.offsets.items() if o >= so}
        B = batch
        self.xb = torch.empty(B, 64, **bf)
        self.aux = torch.empty(B, 48, **f32)
        self.h1 = torch.empty(B, 256, **bf); self.st1 = torch.empty(B, 2, **f32)
        self.a1 = torch.empty(B, 256, **bf)
        self.h2 = torch.empty(B, 256, **bf); self.st2 = torch.empty(B, 2, **f32)
        self.a2 = torch.empty(B, 256, **bf)
        self.h3 = torch.empty(B, 256, **bf)
        self.da = torch.empty(B, 256, **bf)
        self.dh3 = torch.empty(B, 256, **bf)
        self.dh2 = torch.empty(B, 256, **bf)
        self.dh1 = torch.empty(B, 256, **bf)
        self.Pv = layout.views(self.P)
        self.Pbv = layout.views(self.Pb)
        self.Gv = layout.views(self.G)
        self.side = torch.cuda.Stream(device=device) if side_stream else None

    def reset(self, flat_init: torch.Tensor, lr: float):
        self.P.copy_(flat_init)
        self.Pb.copy_(flat_init)
        self.G.zero_(); self.m.zero_(); self.v.zero_(); self.ctrl.zero_()
        self.loss_rep.zero_(); self.Grep.zero_()
        self.lr.fill_(lr)

    @property
    def loss_acc(self):
        return self.loss_rep.sum()

    def full_grad(self) -> torch.Tensor:
        """Full fp32 gradient after fwd_bwd (small params = sum of the R replicas)."""
        g = self.G.clone()
        g[self.L.small_off:self.L.small_off + self.L.n_small] += self.Grep.sum(0)
        return g

    def _fork(self, fn):
        if self.side is None:
            fn()
            return
        cur = torch.cuda.current_stream(self.dev)
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            fn()

    def _join(self):
        if self.side is not None:
            torch.cuda.current_stream(self.dev).wait_stream(self.side)

    def fwd_bwd(self, data, idx, itmax):
        c = self.cfg
        B, BM, BL, BB = self.B, c['BM'], c['BM_LN'], c['BM_B']
        P, Pb, G, Gs = self.Pv, self.Pbv, self.Gv, self.Gs
        rk = dict(R=c['R'], NS=self.L.n_small, num_warps=c['NW_B'])
        strat = self.L.kind == 'strat'
        k_prep[(B // BM,)](data, idx, itmax, self.xb, self.aux, self.ctrl,
                           STRAT=strat, BM=BM, num_warps=c['NW'])
        torch.addmm(Pb['b1'], self.xb, Pb['W1'].t(), out=self.h1)
        k_ln_fwd[(B // BL,)](self.h1, P['g1'], P['be1'], self.st1, self.a1, BM=BL,
                             num_warps=c['NW_LN'])
        torch.addmm(Pb['b2'], self.a1, Pb['W2'].t(), out=self.h2)
        k_ln_fwd[(B // BL,)](self.h2, P['g2'], P['be2'], self.st2, self.a2, BM=BL,
                             num_warps=c['NW_LN'])
        if strat:
            k_head_strat[(B // BB,)](self.a2, P['Wo'], P['bo'], self.aux, self.h2, self.st2,
                                     P['g2'], P['be2'], self.dh2, Gs['b2'], Gs['g2'], Gs['be2'],
                                     Gs['Wo'], Gs['bo'], self.loss_rep, 1.0 / B, BM=BB, **rk)
        else:
            torch.addmm(Pb['b3'], self.a2, Pb['W3'].t(), out=self.h3)
            k_head_adv[(B // BB,)](self.h3, P['Wo'], P['bo'], self.aux, self.dh3, Gs['b3'],
                                   Gs['Wo'], Gs['bo'], self.loss_rep, 1.0 / (5 * B), BM=BB, **rk)
            self._fork(lambda: torch.mm(self.dh3.t(), self.a2, out_dtype=torch.float32,
                                        out=G['W3']))
            torch.mm(self.dh3, Pb['W3'], out=self.da)
            k_ln_bwd[(B // BB,)](self.da, self.h2, self.st2, P['g2'], P['be2'], self.dh2,
                                 Gs['b2'], Gs['g2'], Gs['be2'], BM=BB, **rk)
        self._fork(lambda: torch.mm(self.dh2.t(), self.a1, out_dtype=torch.float32, out=G['W2']))
        torch.mm(self.dh2, Pb['W2'], out=self.da)
        k_ln_bwd[(B // BB,)](self.da, self.h1, self.st1, P['g1'], P['be1'], self.dh1,
                             Gs['b1'], Gs['g1'], Gs['be1'], BM=BB, **rk)
        torch.mm(self.dh1.t(), self.xb, out_dtype=torch.float32, out=G['W1'])
        self._join()

    def update(self):
        n = self.L.size
        BLOCK = self.cfg['EBLOCK']
        k_norm_rep[(triton.cdiv(n, BLOCK),)](self.G, self.Grep, self.mask, self.ctrl, n,
                                             self.L.small_off, R=self.cfg['R'],
                                             NS=self.L.n_small, BLOCK=BLOCK, num_warps=4)
        k_adam[(triton.cdiv(n, BLOCK),)](self.P, self.Pb, self.G, self.m, self.v, self.mask,
                                         self.ctrl, self.lr, self.itmax_dummy, n, BLOCK=BLOCK,
                                         num_warps=4)

    def step(self, data, idx, itmax):
        self.fwd_bwd(data, idx, itmax)
        self.update()
