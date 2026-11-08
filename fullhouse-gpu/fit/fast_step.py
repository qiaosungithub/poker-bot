"""Fastest step: cuBLAS bf16 GEMMs + hand-written CUDA row kernels (rowk.cu).

Per advantage step (critical path):
  prep -> addmm L1 -> ln_fwd -> addmm L2 -> ln_fwd -> addmm L3 -> head_adv -> mm(dh3,W3)
  -> ln_bwd(L2) -> mm(dh2,W2) -> ln_bwd(L1) -> mm(dh1^T,x) -> norm -> adam
with dW3 = dh3^T a2 and dW2 = dh2^T a1 (fp32 out, written into G) on a side stream.
Strategy step: prep -> L1 -> ln_fwd -> L2 -> ln_fwd -> head_strat(+LN2 bwd) -> mm(dh2,W2)
  -> ln_bwd(L1) -> wgrads -> norm -> adam.
Numerics = upstream bf16 autocast (linear operands + outputs bf16, LN / loss / output layer /
weight-grad accumulation fp32).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import gpu_fit as gf

_HERE = Path(__file__).resolve().parent
_EXT = None


def ext():
    """Build (once, cached under fit_opt/.runtime/torch_ext) and return the CUDA extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        build = _HERE / '.runtime' / 'torch_ext'
        build.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
        _EXT = load(name='fit_rowk', sources=[str(_HERE / 'rowk.cu')],
                    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
                    build_directory=str(build), verbose=False)
    return _EXT


class CudaRowStep:
    """Static buffers + launches for one optimizer step (batch B) of the advantage ('adv')
    or strategy ('strat') net. State: P (fp32 master, flat `gf.Layout`), Pb (bf16 copy used
    by the GEMMs), G (fp32 grad of the big weight matrices, zero between steps), S (per-block
    partial sums of the small-param grads, written with plain stores by the row kernels and
    reduced into G by the norm kernel), lossp (per-block loss partials), m, v (Adam),
    ctrl = [norm2, step], lr, loss_buf (loss sum accumulated over the fit)."""

    def __init__(self, layout: gf.Layout, batch: int, device, side_stream=True,
                 fwd=(256, 2), bwd=(256, 4, 1), head=(128, 8)):
        """fwd = (threads, rows per warp) of ln_fwd; bwd = (threads, rows per warp, rows in
        flight per warp) of ln_bwd; head = (threads, rows per warp) of the head kernels."""
        self.L = layout
        self.B = batch
        self.dev = torch.device(device)
        nblk = lambda t, rpw: max(1, batch // ((t // 32) * rpw))
        self.fwd_cfg = (nblk(*fwd), fwd[0])
        self.bwd_cfg = (nblk(bwd[0], bwd[1]), bwd[0], bwd[2])
        self.head_cfg = (nblk(*head), head[0])
        self.k = ext()
        n = layout.size
        f32 = dict(device=self.dev, dtype=torch.float32)
        bf = dict(device=self.dev, dtype=torch.bfloat16)
        self.P = torch.zeros(n, **f32)
        self.Pb = torch.zeros(n, **bf)
        self.G = torch.zeros(n, **f32)
        self.m = torch.zeros(n, **f32)
        self.v = torch.zeros(n, **f32)
        self.mask = torch.as_tensor(layout.mask_np, **f32)
        self.ctrl = torch.zeros(2, **f32)
        self.lr = torch.zeros(1, **f32)
        # small-parameter gradient partials: row = block id of the producing kernel
        self.NB = 256
        self.S = torch.zeros(self.NB, layout.n_small, **f32)
        self.lossp = torch.zeros(self.NB, **f32)
        self.loss_buf = torch.zeros(1, **f32)
        self._pending = False
        so = layout.small_off
        self.Sv = {k: self.S[0, o - so:] for k, o in layout.offsets.items() if o >= so}
        for cfg in (self.bwd_cfg, self.head_cfg):
            assert cfg[0] <= self.NB, cfg
        B = batch
        self.xb = torch.zeros(B, 64, **bf)
        self.aux = torch.zeros(B, 16, **f32)
        self.wrow = torch.zeros(B, **f32)
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
        self.side = torch.cuda.Stream(device=self.dev) if side_stream else None

    @property
    def loss_acc(self):
        """Sum of minibatch losses since reset (incl. a not-yet-reduced pending step)."""
        return self.loss_buf[0] + (self.lossp.sum() if self._pending else 0.0)

    def full_grad(self) -> torch.Tensor:
        """Gradient of the pending step (between fwd_bwd and update; testing only)."""
        g = self.G.clone()
        g[self.L.small_off:] += self.S.sum(0)
        return g

    def reset(self, flat_init: torch.Tensor, lr: float):
        self.P.copy_(flat_init)
        self.Pb.copy_(flat_init)
        self.G.zero_(); self.m.zero_(); self.v.zero_(); self.ctrl.zero_()
        self.loss_buf.zero_(); self.lossp.zero_()
        self.lr.fill_(lr)

    def _fork(self, fn):
        if self.side is None:
            fn()
            return
        self.side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(self.side):
            fn()

    def _join(self):
        if self.side is not None:
            torch.cuda.current_stream(self.dev).wait_stream(self.side)

    def fwd_bwd(self, data, idx, itmax):
        k, P, Pb, G, Sv = self.k, self.Pv, self.Pbv, self.Gv, self.Sv
        ns = self.L.n_small
        fc, bc, hc = self.fwd_cfg, self.bwd_cfg, self.head_cfg
        B = self.B
        strat = self.L.kind == 'strat'
        k.prep(data, idx, itmax, self.xb, self.aux, self.wrow, self.ctrl, int(strat))
        torch.addmm(Pb['b1'], self.xb, Pb['W1'].t(), out=self.h1)
        k.ln_fwd(self.h1, P['g1'], P['be1'], self.st1, self.a1, *fc)
        torch.addmm(Pb['b2'], self.a1, Pb['W2'].t(), out=self.h2)
        k.ln_fwd(self.h2, P['g2'], P['be2'], self.st2, self.a2, *fc)
        if strat:
            k.head_strat(self.a2, P['Wo'], P['bo'], self.aux, self.wrow, self.h2, self.st2,
                         P['g2'], P['be2'], self.dh2, Sv['b2'], Sv['Wo'], self.lossp, ns,
                         1.0 / B, *hc)
        else:
            torch.addmm(Pb['b3'], self.a2, Pb['W3'].t(), out=self.h3)
            k.head_adv(self.h3, P['Wo'], P['bo'], self.aux, self.wrow, self.dh3, Sv['b3'],
                       self.lossp, ns, 1.0 / (5 * B), *hc)
            self._fork(lambda: torch.mm(self.dh3.t(), self.a2, out_dtype=torch.float32,
                                        out=G['W3']))
            torch.mm(self.dh3, Pb['W3'], out=self.da)
            k.ln_bwd(self.da, self.h2, self.st2, P['g2'], P['be2'], self.dh2, Sv['b2'], ns, *bc)
        self._fork(lambda: torch.mm(self.dh2.t(), self.a1, out_dtype=torch.float32, out=G['W2']))
        torch.mm(self.dh2, Pb['W2'], out=self.da)
        k.ln_bwd(self.da, self.h1, self.st1, P['g1'], P['be1'], self.dh1, Sv['b1'], ns, *bc)
        torch.mm(self.dh1.t(), self.xb, out_dtype=torch.float32, out=G['W1'])
        self._join()
        self._pending = True

    def update(self):
        self.k.norm(self.G, self.S, self.mask, self.ctrl, self.lossp, self.loss_buf,
                    self.L.small_off)
        self.k.adam(self.P, self.Pb, self.G, self.m, self.v, self.mask, self.ctrl, self.lr)
        self._pending = False

    def step(self, data, idx, itmax):
        self.fwd_bwd(data, idx, itmax)
        self.update()
