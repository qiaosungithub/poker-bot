"""Thin adapter over the fitting module ``../fit/gpu_fit.py``.

The trainer only talks to this file, so following the fit package's API is a local
change here.

fit package public API used (gpu_fit.py, release v2):
    gf.GPUReservoirSet(n_buffers, capacity, device, seed)
    gf.make_fitter(kind, buffers, batch=, chunk=, backend=, seed=) -> GraphFitter
    GraphFitter.fit(lr, steps) -> FitResult(state_dict, avg_loss, seconds, flat)
        (reads the current seat sizes from buffers.sizes; fresh torch-default init and
         fresh Adam state on every call, like upstream's new net + optimizer per iteration)

Contract of the object returned by :func:`make_fitter` (what the trainer relies on):
    fitter.fit(lr: float, steps: int) -> FitResult with .state_dict (CPU fp32 tensors,
        torch module naming), .avg_loss (mean minibatch loss over all steps), .seconds
    fitter.gen  -- the fitter's CUDA generator (for checkpoint metadata)
"""
from __future__ import annotations

import paths  # noqa: F401  (sys.path / caches)

import gpu_fit as gf

BACKENDS = getattr(gf, "BACKENDS", ("cuda",))


def make_fitter(kind: str, buffers, batch: int, seed: int, backend: str = "cuda",
                chunk: int = 250):
    """kind: 'adv' (6 seat buffers) or 'strat' (1 buffer). `buffers` is a
    gf.GPUReservoirSet whose `.data` tensor must never be reallocated (the CUDA
    graph captures its address)."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown fit backend {backend!r}; the fit package offers {BACKENDS}")
    return gf.make_fitter(kind, buffers, batch=batch, chunk=chunk, backend=backend, seed=seed)


def describe(backend: str) -> str:
    return f"fit gpu_fit.GraphFitter(backend={backend!r}, CUDA graph of 250-step chunks)"
