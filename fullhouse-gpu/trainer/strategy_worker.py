"""Average-strategy-net fitting on a spare GPU, overlapped with later iterations.

Faithfulness argument: upstream re-initialises the strategy net every iteration
and the net feeds neither the traversal nor the deployed (advantage-net) bot, so
the only strategy nets that are ever observable are the ones exported at
checkpoint iterations. Training it only there, on an exact snapshot of the
strategy reservoir taken at that iteration and with that iteration's learning
rate, yields the same distribution of exported nets.

The snapshot is a device-to-device copy into a second ``GPUReservoirSet`` on the
spare GPU (the fitter's CUDA graph captures that tensor's address), taken
synchronously on the main thread before the next iteration can modify the live
reservoir. The fit itself runs in a background thread: it only enqueues CUDA-graph
replays, so it barely competes with the main thread for the GIL.
"""
from __future__ import annotations

import threading
import time
import traceback

import paths  # noqa: F401

import torch

import gpu_fit as gf
from fit_backend import make_fitter


class StrategyWorker:
    def __init__(self, device: str, capacity: int, batch: int, seed: int, backend: str,
                 chunk: int = 250):
        self.dev = torch.device(device)
        self.capacity = capacity
        self.batch = batch
        self.chunk = chunk
        with torch.cuda.device(self.dev):
            self.snap = gf.GPUReservoirSet(1, capacity, self.dev, seed=seed)
            self.fitter = make_fitter("strat", self.snap, batch, seed, backend, chunk)
        self._thread: threading.Thread | None = None
        self.results: dict[int, dict] = {}
        self.errors: list[str] = []
        self.busy_seconds = 0.0

    def warmup(self) -> float:
        """Build kernels + capture the CUDA graph on dummy rows (main thread, before the loop)."""
        t0 = time.monotonic()
        n = min(self.capacity, 2 * self.batch)
        d = self.snap.data
        with torch.cuda.device(self.dev):
            d[:n].zero_()
            d[:n, 46:51] = 1.0           # all actions legal
            d[:n, gf.COL_Y:gf.COL_Y + 5] = 0.2
            d[:n, gf.COL_IT] = 1.0
            d[:n, gf.COL_W] = 1.0
            self.snap.sizes[0] = n
            self.fitter.fit(1e-3, self.chunk)
            d[:n].zero_()
            self.snap.sizes[0] = 0
            torch.cuda.synchronize(self.dev)
        return time.monotonic() - t0

    def submit(self, iteration: int, src: "gf.GPUReservoirSet", lr: float, steps: int,
               on_done) -> float:
        """Snapshot the live 1-buffer strategy reservoir `src` and start fitting in the
        background. Returns seconds the main thread spent (wait for the previous job +
        snapshot copy)."""
        t0 = time.monotonic()
        self.wait()
        size = int(src.sizes[0])
        self.snap.data[:size].copy_(src.data[:size])
        self.snap.sizes[0] = size
        self.snap.total_seen[0] = src.total_seen[0]
        torch.cuda.synchronize(src.device)
        torch.cuda.synchronize(self.dev)
        self._thread = threading.Thread(target=self._run, name=f"strat-fit-{iteration}",
                                        args=(iteration, size, lr, steps, on_done), daemon=True)
        self._thread.start()
        return time.monotonic() - t0

    def wait(self) -> float:
        t0 = time.monotonic()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return time.monotonic() - t0

    def _run(self, iteration: int, size: int, lr: float, steps: int, on_done) -> None:
        t0 = time.monotonic()
        try:
            with torch.cuda.device(self.dev):
                res = self.fitter.fit(lr, steps)
            info = dict(iteration=iteration, loss=res.avg_loss, seconds=res.seconds,
                        size=size, lr=lr, steps=steps)
            self.results[iteration] = info
            on_done(iteration, res.state_dict, info)
        except Exception:
            self.errors.append(traceback.format_exc())
        self.busy_seconds += time.monotonic() - t0
