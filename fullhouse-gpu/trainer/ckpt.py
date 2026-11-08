"""Checkpoint export in exactly the upstream formats, written off the main thread.

* advantage net -> ``<run>_adv/iter_XXXXX.npz`` via upstream ``train._export_advantage_npz``
* strategy net  -> ``<run>/iter_XXXXX.npz``     via upstream ``train._export_strategy_npz``
* small torch file ``<run>/iter_XXXXX.pt``: both state dicts + config + seeds/RNG states

The export functions take torch modules, so the writer owns one template
``DuelingAdvantageNet`` / ``AverageStrategyNet`` built on the main thread at start
(``load_state_dict`` in the writer thread does not touch any RNG).
Every file is written to a temporary name and renamed, so a reader never sees a
torn file. Writer errors are recorded and reported, never raised into the loop.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from pathlib import Path

import paths  # noqa: F401

import numpy as np
import torch

from training.network import AverageStrategyNet, DuelingAdvantageNet
from training.train import _export_advantage_npz, _export_strategy_npz

FEATURE_DIM, TRUNK_DIM, HEAD_DIM, N_ACTIONS = 51, 256, 128, 5

# state_dict key -> upstream advantage-NPZ / traversal parameter name
ADV_SD_TO_NPZ = {
    "trunk.0.weight": "trunk_w0", "trunk.0.bias": "trunk_b0",
    "trunk.1.weight": "trunk_ln0_g", "trunk.1.bias": "trunk_ln0_b",
    "trunk.3.weight": "trunk_w1", "trunk.3.bias": "trunk_b1",
    "trunk.4.weight": "trunk_ln1_g", "trunk.4.bias": "trunk_ln1_b",
    "value_head.0.weight": "val_w0", "value_head.0.bias": "val_b0",
    "value_head.2.weight": "val_w1", "value_head.2.bias": "val_b1",
    "advantage_head.0.weight": "adv_w0", "advantage_head.0.bias": "adv_b0",
    "advantage_head.2.weight": "adv_w1", "advantage_head.2.bias": "adv_b1",
}


def params_from_state_dict(sd: dict) -> dict[str, np.ndarray]:
    """Advantage-net state_dict -> the 16 named arrays used by the traversal
    (same values/order as upstream ``_extract_dueling_weights``)."""
    return {name: np.ascontiguousarray(sd[k].detach().cpu().numpy(), dtype=np.float32)
            for k, name in ADV_SD_TO_NPZ.items()}


def _atomic(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


class CheckpointWriter:
    """Single background thread that performs checkpoint writes in FIFO order."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.adv_dir = Path(str(self.run_dir) + "_adv")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adv_dir.mkdir(parents=True, exist_ok=True)
        # templates built on the caller's thread; RNG state is restored so building
        # them does not shift the global torch stream.
        state = torch.get_rng_state()
        self._adv_net = DuelingAdvantageNet(FEATURE_DIM, TRUNK_DIM, HEAD_DIM)
        self._strat_net = AverageStrategyNet(FEATURE_DIM, TRUNK_DIM, N_ACTIONS)
        torch.set_rng_state(state)
        self._q: queue.Queue = queue.Queue()
        self.errors: list[str] = []
        self.written: list[str] = []
        self.busy_seconds = 0.0
        self._thread = threading.Thread(target=self._loop, name="ckpt-writer", daemon=True)
        self._thread.start()

    # -- public API (main thread) --------------------------------------------
    def write_advantage(self, iteration: int, adv_sd: dict) -> None:
        self._q.put(("adv", iteration, adv_sd))

    def write_strategy_and_pt(self, iteration: int, strat_sd: dict | None, adv_sd: dict,
                              meta: dict) -> None:
        self._q.put(("pt", iteration, (strat_sd, adv_sd, meta)))

    def close(self) -> float:
        """Drain the queue; returns seconds spent waiting."""
        t0 = time.monotonic()
        self._q.put(None)
        self._thread.join()
        return time.monotonic() - t0

    # -- worker -----------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            t0 = time.monotonic()
            try:
                self._do(*job)
            except Exception:  # never kill training because of a write
                self.errors.append(traceback.format_exc())
            self.busy_seconds += time.monotonic() - t0

    def _do(self, kind: str, iteration: int, payload) -> None:
        name = f"iter_{iteration:05d}"
        if kind == "adv":
            self._adv_net.load_state_dict(payload)
            path = self.adv_dir / f"{name}.npz"
            _atomic(path, lambda p: _export_advantage_npz(self._adv_net, p))
            self.written.append(str(path))
            return
        strat_sd, adv_sd, meta = payload
        if strat_sd is not None:
            self._strat_net.load_state_dict(strat_sd)
            path = self.run_dir / f"{name}.npz"
            _atomic(path, lambda p: _export_strategy_npz(self._strat_net, p))
            self.written.append(str(path))
        data = dict(meta)
        data["net"] = adv_sd
        data["strategy_net"] = strat_sd
        path = self.run_dir / f"{name}.pt"
        _atomic(path, lambda p: torch.save(data, p))
        self.written.append(str(path))
