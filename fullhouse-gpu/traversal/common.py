"""Shared helpers: locate the (read-only) upstream source, load LUTs and nets.

The upstream tree is only imported, never edited. Its location is
``$FULLHOUSE_SOURCE``, default ``<repo>/fullhouse-repro/source`` (upstream
fullhouse-bot @ e504793d + ``fullhouse-repro/training_changes.patch``).
Numba's on-disk cache is redirected to ``$FULLHOUSE_GPU_RUNTIME/numba`` (default
``<repo>/fullhouse-gpu/.runtime/numba``) so importing upstream ``cache=True``
functions does not write into the upstream ``__pycache__``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # fullhouse-gpu/traversal
PACKAGE = HERE.parent                           # fullhouse-gpu
RUNTIME = Path(os.environ.get("FULLHOUSE_GPU_RUNTIME", str(PACKAGE / ".runtime")))
(RUNTIME / "numba").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME / "numba"))

SOURCE = Path(
    os.environ.get(
        "FULLHOUSE_SOURCE",
        str(PACKAGE.parent / "fullhouse-repro" / "source"),
    )
)
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import numpy as np  # noqa: E402

N_PLAYERS = 6
N_ACTIONS = 5
FEATURE_DIM = 51

# Order of the 16 dueling-net arrays used by upstream ``worker_run_jit``
# (weights_list/biases_list interleaved) and by the advantage NPZ export.
PARAM_KEYS = (
    "trunk_w0", "trunk_b0", "trunk_ln0_g", "trunk_ln0_b",
    "trunk_w1", "trunk_b1", "trunk_ln1_g", "trunk_ln1_b",
    "val_w0", "val_b0", "val_w1", "val_b1",
    "adv_w0", "adv_b0", "adv_w1", "adv_b1",
)


def load_tables():
    """Return (flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit)."""
    from training._hand_eval_lut import load_or_generate
    from bot.features import load_preflop_equity

    flush_lut, nf_keys, nf_vals = load_or_generate(
        SOURCE / "data" / "hand_eval_lut.npz"
    )
    eq_p, eq_s, eq_o = load_preflop_equity()
    return flush_lut, nf_keys, nf_vals, eq_p, eq_s, eq_o


def params_from_npz(path) -> dict[str, np.ndarray]:
    d = np.load(str(path))
    return {k: np.ascontiguousarray(d[k], dtype=np.float32) for k in PARAM_KEYS}


def params_from_torch(net) -> dict[str, np.ndarray]:
    """Same extraction as upstream ``train._extract_dueling_weights``."""
    from training.train import _extract_dueling_weights

    w, b = _extract_dueling_weights(net)
    out = {}
    for i in range(8):
        out[PARAM_KEYS[2 * i]] = np.ascontiguousarray(w[i], dtype=np.float32)
        out[PARAM_KEYS[2 * i + 1]] = np.ascontiguousarray(b[i], dtype=np.float32)
    return out


def params_random_init(seed: int) -> dict[str, np.ndarray]:
    """Torch default init of DuelingAdvantageNet (what each iteration starts from)."""
    import torch
    from training.network import DuelingAdvantageNet

    torch.manual_seed(seed)
    return params_from_torch(DuelingAdvantageNet(FEATURE_DIM, 256, 128))


def params_tuple(p: dict[str, np.ndarray]) -> tuple:
    """16-tuple in upstream argument order (tw0, tb0, tln0g, ..., aw1, ab1)."""
    return tuple(p[k] for k in PARAM_KEYS)


def worker_lists(p: dict[str, np.ndarray]):
    """(weights_list, biases_list) as built by upstream train for worker_run_jit."""
    weights = [p[PARAM_KEYS[2 * i]] for i in range(8)]
    biases = [p[PARAM_KEYS[2 * i + 1]] for i in range(8)]
    return weights, biases
