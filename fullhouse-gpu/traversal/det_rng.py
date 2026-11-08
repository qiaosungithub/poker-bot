"""Counter/hash-based per-node uniforms shared by the batched traversal and the
deterministic test copy of the reference traversal.

Every traversal node is identified by (seed, traversal id, action history).
The history is folded into a 64-bit key with a splitmix64-style mixer:

    root key  = root_key(seed, tid)
    child key = child_key(parent key, action)

and an opponent node draws its sampling uniform as ``uniform32(key)``, which
mimics ``np.float32(np.random.random())`` (53-bit double in [0, 1) cast to
float32) used by the upstream ``_sample_action``.  Because the uniform depends
only on the node identity, any traversal order (recursive DFS or
level-synchronous BFS) sees exactly the same random numbers.
"""

from __future__ import annotations

import numba as nb
import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_SALT_ROOT = np.uint64(0xD1B54A32D192ED03)
_SALT_U = np.uint64(0x8CB92BA72F3D8DD7)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_ONE = np.uint64(1)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53


@nb.njit(cache=True)
def mix64(z):
    """splitmix64 finaliser (bijective 64-bit mixer)."""
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


@nb.njit(cache=True)
def root_key(seed, tid):
    """Key of the root node of traversal ``tid`` for iteration seed ``seed``."""
    s = mix64(np.uint64(seed) + _GOLDEN)
    return mix64(s ^ (np.uint64(tid) * _GOLDEN + _SALT_ROOT))


@nb.njit(cache=True)
def child_key(key, action):
    """Key of the child reached by ``action`` (0..4) from node ``key``."""
    return mix64(key ^ ((np.uint64(action) + _ONE) * _GOLDEN))


@nb.njit(cache=True)
def uniform32(key):
    """float32 uniform in [0, 1] (like np.float32(np.random.random()))."""
    x = mix64(key ^ _SALT_U) >> _S11
    return np.float32(np.float64(x) * _INV53)


_SALT_DEAL = np.uint64(0x5851F42D4C957F2D)
_S33 = np.uint64(33)


@nb.njit(cache=True)
def deal_seed(seed, tid):
    """31-bit np.random seed for dealing traversal ``tid`` (parallel deals)."""
    return np.int64(mix64(root_key(seed, tid) ^ _SALT_DEAL) >> _S33)
