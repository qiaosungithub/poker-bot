"""TEST COPY of the upstream recursive external-sampling traversal.

Copied from ``fullhouse-repro/source/training/traverse_numba.py``
(``_traverse_external`` / ``run_traversals``) with exactly three changes:

1. The opponent sampling uniform comes either from ``np.random.random()``
   (``use_hash=False`` -- identical RNG consumption to upstream) or from the
   node-keyed hash uniform ``det_rng.uniform32(key)`` (``use_hash=True``).
   ``key`` is threaded through the recursion with ``child_key(key, action)``.
2. Node counters (per street: opponent / traverser / terminal nodes) and the
   number of advantage samples per traversal are recorded.
3. ``run_ref_hash`` takes pre-dealt hands (so the batched traversal can be fed
   the same deals); ``run_ref_np`` deals exactly like upstream run_traversals.

The engine, feature encoder, numba forward and regret matching are the
upstream functions themselves (imported, not copied).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "traversal"))  # fullhouse-gpu/traversal

import common  # noqa: F401  (sets sys.path / NUMBA_CACHE_DIR)

import numba as nb
import numpy as np

from training.game_numba import (
    N_ACTIONS,
    S_PAYOFFS,
    S_STREET,
    S_TERMINAL,
    S_TOACT,
    legal_actions,
    new_hand,
    new_hand_with_stacks,
    step,
)
from training.traverse_numba import (
    FEATURE_DIM,
    _encode_fast,
    _renormalize_strategy,
    _sample_match_stacks,
    _strategy,
)
from det_rng import child_key, root_key, uniform32


@nb.njit(cache=True)
def _sample_action_r(sigma, legal, r):
    """Upstream ``_sample_action`` with the uniform passed in."""
    cumsum = np.float32(0.0)
    last_legal = 0
    for a in range(N_ACTIONS):
        if legal[a]:
            last_legal = a
        cumsum += sigma[a]
        if r < cumsum:
            return a
    if cumsum > np.float32(0.0):
        return last_legal
    raise ValueError("sampled strategy had no cumulative mass")


@nb.njit(cache=True)
def _traverse_ref(
    state, deck, hole_cards, traverser, w,
    out_features, out_advantages, out_count,
    out_strat_features, out_strategies, out_strat_weights, out_strat_count,
    flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
    opponent_exploration, traverser_reach, key, use_hash, stats,
):
    if state[S_TERMINAL] == 1:
        stats[state[S_STREET], 2] += 1
        return np.float32(state[S_PAYOFFS + traverser])

    seat = state[S_TOACT]
    legal = legal_actions(state, seat)

    n_legal = 0
    for a in range(N_ACTIONS):
        if legal[a]:
            n_legal += 1
    if n_legal == 0:
        raise ValueError("game engine returned no legal actions")

    features = _encode_fast(
        state, hole_cards, deck, seat, legal,
        flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
    )

    legal_f = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            legal_f[a] = np.float32(1.0)

    sigma = _strategy(
        features, legal_f,
        w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
        w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15],
    )
    _renormalize_strategy(sigma, legal_f)

    if seat != traverser:
        stats[state[S_STREET], 0] += 1
        if opponent_exploration > np.float32(0.0):
            eps = opponent_exploration
            if eps > np.float32(1.0):
                eps = np.float32(1.0)
            n_legal_f = np.float32(0.0)
            for a in range(N_ACTIONS):
                n_legal_f += legal_f[a]
            if n_legal_f > np.float32(0.0):
                for a in range(N_ACTIONS):
                    sigma[a] = (np.float32(1.0) - eps) * sigma[a] + eps * legal_f[
                        a
                    ] / n_legal_f
            _renormalize_strategy(sigma, legal_f)
        if use_hash:
            r = uniform32(key)
        else:
            r = np.float32(np.random.random())
        action = _sample_action_r(sigma, legal, r)

        child = state.copy()
        step(child, deck, hole_cards, action, flush_lut, nf_keys, nf_vals)
        return _traverse_ref(
            child, deck, hole_cards, traverser, w,
            out_features, out_advantages, out_count,
            out_strat_features, out_strategies, out_strat_weights, out_strat_count,
            flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
            opponent_exploration, traverser_reach, child_key(key, action),
            use_hash, stats,
        )

    stats[state[S_STREET], 1] += 1
    sidx = out_strat_count[0]
    if sidx < out_strat_features.shape[0]:
        for f in range(FEATURE_DIM):
            out_strat_features[sidx, f] = features[f]
        for a in range(N_ACTIONS):
            out_strategies[sidx, a] = sigma[a]
        out_strat_weights[sidx] = traverser_reach
        out_strat_count[0] = sidx + 1

    child_values = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            child = state.copy()
            step(child, deck, hole_cards, a, flush_lut, nf_keys, nf_vals)
            child_values[a] = _traverse_ref(
                child, deck, hole_cards, traverser, w,
                out_features, out_advantages, out_count,
                out_strat_features, out_strategies, out_strat_weights,
                out_strat_count,
                flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
                opponent_exploration, traverser_reach * sigma[a],
                child_key(key, a), use_hash, stats,
            )

    value = np.float32(0.0)
    for a in range(N_ACTIONS):
        value += sigma[a] * child_values[a]

    advantages = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            advantages[a] = (child_values[a] - value) / np.float32(100.0)

    idx = out_count[0]
    if idx < out_features.shape[0]:
        for f in range(FEATURE_DIM):
            out_features[idx, f] = features[f]
        for a in range(N_ACTIONS):
            out_advantages[idx, a] = advantages[a]
        out_count[0] = idx + 1

    return value


@nb.njit(cache=True)
def _alloc(max_samples):
    return (
        np.zeros((max_samples, FEATURE_DIM), dtype=np.float32),
        np.zeros((max_samples, N_ACTIONS), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros((max_samples, FEATURE_DIM), dtype=np.float32),
        np.zeros((max_samples, N_ACTIONS), dtype=np.float32),
        np.zeros(max_samples, dtype=np.float32),
        np.zeros(1, dtype=np.int64),
    )


@nb.njit(cache=True)
def run_ref_hash(
    states0, decks, holes, tid0, traverser, w, seed,
    flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
    opponent_exploration, max_per_trav,
):
    """Deterministic reference over pre-dealt hands; tids are tid0 + i."""
    n = states0.shape[0]
    max_samples = n * max_per_trav
    of, oa, oc, osf, osg, osw, osc = _alloc(max_samples)
    stats = np.zeros((4, 3), dtype=np.int64)
    per_trav = np.zeros(n, dtype=np.int64)
    for i in range(n):
        before = oc[0]
        _traverse_ref(
            states0[i].copy(), decks[i], holes[i], traverser, w,
            of, oa, oc, osf, osg, osw, osc,
            flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
            np.float32(opponent_exploration), np.float32(1.0),
            root_key(seed, tid0 + i), True, stats,
        )
        per_trav[i] = oc[0] - before
    na = oc[0]
    ns = osc[0]
    return (
        of[:na].copy(), oa[:na].copy(), osf[:ns].copy(), osg[:ns].copy(),
        osw[:ns].copy(), stats, per_trav,
    )


@nb.njit(cache=True)
def run_ref_np(
    n_traversals, n_players, traverser, w, seed,
    flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
    opponent_exploration, randomize_stacks,
):
    """Upstream ``run_traversals`` RNG consumption, plus instrumentation."""
    np.random.seed(seed)
    max_samples = n_traversals * 60
    of, oa, oc, osf, osg, osw, osc = _alloc(max_samples)
    stats = np.zeros((4, 3), dtype=np.int64)
    per_trav = np.zeros(n_traversals, dtype=np.int64)
    for i in range(n_traversals):
        dealer = np.random.randint(0, n_players)
        if randomize_stacks:
            starting_stacks = _sample_match_stacks(n_players)
            state, deck, hole_cards = new_hand_with_stacks(
                n_players, dealer, starting_stacks
            )
        else:
            state, deck, hole_cards = new_hand(n_players, dealer)
        before = oc[0]
        _traverse_ref(
            state, deck, hole_cards, traverser, w,
            of, oa, oc, osf, osg, osw, osc,
            flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
            np.float32(opponent_exploration), np.float32(1.0),
            np.uint64(0), False, stats,
        )
        per_trav[i] = oc[0] - before
    na = oc[0]
    ns = osc[0]
    return (
        of[:na].copy(), oa[:na].copy(), osf[:ns].copy(), osg[:ns].copy(),
        osw[:ns].copy(), stats, per_trav,
    )


# ---------------------------------------------------------------------------
# Multiprocess drivers (spawn), mirroring upstream train._run_parallel_traversals


def _hash_worker(args):
    import common as c

    tables = c.load_tables()
    w = c.params_tuple(args["params"])
    return run_ref_hash(
        args["states0"], args["decks"], args["holes"], args["tid0"],
        args["traverser"], w, args["seed"], *tables,
        np.float32(args["eps"]), args["max_per_trav"],
    )


def _np_worker(args):
    import common as c

    tables = c.load_tables()
    w = c.params_tuple(args["params"])
    return run_ref_np(
        args["n"], c.N_PLAYERS, args["traverser"], w, args["seed"], *tables,
        np.float32(args["eps"]), args["randomize_stacks"],
    )


def _concat(results):
    outs = []
    for k in range(5):
        outs.append(np.concatenate([r[k] for r in results], axis=0))
    stats = sum(r[5] for r in results)
    per_trav = np.concatenate([r[6] for r in results])
    return tuple(outs), stats, per_trav


def parallel_ref_hash(params, deals, traverser, seed, eps, n_workers=46,
                      max_per_trav=400):
    """Deterministic reference over all deals, chunked over processes."""
    import multiprocessing as mp

    states0, decks, holes = deals
    n = states0.shape[0]
    bounds = np.linspace(0, n, n_workers + 1).astype(int)
    args = [
        dict(params=params, states0=states0[a:b], decks=decks[a:b],
             holes=holes[a:b], tid0=int(a), traverser=traverser, seed=seed,
             eps=eps, max_per_trav=max_per_trav)
        for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ]
    with mp.get_context("spawn").Pool(len(args)) as pool:
        return _concat(pool.map(_hash_worker, args))


def parallel_ref_np(params, n_traversals, traverser, iteration, eps,
                    n_workers=46, randomize_stacks=True):
    """Instrumented copy with upstream RNG; seeds iteration*1000 + worker."""
    import multiprocessing as mp

    per = n_traversals // n_workers
    rem = n_traversals % n_workers
    counts = [per + (1 if i < rem else 0) for i in range(n_workers)]
    args = [
        dict(params=params, n=c, traverser=traverser,
             seed=iteration * 1000 + i, eps=eps,
             randomize_stacks=randomize_stacks)
        for i, c in enumerate(counts)
    ]
    with mp.get_context("spawn").Pool(n_workers) as pool:
        return _concat(pool.map(_np_worker, args))
