"""GPU-batched external-sampling MCCFR traversal (level-synchronous frontier).

Drop-in replacement for upstream ``traverse_numba.run_traversals`` /
``worker_run_jit`` (fullhouse-repro/source/training/traverse_numba.py): all
root traversals of one Deep CFR iteration are expanded together, one tree
level ("round") at a time:

  round k:  encode the 51-dim features of every live frontier node with the
            upstream ``_encode_fast``  (numba, prange over nodes, CPU)
         -> ONE batched forward of the dueling advantage net on the GPU
         -> regret matching + renormalisation (bit-identical copy of
            ``_strategy``'s tail + upstream ``_renormalize_strategy``);
            opponents sample one action from (1-eps)*sigma + eps*uniform_legal,
            the traverser branches on every legal action; children are made
            with the upstream engine ``step``           (numba, prange, CPU)
  after the last round: values are backed up bottom-up over traverser nodes
  (value = sum_a sigma_a * v_a, advantage = (v_a - value) / 100) and the
  samples are emitted in exactly the order the recursive reference emits them
  (advantage samples in DFS post-order, strategy samples in DFS pre-order,
  traversals in id order).

Randomness: deals (dealer, randomised stacks, deck shuffle) use the upstream
functions with ``np.random`` seeded once per iteration; opponent action
uniforms are counter-based (``det_rng``) keyed by (seed, traversal id, action
history), so the result is independent of traversal order and thread count.

The NN forward never runs on the CPU in production (``forward="torch"``);
``forward="numba"`` (upstream ``_forward_dueling`` per row) exists only for
the bit-exactness test of the tree logic.
"""

from __future__ import annotations

import common  # noqa: F401  (sys.path + NUMBA_CACHE_DIR)

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from common import FEATURE_DIM, N_ACTIONS, N_PLAYERS, load_tables, params_tuple
from det_rng import child_key, deal_seed, root_key, uniform32
from training.game_numba import (
    S_PAYOFFS,
    S_STREET,
    S_TERMINAL,
    S_TOACT,
    STATE_SIZE,
    legal_actions,
    new_hand,
    new_hand_with_stacks,
    step,
)
from training.traverse_numba import (
    _encode_fast,
    _forward_dueling,
    _renormalize_strategy,
    _sample_match_stacks,
)

LEGAL_OFF = 46  # features[46:51] is the legal mask (1.0 / 0.0)


# ---------------------------------------------------------------------------
# numba kernels


@nb.njit(cache=True)
def deal_hands(n, n_players, seed, randomize_stacks):
    """Deal ``n`` hands exactly like upstream run_traversals (same calls)."""
    np.random.seed(seed)
    states = np.empty((n, STATE_SIZE), dtype=np.int32)
    decks = np.empty((n, 52), dtype=np.int8)
    holes = np.empty((n, n_players, 2), dtype=np.int8)
    for i in range(n):
        dealer = np.random.randint(0, n_players)
        if randomize_stacks:
            stacks = _sample_match_stacks(n_players)
            s, d, h = new_hand_with_stacks(n_players, dealer, stacks)
        else:
            s, d, h = new_hand(n_players, dealer)
        states[i] = s
        decks[i] = d
        holes[i] = h
    return states, decks, holes


@nb.njit(parallel=True, cache=True)
def deal_hands_parallel(n, n_players, seed, randomize_stacks):
    """Same per-hand procedure as ``deal_hands`` (upstream calls), but each
    hand re-seeds numba's thread-local MT19937 with ``deal_seed(seed, i)`` so
    the deals are deterministic and independent of the thread count."""
    states = np.empty((n, STATE_SIZE), dtype=np.int32)
    decks = np.empty((n, 52), dtype=np.int8)
    holes = np.empty((n, n_players, 2), dtype=np.int8)
    for i in nb.prange(n):
        np.random.seed(deal_seed(seed, i))
        dealer = np.random.randint(0, n_players)
        if randomize_stacks:
            stacks = _sample_match_stacks(n_players)
            s, d, h = new_hand_with_stacks(n_players, dealer, stacks)
        else:
            s, d, h = new_hand(n_players, dealer)
        states[i] = s
        decks[i] = d
        holes[i] = h
    return states, decks, holes


@nb.njit(cache=True)
def root_keys(n, seed):
    keys = np.empty(n, dtype=np.uint64)
    for i in range(n):
        keys[i] = root_key(seed, i)
    return keys


@nb.njit(parallel=True, cache=True)
def encode_frontier(
    idx, states, tids, decks, holes, traverser,
    flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
    feats_out, is_trav_out, nchild_out,
):
    """Features (upstream _encode_fast) + child counts for live nodes."""
    m = idx.shape[0]
    for i in nb.prange(m):
        j = idx[i]
        st = states[j]
        tid = tids[j]
        seat = st[S_TOACT]
        legal = legal_actions(st, seat)
        n_legal = 0
        for a in range(N_ACTIONS):
            if legal[a]:
                n_legal += 1
        f = _encode_fast(
            st, holes[tid], decks[tid], seat, legal,
            flush_lut, nf_keys, nf_vals, eq_paired, eq_suited, eq_offsuit,
        )
        for k in range(FEATURE_DIM):
            feats_out[i, k] = f[k]
        if seat == traverser:
            is_trav_out[i] = 1
            nchild_out[i] = n_legal
        else:
            is_trav_out[i] = 0
            nchild_out[i] = 1


@nb.njit(cache=True)
def _regret_match(raw_row, legal_f):
    """Bit-identical copy of the post-forward part of upstream ``_strategy``."""
    raw = np.empty(N_ACTIONS, dtype=np.float32)
    for i in range(N_ACTIONS):
        raw[i] = raw_row[i]
    for i in range(N_ACTIONS):
        if raw[i] < np.float32(0.0):
            raw[i] = np.float32(0.0)
    for i in range(N_ACTIONS):
        raw[i] *= legal_f[i]
    total = np.float32(0.0)
    for i in range(N_ACTIONS):
        total += raw[i]
    if total > np.float32(0.0):
        for i in range(N_ACTIONS):
            raw[i] /= total
        return raw
    n_legal = np.float32(0.0)
    for i in range(N_ACTIONS):
        n_legal += legal_f[i]
    for i in range(N_ACTIONS):
        raw[i] = legal_f[i] / n_legal
    return raw


@nb.njit(cache=True)
def _sample_action_r(sigma, legal_f, r):
    """Upstream ``_sample_action`` with the uniform passed in."""
    cumsum = np.float32(0.0)
    last_legal = 0
    for a in range(N_ACTIONS):
        if legal_f[a] > np.float32(0.0):
            last_legal = a
        cumsum += sigma[a]
        if r < cumsum:
            return a
    return last_legal


@nb.njit(parallel=True, cache=True)
def policy_expand(
    idx, states, tids, keys, reach, slot,
    decks, holes, traverser, eps_in,
    feats, raw, is_trav, coff, toff, tbase,
    flush_lut, nf_keys, nf_vals,
    c_states, c_tids, c_keys, c_reach, c_slot, c_term,
    t_feats, t_sigma, t_reach, child_of, root_t, cv,
):
    """Regret matching, opponent sampling / traverser branching, engine step.

    Traverser node i gets id ``tbase + toff[i]``; its children occupy
    ``c_*[coff[i] : coff[i] + n_legal]`` and carry the parent slot
    ``t * 5 + a``.  A child that is terminal writes the traverser payoff into
    ``cv[slot]``; a child chain that reaches a traverser node registers it in
    ``child_of[slot]``.
    """
    m = idx.shape[0]
    for i in nb.prange(m):
        j = idx[i]
        tid = tids[j]
        legal_f = np.empty(N_ACTIONS, dtype=np.float32)
        for a in range(N_ACTIONS):
            legal_f[a] = feats[i, LEGAL_OFF + a]
        sigma = _regret_match(raw[i], legal_f)
        _renormalize_strategy(sigma, legal_f)
        off = coff[i]
        if is_trav[i] == 1:
            t = tbase + toff[i]
            for k in range(FEATURE_DIM):
                t_feats[t, k] = feats[i, k]
            for a in range(N_ACTIONS):
                t_sigma[t, a] = sigma[a]
            t_reach[t] = reach[j]
            if slot[j] >= 0:
                child_of[slot[j]] = t
            else:
                root_t[tid] = t
            k = 0
            for a in range(N_ACTIONS):
                if legal_f[a] > np.float32(0.0):
                    c = off + k
                    k += 1
                    for q in range(STATE_SIZE):
                        c_states[c, q] = states[j, q]
                    child = c_states[c]
                    step(child, decks[tid], holes[tid], a,
                         flush_lut, nf_keys, nf_vals)
                    c_tids[c] = tid
                    c_keys[c] = child_key(keys[j], a)
                    c_reach[c] = reach[j] * sigma[a]
                    c_slot[c] = t * N_ACTIONS + a
                    if child[S_TERMINAL] == 1:
                        c_term[c] = 1
                        cv[t * N_ACTIONS + a] = np.float32(child[S_PAYOFFS + traverser])
                    else:
                        c_term[c] = 0
        else:
            if eps_in > np.float32(0.0):
                eps = eps_in
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
            action = _sample_action_r(sigma, legal_f, uniform32(keys[j]))
            c = off
            for q in range(STATE_SIZE):
                c_states[c, q] = states[j, q]
            child = c_states[c]
            step(child, decks[tid], holes[tid], action, flush_lut, nf_keys, nf_vals)
            c_tids[c] = tid
            c_keys[c] = child_key(keys[j], action)
            c_reach[c] = reach[j]
            c_slot[c] = slot[j]
            if child[S_TERMINAL] == 1:
                c_term[c] = 1
                if slot[j] >= 0:
                    cv[slot[j]] = np.float32(child[S_PAYOFFS + traverser])
            else:
                c_term[c] = 0


_PAR_MIN = 32768  # batches smaller than this are backed up sequentially


@nb.njit(cache=True)
def _backup_node(t, t_sigma, t_feats, child_of, cv, adv, value, size):
    cvt = np.empty(N_ACTIONS, dtype=np.float32)
    sz = 1
    for a in range(N_ACTIONS):
        c = child_of[t * N_ACTIONS + a]
        if c >= 0:
            cvt[a] = value[c]
            sz += size[c]
        else:
            cvt[a] = cv[t * N_ACTIONS + a]
    v = np.float32(0.0)
    for a in range(N_ACTIONS):
        v += t_sigma[t, a] * cvt[a]
    for a in range(N_ACTIONS):
        if t_feats[t, LEGAL_OFF + a] > np.float32(0.0):
            adv[t, a] = (cvt[a] - v) / np.float32(100.0)
    value[t] = v
    size[t] = sz


@nb.njit(cache=True)
def _order_node(t, child_of, size, pre, first):
    po = pre[t] + 1
    qo = first[t]
    for a in range(N_ACTIONS):
        c = child_of[t * N_ACTIONS + a]
        if c >= 0:
            pre[c] = po
            first[c] = qo
            po += size[c]
            qo += size[c]


@nb.njit(parallel=True, cache=True)
def backup_and_order(batches, n_trav, t_sigma, t_feats, child_of, root_t, cv):
    """Bottom-up values/advantages + DFS pre/post-order positions.

    ``batches`` are the T-id boundaries of successive ``policy_expand`` calls;
    a node's children always live in a later batch, so batches are processed
    in reverse (values, subtree sizes: pulled from children) and forward
    (DFS positions: pushed to children).  Large batches run in parallel;
    small ones sequentially (fewer barriers -> robust to CPU contention).
    Arithmetic per node is identical to the reference:
    value = sum_a sigma_a * v_a (fp32, a = 0..4), adv_a = (v_a - value) / 100.
    """
    n_t = batches[-1]
    n_b = batches.shape[0] - 1
    adv = np.zeros((n_t, N_ACTIONS), dtype=np.float32)
    value = np.empty(n_t, dtype=np.float32)
    size = np.empty(n_t, dtype=np.int64)
    for bb in range(n_b - 1, -1, -1):
        lo = batches[bb]
        hi = batches[bb + 1]
        if hi - lo >= _PAR_MIN:
            for t in nb.prange(lo, hi):
                _backup_node(t, t_sigma, t_feats, child_of, cv, adv, value, size)
        else:
            for t in range(lo, hi):
                _backup_node(t, t_sigma, t_feats, child_of, cv, adv, value, size)
    pre = np.empty(n_t, dtype=np.int64)
    first = np.empty(n_t, dtype=np.int64)
    per_trav = np.zeros(n_trav, dtype=np.int64)
    off = 0
    for tid in range(n_trav):
        r = root_t[tid]
        if r >= 0:
            pre[r] = off
            first[r] = off
            per_trav[tid] = size[r]
            off += size[r]
    for bb in range(n_b):  # parents before children
        lo = batches[bb]
        hi = batches[bb + 1]
        if hi - lo >= _PAR_MIN:
            for t in nb.prange(lo, hi):
                _order_node(t, child_of, size, pre, first)
        else:
            for t in range(lo, hi):
                _order_node(t, child_of, size, pre, first)
    post = first + size - 1
    return adv, pre, post, per_trav


@nb.njit(parallel=True, cache=True)
def gather_outputs(t_feats, t_sigma, t_reach, adv, pre, post,
                   out_f, out_a, out_sf, out_s, out_w):
    for t in nb.prange(pre.shape[0]):
        pa = post[t]
        ps = pre[t]
        for k in range(FEATURE_DIM):
            v = t_feats[t, k]
            out_f[pa, k] = v
            out_sf[ps, k] = v
        for a in range(N_ACTIONS):
            out_a[pa, a] = adv[t, a]
            out_s[ps, a] = t_sigma[t, a]
        out_w[ps] = t_reach[t]


@nb.njit(parallel=True, cache=True)
def numba_forward_rows(feats, raw_out, w):
    """TEST ONLY: upstream per-node numba forward, for the bit-exact check."""
    for i in nb.prange(feats.shape[0]):
        x = feats[i].copy()
        legal_f = np.empty(N_ACTIONS, dtype=np.float32)
        for a in range(N_ACTIONS):
            legal_f[a] = x[LEGAL_OFF + a]
        out = _forward_dueling(
            x, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
            w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], legal_f,
        )
        for a in range(N_ACTIONS):
            raw_out[i, a] = out[a]


# ---------------------------------------------------------------------------
# GPU forward


class TorchForward:
    """Batched dueling advantage-net forward on one CUDA device.

    precision: "highest" (fp32 IEEE matmul), "tf32", or "bf16" (autocast).
    The dueling combination is always evaluated in fp32.
    """

    def __init__(self, device="cuda:1", precision="highest"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.precision = precision
        self.p = None

    def set_params(self, params):
        torch = self.torch
        g = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
             for k, v in params.items()}
        self.p = dict(
            w0=g["trunk_w0"], b0=g["trunk_b0"], g0=g["trunk_ln0_g"], e0=g["trunk_ln0_b"],
            w1=g["trunk_w1"], b1=g["trunk_b1"], g1=g["trunk_ln1_g"], e1=g["trunk_ln1_b"],
            wh=torch.cat([g["val_w0"], g["adv_w0"]], 0).contiguous(),
            bh=torch.cat([g["val_b0"], g["adv_b0"]], 0).contiguous(),
            vw1=g["val_w1"], vb1=g["val_b1"], aw1=g["adv_w1"], ab1=g["adv_b1"],
        )
        self.head = g["val_w0"].shape[0]

    def __call__(self, x):
        torch = self.torch
        F = torch.nn.functional
        p = self.p
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.precision == "tf32"
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=self.precision == "bf16"):
                h = F.linear(x, p["w0"], p["b0"])
                h = F.layer_norm(h, (h.shape[1],), p["g0"], p["e0"], 1e-5)
                h = F.leaky_relu(h, 0.01)
                h = F.linear(h, p["w1"], p["b1"])
                h = F.layer_norm(h, (h.shape[1],), p["g1"], p["e1"], 1e-5)
                h = F.leaky_relu(h, 0.01)
                hv = F.leaky_relu(F.linear(h, p["wh"], p["bh"]), 0.01)
                v = F.linear(hv[:, : self.head], p["vw1"], p["vb1"])
                a = F.linear(hv[:, self.head:], p["aw1"], p["ab1"])
            v = v.float()
            a = a.float()
            legal = x[:, LEGAL_OFF:LEGAL_OFF + N_ACTIONS]
            am = a * legal
            n_legal = legal.sum(dim=1, keepdim=True).clamp(min=1.0)
            a_mean = am.sum(dim=1, keepdim=True) / n_legal
            return (v + am - a_mean * legal) * legal
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

# ---------------------------------------------------------------------------
# driver


@dataclass
class TraversalResult:
    features: np.ndarray
    advantages: np.ndarray
    strat_features: np.ndarray
    strategies: np.ndarray
    strat_weights: np.ndarray
    per_trav_adv: np.ndarray
    n_rounds: int
    n_nodes: int
    timing: dict = field(default_factory=dict)
    stats: np.ndarray | None = None  # [street, (opp, trav, terminal)]
    device_tensors: dict | None = None  # optional torch copies (to_device=)

    def as_worker_result(self):
        """5-tuple consumed by upstream ``train._collect_worker_batches``."""
        return (self.features, self.advantages, self.strat_features,
                self.strategies, self.strat_weights)


class _TStore:
    """Growable traverser-node storage, reused across iterations."""

    def __init__(self):
        self.cap = 0
        self.used = 0

    def _alloc(self, cap, keep):
        old = (self.t_feats, self.t_sigma, self.t_reach, self.child_of,
               self.cv) if self.cap else None
        self.t_feats = np.empty((cap, FEATURE_DIM), dtype=np.float32)
        self.t_sigma = np.empty((cap, N_ACTIONS), dtype=np.float32)
        self.t_reach = np.empty(cap, dtype=np.float32)
        self.child_of = np.full(cap * N_ACTIONS, -1, dtype=np.int64)
        self.cv = np.zeros(cap * N_ACTIONS, dtype=np.float32)
        if old is not None and keep:
            self.t_feats[:keep] = old[0][:keep]
            self.t_sigma[:keep] = old[1][:keep]
            self.t_reach[:keep] = old[2][:keep]
            self.child_of[: keep * N_ACTIONS] = old[3][: keep * N_ACTIONS]
            self.cv[: keep * N_ACTIONS] = old[4][: keep * N_ACTIONS]
        self.cap = cap

    def reset(self, cap_hint):
        if cap_hint > self.cap:
            self._alloc(cap_hint, 0)
        else:
            self.child_of[: self.used * N_ACTIONS] = -1
            self.cv[: self.used * N_ACTIONS] = 0.0
        self.used = 0

    def ensure(self, need):
        if need > self.cap:
            self._alloc(int(max(need, self.cap * 1.5)), self.used)


_CHILD_FIELDS = (("states", np.int32, STATE_SIZE), ("tids", np.int32, 0),
                 ("keys", np.uint64, 0), ("reach", np.float32, 0),
                 ("slot", np.int64, 0), ("term", np.int8, 0))


class _GroupRes:
    """Persistent per-group buffers: pinned feature/raw staging, double-buffered
    child frontier arrays, and a CUDA event."""

    def __init__(self, torch_mod):
        self.torch = torch_mod
        self.cap_f = 0
        self.cap_c = [0, 0]
        self.sets = [None, None]
        self.event = torch_mod.cuda.Event(enable_timing=False) if torch_mod else None

    def ensure_feat(self, m):
        if m <= self.cap_f:
            return
        cap = int(max(m * 1.25, 65536))
        if self.torch is not None:
            torch = self.torch
            self.feats_t = torch.empty((cap, FEATURE_DIM), dtype=torch.float32,
                                       pin_memory=True)
            self.raw_t = torch.empty((cap, N_ACTIONS), dtype=torch.float32,
                                     pin_memory=True)
            self.feats = self.feats_t.numpy()
            self.raw = self.raw_t.numpy()
        else:
            self.feats = np.empty((cap, FEATURE_DIM), dtype=np.float32)
            self.raw = np.empty((cap, N_ACTIONS), dtype=np.float32)
        self.is_trav = np.empty(cap, dtype=np.int8)
        self.nchild = np.empty(cap, dtype=np.int64)
        self.cap_f = cap

    def child_set(self, k, n):
        if n > self.cap_c[k]:
            cap = int(max(n * 1.25, 65536))
            self.sets[k] = {
                name: np.empty((cap, w) if w else cap, dtype=dt)
                for name, dt, w in _CHILD_FIELDS
            }
            self.cap_c[k] = cap
        return {name: arr[:n] for name, arr in self.sets[k].items()}


class _Group:
    """Live frontier of one sub-population of traversals [lo, hi)."""

    def __init__(self, res, lo, hi, states0, keys0):
        n = hi - lo
        self.res = res
        self.states = states0[lo:hi]
        self.tids = np.arange(lo, hi, dtype=np.int32)
        self.keys = keys0[lo:hi]
        self.reach = np.ones(n, dtype=np.float32)
        self.slot = np.full(n, -1, dtype=np.int64)
        self.idx = np.arange(n, dtype=np.int64)
        self.flip = 0
        self.m = n


class BatchedTraverser:
    """Reusable traversal engine (keeps tables, pinned buffers, GPU weights).

    n_threads: numba threads (24 = physical cores measured fastest on the
    48-vCPU VM; None keeps numba's default).
    n_groups > 1 splits the traversals into independent sub-populations whose
    rounds are interleaved, so the GPU forward of one group overlaps the CPU
    engine/encoding work of the others.  Results do not depend on n_groups.
    """

    def __init__(self, device="cuda:1", precision="highest", forward="torch",
                 n_threads=24, n_groups=2, parallel_deal=True):
        self.tables = load_tables()
        self.forward_kind = forward
        self.device = device
        self.precision = precision
        self.n_groups = n_groups
        self.parallel_deal = parallel_deal
        if n_threads is not None:
            nb.set_num_threads(n_threads)
        self._tstore = _TStore()
        self._res = []
        if forward == "torch":
            import torch

            self.torch = torch
            self.fwd = TorchForward(device, precision)
            self.stream = torch.cuda.Stream(device=torch.device(device))
        else:
            self.torch = None

    def deal(self, n, seed, randomize_stacks=True):
        fn = deal_hands_parallel if self.parallel_deal else deal_hands
        return fn(n, N_PLAYERS, seed, randomize_stacks)

    # -- one round, split into phases ----------------------------------------
    def _encode(self, g):
        r = g.res
        m = g.idx.shape[0]
        g.m = m
        r.ensure_feat(m)
        flush_lut, nf_keys, nf_vals, eq_p, eq_s, eq_o = self.tables
        encode_frontier(g.idx, g.states, g.tids, self._decks, self._holes,
                        self._traverser, flush_lut, nf_keys, nf_vals,
                        eq_p, eq_s, eq_o, r.feats[:m], r.is_trav[:m],
                        r.nchild[:m])

    def _launch(self, g):
        r = g.res
        m = g.m
        if self.forward_kind == "numba":
            numba_forward_rows(r.feats[:m], r.raw[:m], self._w)
            return
        torch = self.torch
        with torch.cuda.stream(self.stream):
            if self._profile:
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                ev[0].record(self.stream)
            x = r.feats_t[:m].to(self.fwd.device, non_blocking=True)
            if self._profile:
                ev[1].record(self.stream)
            raw = self.fwd(x)
            if self._profile:
                ev[2].record(self.stream)
            r.raw_t[:m].copy_(raw, non_blocking=True)
            if self._profile:
                ev[3].record(self.stream)
            r.event.record(self.stream)
        if self._profile:
            ev[3].synchronize()
            self._tm["h2d"] += ev[0].elapsed_time(ev[1]) / 1e3
            self._tm["fwd"] += ev[1].elapsed_time(ev[2]) / 1e3
            self._tm["d2h"] += ev[2].elapsed_time(ev[3]) / 1e3

    def _wait(self, g):
        if self.forward_kind == "torch":
            g.res.event.synchronize()

    def _expand(self, g):
        r = g.res
        m = g.m
        tm = self._tm
        t0 = time.perf_counter()
        nchild = r.nchild[:m]
        is_trav = r.is_trav[:m]
        coff = np.cumsum(nchild)
        n_child = int(coff[-1])
        coff -= nchild
        toff = np.cumsum(is_trav, dtype=np.int64)
        n_new_t = int(toff[-1])
        toff -= is_trav
        ts = self._tstore
        ts.ensure(ts.used + n_new_t)
        cs = r.child_set(g.flip, n_child)
        if self._stats is not None:
            streets = g.states[g.idx, S_STREET]
            np.add.at(self._stats, (streets, is_trav.astype(np.int64)), 1)
        t1 = time.perf_counter()
        tm["prefix"] += t1 - t0
        flush_lut, nf_keys, nf_vals = self.tables[:3]
        policy_expand(
            g.idx, g.states, g.tids, g.keys, g.reach, g.slot,
            self._decks, self._holes, self._traverser, self._eps,
            r.feats[:m], r.raw[:m], is_trav, coff, toff, np.int64(ts.used),
            flush_lut, nf_keys, nf_vals,
            cs["states"], cs["tids"], cs["keys"], cs["reach"], cs["slot"],
            cs["term"], ts.t_feats, ts.t_sigma, ts.t_reach, ts.child_of,
            self._root_t, ts.cv,
        )
        t2 = time.perf_counter()
        tm["expand"] += t2 - t1
        ts.used += n_new_t
        self._batches.append(ts.used)
        term = cs["term"]
        if self._stats is not None:
            np.add.at(self._stats, (cs["states"][term == 1, S_STREET], 2), 1)
        g.idx = np.flatnonzero(term == 0)
        g.states, g.tids, g.keys = cs["states"], cs["tids"], cs["keys"]
        g.reach, g.slot = cs["reach"], cs["slot"]
        g.flip ^= 1
        tm["compact"] += time.perf_counter() - t2

    # -- main entry ----------------------------------------------------------
    def run(self, params, traverser, n_traversals, eps, seed,
            randomize_stacks=True, deals=None, collect_stats=False,
            profile=False, n_groups=None, to_device=None):
        """One iteration of ``n_traversals`` external-sampling traversals.

        params:  dict of the 16 dueling-net arrays (advantage-NPZ key names).
        seed:    deals and the node-keyed action uniforms derive from it.
        deals:   optional (states0, decks, holes) overriding the dealing.
        profile: synchronous single-group mode that also times H2D / forward
                 / D2H with CUDA events (slower; for the time split only).
        to_device: optional torch device; the five sample arrays are also
                 returned as torch tensors there (result.device_tensors).
        """
        tm = defaultdict(float)
        self._tm = tm
        self._profile = profile and self.forward_kind == "torch"
        t_all = time.perf_counter()
        self._w = params_tuple(params)
        if self.forward_kind == "torch":
            self.fwd.set_params(params)
        self._eps = np.float32(eps)
        self._traverser = np.int32(traverser)
        t0 = time.perf_counter()
        tm["setup"] += t0 - t_all

        if deals is None:
            states0, decks, holes = self.deal(n_traversals, seed, randomize_stacks)
        else:
            states0, decks, holes = deals
            n_traversals = states0.shape[0]
        self._decks, self._holes = decks, holes
        n = n_traversals
        t1 = time.perf_counter()
        tm["deal"] += t1 - t0

        keys0 = root_keys(n, seed)
        self._tstore.reset(n * 48)
        self._root_t = np.full(n, -1, dtype=np.int64)
        self._batches = [0]
        self._stats = np.zeros((4, 3), dtype=np.int64) if collect_stats else None
        k = 1 if self._profile else (n_groups or self.n_groups)
        k = max(1, min(k, n))
        while len(self._res) < k:
            self._res.append(_GroupRes(self.torch))
        bounds = np.linspace(0, n, k + 1).astype(np.int64)
        groups = [_Group(self._res[i], int(bounds[i]), int(bounds[i + 1]),
                         states0, keys0) for i in range(k)]
        tm["setup"] += time.perf_counter() - t1

        n_rounds = 0
        n_nodes = 0

        def encode_launch(g):
            t0 = time.perf_counter()
            self._encode(g)
            t1 = time.perf_counter()
            self._launch(g)
            tm["encode"] += t1 - t0
            tm["launch"] += time.perf_counter() - t1

        for g in groups:
            encode_launch(g)
        live = groups
        while live:
            n_rounds += 1
            nxt = []
            for g in live:
                n_nodes += g.m
                t0 = time.perf_counter()
                self._wait(g)
                tm["gpu_wait"] += time.perf_counter() - t0
                self._expand(g)
                if g.idx.shape[0] > 0:
                    encode_launch(g)
                    nxt.append(g)
            live = nxt

        t0 = time.perf_counter()
        ts = self._tstore
        n_t = ts.used
        adv, pre, post, per_trav = backup_and_order(
            np.asarray(self._batches, dtype=np.int64), n, ts.t_sigma,
            ts.t_feats, ts.child_of, self._root_t, ts.cv)
        t1 = time.perf_counter()
        tm["backup"] += t1 - t0
        out_f = np.empty((n_t, FEATURE_DIM), dtype=np.float32)
        out_a = np.empty((n_t, N_ACTIONS), dtype=np.float32)
        out_sf = np.empty((n_t, FEATURE_DIM), dtype=np.float32)
        out_s = np.empty((n_t, N_ACTIONS), dtype=np.float32)
        out_w = np.empty(n_t, dtype=np.float32)
        gather_outputs(ts.t_feats, ts.t_sigma, ts.t_reach, adv, pre, post,
                       out_f, out_a, out_sf, out_s, out_w)
        t2 = time.perf_counter()
        tm["gather"] += t2 - t1
        dev_t = None
        if to_device is not None:
            torch = self.torch
            dev = torch.device(to_device)
            dev_t = {name: torch.from_numpy(arr).to(dev) for name, arr in zip(
                ("features", "advantages", "strat_features", "strategies",
                 "strat_weights"), (out_f, out_a, out_sf, out_s, out_w))}
            torch.cuda.synchronize(dev)
            tm["to_device"] += time.perf_counter() - t2
        tm["total"] = time.perf_counter() - t_all
        return TraversalResult(out_f, out_a, out_sf, out_s, out_w, per_trav,
                               n_rounds, n_nodes, dict(tm), self._stats, dev_t)


_DEFAULT = {}


def run_iteration(params, traverser, n_traversals=30000, eps=0.1, seed=0,
                  device="cuda:1", precision="highest", forward="torch",
                  randomize_stacks=True, **kw):
    """Functional wrapper around a cached ``BatchedTraverser``.

    Returns a TraversalResult; ``result.as_worker_result()`` is the 5-tuple
    (features, advantages, strat_features, strategies, strat_weights) that
    upstream ``train._collect_worker_batches`` consumes (pass ``[tuple]``).
    """
    key = (device, precision, forward)
    if key not in _DEFAULT:
        _DEFAULT[key] = BatchedTraverser(device, precision, forward)
    return _DEFAULT[key].run(params, traverser, n_traversals, eps, seed,
                             randomize_stacks=randomize_stacks, **kw)
