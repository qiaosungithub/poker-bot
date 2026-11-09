# gpu-trav: GPU-batched external-sampling traversal for Fullhouse Deep CFR

Location on the VM: `qiaos-4a100-2:~/work/poker-fast/gpu_trav/` (local git history in the
workstation artifact dir `gpu_trav_work/`, 4 commits, nothing pushed).

## Outcome

One Deep CFR iteration of **30,000 external-sampling traversals takes about 0.49 s**
(median of 8; mean 0.52 s; best 0.45 s; about 61,000 traversals/s). The
measurement uses the released advantage net (`source/data/deep_cfr_model.npz`, late-run tree
size: 3.77 M decision nodes and 1.24 M advantage samples per iteration) with an **fp32 'highest'**
GPU forward on one A100 (cuda:1) and 24 numba CPU threads. The budget was 1.5 s; the upstream
baseline is about 6 s (46 CPU workers, numba forward on CPU), so this is about **12x faster**. All
neural-network evaluations run as batched GPU forwards; the CPU only runs the game engine, the
feature encoder, regret matching and the value backup.

Correctness:
- With a deterministic per-node random source and the upstream numba forward, the batched
  traversal reproduces the recursive reference **bit for bit**. This covers all five sample arrays,
  their order, the per-traversal counts and the per-street node counts, on 30,000 traversals for a
  random-init net, a partly trained net (baseline iteration 4) and the released net.
- With the production fp32 GPU forward, **no traversal out of 30,000 diverges** for any of the three
  nets; the remaining differences are float rounding (tolerances below).
- With production-style randomness, 149 statistics agree with the **unmodified upstream**
  `worker_run_jit`: 6 x 30k traversals per side for each of 3 nets, plus a 16 x 30k follow-up on the
  released net. Max |Welch t| is 3.4 on the original runs and 2.6 on the follow-up; no metric exceeds 4.

## Files

| File | Purpose |
|---|---|
| `gpu_traverse.py` | The module. `BatchedTraverser(device, precision, n_threads=24, n_groups=2).run(params, traverser, n_traversals, eps, seed)` returns a `TraversalResult`; `run_iteration(...)` is a functional wrapper. |
| `common.py` | Locates the read-only upstream tree (`sys.path`), redirects `NUMBA_CACHE_DIR` into this dir, loads LUTs/equity tables, and converts nets to the 16-array param dict (`params_from_npz`, `params_from_torch`, `params_random_init`). |
| `det_rng.py` | Counter-based uniforms keyed by (seed, traversal id, action history) (splitmix64), plus per-traversal deal seeds. |
| `ref_det.py` | TEST COPY of upstream `_traverse_external` / `run_traversals`. Only two changes: the uniform source is switchable (np.random or hash), and it has instrumentation. Also multiprocess drivers. |
| `test_exact.py` | Exact-equality / tolerance test against `ref_det` (same deals, same hash uniforms). |
| `test_invariance.py` | Results vs n_groups / threads; parallel deals vs thread count. |
| `compare_stats.py` | Statistical comparison against unmodified upstream `worker_run_jit`. |
| `bench.py` | Timing and time split. `example_integration.py` feeds results through upstream `_collect_worker_batches` + `ReservoirBuffer`. |
| `run_*_suite.sh`, `results/*.json` | The exact runs behind every number in this report. |

## Algorithm (level-synchronous frontier)

All 30k root traversals of an iteration expand together, one tree level ("round") at a time;
there are about 30 rounds per iteration.

1. **Deal** (numba prange): per traversal, `dealer = randint(6)`, upstream `_sample_match_stacks`, then
   `new_hand_with_stacks` (deck shuffle). These are the same calls as upstream `run_traversals`;
   numba's thread-local MT19937 is re-seeded per traversal from `deal_seed(seed, tid)`.
2. **Encode** (numba prange over live frontier nodes): upstream `legal_actions` + `_encode_fast`
   write the 51-dim features straight into a pinned host buffer.
3. **GPU forward**: one batched H2D copy, then the dueling net (trunk Linear-LN-LeakyReLU x2, fused
   value/advantage first layers, dueling combination in fp32), then D2H of the 5 raw outputs.
4. **Policy + expand** (numba prange):
   - Regret matching is a bit-identical copy of the tail of `_strategy`, followed by upstream
     `_renormalize_strategy`.
   - Opponent nodes mix to (1-eps)*sigma + eps*uniform_legal, renormalise, and sample one action
     with the upstream `_sample_action` logic; the uniform is `uniform32(key)`.
   - Traverser nodes record (features, sigma, reach) and branch on every legal action, with reach
     multiplied by sigma[a] in fp32, as in the reference.
   - Children are produced with upstream `step`. A terminal child writes `float32(payoff[traverser])`
     into its parent's slot; a child that reaches a traverser node registers itself as that slot's
     child.
5. **Backup + order** (after the last round, in reverse batch order):
   - Each traverser node computes `value = sum_a sigma_a * v_a` (fp32, a = 0..4) and
     `adv_a = (v_a - value)/100` for legal a, pulling child values.
   - DFS pre-order and post-order positions come from the subtree sizes. Output is emitted
     **exactly in the reference order**: advantage samples in post-order, strategy samples in
     pre-order, traversals by id.
6. **CPU/GPU pipelining**: the traversals are split into `n_groups = 2` independent groups whose
   rounds are interleaved, so one group's GPU forward runs while the CPU encodes/expands the other.
   This hides the GPU entirely (exposed GPU wait is about 1 ms per iteration).
   Frontier/child arrays, pinned buffers and traverser-node storage persist across iterations, so
   there are no page faults in steady state.

Output: `res.as_worker_result()` is the 5-tuple `(features, advantages, strat_features,
strategies, strat_weights)` of float32 numpy arrays, unfiltered like upstream `run_traversals`.
Pass `[tuple]` to upstream `train._collect_worker_batches`, which applies the `weight > 1e-8`
strategy filter. `example_integration.py` runs this path for 3 iterations into 5M-capacity
`ReservoirBuffer`s and asserts the counts. Optionally `to_device="cuda:k"` also returns the
five arrays as torch tensors (`res.device_tensors`).

Integrator usage:
```python
from gpu_traverse import BatchedTraverser
import common
trav = BatchedTraverser(device="cuda:1", precision="highest", n_groups=2)   # once
params = common.params_from_torch(net)          # DuelingAdvantageNet -> 16 arrays
res = trav.run(params, traverser=(it - 1) % 6, n_traversals=30000,
               eps=_exploration_epsilon(it, 250, 0.1, 0.02, 250), seed=base_seed * 1000 + it)
_collect_worker_batches([res.as_worker_result()], buffers[traverser], strategy_buffer, it)
```

## Performance

Unit: wall seconds for one `run()` of 30,000 traversals (one iteration), released net, eps 0.02,
on qiaos-4a100-2 (4x A100-40GB, 48 vCPU). The first call compiles the kernels and is excluded.

| Configuration | mean | median / min | traversals/s (mean) |
|---|---|---|---|
| upstream baseline (46 CPU workers, numba fwd) | ~6 | | ~5,000 |
| v1: no pipelining, fresh buffers each round | 1.18 | 1.08 min | 25,400 |
| final, fp32 highest, 2 groups, 24 threads (`results/bench_v3_highest_g2.json`) | 0.522 | 0.489 / 0.485 | 57,500 |
| same, quiet machine, 48 threads (v2, `bench_final_highest_g2.json`) | 0.495 | 0.453 min | 60,700 |
| 1 group (no overlap) | 0.647 | 0.585 min | 46,400 |
| 3 groups | 0.525 | 0.481 min | 57,100 |
| TF32 forward, 2 groups | 0.523 | 0.483 min | 57,400 |
| numba threads 12 / 24 / 36 / 48 (v2) | 0.742 / 0.486 / 0.575 / 0.495 | | |
| partly trained net (iter 4, eps 0.0988), quiet reps | 0.47-0.53 | | |
| random-init net (1.1 M nodes, eps 0.1) | 0.161 | 0.141 min | 186,700 |
| with `to_device` upload of the samples | +0.12 | | |

**Time split** for the final configuration (mean per iteration, seconds; `bench_v3_highest_g2.json`):
- CPU encode (legal actions + `_encode_fast`): 0.150
- CPU policy + engine step (`policy_expand`): 0.147
- Prefix sums / child-buffer setup: 0.035
- Frontier compaction: 0.019
- Torch launch (CPU time to enqueue H2D + forward + D2H): 0.062
- **Exposed GPU wait: 0.001**
- Backup + DFS ordering: 0.051
- Output gather: 0.039
- Deal: 0.005
- Setup: 0.009

**GPU-side split** (synchronous profile mode, 1 group, CUDA events; `bench_final_profile.json`),
per iteration over 3.78 M rows: H2D 0.066 s, forward 0.157 s, D2H 0.006 s. Peak GPU memory is
0.84 GB with 2 groups (1.65 GB with 1 group).

TF32 gives no wall-clock gain because the GPU is already hidden behind the CPU work. **Production
should use fp32 'highest'**: it is as fast and closest to the reference.

## Validation

### (i) Exact equality with a shared deterministic random source

Setup: both sides get the same deals (`deal_hands`, seeded). Every opponent node draws
`uniform32(key)`, with `key` folded from (seed, traversal id, action history). The reference side
is `ref_det.py`, a test copy of the upstream recursion run in 46 processes; the batched side is
`gpu_traverse`. For each traversal, the comparison checks equal counts and bit-identical features
(a fork would show up there); it then compares every array on the aligned traversals.
Results are in `results/exact_*.json`; all runs use 30,000 traversals.

| Forward on batched side | Net (traverser, eps) | Adv samples | Result |
|---|---|---|---|
| upstream numba `_forward_dueling` | random init (0, 0.10) | 238,930 | **all 5 arrays bit-identical incl. order; per-traversal counts and per-street node counts equal** |
| upstream numba | iter-4 net (3, 0.0988) | 1,261,496 | bit-identical |
| upstream numba | released (5, 0.02) | 1,243,696 | bit-identical |
| upstream numba, 3 groups | released (1, 0.05) | 1,222,033 | bit-identical |
| upstream numba, final hybrid backup | released (4, 0.03) | 1,226,231 | bit-identical |
| negative control: batched side seed+1 | released, 5k traversals | 210,590 vs 211,795 | **fails as required**: 4,902 / 5,000 traversals diverge |

With the **GPU fp32 'highest'** forward (cuBLAS matmul and LayerNorm reductions round differently
from numba's BLAS gemv), on 30,000 traversals per net:

| Net | Diverged traversals | Features | Advantages: max abs diff (p99.99 per row) | Strategies: max abs diff | Reach weights: max abs diff | Strategy-filter flips | Bit-identical samples |
|---|---|---|---|---|---|---|---|
| random init | **0** | identical | 7.0e-4 (3.3e-4) | 3.8e-6 | 2.0e-6 | 0 | 80,469 / 238,930 |
| iter 4 | **0** | identical | 0.035 (5.8e-3) | 3.8e-4 | 3.9e-5 | 0 | 439,785 / 1,261,496 |
| released | **0** | identical | 0.0070 (1.0e-3) | 0.67 in 1 row; 130 rows > 1e-4 | 3.2e-5 | 0 | 488,641 / 1,243,696 |

Advantages are in units of 100 chips; their std is about 30. The single 0.67 strategy difference
is the regret-matching discontinuity: when every regret is <= 0 the strategy is uniform over legal
actions, and a 1e-7 sign flip of one regret makes it one-hot. Under TF32, 80 / 447 / 268
traversals diverge (0.3-1.5 %); under bf16, 1,069 / 6,492 / 4,475 (3.6-21.6 %).

### (ii) Statistical agreement with production RNG vs unmodified upstream

- Reference: upstream `traverse_numba.worker_run_jit`, unmodified, launched like
  `train._run_parallel_traversals` (46 spawn workers, seeds `iteration*1000 + w`).
- Instrumented copy: `ref_det.run_ref_np` with the same seeds was **bit-identical to upstream in
  every run** (22 reps x 46 workers), so its per-traversal and per-street counters describe the
  upstream runs exactly.
- Batched side: production mode (hash uniforms, parallel deals, 2 groups), different seeds.
- 30,000 traversals per replicate.
- 149 metrics: samples per traversal (mean, std, quantiles, zero fraction); opponent / traverser /
  terminal node counts per street; advantage mean and std per action; legal fraction; sigma mean
  and zero fraction per action; mean and std of the 46 non-mask features; strategy samples kept;
  reach-weight mean, log-mean and quantiles.
- Statistic: Welch t between replicate means (`results/stats_*.json`).

| Net (eps), reps per side | fp32 highest: max abs t / count with abs t > 3 | TF32 | bf16 |
|---|---|---|---|
| released (0.02), 6 | 3.51 / 2 (features 30 and 41 std) | 3.54 / 2 | **6.66 / 7** |
| released (0.02), 16 fresh seeds | **2.63 / 0** (features 30/41: t = 0.81 / 1.08) | n/a | n/a |
| iter 4 (0.0988), 6 | 3.36 / 1 | 3.27 / 1 | **6.63 / 1** |
| random init (0.10), 6 | 2.91 / 0 | 2.76 / 0 | 2.81 / 0 |

Headline values on the released net, 16 reps, reference vs batched:
- Advantage samples per traversal: 41.46 +- 0.22 vs 41.40 +- 0.17.
- Strategy samples kept per traversal: 6.245 vs 6.252.
- Mean advantage of CALL: -2.320 vs -2.288.
- Median kept reach weight: 0.0625 vs 0.0627.

No upstream worker hit its 60-samples-per-traversal output cap. bf16 shifts the
strategy statistics of trained nets (sigma mean and zero fraction for CALL / BETPOT, |t| about 6.6),
so **bf16 is not acceptable for a faithful run**; TF32 passes but has no speed benefit.

Invariance (`results/invariance.log`):
- Parallel deals are identical with 48 and 7 threads.
- Traversal results are identical with 48 and 13 threads.
- With the numba forward, 1 group and 3 groups give bit-identical results.
- With the GPU forward, 1 vs 2 groups differ only by cuBLAS batch-size-dependent rounding:
  0 diverged traversals, max strategy difference 1.3e-5.

## Caveats and deviations (read before integrating)

1. **Random streams differ from upstream; the distribution is the same.**
   - Opponent action uniforms come from a splitmix64 hash of (seed, traversal id, action history)
     instead of one MT19937 stream per worker.
   - Deals re-seed numba's MT19937 per traversal.
   - Same distributions (verified in (ii)), but a run is not seed-for-seed comparable with an
     upstream run.
2. **Upstream truncates at `n_traversals*60` samples per worker; this is not replicated.** The cap
   was never hit in 22 x 46 upstream worker runs.
3. **GPU results depend on batch composition.** cuBLAS picks kernels by row count, so outputs
   depend on `n_groups` at the 1e-6 level. They are deterministic for a fixed (net, seed, n_groups)
   and independent of the thread count.
4. **CPU contention.** The traversal uses all cores for about 0.3 s per iteration.
   - With another process loading the VM (load average 35-43), earlier reps took 1.3-1.9 s.
   - The fully parallel backup was hit hardest (0.7 s, from barrier waits); the final version backs
     up small batches sequentially and stays at about 0.05 s.
   - At load average 9-13 the final code measured 0.49 s median for the released net, but 0.77 s
     for the iter-4 net and 0.82 s with `to_device`. Those two were measured while the load was
     rising, so their encode/expand times are inflated.
   - Keep other CPU-heavy work (checkpoint compression, a CPU fit) off the box during traversal, or
     lower `n_threads`.
5. **Process-global side effects of importing.** `import gpu_traverse` inserts the upstream source
   into `sys.path`, sets `NUMBA_CACHE_DIR` (setdefault, to `gpu_trav/.numba_cache`) and calls
   `nb.set_num_threads(24)` in the constructor. If numba was imported before `common`, upstream
   `cache=True` functions use the upstream `__pycache__` instead.
6. **Unsupported options.** `randomize_stacks=False` is implemented (upstream `new_hand`) but not
   tested; n_players is fixed at 6.

## Not verified

- Integration into a full training loop with the fit-opt GPU buffers. Only the upstream
  `ReservoirBuffer` path was exercised (`example_integration.py`, 3 iterations, counts asserted).
- The contents of `to_device` tensors. Only the timing was measured; the code is a plain
  `torch.from_numpy(arr).to(dev)`.
- Behaviour when the traversal shares the CPU with the integrator's other phases. Only
  standalone timings were taken, some under unrelated external load.
- Nets from mid/late training other than the released net and the iteration-4 net.
