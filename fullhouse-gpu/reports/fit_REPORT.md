# fit-opt: GPU-resident reservoir buffers + fast Deep CFR network fitting

Sub-agent A of the Fullhouse 20-minute reproduction. Target: <= 0.2 ms per optimizer step for the
advantage-net fit (B = 8192, 10,000 steps) on one A100, faithful to upstream
`train_network` / `train_strategy_network` (as reimplemented in `fullhouse-repro/jax_backend.py`).
Baseline: 0.87 ms/step (JAX fp32 'highest'), 0.47 ms/step (JAX TF32).

## Result

| | ms / step | 10k-step adv fit | 3k-step strategy fit |
|---|---|---|---|
| JAX fp32 (baseline, plan.md) | 0.87 | 8.7 s | |
| JAX TF32 (baseline, plan.md) | 0.47 | 4.7 s | |
| **chosen: `backend='cuda'` (cuBLAS bf16 GEMMs + custom CUDA row kernels, one CUDA graph per 250 steps)** | **0.175 adv / 0.137 strat** | **1.75 s** (first fit incl. graph capture 2.16 s) | **0.41 s** (first 0.50 s) |

Measured end to end through the public API (`fitbench.py`; repeated after the traversal run ended:
1.749 s) on full-size buffers (6 x 5M-row
advantage buffers holding the realistic seat sizes 1.87M/2.37M/1.29M/1.42M/1.32M/1.38M, and one 5M-row
strategy buffer), A100-SXM4-40GB, GPU 0, while another agent was running traversal on GPUs 1-3
and all CPU cores. Sampling (seat choice + row indices + per-batch max iteration) is inside the graph.

## Public API (`gpu_fit.py`)

```python
import gpu_fit as gf
adv = gf.GPUReservoirSet(6, 5_000_000, 'cuda:0', seed=...)      # one per buffer family
strat = gf.GPUReservoirSet(1, 5_000_000, 'cuda:0', seed=...)
adv.add_batch(seat, features[n,51], advantages[n,5], iteration)  # numpy or torch, any device
strat.add_batch(0, features, strategy_targets, iteration, sample_weights=reach)
fa = gf.make_fitter('adv', adv)        # create ONCE; captures the CUDA graph on first fit
fs = gf.make_fitter('strat', strat)
res = gf.fit_advantage(fa, lr=gf.lr_schedule(it))   # 10k steps, fresh torch-default init
res.state_dict   # CPU fp32 tensors, torch module naming (DuelingAdvantageNet)
res.avg_loss     # mean training loss over all steps (upstream return value)
res.flat         # flat fp32 params on device (gf.export_state_dict(gf.ADV, flat) -> state_dict)
res = gf.fit_strategy(fs, lr=...)    # 3k steps, AverageStrategyNet
adv.state_dict(seat) / adv.load_state_dict(seat, d)   # upstream ReservoirBuffer keys (resume)
```

`make_fitter(kind, buffers, batch=8192, chunk=250, backend='cuda'|'cublas_triton', seed=0)`.
If the smallest occupied seat has fewer rows than the batch (very first iteration) the batch is capped
at that size exactly as `jax_backend` does, and an eager (non-graph) path is used for that fit.

## Faithfulness

* Fresh init every fit, matching torch defaults: nn.Linear weight and bias U(+-1/sqrt(fan_in)),
  LayerNorm gamma 1 / beta 0 (T1 compares moments/bounds against real torch modules and a negative
  control with a wrong bound).
* Adam b1 0.9, b2 0.999, eps 1e-8 (torch formula with bias correction), global grad-norm clip 1.0
  with torch `clip_grad_norm_` semantics (T3: 20 steps vs `torch.optim.Adam` + `clip_grad_norm_`,
  max abs diff 3.6e-7).
* Per minibatch: one seat chosen with probability proportional to its buffer size, rows uniform with
  replacement within it; empty seats are never chosen.
* Advantage targets normalised per sample over legal actions; weight (iter / max_iter_in_batch)^1.5,
  times stored reach weight for the strategy net; weighted MSE / masked cross-entropy;
  dueling combination and leaky-ReLU slope 0.01, LN eps 1e-5 as upstream.
* Reservoir `add_batch`: fill free slots in order, then the k-th further row draws
  r = floor(U * (total_seen + k)) and replaces slot r if r < capacity; when several rows of one batch
  hit the same slot the last one wins (numpy fancy-assignment order). T4 checks bit-exact equality with
  upstream `training/buffer.py::ReservoirBuffer.add_batch` given the same uniforms, including a
  forced-collision case. Random numbers come from a torch CUDA generator, not numpy's global RNG.

## Correctness validation

* **Loss/gradient vs `jax_backend.objective` (fp32 'highest') on identical batches** (`test_gpu_fit.py`):
  * fp32 manual forward/backward (`fwd_bwd_flat`) vs JAX: loss rel err <= 1e-7, gradient rel err
    1.5-1.8e-7 (adv and strategy, 3 trials each, incl. perturbed parameters);
  * T5, the actual fused training step of each backend vs JAX: `cuda` backend gradient rel err
    3.6e-3 (adv) / 4.1e-3 (strat), i.e. smaller than an honest torch bf16-autocast implementation
    on the same batch (5.1e-3 / 6.0e-3); loss rel err 0.6-1.1e-4; the update equals the fp32
    reference clip+Adam update to 1.2e-7. `cublas_triton` backend: 3.6e-3 / 5.2e-3, same update match.
  * negative controls (wrong leaky-ReLU slope, wrong init bound) are detected by the same thresholds.
* **Full fits on a realistic buffer** (`validate_fit.py`): 3 seeds each, the realistic buffer produced by the upstream
  JAX pipeline (`datagen.py`: 8 iterations, seat sizes 1.87M/2.37M/1.29M/1.42M/1.32M/1.38M, strategy
  buffer 1.72M rows), lr = `lr_schedule(9)` = 5.05e-4, both nets scored with the fp32 reference objective
  on the same 48 evaluation minibatches drawn with the training sampling protocol
  (`release_v2/validate_{adv,strat}_cuda.json`):

  | | GPU v2 (bf16, `cuda`) eval loss | JAX fp32 reference eval loss | GPU train avg loss | JAX train avg loss |
  |---|---|---|---|---|
  | advantage, 10k steps | 0.31311 / 0.31309 / 0.31137 (mean 0.31252) | 0.31289 / 0.31147 / 0.31197 (mean 0.31211) | 0.30985 / 0.30903 / 0.30957 (mean 0.30948) | 0.30897 / 0.31042 / 0.30901 (mean 0.30947) |
  | strategy, 3k steps | 0.07903 / 0.07914 / 0.07895 (mean 0.07904) | 0.07912 / 0.07908 / 0.07892 (mean 0.07904) | 0.08139 / 0.08142 / 0.08131 | 0.08139 / 0.08148 / 0.08130 |

  The GPU-vs-JAX difference of the means (adv eval +0.0004, train avg +0.00001; strategy 0.00000) is
  smaller than the seed-to-seed spread (adv eval range 0.0017 for GPU, 0.0014 for JAX), i.e. the final
  loss matches the reference within noise. (An earlier validation of the v1 kernels gave GPU adv eval
  0.31290 / 0.31301 vs JAX 0.31253 / 0.31149 — same picture.)

## Buffer add_batch cost (1.3M rows, full-size 5M-row buffers, A100)

| case | time |
|---|---|
| host numpy -> GPU, buffer not full (append) | 64 ms |
| host numpy -> GPU, buffer full, total_seen 25M (reservoir replacement) | 131 ms |
| GPU tensors, buffer full, total_seen 25M | 2.4 ms |
| GPU tensors, buffer full, total_seen 5M | 3.9 ms |
| strategy buffer with sample weights, host numpy | 64 ms |

The host-path times are consistent with a pageable host->device copy of ~300 MB (features + targets);
nothing is re-uploaded between iterations (buffers stay resident; 6x5M advantage rows = 7.7 GB, strategy 1.3 GB).

## Variants tried (advantage net, B = 8192, A100, ms per optimizer step)

| variant | ms/step | notes |
|---|---|---|
| torch nn.Module eager, fp32 | 5.14 | CPU-launch bound (CPUs shared with traversal) |
| torch nn.Module eager, TF32 | 3.06 | |
| torch nn.Module eager, bf16 autocast | 6.06 | autocast adds launches |
| manual fwd/bwd eager, fp32 / bf16 | 3.13 / 6.09 | |
| manual fwd/bwd, bf16, captured in a CUDA graph | 1.109 | |
| torch.compile (max-autotune-no-cudagraphs) + CUDA graph, bf16 | 0.448 | fp32 1.159; TF32 errored |
| Triton: 8 fused full-width kernels (GEMM + LN epilogues) (`triton_fit.py`) | 0.357-0.39 | 1 CTA/SM, latency bound |
| Triton row-fused megakernel (`triton_fit2.py`) | 0.37-0.40 | |
| cuBLAS GEMMs + Triton row kernels + side-stream wgrad (`triton_fit3.py`, backend `cublas_triton`) | 0.335 | |
| cuBLAS + CUDA row kernels, v1: per-block global atomics for small-param grads | 0.315 (full fit) | the version vendored by integrate for its first timed run |
| **cuBLAS + CUDA row kernels, v2: per-block partial sums with plain stores + reduction in the norm kernel** | **0.173-0.175** (graph) / **0.175** (full fit) | chosen (`backend='cuda'`), frozen in `release_v2/` |
| v2 + launch-config sweep (head 128 thr x 8 rows/warp, LN-bwd 256 x 4 x 1, LN-fwd 256 x 2) | 173.4-184.5 us across configs | best config is the v2 default |
| v3 = v2 + gather/prep of step k+1 on a side stream (double-buffered inputs) | 0.1749 (full fit, vs 0.1748) | no gain: the step is SM-throughput bound across the overlapping streams, not dependency bound; not adopted (kept in `experiments/prefetch_v3/`) |

Per-kernel profile of the chosen step (adv, torch profiler inside the graph, sum 232 us of kernel time
of which some overlaps on the side stream): cuBLAS GEMMs ~127 us (256x128-tile kernels ~57 us,
128x64-tile weight-gradient kernels ~34 us + their split-K reduce 15 us, two smaller GEMM kernels ~21 us),
head (output layer + loss + its backward) 30 us, 2 x LN backward 33 us, 2 x LN forward 16 us,
grad-norm/reduction 12 us, sample gather/prep 8 us, Adam 5.5 us.

Nsight Compute on the chosen kernels (adv): head kernel 31 us at 159 registers/thread, 18.75%
theoretical / 14% achieved occupancy (register + shared-memory limited, 0.79 waves); LN backward
14.3 us each at 27% achieved occupancy; grad-norm/reduction kernel 11.8 us (0.31 waves, 19% occupancy,
latency bound); LN forward 9.6-10.3 us; gather/prep 8.3 us. These (~85 us of row-kernel time, all far from
bandwidth limits) are the remaining headroom if more speed is ever needed (e.g. splitting the head kernel's
register-heavy output-layer gradient accumulation, more rows in flight per warp in LN backward,
a two-level S reduction); the GEMMs (~127 us incl. split-K reduce, partly overlapped) are cuBLAS-latency bound.

Key findings:
* The GEMMs are tiny ([8192,256] x [256,256] bf16 = 1 GFLOP) and latency bound (13-17 us each, ~70 TFLOP/s);
  the A100 itself is healthy (269 TFLOP/s on large GEMMs, 1.37 TB/s copy).
* Column reductions of small-parameter gradients (biases, LN gains, output layer) were the hidden cost:
  Triton `tl.sum(axis=0)` ~20 us per kernel, and contended global `atomicAdd` flushes made the CUDA row
  kernels scale with the number of blocks (head 54-211 us). Writing per-block partials to a [256, n_small]
  scratch with plain stores and reducing them in the (already needed) grad-norm kernel removed it: 0.315 -> 0.175 ms.
* Full-width fused GEMM+LN Triton kernels cannot fill the GPU at this size (1 CTA/SM, 12.5% occupancy).
* Data parallel across GPUs was not pursued: one step is ~0.17 ms of latency-bound work, so cross-GPU
  all-reduce per step would cost more than it saves; the other GPUs are busy with traversal anyway.

## Caveats

* Numerics: GEMM operands/outputs in bf16 (allowed by the plan), LayerNorm statistics, loss, output layer,
  gradients, Adam state and master weights in fp32. Gradient error vs fp32 JAX ~4e-3 relative (smaller than
  torch bf16 autocast). Final-loss agreement: see validation above.
* Random streams differ from upstream (torch CUDA generators instead of numpy / torch CPU RNG), so runs are
  statistically, not bitwise, equivalent.
* The CUDA extension (`rowk.cu`) is JIT-built with `torch.utils.cpp_extension.load` on first use
  (~1-2 min, needs nvcc + ninja), cached in `<module dir>/.runtime/torch_ext`; compiled for sm_80 only.
* `make_fitter` must be created once per buffer set and kept; the buffer tensor must not be reallocated
  (the graph holds its address). Graph capture happens on the first fit (~0.4 s extra).
* Kernel launch configs assume B = 8192 (other batch sizes work but use <= 256 blocks per row kernel;
  the eager fallback is used whenever the batch is capped).

## Files

Frozen, tested snapshot handed to the integrator: `~/work/poker-fast/fit_opt/release_v2/`
(sha256 prefixes gpu_fit.py 545fe00bcb1c9ffe, fast_step.py a1a39e73a0245dd6, rowk.cu 19bb14b8366c3099;
the copies in `fit_opt/` are identical). To vendor: copy `gpu_fit.py`, `fast_step.py`, `rowk.cu`
(plus `triton_fit.py`, `triton_fit3.py` only for `backend='cublas_triton'`) into one directory; have
`~/venvs/poker/bin` (ninja) and `/usr/local/cuda/bin` (nvcc) on PATH and `CUDA_HOME=/usr/local/cuda`;
build once before timing with `python -c "import fast_step; fast_step.ext()"` (output in
`<dir>/.runtime/torch_ext`; delete an older build dir from v1 first).

* `gpu_fit.py` — public API: layout, init, buffers, reference objective, fitter (required).
* `fast_step.py`, `rowk.cu` — chosen backend `cuda` (required).
* `triton_fit.py`, `triton_fit3.py` — optional backend `cublas_triton` (needs triton); `triton_fit2.py` bench only.
* `test_gpu_fit.py` — T1-T6 correctness tests; `fitbench.py` — end-to-end timing + add_batch cost;
  `validate_fit.py` — full fits vs JAX reference on the realistic buffer; `kprof3.py`, `sweep.py`,
  `bench.py` — per-kernel profiles, launch-config sweeps, variant benchmarks; `datagen.py` — realistic buffer.
