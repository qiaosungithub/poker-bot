# Fullhouse Deep CFR on 4xA100 in 16 (run 1) and 10 (run 2) minutes: integration, timed runs, six-player confirmation

Sub-agent C ("integrate"), 2026-11-08. VM `qiaos-4a100-2` (4x A100-SXM4-40GB, 48 vCPU),
code in `~/work/poker-fast/trainer/` (local git history in the workstation artifact dir
`integrate_work/trainer`, nothing pushed).

## Outcome

- **Run 1 (fit-opt snapshot v1):** the faithful 250-iteration training finished in
  **974.4 s for the 250-iteration loop** and **982.3 s = 16.4 min for the whole process**
  (process creation, imports, numba JIT and CUDA-graph warm-up included, up to the last
  checkpoint file). The original run took 2 h 05 min (TPU fit + 24 CPU workers); the original
  code on this VM projects to about 95 min.
- **Run 1 strength:** passes the predeclared criterion. The checkpoint chosen on the selection
  seeds is iteration 20. On the 1,024 confirmation blocks it scores **60.63**; the paired
  difference against the released model is **+2.04, 95% CI [+1.19, +2.90]** (criterion: lower
  bound > -2). The original reproduction scored 60.94, difference +2.35 [+1.51, +3.19]. The
  final checkpoint (250) scores 54.39, difference -4.20 [-5.09, -3.30]; the original's final
  checkpoint scored 54.42, difference -4.17 [-5.06, -3.28].
- **Run 2 (fit-opt release v2, same config and seed):** the loop took **612.4 s** and the whole process
  **619.9 s = 10.3 min**. Selection picks iteration 40. On the confirmation seeds it scores
  **63.35**, paired difference vs released **+4.76, 95% CI [+3.88, +5.65]** (criterion passed).
  Its final checkpoint scores 53.13, difference -5.46 [-6.34, -4.59].
- The two runs differ only in the fit kernels and therefore in floating-point rounding; with the
  same seeds they diverge after iteration 1 (iteration 1 is identical, 523,110 samples in both).
  Both are valid independent samples of the faithful training procedure. Both selected
  checkpoints pass the criterion and are at least as strong as the original reproduction's
  selected checkpoint (+2.35). Both final checkpoints are clearly weaker than the released model
  (-4.20 and -5.46), like the original's final checkpoint (-4.17).

## Timings

All times are wall-clock seconds measured on the VM. "Loop" = iterations 1..250, from the start
of iteration 1 to the end of iteration 250's checkpoint hand-off; the line "incl. final" adds
waiting for the background strategy fit and file writes of iteration 250. "Process" starts at
the creation of the Python process (read from `/proc`), so it includes interpreter start,
imports, buffer allocation and all warm-ups.

| Measurement | Run 1 (v1 fit) | Run 2 (v2 fit) |
|---|---:|---:|
| Process start -> loop start (imports, 9.2 GB buffer allocation, numba + graph warm-up) | 7.1 | 7.1 |
| 250-iteration loop | **974.4** | **612.4** |
| Loop incl. final strategy fit + writes | 975.2 | 612.9 |
| **Whole process, start -> last checkpoint file** | **982.3 (16.4 min)** | **619.9 (10.3 min)** |
| Saving the 9.2 GB replay buffers afterwards (outside the timed part) | 76.6 | 76.5 |

Per-iteration split, run 1 (seconds; `runs/full_metrics.jsonl`):

| Phase | Total | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Traversal, 30,000 roots (GPU forward on cuda:1, numba engine on 24 threads) | 158.2 | 0.633 | 0.602 | 0.830 | 1.558 |
| Reservoir insertion (host -> cuda:2, reservoir replacement) | 27.4 | 0.110 | 0.110 | 0.130 | 0.156 |
| Advantage fit, 10,000 steps, batch 8,192 (cuda:2) | 788.6 | 3.155 | 3.155 | 3.156 | 3.201 |
| Checkpoint hand-off (every 10 iterations; snapshot copy + queueing) | 0.1 | 0.001 | 0.000 | 0.006 | 0.006 |
| **Iteration** | 974.3 | **3.897** | 3.870 | 4.092 | 4.866 |

Inside the traversal (mean s/iteration): policy + engine step 0.208, encode 0.199, torch launch
0.059, backup 0.051, gather 0.041, prefix sums 0.037, compaction 0.021, exposed GPU wait 0.001.
The 25 strategy fits (3,000 steps each, on cuda:3) took 0.82-0.85 s each and ran in the
background; they never delayed an iteration (the final one drained in 0.83 s after the loop).
Checkpoint writes took 0.2 s in total on the writer thread.

Per-iteration split, run 2 (`runs/full_v2_metrics.jsonl`): traversal mean 0.594 s (median 0.590,
max 0.911), reservoir insertion 0.107, advantage fit **1.747** (median 1.747, max 1.767; 0.175 ms per
step), checkpoint hand-off 0.001; **iteration mean 2.449 s** (median 2.446, p95 2.601, max 2.796).
Strategy fits took about 0.45 s each in the background (11.3 s total). Run 2 produced 322,525,895
advantage samples (43.0 per traversal).

## What runs where

| Work | Where | Code |
|---|---|---|
| Traversal: level-synchronous batched external sampling, fp32 'highest' GPU forward | cuda:1 + 24 numba threads | sub-agent B, `../gpu_trav/gpu_traverse.py` (imported read-only) |
| Six 5M advantage reservoirs + 5M strategy reservoir, fp32 rows on the GPU | cuda:2 | sub-agent A, `GPUReservoirSet` |
| Advantage fit (fresh init, Adam, clip, 10k steps, one CUDA graph per 250 steps) | cuda:2 | sub-agent A, `GraphFitter(backend='cuda')`, frozen copy in `vendor_fit/` (run 1) and `vendor_fit_v2/` (run 2) |
| Strategy fit at checkpoint iterations, from a device-to-device snapshot of the strategy reservoir | cuda:3, background thread | `strategy_worker.py` |
| Checkpoint export in the upstream formats | writer thread | `ckpt.py` |
| GPU 0 | left to fit-opt | |

## Files

| File | Purpose |
|---|---|
| `train_fast.py` | The trainer (upstream `train_deep_cfr` loop). `./run.sh CONFIG` sets PATH/CUDA_HOME and records launch/exit epochs. |
| `fit_backend.py` | Thin adapter over fit-opt's `gpu_fit.make_fitter` / `GraphFitter.fit(lr, steps)`. |
| `strategy_worker.py` | Strategy-net fit on a spare GPU from a reservoir snapshot, overlapped with later iterations. |
| `ckpt.py` | Background writer; calls upstream `_export_advantage_npz` / `_export_strategy_npz`. |
| `paths.py` | sys.path + cache dirs; selects the vendored fit code (`FIT_VENDOR=vendor_fit_v2`). |
| `vendor_fit.sh`, `vendor_fit*/MANIFEST.sha256` | Frozen copies of fit-opt's files with their own prebuilt CUDA extension. |
| `check_run.py` | Verifies checkpoints: keys/shapes/dtypes equal to the released NPZ, play through upstream `DeepCFRLookup`, `.pt` == NPZ; includes a negative control. |
| `summarize_run.py` | Time split and sample statistics from `runs/<run>_metrics.jsonl`. |
| `configs/{smoke,timing10,vcheck,vcheck_v2,full,full_v2}.json` | Run configurations. |
| `evalkit/` | Unmodified copies of the frozen benchmark scripts, `run_eval.py` (protocol driver = `supervise.py` logic), results in `eval_full/` and `eval_full_v2/`. |
| `runs/full{,_v2}/` | Strategy NPZ + `.pt` per checkpoint, `latest.pt` with replay buffers; `runs/full{,_v2}_adv/` advantage NPZs. |
| `logs/` | `full.log`, `full_v2.log`, `eval_full.log`, `eval_full_v2.log`, smoke/timing logs. |

## Faithfulness

Kept exactly as in upstream `train_deep_cfr` + `training_changes.patch` with `config_main.json`:
250 iterations; traverser (t-1) mod 6; 30,000 root traversals; eps = `_exploration_epsilon(t, 250,
0.10, 0.02, 250)`; randomized stacks, random dealer; regret matching + `_renormalize_strategy`;
opponents sample (1-eps)·sigma + eps·uniform; traverser explores all legal actions; advantage
targets (v_a - v)/100; strategy samples kept if reach > 1e-8; reservoirs 6 x 5,000,000 +
5,000,000 with `add_batch` semantics; a new torch-default-initialised DuelingAdvantageNet and a
new Adam each iteration; lr 1e-4 + 9e-4·t/20 for t <= 20, else 1e-3; 10,000 steps of batch 8,192;
fit only if the reservoirs hold >= 2·batch rows; one seat per minibatch chosen proportional to
its size; per-sample advantage normalisation; weight (it/max it)^1.5; MSE; clip 1.0; strategy net
fresh, 3,000 steps, same lr, CE weighted by reach; checkpoints every 10 iterations; advantage NPZ
in `_export_advantage_npz` format (the deployed bot uses it). The iteration-1 traversal uses the
same initial network as the original run (`torch.manual_seed(20261107)` then
`DuelingAdvantageNet()`); iteration 1 therefore produced 523,110 advantage / 160,667 strategy
samples and loss 0.2533 vs 523,780 / 161,949 / 0.2526 in the original code on this VM.

**Deviations from upstream (all of them):**

1. **Random streams.** The run is not seed-for-seed comparable with an upstream run; all
   distributions are the same.
   - Opponent action uniforms come from a splitmix64 hash of (seed, traversal id, action
     history) instead of one numpy MT19937 stream per worker. Deals re-seed numba's MT19937 per
     traversal. Traversal seed = 20261107·1000 + t; upstream uses t·1000 + worker index.
   - Reservoir replacement draws come from torch CUDA generators (seeds 20261108, 20261109)
     instead of numpy's global RNG.
   - Network initialisation and minibatch sampling come from the fitters' CUDA generators (seeds
     20261110 adv, 20261111 strategy) instead of torch's global CPU RNG (init) and a JAX PRNGKey
     drawn from numpy (sampling). Row ids are `randint(2^62) mod size` (bias ~1e-11).
2. **Traversal implementation.** Level-synchronous batched traversal on the GPU (sub-agent B)
   instead of per-worker recursion with a numba CPU forward. With a shared deterministic random
   source it is bit-identical to the reference recursion when using the numba forward; with the
   fp32 cuBLAS forward used here, 0 of 30,000 traversals diverged in B's tests (rounding only).
   Upstream's per-worker cap of n_traversals·60 samples is not replicated (it was never hit).
3. **One `add_batch` per iteration** instead of one per worker result (24 in the original run).
   Sequential reservoir semantics make this equivalent.
4. **Fit numerics.** fit-opt's fused CUDA step: GEMM operands and linear outputs in bf16,
   LayerNorm / loss / output layer / weight-gradient accumulation and Adam in fp32. This matches
   upstream's own CUDA path (bf16 autocast). The original reproduction used the JAX fp32
   'highest' backend. fit-opt's validation on a real 8-iteration buffer: advantage eval loss
   0.3129-0.3131 vs JAX fp32 0.3115-0.3125.
5. **Strategy net only at checkpoint iterations** (every 10th and 250), trained on cuda:3 from a
   snapshot of the strategy reservoir taken at that iteration, with that iteration's lr, while the
   next iterations run. Upstream trains it every iteration, but it is re-initialised every time and
   feeds neither the traversal nor the deployed advantage-net bot, so the exported nets have the
   same distribution. Consequence: the `strat_loss` printed in the per-iteration log is the loss of
   the most recent checkpoint fit, tagged `(itN)`.
6. **Checkpoints.** No 9 GB replay-buffer file every 10 iterations (upstream with
   `save_resume_buffers=true` rewrites `latest.pt` with all buffers at every checkpoint). The
   buffers are saved once after the timed run, in upstream `state_dict` format (advantage buffers
   without `sample_weights`, as upstream). The per-checkpoint `.pt` holds both state dicts,
   config, seeds and RNG states (numpy, torch, and the four CUDA generators). The trainer has no
   `--resume` path; resuming from these files is not implemented or tested.
7. **Metadata only:** `.pt` `config.deployment` says `advantage_npz` (upstream writes
   `average_strategy`); the log line has extra timing fields.

Training statistics agree with the original run: 324,950,178 advantage samples over 250
iterations (original: 325,242,386), i.e. 43.3 per traversal; the reservoirs are full from about
iteration 22 on; the final advantage loss is 0.164.

## Six-player benchmark

Protocol (from `docs/research/fullhouse.md`, `benchmarks/six_player_2026-11-06/protocol.txt`
and `fullhouse-repro/supervise.py`): only the Fullhouse slot changes; opponents cfr_lite64,
hydra32, solver_like, maniac, station; 100 BB stacks, 50/100 blinds, up to 200 hands, payouts
[100, 60, 40, 25, 10, 0] with exact survivor ICM. A block = one seeded roster permutation x 6
cyclic seat rotations; the block mean is the statistical unit. Selection: 256 blocks from seed
11,300,000, every 10th advantage NPZ, best mean selected. Confirmation: 1,024 blocks from seed
12,300,000 for the selected and the final checkpoint, plus the reference on the same seeds. Paired
statistic: per-block candidate minus reference; 95% CI = mean ± 1.96·sd/sqrt(blocks). Criterion:
lower bound > -2 points. Each candidate NPZ is exported to FP32 ONNX and validated against the
upstream NumPy bot on 1,024 retained states (`export_validate.py`; max abs error over run 1's 25
exports 1.1e-4 and over run 2's 25 exports 1.2e-5; limit 1e-3).

**Reference.** `evaluate.py` without `--onnx` uses `latency-survey/fullhouse_fp32.onnx`, the
FP32 ONNX export of the released `deep_cfr_model.npz`. I regenerated it from the released NPZ
with `fullhouse_onnx.export`; its SHA256 equals the frozen value `7b52be99…`.

**Pipeline check.** Environment: separate venv with Python 3.12.3 (original: 3.11.0rc1), numpy
2.2.6, onnx 1.23.2, onnxruntime 1.23.2, eval7 0.1.11, pokerkit 0.7.7. The benchmark scripts are
byte-identical to the frozen ones (multiway_benchmark.py `43030570…`, low_latency_compare.py
`37141023…`, run_benchmark.py `4194d5f0…`); the bot is a `git archive` of upstream `e504793d`.
- The reference on the selection seeds scored **59.18956468908845**. The original's selection
  reference, derived as 60.769180244512405 - 1.5796155554239597, is 59.189564689088445.
- Pairing my reference blocks with the ORIGINAL run's iteration-210 selection blocks reproduces
  the original paired result exactly: +1.5796155554239597, se 0.8882248950830001.
- The reference on the confirmation seeds scored 58.5894 (original: 60.9397 - 2.3504 = 58.5894).

So the evaluation reproduces the original numbers to floating-point precision.

### Run 1 selection (256 blocks, seed 11,300,000; reference 59.19)

| Iteration | Run 1 mean | Run 1 paired diff [95% CI] | Original run mean | Original diff |
|---:|---:|---|---:|---:|
| 10 | 50.77 | -8.42 [-10.28, -6.57] | 54.66 | -4.53 |
| 20 | **61.54** | **+2.35 [+0.52, +4.18]** | 57.27 | -1.92 |
| 30 | 56.26 | -2.93 [-4.81, -1.05] | 51.94 | -7.25 |
| 40 | 56.39 | -2.80 [-4.60, -0.99] | 56.49 | -2.70 |
| 50 | 55.31 | -3.88 [-5.82, -1.94] | 59.00 | -0.19 |
| 60 | 55.79 | -3.40 [-5.23, -1.57] | 53.89 | -5.30 |
| 70 | 56.73 | -2.46 [-4.21, -0.71] | 50.89 | -8.30 |
| 80 | 57.00 | -2.19 [-4.07, -0.30] | 58.97 | -0.22 |
| 90 | 52.60 | -6.59 [-8.52, -4.67] | 57.36 | -1.83 |
| 100 | 57.33 | -1.86 [-3.67, -0.06] | 59.59 | +0.40 |
| 110 | 52.68 | -6.51 [-8.29, -4.73] | 52.91 | -6.28 |
| 120 | 55.40 | -3.79 [-5.77, -1.81] | 56.66 | -2.53 |
| 130 | 56.27 | -2.92 [-4.71, -1.13] | 57.75 | -1.44 |
| 140 | 57.89 | -1.30 [-3.12, +0.53] | 60.48 | +1.29 |
| 150 | 53.84 | -5.35 [-7.22, -3.48] | 56.93 | -2.26 |
| 160 | 58.91 | -0.28 [-1.97, +1.42] | 54.63 | -4.56 |
| 170 | 52.81 | -6.38 [-8.31, -4.45] | 51.11 | -8.08 |
| 180 | 52.39 | -6.79 [-8.60, -4.99] | 56.58 | -2.61 |
| 190 | 54.81 | -4.38 [-6.20, -2.56] | 55.85 | -3.34 |
| 200 | 56.42 | -2.77 [-4.50, -1.05] | 57.03 | -2.16 |
| 210 | 53.04 | -6.15 [-7.89, -4.42] | **60.77** | **+1.58** |
| 220 | 60.08 | +0.89 [-0.90, +2.68] | 51.66 | -7.53 |
| 230 | 54.32 | -4.86 [-6.67, -3.06] | 53.51 | -5.68 |
| 240 | 59.62 | +0.43 [-1.35, +2.21] | 52.90 | -6.29 |
| 250 | 55.58 | -3.61 [-5.50, -1.71] | 54.44 | -4.75 |
| mean of 25 | 55.75 | | 55.73 | |

Strength fluctuates strongly between checkpoints in both runs. The mean over the 25 checkpoints
is 55.75 here and 55.73 in the original. Which iteration is best differs (20 here, 210 there),
as expected for an unstable run with different random streams.

### Run 2 selection (256 blocks, seed 11,300,000; reference 59.19)

| Iteration | Run 2 mean | Run 2 paired diff [95% CI] |
|---:|---:|---|
| 10 | 53.56 | -5.63 [-7.53, -3.73] |
| 20 | 62.50 | +3.31 [+1.46, +5.17] |
| 30 | 51.30 | -7.89 [-9.63, -6.14] |
| 40 | **62.92** | **+3.73 [+1.93, +5.53]** |
| 50 | 57.52 | -1.67 [-3.61, +0.27] |
| 60 | 57.37 | -1.82 [-3.56, -0.07] |
| 70 | 57.92 | -1.27 [-3.16, +0.62] |
| 80 | 57.68 | -1.51 [-3.30, +0.28] |
| 90 | 55.24 | -3.95 [-5.66, -2.23] |
| 100 | 54.95 | -4.24 [-6.07, -2.40] |
| 110 | 59.89 | +0.70 [-1.17, +2.57] |
| 120 | 59.21 | +0.02 [-1.84, +1.89] |
| 130 | 60.39 | +1.21 [-0.45, +2.86] |
| 140 | 56.52 | -2.67 [-4.42, -0.92] |
| 150 | 48.93 | -10.26 [-11.95, -8.58] |
| 160 | 57.54 | -1.65 [-3.50, +0.20] |
| 170 | 55.85 | -3.34 [-5.12, -1.57] |
| 180 | 48.92 | -10.27 [-12.07, -8.48] |
| 190 | 53.33 | -5.86 [-7.59, -4.13] |
| 200 | 57.71 | -1.48 [-3.24, +0.28] |
| 210 | 55.80 | -3.39 [-5.03, -1.74] |
| 220 | 55.11 | -4.08 [-5.88, -2.28] |
| 230 | 54.72 | -4.47 [-6.23, -2.71] |
| 240 | 56.13 | -3.06 [-4.89, -1.23] |
| 250 | 53.28 | -5.91 [-7.64, -4.18] |
| mean of 25 | 56.17 | |

Run 2 is just as unstable: 48.9-62.9 across checkpoints, mean 56.17 (run 1 55.75, original 55.73).
Its best selection checkpoints are 40 (62.92) and 20 (62.50).

### Confirmation (1,024 blocks, seed 12,300,000; reference 58.59 in every row)

| Model | Mean [95% CI] | Paired diff vs released [95% CI] | Lower bound > -2? |
|---|---|---|---|
| Released Fullhouse (reference) | 58.59 [57.87, 59.31] | — | — |
| **Run 1, selected iteration 20** | **60.63 [59.94, 61.33]** | **+2.04 [+1.19, +2.90]** | **yes** |
| Run 1, final iteration 250 | 54.39 [53.64, 55.14] | -4.20 [-5.09, -3.30] | no (not the criterion) |
| **Run 2, selected iteration 40** | **63.35 [62.67, 64.04]** | **+4.76 [+3.88, +5.65]** | **yes** |
| Run 2, final iteration 250 | 53.13 [52.44, 53.82] | -5.46 [-6.34, -4.59] | no (not the criterion) |
| Original reproduction, selected iteration 210 | 60.94 [60.23, 61.65] | +2.35 [+1.51, +3.19] | yes |
| Original reproduction, final iteration 250 | 54.42 [53.68, 55.16] | -4.17 [-5.06, -3.28] | no |

Run 1's selected checkpoint and the original's selected checkpoint differ by 0.31 points in
paired difference. Their CIs overlap almost completely. Their final checkpoints differ by 0.03
points. Run 2's selected checkpoint is stronger (+4.76). Its CI does not overlap the original's
(+1.51..+3.19), so this is a real difference between checkpoints, not noise in the evaluation. It
is consistent with the large checkpoint-to-checkpoint spread seen in all three runs, and with the
fact that the best of 25 noisy checkpoints is a selected maximum. Confirmation runs had zero
runtime errors: run 1 had 851,395 hands for iteration 20 and 790,191 for iteration 250; run 2 had
909,308 and 786,571. Each confirmation arena took 98-121 s with 46 workers. Run 2's evaluation
recomputed the reference arenas, and their raw per-block records are byte-identical to run 1's,
so the evaluation is deterministic.

## Verification

- Smoke run (4 iterations, 2,000 traversals, 5,000-row reservoirs so reservoir replacement is
  exercised): end to end with exports; `check_run.py` passed. It checks keys, shapes and dtypes
  against the released NPZ, play through `DeepCFRLookup` on 256 corpus states, and `.pt` == NPZ
  bit for bit, with a negative control that must fail.
- `check_run.py` also passed on the vcheck runs for both fit versions, and `export_validate.py`
  passed for all 50 NPZs of runs 1 and 2.
- Full-scale 10-iteration timing run before the real run: 3.74-4.10 s per iteration, the same
  as in the real run.
- The evaluation pipeline reproduces the original reference and paired numbers exactly (above).

## Not verified / caveats

- The exact per-node random streams differ from upstream (deviation 1). Equivalence is
  statistical and comes from sub-agent B's validation and the matching training statistics,
  not from bit equality.
- Resume from the saved `latest.pt` was not implemented or tested.
- Process timing excludes the one-off CUDA-extension build (about 2 min, done by
  `vendor_fit.sh` before the run) and Python bytecode/numba/inductor caches that already existed
  on disk. A first run on a completely fresh machine would add those compile times. The numba
  traversal cache was warm; its JIT warm-up inside the timed process took 0.5 s.
- The evaluation uses Python 3.12.3 instead of 3.11.0rc1. The reference numbers match exactly,
  so this does not affect results.
- The selections pick iterations 20 (run 1) and 40 (run 2), early checkpoints. The confirmation seeds are disjoint from
  the selection seeds, so the confirmation results (+2.04, +4.76) are unbiased estimates for
  those checkpoints. It does not mean training past iteration 20 is useless in general: as in the
  original run, strength varies a lot between neighbouring checkpoints.
