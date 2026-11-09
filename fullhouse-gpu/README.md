# fullhouse-gpu: the Fullhouse Deep CFR reproduction on 4x A100

This package trains the upstream Fullhouse Deep CFR bot
([advitrocks9/fullhouse-bot](https://github.com/advitrocks9/fullhouse-bot) @
`e504793d`, plus `../fullhouse-repro/training_changes.patch`) with the **original
configuration** (250 iterations x 30,000 external-sampling traversals, six 5M
advantage reservoirs + a 5M strategy reservoir, 10,000 advantage-fit steps and
3,000 strategy-fit steps of batch 8,192 per iteration) on one 4x A100-40GB machine.
All neural-network work runs on the GPUs. The whole 250-iteration training
process takes **10.3 minutes** (run 2); the original TPU reproduction took
2 h 5 min.

It replaces the original TPU/JAX training path, which is removed on this branch
(see `main` at commit `7356b42` for the TPU launcher and its provenance). The
TPU run's measured results stay in `../fullhouse-repro/` as the reference these
runs are compared against.

Research reproduction only: the November 8 organizer email forbids pre-written
bots for the event (`../docs/competition/rules.md`).

## Layout

| Path | Content |
|---|---|
| `traversal/` | `gpu_traverse.py`: level-synchronous batched external-sampling traversal. All 30k root traversals expand one tree level at a time; each level is one batched fp32 GPU forward, while the upstream numba engine / feature encoder / regret matching run on the CPU cores. `det_rng.py`: counter-based uniforms. `common.py`: locates the upstream source, loads LUTs, converts nets. |
| `fit/` | `gpu_fit.py`: GPU-resident reservoir buffers (`GPUReservoirSet`, upstream `add_batch` semantics) and the CUDA-graph fitter. `fast_step.py` + `rowk.cu`: cuBLAS bf16 GEMMs + custom CUDA row kernels (backend `cuda`, the one used). `triton_fit*.py`: optional `cublas_triton` backend. Byte-identical to the code that trained run 2 (`MANIFEST.sha256`); its docstrings still use development names (`fit_opt/`, `jax_backend` = `tests/fit/jax_reference.py`). |
| `trainer/` | `train_fast.py`: the training loop (upstream `train_deep_cfr`). `strategy_worker.py`: strategy-net fits on a spare GPU. `ckpt.py`: background checkpoint writer in upstream formats. `check_run.py`: checkpoint verifier. `summarize_run.py`: time split from metrics. `run.sh`: launcher. `configs/`: run configurations. `paths.py`: import paths and caches. |
| `eval/run_eval.py` | Six-player selection / confirmation protocol driver over `../fullhouse-repro/export_validate.py` and `../fullhouse-repro/evaluate.py`. |
| `tests/traversal/` | Exact-equality test vs a test copy of the recursive reference (`ref_det.py`, `test_exact.py`), invariance, statistical comparison vs unmodified upstream, benchmarks, suite scripts. |
| `tests/fit/` | `test_gpu_fit.py` (T1-T6), `fitbench.py`, `validate_fit.py`, and `jax_reference.py`: the original run's fp32 JAX fitting code, kept only as a test oracle. |
| `reports/` | The three engineering reports: `trainer_REPORT.md` (integration, timed runs, benchmark), `traversal_REPORT.md`, `fit_REPORT.md`. |
| `results/` | Per-iteration metrics, run summaries, checkpoint SHA256 manifests, eval outputs, traversal and fit validation records. See `results/README.md`. |

## Environment

Measured on GCP VM `qiaos-4a100-2` (4x A100-SXM4-40GB, 48 vCPU / 24 cores,
334 GB RAM), training environment `~/venvs/poker`:

- Python 3.12.3, torch 2.11.0+cu128, numba 0.68.1, numpy 2.2.6, and the
  upstream training dependencies (eval7 etc.); the Triton backend uses the
  triton bundled with torch.
- CUDA toolkit with `nvcc` at `$CUDA_HOME` (default `/usr/local/cuda`) and
  `ninja` on `PATH`: `fit/rowk.cu` is JIT-compiled with
  `torch.utils.cpp_extension.load` for `sm_80`. It builds **once**
  into `fit/.runtime/torch_ext` (about 2 min on an idle machine; 5.8 min in the
  branch check while another job loaded all CPUs); later runs load it.
- 4 GPUs: the configs use `cuda:1` for traversal, `cuda:2` for the reservoirs
  and the advantage fit, `cuda:3` for strategy fits (`devices` in the config).
  The replay reservoirs take 9.2 GB on the fit GPU.
- JAX is needed only for the oracle tests (`tests/fit/test_gpu_fit.py` T2/T5,
  `validate_fit.py`); the trainer never imports it.

The upstream source must exist at `../fullhouse-repro/source` (reconstruct it
from the pinned submodule + patch as in `../docs/repository.md`, or clone
upstream at `e504793d` there and `git apply ../training_changes.patch`), or set
`FULLHOUSE_SOURCE`. Environment variables and defaults:

| Variable | Default | Used by |
|---|---|---|
| `FULLHOUSE_SOURCE` | `<repo>/fullhouse-repro/source` | trainer, traversal, tests |
| `FULLHOUSE_GPU_RUNTIME` | `<repo>/fullhouse-gpu/.runtime` | numba / inductor / triton caches |
| `PYTHON` | `$HOME/venvs/poker/bin/python` | `trainer/run.sh`, `fit/build_ext.sh`, test suites |
| `CUDA_HOME` | `/usr/local/cuda` | extension build |
| `EVAL_PY` | the current python | `eval/run_eval.py` (the pinned eval environment) |
| `FULLHOUSE_REPRO` | `<repo>/fullhouse-repro` | `eval/run_eval.py` |

## Train

```bash
cd fullhouse-gpu
fit/build_ext.sh                                   # once: checks MANIFEST, builds the extension
cd trainer
./run.sh configs/smoke.json --run-name my-smoke    # 4 iterations, 2k traversals, 5k-row reservoirs
python check_run.py runs/my-smoke                  # formats, bot playability, .pt == NPZ (+ negative control)
nohup ./run.sh configs/full.json --run-name my-full > my-full.log 2>&1 &   # the faithful 250 iterations
python summarize_run.py runs/my-full               # time split / samples
```

Outputs go to `trainer/runs/<name>/` (ignored by Git; a run directory must be
new or empty): every 10 iterations an advantage NPZ in upstream
`_export_advantage_npz` format at `runs/<name>_adv/iter_XXXXX.npz` (**the
deployed bot uses this file**), a strategy NPZ and a `.pt` with both state dicts,
config, seeds and RNG states; `runs/<name>_metrics.jsonl` and
`runs/<name>_summary.json`; with `save_final_buffers` (full configs) a 9.2 GB
`latest.pt` with the replay buffers, written after the timed part.

## Evaluate

The protocol is the original one (`../benchmarks/six_player_2026-11-06/protocol.txt`,
`../docs/research/fullhouse.md`): only the Fullhouse slot changes; opponents
`cfr_lite64`, `hydra32`, `solver_like`, `maniac`, `station`; 100 BB stacks,
50/100 blinds, up to 200 hands, payouts `[100,60,40,25,10,0]` with exact
survivor ICM. A block = one seeded roster permutation x 6 seat rotations; the
block mean is the unit. Selection: 256 blocks from seed 11,300,000 for every
10th advantage NPZ, best mean selected. Confirmation: 1,024 blocks from seed
12,300,000 for the selected and the final checkpoint, plus the released model
on the same seeds. Criterion: paired 95% lower bound (candidate - released)
> -2 points.

Requirements: the `latency-survey/fullhouse-bot` submodule
(`git submodule update --init latency-survey/fullhouse-bot`), the reference
ONNX `latency-survey/fullhouse_fp32.onnx` (not in Git; regenerate it from the
released NPZ and check SHA256 `7b52be99fbb20806669cba91deac58e3f0a03ec8a2169720e273ac08a67e1ef1`):

```bash
cd latency-survey && $EVAL_PY -c "from fullhouse_onnx import export; export('fullhouse-bot/data/deep_cfr_model.npz', 'fullhouse_fp32.onnx')"
sha256sum fullhouse_fp32.onnx
```

and an eval environment with numpy 2.2.6, onnx 1.23.2, onnxruntime 1.23.2,
eval7 0.1.11, pokerkit 0.7.7 (the runs used Python 3.12.3; the original used
3.11.0rc1 and the reference numbers agree exactly). Then:

```bash
export EVAL_PY=/path/to/eval-venv/bin/python
$EVAL_PY fullhouse-gpu/eval/run_eval.py \
  protocol fullhouse-gpu/trainer/runs/my-full_adv /new/output/dir --workers 46
```

It exports and validates each NPZ as FP32 ONNX (1,024 retained states, max abs
error < 1e-3), runs the reference and candidate arenas, appends to
`results.jsonl` and writes `status.json`. A confirmation arena takes about
100-120 s with 46 workers.

## Tests

Each suite needs the training environment (`PYTHON`, default
`$HOME/venvs/poker/bin/python`) and the upstream source; new outputs go to
`.runtime/test-results/`, never into the committed `results/`.

```bash
tests/fit/run_fit_tests.sh            # fit T1-T6: init, loss/grad vs the fp32 JAX oracle, clip+Adam,
                                      # reservoir add_batch == upstream, fused step, graph fitter
python tests/traversal/test_exact.py --n 2000 --net random:0 --forward numba --workers 8 --device cuda:1
python tests/traversal/test_exact.py --n 2000 --net ../fullhouse-repro/source/data/deep_cfr_model.npz \
  --forward torch --precision highest --traverser 5 --eps 0.02 --seed 1003 --workers 8 --device cuda:1
python tests/traversal/test_exact.py ... --negative-control     # must report diverged traversals
tests/traversal/run_exact_suite.sh    # the full 30k-traversal suites behind reports/traversal_REPORT.md
tests/traversal/run_stats_suite.sh    # (set ITER4_NPZ to include the partly trained net)
tests/traversal/run_bench_suite.sh
FITOPT_BUFFER=trainer/runs/<run>/latest.pt python tests/fit/validate_fit.py adv 3   # full fits vs JAX
python tests/fit/fitbench.py          # end-to-end fit timing on full-size buffers
```

`test_exact.py` prints (and with `--json` writes) a verdict; it does not set the
exit code, so read it: with `--forward numba` `exact_all` must be `true` (every
sample array bit-identical to the recursive reference, including order), with
`--forward torch` `n_diverged_traversals` should be 0 and `tolerance_on_aligned`
gives the float differences; the negative control must show diverged traversals.
`run_fit_tests.sh` prints `[PASS]`/`[FAIL]` per check.

## Measured timings (qiaos-4a100-2, 2026-11-08)

Wall-clock seconds. "Process" starts at the creation of the Python process
(from `/proc`) and ends when the last checkpoint file is written: it includes
interpreter start, imports, the 9.2 GB buffer allocation and all warm-ups, but
not the one-off extension build or caches already on disk (see caveats).

| Measurement | Run 1 (fit v1) | Run 2 (fit v2, `fit/`) |
|---|---:|---:|
| Process start -> loop start | 7.1 | 7.1 |
| 250-iteration loop | 974.4 | 612.4 |
| **Whole process, start -> last checkpoint file** | **982.3 s = 16.4 min** | **619.9 s = 10.3 min** |
| Saving the replay buffers afterwards (untimed) | 76.6 | 76.5 |

Per iteration (mean / median / max, seconds; `results/runs/*_metrics.jsonl`):

| Phase | Run 1 | Run 2 |
|---|---|---|
| Traversal, 30,000 roots (GPU forward on cuda:1 + 24 numba threads) | 0.633 / 0.602 / 1.558 | 0.594 / 0.590 / 0.911 |
| Reservoir insertion (host -> cuda:2) | 0.110 / 0.110 / 0.156 | 0.107 / 0.107 / 0.137 |
| Advantage fit, 10,000 steps x 8,192 | 3.155 / 3.155 / 3.201 (0.32 ms/step) | 1.747 / 1.747 / 1.767 (0.175 ms/step) |
| Checkpoint hand-off | 0.001 | 0.001 |
| **Iteration** | **3.897** / 3.870 / 4.866 | **2.449** / 2.446 / 2.796 |

Strategy fits (3,000 steps, every 10th iteration, on cuda:3 in the background)
took about 0.83 s (run 1) and 0.45 s (run 2) each and never delayed an iteration.
For comparison, the original code on this VM (46 CPU workers, JAX fp32 fit)
measured 22-24 s per iteration (about 95 min projected); the original TPU run
logged 23.05 s median per iteration and 2 h 5 min 23 s in total.

## Results (six-player benchmark, custom research points)

Confirmation, 1,024 blocks from seed 12,300,000; the released model scores
58.59 [57.87, 59.31] on these seeds. Paired difference = candidate - released.

| Model | Mean [95% CI] | Paired difference [95% CI] | Criterion (lower bound > -2) |
|---|---|---|---|
| **Run 2, selected iteration 40** | **63.35 [62.67, 64.04]** | **+4.76 [+3.88, +5.65]** | passed |
| Run 2, final iteration 250 | 53.13 [52.44, 53.82] | -5.46 [-6.34, -4.59] | not the criterion |
| **Run 1, selected iteration 20** | **60.63 [59.94, 61.33]** | **+2.04 [+1.19, +2.90]** | passed |
| Run 1, final iteration 250 | 54.39 [53.64, 55.14] | -4.20 [-5.09, -3.30] | not the criterion |
| Original TPU reproduction, selected iteration 210 | 60.94 [60.23, 61.65] | +2.35 [+1.51, +3.19] | passed |
| Original TPU reproduction, final iteration 250 | 54.42 [53.68, 55.16] | -4.17 [-5.06, -3.28] | not the criterion |

Selection (256 blocks from 11,300,000; released model 59.19): the 25
checkpoints range 50.8-61.5 (mean 55.75) in run 1, 48.9-62.9 (mean 56.17) in
run 2 and 50.9-60.8 (mean 55.73) in the original run. Strength fluctuates far
more between neighbouring checkpoints than the per-checkpoint CI (about +-1.4),
the best iteration differs between runs (20, 40, 210), and the final checkpoint
is weaker than the released model in all three runs. The selected confirmation
results are unbiased for those checkpoints (disjoint seeds), but each is the best
of 25 noisy checkpoints. Full tables: `reports/trainer_REPORT.md`; raw records:
`results/eval/`.

The evaluation reproduces the original numbers to floating-point precision:
the reference scores 59.18956 (selection) and 58.5894 (confirmation) exactly as
in the original run, and pairing these reference blocks with the original
iteration-210 selection blocks reproduces its paired result (+1.57962) exactly.

Training statistics agree with the original run: run 1 collected 324,950,178
advantage samples (original 325,242,386; about 43 per traversal), the
reservoirs are full from about iteration 22, and the final advantage loss is
0.164 (run 1) / 0.165 (run 2).

## Deviations from upstream

The algorithm and hyper-parameters are upstream's; these are all the
differences (details in `reports/trainer_REPORT.md` §Faithfulness):

1. **Random streams.** Not seed-for-seed comparable with an upstream run; the
   distributions are the same. Opponent-action uniforms come from a splitmix64
   hash of (seed, traversal id, action history); deals re-seed numba's MT19937
   per traversal (traversal seed = 20261107 x 1000 + t); reservoir replacement,
   network initialisation and minibatch sampling use torch CUDA generators
   (seeds 20261108-20261111). The iteration-1 network is upstream's
   (`torch.manual_seed(20261107)` then `DuelingAdvantageNet()`).
2. **Batched traversal** instead of per-worker recursion with a numba CPU
   forward. With a shared deterministic random source it is bit-identical to the
   reference recursion (numba forward); with the fp32 GPU forward 0 of 30,000
   traversals diverged in the tests (rounding only). Upstream's per-worker cap
   of 60 samples per traversal is not replicated (never hit).
3. **One `add_batch` per iteration** instead of one per worker result
   (equivalent under sequential reservoir semantics).
4. **Fit numerics:** GEMM operands/outputs bf16; LayerNorm, loss, output layer,
   gradient accumulation and Adam fp32 (upstream's own CUDA path uses bf16
   autocast; the original reproduction used JAX fp32). On a realistic buffer
   the final eval loss matches the fp32 reference within seed noise.
5. **Strategy net only at checkpoint iterations** (every 10th and 250), from a
   snapshot of the strategy reservoir, on a spare GPU. It is re-initialised
   every iteration and feeds neither traversal nor the deployed advantage-net
   bot, so the exported nets have the same distribution. The per-iteration
   `strat_loss` is that of the latest checkpoint fit.
6. **Checkpoints:** the 9.2 GB replay buffers are saved once after the run, not
   rewritten every 10 iterations.
7. Metadata only: `.pt` `config.deployment` is `advantage_npz`; extra timing
   fields in the log.

## Caveats

- **Warm caches.** The process timings exclude the one-off CUDA extension build
  (about 2 min idle) and reuse numba / Python bytecode caches already on disk; a
  first run on a fresh machine adds those compile times (the numba warm-up inside
  the timed process was 0.5 s with a warm cache).
- **Resume is not implemented.** `train_fast.py` has no `--resume`; the `.pt`
  files carry RNG states and `latest.pt` the buffers, but resuming from them is
  untested.
- **Random streams differ** from upstream (deviation 1): equivalence is
  statistical (149 traversal statistics, matching training statistics), not
  bitwise.
- **CPU contention** slows the traversal (it uses all cores for about 0.3 s per
  iteration); keep other CPU-heavy work off the machine during timed runs.
- Run 1 used the fit v1 kernels, which are not in this package (hashes in
  `results/runs/full_summary.json`); `configs/full.json` with the committed
  `fit/` reproduces run 2's code path.
- These are six-player research numbers against five fixed bots, not official
  contest points, eight-player results or a global ranking.
