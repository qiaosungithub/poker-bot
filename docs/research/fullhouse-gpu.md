# Fullhouse Deep CFR on 4x A100: the GPU reproduction

Verified 2026-11-08 (branch `gpu-training`). Research reproduction, not an
official tournament submission; read the
[event-time coding restriction](../competition/rules.md) before planning an entry.

The code, commands, environment and full method are in
[`fullhouse-gpu/README.md`](../../fullhouse-gpu/README.md). This page is the
short research summary and the map from this repository to the artifacts.

## What it is

The upstream Fullhouse Deep CFR training
([advitrocks9/fullhouse-bot](https://github.com/advitrocks9/fullhouse-bot) @
`e504793d` + `fullhouse-repro/training_changes.patch`) with its **original
configuration** — 250 iterations x 30,000 external-sampling traversals, six 5M
advantage reservoirs and a 5M strategy reservoir, 10,000 advantage-fit steps
and 3,000 strategy-fit steps of batch 8,192 per iteration, randomized stacks,
exploration 0.10 -> 0.02 — re-engineered to run on one 4x A100-40GB machine
with all neural-network work on the GPUs:

- traversal: all 30,000 root traversals of an iteration expand one tree level
  at a time; each level is one batched fp32 GPU forward, the upstream numba
  engine and feature encoder run on the CPU cores;
- reservoirs stay resident on a GPU with upstream `add_batch` semantics;
- the advantage fit is one CUDA graph per 250 steps of custom kernels (bf16
  GEMMs, fp32 LayerNorm / loss / Adam);
- the strategy net is fitted only at checkpoint iterations on a spare GPU.

It replaces the original JAX/TPU training path, whose launcher was removed on
this branch. That run's results remain in `fullhouse-repro/` as the reference
([fullhouse.md](fullhouse.md)); its full provenance is on `main` @ `7356b42`.

## Speed

Wall clock on GCP VM `qiaos-4a100-2` (4x A100-SXM4-40GB, 48 vCPU), process
creation to the last checkpoint file, warm compile caches:

| Run | Fit kernels | Whole process | Mean iteration (traversal / insert / advantage fit) |
|---|---|---:|---|
| Run 1 | release v1 | **982.3 s = 16.4 min** | 3.897 s (0.633 / 0.110 / 3.155) |
| Run 2 | release v2 (`fullhouse-gpu/fit/`) | **619.9 s = 10.3 min** | 2.449 s (0.594 / 0.107 / 1.747) |
| Original TPU run | JAX on 1 TPU + 24 CPU workers | 2 h 5 min 23 s (logged) | 23.05 s median |

## Strength (six-player benchmark, custom research points)

Same protocol as the original run (selection 256 blocks from seed 11,300,000
for every 10th checkpoint; confirmation 1,024 blocks from 12,300,000; criterion:
paired 95% lower bound vs the released model > -2). Confirmation results; the
released model scores 58.59 on these seeds:

| Model | Mean | Paired difference vs released [95% CI] |
|---|---:|---|
| GPU run 2, selected iteration 40 | 63.35 | **+4.76 [+3.88, +5.65]** |
| GPU run 1, selected iteration 20 | 60.63 | **+2.04 [+1.19, +2.90]** |
| Original TPU run, selected iteration 210 | 60.94 | +2.35 [+1.51, +3.19] |
| GPU run 2, final iteration 250 | 53.13 | -5.46 [-6.34, -4.59] |
| GPU run 1, final iteration 250 | 54.39 | -4.20 [-5.09, -3.30] |
| Original TPU run, final iteration 250 | 54.42 | -4.17 [-5.06, -3.28] |

All three selected checkpoints pass the criterion; all three final checkpoints
are weaker than the released model. Strength swings widely between neighbouring
checkpoints in every run (selection means 48.9-62.9), and the best iteration
differs between runs (20, 40, 210), so a selected checkpoint is the best of 25
noisy snapshots. The evaluation pipeline reproduces the original reference
numbers to floating-point precision. Do not compare these points with the
official 50/30/20 payout scale or treat them as an eight-player result.

## Faithfulness

Same algorithm and hyper-parameters; different random streams (hash-keyed
opponent uniforms, per-traversal deal seeds, torch CUDA generators), so runs are
statistically, not bitwise, comparable with upstream. The batched traversal is
bit-identical to the recursive reference when both use the same deterministic
random source and the numba forward; with the fp32 GPU forward none of 30,000
traversals diverged. Fit numerics follow upstream's own bf16-autocast CUDA path.
Full list: `fullhouse-gpu/README.md` §Deviations.

## Artifacts

In Git (`fullhouse-gpu/`): code, configs, the three engineering reports
(`reports/`), per-iteration metrics, run summaries, console logs, six-player
outputs with raw seed blocks, traversal and fit validation records
(`results/`, described in `results/README.md`).

Not in Git: checkpoints, ONNX exports and replay buffers stay on
`qiaos-4a100-2:~/work/poker-fast/trainer/` (`runs/full*`, `runs/full_v2*`,
`evalkit/eval_full*/`). SHA256 of every checkpoint:
`fullhouse-gpu/results/runs/checkpoints_full{,_v2}.sha256`. The selected
checkpoints are also listed in [model-artifacts.json](model-artifacts.json).

## Not verified

- Resume of the GPU trainer (`latest.pt` holds the buffers; no resume path).
- First-run timing on a fresh machine (extension build and numba compile excluded).
- Whether the selected checkpoints' advantage holds outside this fixed
  five-opponent table.
