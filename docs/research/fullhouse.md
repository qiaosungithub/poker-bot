# Fullhouse reproduction and measurements

Verified 2026-11-08. Training completed on 2026-11-07. This is a research
reproduction, not an official tournament submission. Read the
[event-time coding restriction](../competition/rules.md) before planning an entry.

## Result and checkpoint selection

The selected reproduction meets the predeclared comparison criterion and
outperforms the released model **in this fixed six-player benchmark**:

| Model | Mean points | 95% CI | Paired difference vs released |
|---|---:|---|---|
| Released Fullhouse | 58.5894 | [57.8663, 59.3124] | Reference |
| Reproduction, selected iteration **210** | **60.9397** | **[60.2345, 61.6450]** | **+2.3504**, CI **[+1.5060, +3.1947]** |
| Reproduction, final iteration 250 | 54.4207 | [53.6839, 55.1574] | -4.1687, CI [-5.0593, -3.2781] |

Use iteration 210. Training was unstable across checkpoints; additional
iterations did not monotonically improve strength. The latest model is not the
best model. The success criterion was a paired 95% lower confidence bound above
-2 points; it was declared before the confirmation evaluation.

## What was reproduced

Upstream: <https://github.com/advitrocks9/fullhouse-bot>, commit
`e504793d480b1b975f25258d25939b45c6dbd5a4`.

- Random network initialization and self-play; released weights were only an
  evaluation reference, not initialization or training targets.
- CPU external-sampling MCCFR traversal, accelerated by Numba, and a JAX/TPU
  implementation of the upstream network fitting procedure.
- Advantage network: 51 input features, 256/256 shared trunk with LayerNorm and
  LeakyReLU, separate 128-hidden-unit value/advantage heads, five abstract action
  outputs; 146,694 parameters. These outputs are regrets/advantages used to form
  action probabilities, not a conventional DQN Q-learning run.
- A separate average-strategy network was also trained. The reported bot uses
  the **advantage-network** export, matching the released deployment path.
- 250 iterations, 30,000 root traversals each: **7,500,000 traversals**. One
  traversal can generate many training examples; the run collected 325,242,386
  advantage examples in total, not that many independent poker games.
- Per iteration: 10,000 advantage fit steps and 3,000 strategy fit steps, batch
  size 8,192; six per-seat advantage reservoirs of five million samples each and
  a five-million-sample strategy reservoir. Learning rate warms up to 0.001.
- Exploration moves from 0.10 to 0.02. Starting stacks are randomized. Training
  rewards are chip payoffs, not the official tournament ICM objective.
- Existing equity features, original <=12 BB preflop overlay, feature encoding
  and action logic were retained. FP32 ONNX serves the exported network.

The released checkpoint has no published historical training manifest. This
reproduces the available method and evaluates against its weights; it is not a
claim of reconstructing every detail of the original historical run.

## Strength protocol and speed

See the immutable [original protocol](../../benchmarks/six_player_2026-11-06/protocol.txt).
For this comparison only the Fullhouse model slot changed. The other five bots
were `cfr_lite64`, `hydra32`, `solver_like`, `maniac` and `station`.

- Standard Hold'em, six players, 10,000 chips each, fixed 50/100 blinds, up to
  200 hands; exact survivor ICM, custom payouts `[100,60,40,25,10,0]`.
- Selection: 256 seed blocks starting at 11,300,000, six cyclic seat rotations
  per block, evaluated every ten iterations. The best checkpoint was chosen
  before confirmation.
- Confirmation: a separate 1,024 blocks starting at 12,300,000, six rotations
  per block: **6,144 tournaments per model**. Compare matching seed blocks to
  compute paired confidence intervals. The independent unit is a block, not
  each hand or rotation.
- The selected confirmation covered 828,846 hands in **95.18 seconds**, with
  24 CPU workers and zero runtime errors. The reference covered 820,623 hands
  in **87.74 seconds**. These are parallel simulator throughput measurements,
  not single-decision latency.
- Strength matches impose no action deadlines. The shared adapter normalizes
  some unavailable/out-of-range raises; official play rejects such actions.

Do not interpret 60.94 as a win percentage or compare it numerically to the
official 50/30/20 payout scale. This experiment does not establish eight-player
compatibility, performance on the unreleased variant, or a global poker ranking.

## Training speed

The job used **24 CPU workers plus one local TPU device** for network fitting.
It is not a CPU-only training timing.

| Measurement | Observed value |
|---|---:|
| Logged training duration, including training/checkpoint work | 7,523 s = **2 h 5 min 23 s** |
| Start through final confirmation completion | 7,855.99 s = **2 h 10 min 56 s** |
| Median logged iteration compute duration | **23.05 s** |
| Median logged traversals/s during iteration work | **1,301.5** |
| Overall root-traversal throughput, including training duration | Approximately **997/s** |

The logged iteration duration excludes later checkpoint-writing overhead;
do not multiply it by 250 and label that the full wall time. Timing comes from
`train.log`, `process.json` and `status.json`. No reproduction training process
was running when inspected on 2026-11-08.

## Fresh CPU decision latency, 2026-11-08

| Model | Median | P99 | P99.9 | Maximum | Calls >1 ms |
|---|---:|---:|---:|---:|---:|
| Released | 0.225 ms | 0.332 ms | 0.397 ms | 6.118 ms | 2 / 30,000 |
| Reproduced iteration 210 | **0.224 ms** | **0.331 ms** | **0.394 ms** | **3.501 ms** | **1 / 30,000** |

The observed speed is effectively the same. Neither model has a measured hard
1 ms worst-case guarantee. Do not attribute the rare long calls to a particular
cause without further evidence; both wall time and thread CPU time were kept.

Measurement protocol:

- AMD EPYC 7B12, pinned to logical CPU 2, one ONNX intra-op and inter-op thread.
- Frozen Python 3.11.0rc1 environment: NumPy 2.2.6, ONNX Runtime 1.23.2,
  eval7 0.1.11, PokerKit 0.7.7.
- Reused the exact retained corpus of 10,000 six-seat legal states, 2,500 per
  street, with starting stacks of 5–200 BB.
- Three passes per model, 200 warm-up calls before each pass, garbage collection
  disabled during timing. Model order alternated between passes.
- Timed complete `state dict -> features -> FP32 model -> action dict`, including
  action logic. Excluded startup, official SDK conversion, memory serialization,
  JSON, IPC and game simulation. Action-kind checks occurred outside timing.
- Saved all wall/thread CPU timings, corpus and model hashes, source hashes,
  environment and per-street statistics. Both raw and summarized results were
  independently cross-checked after the run.

Evidence:
[result.json](../../fullhouse-repro/latency/2026-11-08/result.json),
[raw timings](../../fullhouse-repro/latency/2026-11-08/timings.npz),
[measurement script](../../fullhouse-repro/measure_latency.py).
The original November 6 value of 0.343 ms P99 belongs to the released checkpoint
on that historical run; it remains unchanged in the frozen benchmark table.

## November 9 build-box latency comparison

An independent environment was installed on the organizer's EC2 `m7i.large`
build box. On the same retained corpus, source and FP32 models, the selected
iteration 210 measured median **0.121 ms**, P99 **0.196 ms**, maximum **0.446 ms**,
and **0/30,000 calls over 1 ms**. A new Python 3.12 research-server run measured
median 0.222 ms and P99 0.363 ms. Python builds and host load differ; the research
CPU's sibling became busy during the run. These values describe the measured
hosts, not isolated CPU speed or guaranteed production timing. Official SDK and
IPC are excluded. See the [full report and reusable environment](../../fullhouse-repro/latency/2026-11-09-buildbox-comparison/REPORT.md).

## Artifacts and validation

Paths are relative to `poker-bot/`:

| Artifact | Role |
|---|---|
| `fullhouse-repro/runs/main_adv/iter_00210.npz` | Recommended weights, 590,758 bytes |
| `fullhouse-repro/eval/iter_00210.onnx` | Recommended FP32 inference model, 588,622 bytes |
| `fullhouse-repro/runs/main/iter_00210.pt` | Selected Torch model/config export, without full replay buffers |
| `fullhouse-repro/runs/main/latest.pt` | Final iteration 250 checkpoint with replay buffers, about 9.20 GB |
| `fullhouse-repro/status.json` | Authoritative final selection and paired confirmation summary |
| `fullhouse-repro/objective.json` | Frozen initial plan; its status text is historical, not live |
| `fullhouse-repro/config_main.json` | Training configuration |
| `fullhouse-repro/eval/*_confirmation.json` and `.jsonl` | Summaries and raw seed blocks |
| `fullhouse-repro/results.jsonl` | 25 selection records and two confirmation records |
| `fullhouse-repro/launch_manifest.json`, `launch_sources.zip` | Original launch provenance; do not rewrite after later maintenance |
| `fullhouse-repro/environment312.txt` | Training environment versions |
| `fullhouse-repro/storage_relocation.json` | Verified migration into work-local runtime storage |

Selected NPZ SHA256:
`c476a83f39c4840b54e9599324f6c8b9f43a917d870412bf5325d24193ff57cd`.
Selected ONNX SHA256:
`2f2455adf06f22a3a6d87bef1b330e4ea80438a6567943abfab2762d6cf0db32`.
Released NPZ SHA256:
`1102326b68da95564de147106612df71cb891b42f0726ba0212d3b9a5bcae295`.

Recorded validation includes engine/feature/reach tests, Torch/JAX forward,
gradient and optimizer checks, and bit-identical two-iteration uninterrupted vs
resumed training. The selected ONNX validation used 1,024 states; maximum
absolute output difference was 1.073e-5. The storage migration also reran 48
six-player tournaments with zero errors and confirmed unchanged selected weights.
See `backend_validation.json`, `resume_validation.json`,
`eval/iter_00210.onnx.validation.json` and `storage_relocation_validation.json`.

**Resume limitation:** replay state is retained at iteration 250 in `latest.pt`.
Iteration 210 does not have a full saved replay reservoir. Loading its weights
is possible; claiming an exact continuation of the entire iteration-210
training state is not.

## Commands for a fresh agent

Read-only status:

```bash
cd /kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot
cat fullhouse-repro/status.json
tail -n 8 fullhouse-repro/train.log
```

Repeat latency into a **new, nonexistent output directory**:

```bash
.runtime/envs/poker-hu-validation/bin/python fullhouse-repro/measure_latency.py \
  --output-dir fullhouse-repro/latency/my-new-run --cpu 2 --repeats 3
```

Run a small research arena check, again with new output paths:

```bash
mkdir -p fullhouse-repro/eval/manual
.runtime/envs/poker-hu-validation/bin/python fullhouse-repro/evaluate.py \
  --onnx fullhouse-repro/eval/iter_00210.onnx \
  --output fullhouse-repro/eval/manual/iter210-smoke.json \
  --blocks 8 --seed 22300000 --workers 4
```

This command is a smoke test, not evidence equivalent to the 1,024-block
confirmation. For a new strength comparison, run a separate reference command
without `--onnx` on the same block seeds, predeclare the sample size, and analyze
paired block differences. Do not tune on confirmation seeds and call them held
out afterward. No new strength run was needed for the documentation update.

For new training, first verify TPU ownership through workspace infra procedures,
choose a new run directory and configuration, and smoke-test/restart-test the
change. `launch.py CONFIG` performs training; `supervise.py` uses the existing
main configuration and may resume the old run. Neither is a status command.
