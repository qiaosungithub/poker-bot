# Fullhouse latency on the Boston build box

Measured November 9, 2026. The organizer's build box ran the complete warm
Fullhouse research decision function faster than the research server in this
measurement. Neither measurement includes the official SDK or production IPC.

## Results

All times are milliseconds. Each row contains 30,000 calls, covering three
passes through the same 10,000-state corpus. Lower latency is better.

| Host | Model | Median | P99 | P99.9 | Maximum | Calls >1 ms |
|---|---|---:|---:|---:|---:|---:|
| Boston build box | Released Fullhouse | 0.121 | 0.195 | 0.218 | 0.419 | 0 / 30,000 |
| Boston build box | Reproduced iteration 210 | 0.121 | 0.196 | 0.218 | 0.446 | 0 / 30,000 |
| Research server | Released Fullhouse | 0.225 | 0.409 | 0.740 | 8.466 | 16 / 30,000 |
| Research server | Reproduced iteration 210 | 0.222 | 0.363 | 0.519 | 4.305 | 6 / 30,000 |

For the selected iteration 210, the research server's observed median and P99
were 1.83 and 1.85 times the build box's values. These are host-environment
comparisons, not isolated CPU speed ratios. No observed overrun on the build box
is not a hard worst-case guarantee or an official acceptance result.

## Machines and comparability

| Property | Boston build box | Research server |
|---|---|---|
| Host | `boston-team-95` | `t1v-n-21fe7514-w-0` |
| Instance | EC2 `m7i.large`, verified through instance metadata | GCP research host; instance type not checked |
| CPU | Intel Xeon Platinum 8488C | AMD EPYC 7B12 |
| Exposed logical CPUs | 2, one core with two threads | 240, two sockets, 60 cores/socket, two threads/core |
| Selected logical CPU | 0 | 1 |
| Python | 3.12.3, GCC 13.3.0 build | 3.12.16, Clang 23.1.3 build |
| OS | Ubuntu 24.04.5; AWS Linux kernel 7.0.0-1014 | GCP Linux kernel 5.19.0-1022; glibc 2.35 |
| Same-core sibling utilization during the run | 1.45% | 25.06% |

The research CPU and its sibling appeared idle during the pre-run sample, but
the sibling became busy during the measurement. The research host's overall
load was approximately 67, versus approximately zero before the build-box run.
No CPU steal was reported in these samples. Wall and thread CPU times were
both retained; the long research-host calls also appear in thread CPU time, so
the evidence does not justify attributing them only to descheduling. Python
builds, kernels, CPU architecture and co-tenant load differ.

The organizer's [Getting started page](https://pokerbot.sequoiacap.com/tournament/competitions/abb5539f-60ce-48b6-8925-c0c56a3102e3/start)
describes the supplied EC2 machine as similar to the tournament runtime.
It does not establish the production CPU model, allocation, host load or
microVM overhead. The build box is a useful additional validation target;
we cannot prove that its absolute latency is closer to production than the
research server's. More cores help parallel experiments, not a single-thread
decision's latency.

## Protocol and evidence

- Used the unchanged [measurement script](../../measure_latency.py) and the
  existing ONNX adapter. Timing includes state-dict feature construction,
  FP32 network inference, policy logic and action-dict construction.
- Excludes startup, official SDK conversion, memory serialization, JSON, IPC,
  game simulation and deadline enforcement. This is a six-seat Hold'em corpus,
  not the unreleased Boston variant or an eight-seat compatibility test.
- The corpus contains 2,500 states per street, with 5–200 BB stacks. There are
  200 warm-up calls before every pass; model order alternates between passes.
  Garbage collection is disabled during timing. Returned action kinds are
  checked outside the measured interval; official action bounds are not tested.
- One pinned logical CPU per run. ONNX Runtime uses one intra-op and one
  inter-op thread, sequential execution and CPUExecutionProvider. Numeric
  thread environment variables are set to one. `LD_PRELOAD` is removed for the
  measurement process on both hosts.
- NumPy 2.2.6, ONNX Runtime 1.23.2, eval7 0.1.11 and PokerKit 0.7.7 match on both
  hosts. All 13 installed package versions are pinned in
  [requirements-lock.txt](requirements-lock.txt). These are research environment
  pins; Boston's official requirements were not yet available.
- All 12 source/data/model files were SHA256-verified before running; see
  [source-manifest.json](source-manifest.json). Only this approximately 3 MB
  benchmark archive was copied to the team's machine, with no credentials or
  training replay checkpoint.
- [buildbox312/result.json](buildbox312/result.json) and
  [local312/result.json](local312/result.json) contain exact results and source
  hashes. Each sibling `timings.npz` retains every wall and thread CPU sample;
  `host.json` retains machine, package, load and CPU counter evidence.
- [audit.json](audit.json) independently recomputes aggregate statistics from
  sorted raw timing arrays, checks all sample counts and per-street totals,
  and verifies matching source, corpus and model hashes and package versions.

## Reuse the installed environment

The remote environment is `/home/ubuntu/work/fullhouse-latency/.venv`.
The frozen benchmark source is in the surrounding `fullhouse-latency` directory.
It is separate from the future `~/work/starter-kit`. Codex login is unnecessary
for these tests. Run from this research server with a new output directory:

```bash
ssh -o StrictHostKeyChecking=accept-new no-limit-2026-build \
  'env -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /home/ubuntu/work/fullhouse-latency/.venv/bin/python \
  /home/ubuntu/work/fullhouse-latency/run_measurement.py \
  --root /home/ubuntu/work/fullhouse-latency \
  --output-dir /home/ubuntu/work/fullhouse-latency/results/my-new-run --cpu 0'
```

The [wrapper](run_measurement.py) verifies its manifest before measuring,
records host information, and refuses to overwrite output. The local matching
environment is `.runtime/envs/fullhouse-buildbox-comparison312/`; its frozen
source snapshot is `.runtime/fullhouse-buildbox-20261109/source/`. Both are
outside Git. The canonical source/model paths and hashes are in the manifest.

## Starter-kit availability

Rechecked at **2026-11-09 03:17:59 UTC**. An authenticated competition request
returned `variant: null` and `variant_announced_at: null`. The starter-kit URL
returned by the team's machine API returned HTTP 404 with `not_found`.
This agrees with the portal's statement that the kit becomes available at the
variant announcement. No kit was installed or smoke-tested. Private evidence:
`.runtime/private-web/kit-check-20261109T031759Z.json`.
