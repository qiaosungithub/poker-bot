# Handwritten standard Hold'em policies

This runner compares existing six-player strategies, including explicitly named
Monte Carlo budget variants, against both Fullhouse neural models. It imports
the frozen `hu-benchmark/multiway_benchmark.py` rules adapter; it does not replace
the engine or alter the older benchmark.

Read [the predeclared plan](PLAN.md) and
[the result report](../benchmarks/heuristic_holdem_2026-11-09/REPORT.md).
Candidate discovery, source commits and exclusions are recorded in
[survey.json](../benchmarks/heuristic_holdem_2026-11-09/survey.json).

## Setup and interface

Use the existing `.runtime/envs/poker-hu-validation/bin/python` environment.
Exact packages are in the benchmark's `environment.txt`. A fresh clone needs
`git submodule update --init --recursive`, the work-local evaluation environment,
the released FP32 ONNX export and (for the reproduced control) iteration 210.
Models are local artifacts, not backed up by this Git repository; see
[the Fullhouse guide](../docs/research/fullhouse.md) and
[artifact hashes](../docs/research/model-artifacts.json).

Every candidate uses `decide(public_state_dict) -> action_dict`. Public bot IDs
are anonymous physical-seat identifiers. Hole cards of other players are never
included. Each instance gets a private module namespace, opponent memory and
random stream; a table resets memory. Only txnujk's deterministic import-time
equity table and read-only ONNX sessions are cached across instances.

`policies.py` documents every sampling edit. No upstream source file is edited.
The loader observes caught exceptions without changing successful decision paths.
The existing game adapter clamps raises and converts unavailable raises to
check/call; raw results retain all such corrections. This behavior differs from
an official sandbox rejecting an illegal action.

## Commands

Run from the repository root. Output directories must not already exist.

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m unittest -v heuristic_holdem.test_benchmark

PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m heuristic_holdem.benchmark latency --names solver_like,tournament_specialist \
  --per-street 2500 --repeats 3 --output benchmarks/my-new-latency

PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m heuristic_holdem.benchmark screen --names solver_like,tournament_specialist \
  --blocks 128 --seed 31100000 --workers 24 --output benchmarks/my-new-screen

PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m heuristic_holdem.analyze benchmarks/my-new-screen
```

`confirmation` requires exactly six comma-separated policy names and plays all
six together. `trajectory` measures complete decisions during sequential tables,
including long histories, surviving 2–5 player states and evolving bot memory.
Timing and parallel strength runs must not overlap. See each retained manifest
for the exact commands/configurations used in the reported experiment.

The score scale is the older custom six-player research scale, not the No Limit
50/30/20 schedule. Do not compare numbers across different lineups. No result
establishes a global best handwritten bot or hard worst-case 1 ms latency.
