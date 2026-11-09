# Standard Hold'em handwritten strategies: 1 ms CPU survey

Observed November 9, 2026. This is a six-player standard no-limit Texas Hold'em
research benchmark. It is separate from the Bay Area variant and official
No Limit scoring. Sources and all measurements are pinned; the old November 6
comparison is unchanged.

## Finding

**TournamentSpecialist is the practical recommendation for this field and ICM
score.** It had the highest mean score, ran unchanged from upstream, and had no
calls over 1 ms in the two final timing tests (30,000 corpus calls plus 14,578
actual-game calls). Its actual-game P99 was 0.084 ms and maximum 0.455 ms.

**txnujk-32 is a statistically unresolved strength alternative**, with a lower
mean but no significant paired difference from TournamentSpecialist. It uses a
32-sample cap introduced here; the upstream original did not meet the timing
requirement. Its actual-game P99 was 0.541 ms, with 11 / 17,913 calls over 1 ms.
Both handwritten policies scored above both neural controls, including after
adjusting confidence intervals for all 15 pairwise comparisons.

This establishes neither a global best handwritten poker policy nor a hard
worst-case latency bound. The published Fullhouse checkpoint had the highest
mean final chips in the same experiment. The ICM-point ranking and chip ranking
answer different questions; do not call the points a win rate or equate them
with cash-game bb/100.

## Independent confirmation: all six policies at the same table

12,288 tables, 2,404,883 hands; 2,048 independent deal-seed blocks with six seat
rotations per block. Higher points are better. Every table distributes 235
points; equal share is 39.1667. Mean confidence intervals below are marginal
95% intervals over block means. Timing comes from a separate sequential run,
not the parallel strength tournament.

| Policy | Type | Mean points [95% CI] | Mean final chips | Actual-game P99, ms | Actual-game calls >1 ms |
|---|---|---:|---:|---:|---:|
| `tournament_specialist` | handwritten | 45.13 [44.72, 45.54] | 8,098 | 0.084 | 0 / 14,578 |
| `txnujk_32` | handwritten | 44.46 [43.97, 44.95] | 10,662 | 0.541 | 11 / 17,913 |
| `fullhouse_reproduced` | learned reference | 42.66 [42.08, 43.23] | 13,006 | 0.765 | 53 / 15,424 |
| `fullhouse_released` | learned reference | 42.26 [41.66, 42.85] | 13,885 | 0.863 | 81 / 13,269 |
| `solver_like` | handwritten | 34.88 [34.41, 35.36] | 7,164 | 0.084 | 0 / 11,485 |
| `cfr_lite_32` | handwritten | 25.61 [25.06, 26.16] | 7,185 | 0.411 | 3 / 8,465 |

Starting chips were 10,000 each. Ending chips include zero for eliminated
players, while those players retain their placement payouts. Final chip means
therefore need not rank policies in the same order as ICM/placement points.
A fixed 200-hand cap and fixed blinds also affect the incentives.

| Paired points difference | Mean | Simultaneous 95% CI, all 15 pairs |
|---|---:|---:|
| TournamentSpecialist − txnujk-32 | +0.67 | [-0.36, +1.70] |
| TournamentSpecialist − released Fullhouse | +2.87 | [+1.70, +4.04] |
| TournamentSpecialist − reproduced Fullhouse | +2.48 | [+1.35, +3.60] |
| txnujk-32 − released Fullhouse | +2.20 | [+0.93, +3.47] |
| txnujk-32 − reproduced Fullhouse | +1.80 | [+0.55, +3.05] |

The two leading handwritten strategies cannot be separated by these data.
The two Fullhouse models also cannot be separated in this field. The earlier
November 6 lineup was different; its scores must not be compared directly to
this table.

Machine-readable results: [table.csv](table.csv),
[full paired comparisons and audit](confirmation/audit.json),
[raw block/table records](confirmation/blocks.jsonl).

## What the recommended policies actually do

- **TournamentSpecialist** is an existing handcrafted opponent in the Fullhouse
  repository, not a newly discovered verified tournament champion. It splits
  preflop behavior into short (below 15 big blinds), medium (15–30), and deep
  stacks. It uses explicit opening/calling/shoving hand sets, adjusts aggression
  when leading in chips or facing short stacks, then groups postflop hands into
  strong, medium, draw and air. Fixed bet sizes, pot-odds thresholds and
  deterministic mixed actions choose the move. It does not run a neural
  network, Monte Carlo equity, a poker solver, or an exact ICM optimizer.
  [Pinned source](https://github.com/advitrocks9/fullhouse-bot/blob/e504793d480b1b975f25258d25939b45c6dbd5a4/bots/adversarial/tournament_specialist.py).
- **txnujk-32** loads the submitted `floorfix` heuristic, not the author's separate
  trained CFR variant. It uses 169 starting-hand equity classes, position,
  opponent action summaries, sampled showdown equity, draw outs and pot-odds
  thresholds. Strong made hands build pots; marginal one-pair hands control pot
  size and avoid committing most of the stack. The only strategy-budget edit
  here caps postflop equity sampling at 32. The deterministic import-time
  preflop equity table is cached across instances; opponent state is not shared.
  [Pinned source](https://github.com/txnujk09/fullhouse-poker-bot/blob/4fe154318949aa60430682416f148733c6f0d564/bots/floorfix/bot.py).
- **solver_like** uses fixed hand ranges, made-hand/draw categories and mixed
  rules. Its name does not mean it solves the current game.
- **CFR-Lite-32** scores actions with handcrafted equity/expected-value formulas
  and positive-regret weighting within one decision. It is not a trained CFR
  blueprint or reinforcement-learning model. Its sampling budget is capped.
- **Fullhouse controls** retain the released policy and our selected iteration
  210 policy, using their FP32 ONNX exports and existing decision wrapper. They
  are learned references, not members of the handwritten candidate search.

The exact original paths, hashes and sampling edits are in
[the loader](../../heuristic_holdem/policies.py) and
[confirmation manifest](confirmation/manifest.json).

## Search and latency exclusions

Nine handwritten families and 21 original/budget versions were timed. Static
hand-equity tables and simulation were allowed; trained policies were excluded
from the handwritten group. Search queries, pinned repositories, classification
and other excluded sources are in [survey.json](survey.json). Event placements
quoted by project authors were not treated as independently verified strength.

The following 400-state pilot numbers are screening evidence, not final latency
bounds. Fullhouse-source opponents use the pinned repository above.

| Family | Original pilot P99, ms | Budget result and disposition |
|---|---:|---|
| TournamentSpecialist | 0.052 | Original qualified; final recommendation |
| solver_like | 0.042 | Original qualified |
| strong | 1.495 | Rejected |
| CFR-Lite | 1.535 | 32 qualified; 64 later failed tail-frequency filter |
| Hydra | 5.352 | 32 had P99 0.653 but 1/400 calls >1 ms; rejected conservatively |
| txnujk floorfix | 4.670 | 32 and 64 passed pilot; 32 had higher screening score |
| Vikram submitted v12 | 8.185 | 32 looked fast on corpus, but failed real-game river timing |
| Pantheon final v16 | 39.594 | Corrected 32-sample version still P99 3.030 ms; rejected |
| fold-equity | 72.549 | 32-sample version still P99 2.433 ms; rejected |

Relevant additional primary sources:
[Vikram report](https://github.com/Vikknows/Fullhouse-Hackathon/blob/cb16737fea04805f2ab1bd7f6a6de6f9bb0ea974/PROJECT_README.md),
[Pantheon project](https://github.com/lucashsu007-create/fullhouse-hackathon-write-up-repo/tree/8b98ee3efce3b224177609352258dc86db901cce),
[fold-equity](https://github.com/HighRiskLowYield/fold-equity/tree/f427cb5cd429123e14da3626d8f440f7f9ffd4dd).

Vikram-32 led the strength screen (60.65 points against the five screening
opponents), but an intact exact-enumeration path in heads-up river play made
its real-game river P99 **2.522 ms**, with **715 / 15,475** overall calls above
1 ms. It was excluded before confirmation. CFR-Lite-64 had **110 / 30,000**
corpus calls above 1 ms and was replaced by its already-screened 32 version.
These exclusions are retained in [the frozen selection](selection.json).

The initial Pantheon budget pilot missed function-default iteration caps.
Those pilot records are retained as superseded; the corrected measurement in
`latency_pantheon_corrected/` caps both defaults and global settings. No strength
results informed that correction. Pantheon also hit two caught position-lookup
exceptions on the corpus; it is not a validated compatible finalist.
A single pilot outlier excluded Hydra-32 and Vikram-64 under the predeclared
filter. This conservative search may miss other qualifying implementations;
it is not an exhaustive ranking of all public poker programs or C++ ports.

## Timing method and remaining limits

AMD EPYC 7B12, one pinned logical CPU (2), Python 3.11.0rc1, single-threaded
libraries, warm imports/evaluator, garbage collection disabled during timing.
Measure the complete public state dictionary to action dictionary function,
including features, history processing and sampling. Exclude state construction,
engine work, random-stream isolation, startup, JSON serialization and IPC.
No official sandbox or hardware was measured.

The predeclared statistical filter requires overall **and each street's P99
below 1 ms**, plus **at most 0.1%** of all sampled wall-clock calls over 1 ms.
This explicitly allows rare overruns and is not an every-action deadline
promise. Thread-CPU times and wall times are both retained.

Final corpus timing: 10,000 legal states, 2,500 per street, three passes = 30,000
calls per policy. Actual-game timing: a disjoint 12-block × 6-rotation run on
the final lineup (72 tables, 81,134 decisions), with up to 200 historical
actions, evolving bot memory and fewer surviving players. Timing and parallel
strength runs did not overlap.

| Policy | Corpus P99, ms | Corpus calls >1 ms | Actual-game max, ms | Statistical filter |
|---|---:|---:|---:|---|
| `tournament_specialist` | 0.042 | 0 / 30,000 | 0.455 | Pass |
| `txnujk_32` | 0.433 | 9 / 30,000 | 3.599 | Pass |
| `fullhouse_reproduced` | 0.353 | 7 / 30,000 | 2.258 | Fail; strength reference only |
| `fullhouse_released` | 0.344 | 8 / 30,000 | 3.506 | Fail; strength reference only |
| `solver_like` | 0.040 | 1 / 30,000 | 0.438 | Pass |
| `cfr_lite_32` | 0.388 | 6 / 30,000 | 1.616 | Pass |

Fullhouse reference policies pass the corpus timing filter but fail the
actual-game tail-frequency filter. They remain strength controls, without time
penalties. This is additional evidence beyond the old corpus-only timing;
retain the older measurements with their original scope.
TournamentSpecialist had zero measured overruns across the two final tests,
which provides useful margin but cannot prove an absolute worst-case bound on
a shared CPU. txnujk-32 and CFR-Lite-32 had measured overruns and should not be
described as satisfying a strict every-call 1 ms requirement without changes.

## Exact game and statistical protocol

[PLAN.md](../../heuristic_holdem/PLAN.md) was written before measurements.
The runner imports the frozen `hu-benchmark/multiway_benchmark.py` adapter
(PokerKit 0.7.7) without editing it. Six players start at 10,000 chips (100 BB),
blinds 50/100, no ante or rake, standard 52-card Hold'em, at most 200 hands per
table. Blinds stay fixed. Payouts are **[100, 60, 40, 25, 10, 0]**. Eliminations
receive their placement awards; surviving stacks at the cap receive exact ICM
against the remaining prizes. This is our custom research scoring, not the
No Limit contest's published payout schedule.

Each bot receives only its own hole cards and public information. Identity is
an anonymous physical-seat ID. Modules/opponent memory/random streams are
isolated between instances and reset per table. Only deterministic preflop
lookup data and read-only neural inference sessions are cached. Seat order is
randomized within each seed block, then cyclically rotated six ways; all six
legs reuse the same underlying deal seeds. Blocks, not hands or rotations, are
the independent units for standard errors.

Screening: seven eligible candidates each played the same five controls
(released Fullhouse, solver_like, Hydra-32, station, maniac), 128 blocks per
candidate, seeds 31,100,000–31,100,127. Duplicate policies had independent state.
That produced **5,376 tables / 803,855 hands in 102.18 seconds** on 24 workers.
Keep the highest-scoring version per family and the four leading eligible
families, with latency-based replacement before confirmation. Screening means
are not comparable directly to the final table because the lineup changed.

Confirmation: the four frozen handwritten finalists and both neural controls
played together, seeds **31,200,000–31,202,047**, 2,048 blocks, **12,288 tables**.
Runtime **281.73 seconds (4.70 minutes)** on 24 CPU workers. No additional blocks
were added after inspecting the scores. CIs use block standard errors; pairwise
simultaneous CIs use a Bonferroni normal critical value for all 15 pairs.
Residual uncertainty is reported; variance never literally disappears.

### Action compatibility

The shared adapter normalizes legal raise sizes and converts a raise when
raising has not reopened into a call/check. This is the same rule for all
policies, not an official timeout/illegal-action penalty. In confirmation, all
recorded corrections were of the latter kind:

| Policy | Decisions | Raise not reopened → call/check |
|---|---:|---:|
| `tournament_specialist` | 2,409,419 | 2,338 |
| `txnujk_32` | 3,071,830 | 2,347 |
| `fullhouse_reproduced` | 2,689,137 | 2,234 |
| `fullhouse_released` | 2,458,786 | 2,476 |
| `solver_like` | 2,158,175 | 1,128 |
| `cfr_lite_32` | 1,620,134 | 4,773 |

Consequently these results evaluate the policies behind this adapter. A raw
upstream file is not certified as a valid official submission. Both the
strength screen and confirmation recorded **zero caught decision exceptions**;
neural emergency fallback was configured to raise rather than hide an error.

## Reproduction and retained evidence

Read [the runner guide](../../heuristic_holdem/README.md) for setup. All new
models/environments/caches remain under this workspace. Model binaries are
local, ignored artifacts; a fresh Git clone needs the released model and
iteration-210 artifacts listed in
[model-artifacts.json](../../docs/research/model-artifacts.json).
Nine source submodules are pinned in the parent repository.

Exact replay of the final lineup, writing to a new directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m heuristic_holdem.benchmark confirmation \
  --names txnujk_32,tournament_specialist,solver_like,cfr_lite_32,fullhouse_released,fullhouse_reproduced \
  --blocks 2048 --seed 31200000 --workers 24 \
  --output benchmarks/my-new-confirmation

PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m heuristic_holdem.verify
```

Each stage retains exact configuration/seed arguments. The verification command
checks the frozen November 9 data; use `heuristic_holdem.analyze` for a new run.

- [survey.json](survey.json): discovery, source commits, classification, exclusions.
- [selection.json](selection.json): final roster frozen before confirmation.
- [runner_sources.zip](runner_sources.zip): byte-for-byte as-run source snapshot,
  protocol, selection, survey and environment.
- [environment.txt](environment.txt), [host.json](host.json): packages and host.
- `latency*/timings.npz`: individual wall/CPU samples and phases; actual-game
  archives also retain history lengths and table outcomes.
- `screen/blocks.jsonl`, `confirmation/blocks.jsonl`: original table records,
  seats, stacks, eliminations, scores, actions and normalization counts.
- [audit.json](audit.json): independently recomputed ICM/placement scores and
  quantiles, source/model hashes, clean pinned sources, negative score control.
  **421,697** timing calls were checked against the summaries.
- [validation.txt](validation.txt): six isolation/cap/replay/diagnostic tests and
  seven existing rules/side-pot/ICM tests passed.

The independent scorer enumerates finish permutations instead of reusing the
engine's ICM implementation. It verifies 60,000 total chips and 235 points per
table, elimination awards, seat balance, block seeds/means and raw-data hashes.
A deliberately corrupted score is rejected. Selected/reference model artifacts
were hash-checked without modifying them.
