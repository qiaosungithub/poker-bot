# Bay Area environment and public-bot comparison

This directory runs the November 6 Bay Area **three-card discard/reveal** game.
It integrates the participant-published competition engine at a pinned commit,
adds a step interface and independent rule/settlement tests, and runs public
strategies through the actual SDK host/guest process protocol. It does not claim
to reproduce the organizer's microVM hardware or the unknown Boston variant.

## Install and test

From the repository root, with Python 3.12 available:

```bash
bash bay_area/setup.sh python3.12
.runtime/envs/poker-bay-area312/bin/python -m pytest bay_area/tests -q
```

On the current research server, an existing Python 3.12 executable can bootstrap
the environment:

```bash
bash bay_area/setup.sh .runtime/envs/poker-fullhouse-train312/bin/python
```

The setup uses `.runtime/` for the environment, cache and temporary files. It
installs only dependencies needed by the inspected strategies; this is not the
complete official submission package allowlist. Exact installed versions are
retained with the benchmark artifacts.

## Environment API

`BayAreaEnv.state` is the real SDK `GameState` for the next player, or `None` at
termination. `step((action, memory))` applies the action and advances to the next
decision. Each seat has separate memory, reset for each table. `result()` returns
the engine result and the actual payout/ICM points, not chip-proportional scores.

```python
from bay_area.environment import BayAreaEnv
from poker_agent_sdk import Action

env = BayAreaEnv("example-seed")
while env.state is not None:
    state = env.state
    if state.phase == "select":
        action = Action("select", cards=(2, 0))
    elif state.can("check"):
        action = Action.check()
    else:
        action = Action.fold()
    env.step((action, b""))
result, points = env.result()
```

Defaults: eight seats, 2,000 chips, 10/20 initial blinds, the kit's twelve blind
levels changing every 20 hands, 250-hand cap, payouts 50/30/20 followed by zeros.
The board has the usual flop/turn/river. Before flop betting, every live player,
including all-ins, simultaneously discards one of three cards and exposes one
retained card. Discarded cards remain dead and private. The engine handles legal
raises, side pots, refunds, automatic reveals, tied eliminations and exact ICM.

The tests cover all six selection combinations, concealed simultaneous choices,
dead-card exclusion, all-in selection, fallback actions, memory semantics,
timeout/event replay, independently calculated ICM, and unequal-stack side pots
checked against a separate eval7 showdown/pot-layer oracle.

## Run the comparisons

```bash
PYTHONHASHSEED=0 .runtime/envs/poker-bay-area312/bin/python -m bay_area.arena \
  --config bay_area/configs/screen.json --out benchmarks/bay-area-screen-new --workers 16
.runtime/envs/poker-bay-area312/bin/python -m bay_area.analyze benchmarks/bay-area-screen-new

PYTHONHASHSEED=0 .runtime/envs/poker-bay-area312/bin/python -m bay_area.arena \
  --config bay_area/configs/confirmation.json --out benchmarks/bay-area-confirm-new --workers 16
.runtime/envs/poker-bay-area312/bin/python -m bay_area.analyze benchmarks/bay-area-confirm-new
```

Output directories must be new; existing evidence is never overwritten. The
confirmation configuration contains the finalists selected in the retained
screening run. It is not automatically reselected by the command above.

`candidates.json` records source repositories, commits, paths and source hashes.
There are **three repository groups and sixteen versions, not sixteen entrants**.
Their official placements are unknown. The denfujita README still discusses v6,
while its pinned source and commit implement v7. James's root README describes
a rehearsal engine, but the included `bots/` implementations explicitly support
the revealed Bay Area game. No strategy parameters or source files were tuned
for this comparison. Obviously weak SDK examples, obsolete versions, duplicate
source and special-prize-only candidates were excluded in advance.

The [predeclared plan](PLAN.md) fixes the screening, selection and confirmation
procedure. Finalists all play together, one copy each; no weak background bots
inflate the score. A seed block contains all eight seat rotations of one random
ordering. Confidence intervals use block means, and comparisons use paired
block differences. The final report includes simultaneous intervals across all
28 pairs, so selection of the highest mean does not by itself establish superiority.

## Timing and reproducibility boundaries

- Main scores include actual SDK errors/timeouts/illegal-action fallback, its
  **1 ms decision + 2 ms delivery grace + 100 ms table bank**, and SDK memory
  validation/truncation. The 2 ms grace is from this kit's `BotProcess` default;
  it is not a newly inferred official inference budget.
- Each worker runs one table at a time with one separate guest process per
  strategy. Host and guests share a two-CPU affinity set. Subprocesses receive a
  clean environment and one-thread numerical-library settings.
- `seeded_guest.py` wraps otherwise unchanged bot sources to seed global,
  module-owned and instance-owned PRNGs for each request. This adds small timed
  overhead. Existing internal wall-clock computation guards remain unchanged,
  so results need not replay bit-for-bit on differently loaded hardware.
- The wrapper exposes only ordinary SDK observations and the seat's memory to
  a bot. Seed strings do not contain hidden cards or opponent identities.
- Separate `measure_latency.py` measurements time the original `Agent.act` on
  real played states on one CPU, excluding seeding, SDK conversion, IPC and
  startup. They retain raw samples by betting street/selection phase. That P99
  must not be read as a complete service deadline guarantee.
- Plain local guest processes are for trusted inspected research code; they do
  not enforce the production network/filesystem sandbox.

## Files and evidence

| Path | Purpose |
|---|---|
| `environment.py` | Small step API backed by the pinned engine |
| `arena.py`, `seeded_guest.py` | Parallel SDK matches and reproducible PRNG setup |
| `analyze.py` | Raw-record audit, independent scoring and clustered uncertainty |
| `audit.py` | Recompute scores/timings and verify exact source hashes |
| `measure_latency.py` | Separate source-function CPU timing |
| `configs/`, `candidates.json`, `PLAN.md` | Exact fields, seeds, source provenance and selection plan |
| `../external/` | Three pinned public repositories; no strategy edits |
| `../benchmarks/bay_area_2026-11-08/` | Frozen smoke, screen, selection, confirmation and latency evidence |

The environment is suitable for research and local evaluation. Public code
availability does not settle the event's rule against pre-written competition
code. See [the competition documentation](../docs/competition/rules.md).
