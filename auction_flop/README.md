# Auction Flop: shared strategies and fixed opponents

This directory publishes the other event agent's completed strategy versions,
committed research modules, and **auction-flop-v1**, a permanent comparison
against four selected opponents. See [results](REPORT.md) and the
[predeclared protocol](benchmark/PLAN.md).

## Strategies to read

| Source | Strategy |
| --- | --- |
| [strategies/variant](strategies/variant/agent.py) | Combinatorial auction preferences and bounded hand rules |
| [strategies/tournament](strategies/tournament/agent.py) | Tournament risk and auction budget allocation |
| [strategies/ranges](strategies/ranges/agent.py) | Position, ranges, draw-aware betting and deterministic mixing |
| [strategies/exploit](strategies/exploit/agent.py) | Within-table opponent updates and selective exploitation |
| [strategies/equity](strategies/equity/agent.py) | Bounded equity estimates and auction intervention |
| `hybrid_tournament_auction` in [strategies.json](strategies.json) | Variant betting plus tournament auction; both original donors receive public observations with separate byte memories |

These are the exact measured v1 source bytes, including their existing limitations.
Original event commit IDs and hashes are recorded in `strategies.json`. The hybrid
is a benchmark composition, available through `--reference`; its production
modular implementation is in [research/modules/bot](research/modules/bot/agent.py).
Direct-call benchmark timing is not a production deadline qualification.

[research/](research/provenance.json) contains committed snapshots from all nine
event branches available at publication: the five policy families, shared modules,
auction history, betting inference, and the joint-policy design. It includes tests,
contracts, ablations and their original evidence where committed. These are
review material, not additional selected opponents. Historical absolute paths in
those snapshots refer to the originating workspace. The joint branch had design
documents but no committed implementation at capture. Each branch revision is
fixed in the provenance manifest; subsequent work is not silently imported.

## New modular candidates

Nine later ready-to-run arms are in [candidate_catalog.json](candidate_catalog.json) and
`candidates/`. They include the strong modular baseline, history/inference 2x2,
tight/loose preflop, and joint ON/OFF. These newer snapshots supplement the earlier
`research/` capture. Run any one with `--candidate auction_flop/candidates/NAME`.
Their fixed-field comparison is pending; the frozen opponents remain unchanged.

## Run after cloning

Use Python 3.12 on Linux and run commands from the repository root. No model,
private event directory, account, submodule checkout or API key is needed.
The runner loads the byte-for-byte released SDK under `vendor/` directly.

```bash
python3.12 -m venv .runtime/auction-benchmark
.runtime/auction-benchmark/bin/python -m pip install -r auction_flop/requirements.txt
.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run verify
.runtime/auction-benchmark/bin/python -m unittest -v auction_flop.benchmark.test_benchmark

# A published baseline, 4,096 tables against the permanent opponents.
.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run evaluate \
  --reference hybrid_tournament_auction --workers 8 --out .runtime/baseline-run

# Your folder must contain agent.py with the official Agent interface.
.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run evaluate \
  --candidate /path/to/my-bot --workers 8 --out .runtime/my-bot-run

.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run audit .runtime/my-bot-run
.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run compare \
  .runtime/my-bot-run .runtime/baseline-run
```

All outputs must be new directories outside the candidate's source folder.
Add `--smoke-blocks 2` for a 64-table wiring check; it is explicitly a non-score
run and is rejected by `compare`. The left-minus-right comparison reports a
paired difference in table points with a 95% interval.

Candidates must keep table state in the SDK byte memory and use deterministic
state/memory-derived randomness for reproducible comparisons. The frozen
policies meet this condition. A stochastic candidate using unseeded process
globals does not have a reproducible score. The runner is for trusted local
research code, not an untrusted-code sandbox.

The new selection experiment can be reproduced without changing the frozen
opponents:

```bash
.runtime/auction-benchmark/bin/python -m auction_flop.benchmark.run selection \
  --workers 8 --out .runtime/repeat-selection
```

## Permanent comparison contract

The opponents are `hybrid_tournament_auction`, `variant`, `tournament`, and
`ranges`. Four panels each contain one candidate plus seven opponent instances.
Each selected strategy appears once in one panel and twice in the other three,
giving equal exposure overall. Each panel has 128 independent seeds and eight
cyclic seat rotations. Sources and memory are isolated per instance/table.

Rules: eight seats, 2,000 ordinary chips, 1,000 auction chips, 250-hand cap,
released blind progression, fixed 50/30/20 payouts and ICM at the cap. The main
metric is candidate points per table, higher is better. The table distributes
100 points; an identical eight-copy policy field has symmetry expectation 12.5.
Do not compare scores across different opponent fields or benchmark versions.

[v1.json](benchmark/v1.json) and its SHA256 file lock opponent source, SDK,
rules, runner, weights and seed schedules. The runner refuses changed files,
incomplete results, incorrect rotations, invalid payouts and failed decisions.
Raw tables, source snapshots, manifests and independent score audits accompany
each result. Source paths in a run manifest are historical; auditing retained
results uses their adjacent snapshots and works after moving the checkout.

Never update v1 in place. Keep future improvements in new candidate folders.
A new opponent set or ruleset needs a separately named benchmark version, while
v1 remains runnable. The public fixed seeds are for regression tracking; use
separately named fresh holdouts for claims beyond this benchmark. Parallel
strength runs have no clock penalties; production SDK/IPC latency is separate.

The losslessly compressed [evidence/](evidence/) preserves the preceding event
experiments and the fresh 4,096-table selection. It also retains all four
calibration runs against the new fixed field. Historical manifests preserve
their original source paths and uncompressed table hashes.
