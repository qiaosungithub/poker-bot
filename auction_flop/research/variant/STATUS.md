# Variant strategy v1

Fresh event implementation after the user-provided live reveal. Official kit
read at release, 2026-11-09T23:14:42Z. Production files are `agent.py` and
`auction.py`; `package.py` builds a source-only ZIP with the official checker.

Auction: score all ten complete three-card boards, then form minimum-forcing
bids for two strong target boards in three anonymous public-budget scenarios.
Compare expected complete-board utility minus selected-card-only cost. The
currency opportunity-cost term depends on useful remaining auctions, normal
stack depth, pot size, all-in status and hand cap; normal chips and auction
currency never mix in accounting. Bid-vector sums never exceed legal balance.

Betting: fresh position/stack-depth preflop rules, numeric eval7 categories,
private-card contribution, draw and board-danger adjustments, pot-odds margins,
bounded value betting and rare deterministic bluffs. No learned model, no
per-table memory, no cross-table state. Hand utility is a proxy, not exact equity.

Validation: nine targeted tests; twelve direct official-engine tables across
2/3/8 seats, 6,458 decisions with zero illegal/exception failures. Auction
policy-only P99 0.300 ms; all betting-phase P99 below 0.052 ms. These direct
measurements precede a small zero-opponent search/bubble-rule correction.
The final exact source is validated separately by the process smoke below.

Official BotPool + guest validation uses Python 3.12, eval7 0.1.11 and the new
released SDK modules supplied through PYTHONPATH, while the root prepares the
shared fully pinned environment. Six heads-up tables have a zero time bank and
20-hand cap; three eight-seat tables use 100 ms bank and 100-hand cap. Delivery
grace is the SDK default 2 ms. Raw transport timings, source hashes, failures,
seeds and phase summaries are in `evidence/process_smoke.json`. This is local
process testing, not an official production microVM acceptance result.

Package identity/path and exact two source hashes: `evidence/package.json`.
Frozen richer auction baseline: `experiments/rich_kernel_v1.py`.

Next: compare changed outcomes against zero bidding, free-ride where a desired
card is selected anyway, and replace arbitrary opposition profiles with
public-card/range-aware scenarios. Freeze v1 for common-field benchmarking
before evaluating any refinement on fresh seeds.

## Fully pinned environment recheck and shared qualification CLI

The root's official environment has now checked the same v1 ZIP and run all
nine process-smoke tables. Zero decision failures; auction P99 0.462 ms and
maximum 0.517 ms. One ordinary betting request measured 1.105 ms, within the
SDK delivery grace; this is retained rather than called a strict 1 ms pass.
Raw final-source data: `evidence/process_smoke-pinnedenv.json`.

`smoke_process.py` is now a reusable CLI for root's independent remote checks:

```
python smoke_process.py --bot-dir FROZEN_BOT --kit-dir OFFICIAL_KIT \
  --output FRESH_REPORT.json --acceptance-tables 6 --eight-seat-tables 3
```

It hashes every frozen input file before/after, rejects symlinks, records the
actual eval7/Python/module identity, zero-bank acceptance settings and 2 ms
delivery grace, captures SDK-reported transport times/errors, and exits nonzero
on any hero failure, startup failure or source mutation. The input bot is never
modified. The CLI itself was exercised on two acceptance tables and one
8-seat table with a fresh output record.
