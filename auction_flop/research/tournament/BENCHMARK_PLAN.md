# Event benchmark implementation plan

Owner: tournament_manager, after the first runnable tournament baseline.
Design only before release; implement against the actual Boston starter kit.
Inspected read-only: bay_area/{arena,environment,analyze}.py, bay_area/PLAN.md,
and heuristic_holdem/PLAN.md. Their architecture is useful; their variant,
constants, old policies and timing grace must not be assumed for Boston.

## Two execution modes and their claims

1. Fast strength mode uses the released official table/variant engine directly
   with a narrow in-process SDK adapter. It owns separate seat memory, constructs
   the same GameState, validates replies using the official protocol, and calls
   the official apply/fallback path. It never reimplements betting, variant
   transitions, side pots or settlement. Clock penalties are disabled for fast
   parallel strength comparisons, clearly marked in every result.
2. SDK smoke/confirmation uses the actual host/guest process protocol and released
   BotProcess/BotPool/play_table equivalents. Run the same deterministic bots,
   config and seed on both modes; compare decisions and final settlement. Check
   exception, illegal action, memory truncation and timeout/event-replay parity
   with deliberate faulty controls. A direct-mode score is not upload acceptance.
3. Separate low-load timing captures real played states and explicit memory,
   measures complete act() wall and CPU time, and then SDK service latency on the
   build box. Report both scopes separately. Use actual kit grace/bank settings;
   do not inherit the Bay Area 2 ms grace. Acceptance smoke must include zero
   bank if the released protocol permits that setting.

Prefer a small SDK smoke immediately over building an elaborate wrapper. If
in-process parity is costly, begin all comparisons with the official CLI/API;
optimize only after one measured complete table.

## Protocol and field

Use the live variant and official TableRules: eight seats, actual starting
stacks, blind schedule, hand cap, payouts and timing. Record every effective
config; never hard-code the old 16,000-chip/100-point conservation constants.
Freeze one eight-seat mixed field containing the five event branches plus three
clearly identified released examples or freshly written controls. Keep the same
three controls across cycles. This mixed field is the main comparison.

Complement it with one-candidate-at-a-time screens against seven fixed controls
when time allows, using the same seats/deals across candidate substitutions.
Controls should include passive, aggressive and incumbent-strength behavior,
not merely seven check/fold bots. Label control screens separately: they measure
exploitation of that field, not general superiority. Save old champion snapshots
as fixed references after the first cycle; do not let all controls change with
all candidates simultaneously.

For each independent seed block, shuffle the eight seat entries once and play
all eight cyclic rotations with the same underlying deal seed. Candidate RNG
must depend only on permitted observation/own memory, never hidden cards or
opponent labels. Duplicate control copies have distinct entries and isolated
memory. Every bot sees anonymous seat IDs; no strategy-name leakage.

## Fixed budget per cycle

- Smoke: 1 block, full official hand cap unless failure debugging requires a
  separately labeled shorter run. It establishes execution, not strength.
- First screen: 16 independent blocks x 8 rotations = 128 mixed tables.
- Routine screen: 32 blocks x 8 = 256 mixed tables, subject to measured throughput.
- Confirmation: 64 new blocks x 8 = 512 tables for a frozen selected candidate
  field; grow sample size only in a later predeclared cycle, not by testing until
  significance appears. If speed permits, use 128 blocks as planned in advance.
- Target new comparison every 10–15 minutes. Benchmark wall-time target under
  4 minutes per routine cycle; report smaller samples honestly if necessary.

Screen seed namespace: boston-20261109-screen-<cycle>-<block>.
Confirmation namespace: boston-20261109-confirm-<cycle>-<block>.
Reserve confirmation namespaces before tuning and do not reuse them for further
parameter selection. A later refined version needs new confirmation blocks.

## Scoring, auditing and uncertainty

Use official table_points and retain final stacks, bust order, tied eliminations,
actual payouts, hand count and every seat's score. Independently recompute ICM
for survivors and tied finishing payouts; check sum of stacks equals initial
chip supply and sum of all scores equals total table payouts. If variant rules
add/remove chips, derive and record the correct conservation law first.

Average rotations within each seed block, then calculate mean points, block
standard error and 95% intervals. Pairwise differences use matched block means;
for the five candidate families report simultaneous intervals across all ten
pairs. Report unresolved leaders when intervals overlap zero; do not name a
proven winner from sample means alone. Also report chips, bust/paid-place rates,
full-table completion and hand-cap frequency, but select by official points.
Tournament-wide Fibonacci weighting is distinct from per-table points.

Do not drop failures, slow calls, bad seeds or crashed blocks from comparisons.
Report incomplete runs separately; require comparable completed blocks for a
ranking. Preserve every exception and official fallback count by candidate and
phase. Independent audit must reject a deliberately corrupted score record.

## Artifacts and frozen inputs

Each cycle creates a new directory, never overwriting older measurements:
.runtime/competition-20261109/benchmarks/<UTC>-<cycle>-<mode>/

Before launch, snapshot each bot's complete submission directory to immutable
cycle-local inputs. Record branch, commit, dirty status and SHA256 of actual
bytes; branch HEAD alone is insufficient. Never benchmark a worker's live file
while it is being edited. Retain runner/kit hashes, Python/package versions,
CPU affinity, process count, effective config, seed plan and timing scope.

Outputs: manifest.json, tables.jsonl, summary.json, table.csv, failures.jsonl,
and raw timing samples plus source snapshot. Include completion timestamps,
counts and raw data hashes. Final summary states sample size, independent
blocks, points/CI, paired incumbent difference, P99/max/over-budget count,
illegal/exception/timeout counts and what the next iteration will change.

## Initial status

At 2026-11-09T23:05:54Z the watcher still recorded variant=null and kit HTTP 404.
No RELEASE.json was present during this planning task. No implementation was
written or existing frozen benchmark altered.
