# SQA auction-flop-v1: macOS replication

`icm_defend_v7` scored **12.8466796875** points per table, 95% interval
**[12.2826289925, 13.4107303825]**, across all 4,096 scheduled tables and 512
independent blocks. There were zero failed decisions. The full independent
audit passed. Against the retained `hybrid_tournament_auction` reference,
the paired difference is **-2.2412109375**, 95% interval
**[-3.0533876650, -1.4290342100]**.

This is a **macOS portability-adapted replication**, not Linux qualification.
The frozen runner sets `official_score: true` for a complete schedule; that field
does not certify the platform. The separate unmodified Linux run remains the
standard-platform confirmation. At the recorded partial checkpoint, all 2,120
available Linux rows matched these Mac rows exactly after removing only elapsed
seconds. This partial comparison is not a completed Linux score.

The snapshot comes from main commit
`481d1f05eb121762dffaf493b55a0c88fd9584af`. Frozen source files, game rules, seeds,
opponent policies, source hashes, scoring, and Python hash seed were unchanged.
The adapter only supplies missing `os.sched_getaffinity` (all logical CPUs) and
`os.sched_setaffinity` (no-op). It preserves the runner's `PYTHONHASHSEED=0` and
installs the same shims in spawned workers. This changes CPU pinning, not game
logic. Direct-call timing is not production latency evidence.

- Candidate SHA256: `30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`.
- Adapter SHA256: `62f8fde5b1b195ee70f2ebed88de647335ffd93d4350a63b1805ed8b7963ef26`.
- Full platform and command provenance: `platform-adapter.json`.
- Independent audit: `audit-macos.log`.
- Paired reference comparison: `compare-hybrid-macos.log`.
- Cross-platform partial check: `cross-platform-partial.json`.
- Raw schedule and immutable policy snapshots: `tables.jsonl.gz`, `manifest.json`, `sources/`.

The subsequently published SQA `preflop_loose` result is **16.5625** points,
95% interval **[15.9753235327, 17.1496764673]**, 4,096 tables, zero failures.
Its exact published artifact is
[`candidate-preflop_loose/summary.json`](https://github.com/qiaosungithub/poker-bot/blob/c5ec82ae4b1ba0abc35395cfc2127996b25be69b/auction_flop/evidence/candidate-preflop_loose/summary.json),
at commit `c5ec82ae4b1ba0abc35395cfc2127996b25be69b`, Git blob
`24b5617822ba91eca75c2ddcaddc34a081b4471b`.
No paired comparison against that newer candidate was calculated here.

Earlier Dianne-field scores use different opponents and must not be mixed into
this benchmark's rankings.
