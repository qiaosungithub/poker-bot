# Boston 2026 event program

Manager objective: deliver five independently implemented handwritten strategies,
compare them on the released official variant, and refine within the 160-minute
reveal-to-final-snapshot window. No neural-network training. All competition
code starts after the official reveal. Existing research informs ideas only.

Repository: isolated event repository within the poker workspace. Each strategy
has its own Git branch/worktree. Do not push unless the user explicitly requests it.

| Branch | Strategy family | Initial hypothesis |
| --- | --- | --- |
| strategy/tournament | Tournament risk and ICM | Stack/payout-aware play beats chip-only heuristics |
| strategy/equity | Equity and pot odds | Bounded equity estimates improve calls and value bets |
| strategy/ranges | Range and mixed strategy | Position/range advantage and controlled bluffing generalize |
| strategy/exploit | Within-table exploitation | Conservative opponent updates improve weak-field performance |
| strategy/variant | Variant-specific decisions | Price the new mechanic explicitly rather than applying Hold'em blindly |

Release watcher: ../.runtime/competition-20261109/watch_release.py
Snapshots/log: ../.runtime/competition-20261109/
Official kit (once released): ../.runtime/competition-20261109/starter-kit/
Worktrees: ../.runtime/competition-20261109/branches/<family>/

Milestones relative to confirmed reveal: +15m first runnable bot; +25m all five
v1 candidates; +35m first matched comparison; +45m validated initial candidate.
Then compare/refine every 10–15m with fresh confirmation seeds and reserve a
validation margin before the actual round snapshot deadlines.

Benchmark contract: official variant, eight seats if supported, actual config
and payouts, fixed seed blocks with cyclic seat rotations, per-table memory
isolation, failure/illegal/timeout counters, points and uncertainty. Preserve
raw table records and source hashes. Separate tuning seeds from confirmation.
Measure decision latency separately from parallel strength work and on the
organizer build box; SDK/IPC inclusion must be stated.

Live observation at 2026-11-09T23:02Z: authenticated Boston team Timothy;
variant and announced timestamp null; starter download HTTP 404. Live
round_weighting=fibonacci, unlike older prose docs. Reconcile at reveal.
