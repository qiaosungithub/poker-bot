# Opponent-adaptive candidate

Written during the Boston event on November 9, 2026. This candidate extends our
new `value_heuristic` implementation with table-local opponent statistics.

- A fixed 172-byte memory stores folds, calls, checks, raises, observed auction
  participation, last flop-chip balance and total visible spending for ten seats.
- Priors shrink sparse samples. Call thresholds respond to the current street's
  last raiser. Opening ranges expand against frequent folders. Occasional small
  heads-up steals require observed folding and a free check option.
- The auction compares all ten possible flops. Observed balance reductions
  estimate total spending pressure, blended with the default first-three flop.
  High-value opportunities can justify more spending; quiet fields allow cheap
  bids. Lifetime budgeting and all-in urgency remain explicit.
- Memory includes a hand/history cursor. Re-delivering the same state with its
  returned memory does not count the events again. Balance changes are counted
  from their last observed level. Empty, truncated or unknown memory resets.
  Recovery scans at most 256 events to bound decision work.

Only public behavior is used. Total auction spending cannot identify individual
card bids, and the estimate is intentionally a coarse price proxy. Betting
strength uses the original event-time hand/texture heuristic and has no explicit
opponent range enumeration. A high fold rate also includes preflop folds, so the
bluff adjustment is conservative and may still be overgeneralized.

Validation at handoff

- Python 3.12 syntax compilation passed.
- Official static submission checker passed.
- Runtime and timing measurements await the shared eval7 installation and the
  parent's centralized tournament comparison. The design target is below
  0.4 ms per decision, which has not yet been measured.

No external strategy source was copied. The base implementation was authored by
our value-heuristic subagent during this event.
