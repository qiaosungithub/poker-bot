# Classic policy cross-review before further development

Source: event revision `0f84eb45a7501d1cc24c423244b298134c4a4921`.
Comparator: published `corrected_preflop_loose_v2`. No strategy source was edited.
At the time of this review the 4,096-table strength run had not started: steering
requested cross-review before new development and bulk comparisons. The full
run subsequently completed; see the current REPORT.md for its measured result. A 32-table interface
smoke completed with 5,395 candidate decisions and zero failed decisions; it is
not a ranking result.

## Observed differences

The candidates share the tournament auction family, but their complete betting
policies differ. Classic uses its regenerated 169-class preflop equity table,
opponent action classes, 32-sample multiway postflop equity, categorical stack
commitment, and explicit street-specific pot control. The current best uses a
rank-based preflop proxy with blind/position adjustments and a corrected
postflop quality proxy with payout/bubble margins. A full comparison would not
isolate a single auction or equity-estimation mechanism.

Pure helper probes are retained in `decision-feature-probes.json`. They are not
native-game outcomes or estimates of tournament frequency:

| Hole / board | Classic output | Current best output |
| --- | --- | --- |
| `2c 3d` / `7s 7h 7d` | Made category 3, commitment 2 | Quality 0.31, trips category |
| `2c 6d` / `As Js 8s 4s` | Flush draw true despite no private spade | Quality 0.01, draw false |
| `Ah Kd` / `Ac 7d 2s` | Top-pair category 2, commitment 1 | Quality 0.675, pair category |

In `classic_policy.py`, the stack-off branch (cost at least 55% of total current
stack) calls whenever commitment is at least 2 and otherwise folds. It ignores
the equity estimate already computed for that decision. The large-call branch
also calls every commitment-2 hand without an equity threshold. Thus a public
trips classification can bypass a poor equity estimate, while top pair folds
under stack-off sizing regardless of its sampled equity. The latter is a
deliberate conservative rule, not automatically an error: its value depends on
opponent ranges and tournament risk.

The draw detector counts any suit appearing four times across private and board
cards. In the second probe it therefore assigns a nine-out flush draw to an
entirely public four-flush. The continuation branch can use this outs-based
estimate independently of the lower Monte Carlo estimate. A fifth board spade
does not give the hero a private flush advantage. The existing coarse straight
draw detector also treats some wheel/gutshot shapes as an eight-out draw, as
already acknowledged in the source-to-rule report.

These source predicates motivate targeted review; the probes alone do not
establish which change improves tournament payout.

## Four separate refinement hypotheses

1. Add a private-contribution/equity check only to commitment-2 large calls,
   targeting public trips and weak one-card made hands; retain other branches.
2. Correct private draw contribution and dead-card outs only, preserving the
   remaining action thresholds and the same auction.
3. Adjust only unopened position/blind ranges, retaining the preflop equity
   table and all postflop rules; keep heads-up blinds distinct from button steals.
4. Add only a bounded payout/bubble margin to costly calls, using actual payouts
   and eligible ordinary-chip pots, without changing auction spending.

Each is a hypothesis for a separately frozen candidate after reflection. Keep
the current classic source as a control and preserve auction-flop-v1 opponents,
rules, seed schedule, and historical results. Do not infer the value of history
from whole-policy losses that also replace auction/value selection.
