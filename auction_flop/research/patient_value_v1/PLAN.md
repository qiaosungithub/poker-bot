# Patient value policy and key-card auction

User request: mostly conservative betting, but pursue value when genuinely
strong; recognize key auction cards rather than splitting bids equally.
Keep all work independent of other agents, preserve every existing candidate,
and do not upload. The experiment-loop workflow remains in use.

## Fixed designs before results

1. `simple_keycard_v1`: exact simple_tuned_v2 betting and total auction budget;
   only replace equal allocation with one-card replacement-loss weights.
2. `patient_value_v1`: the same key-card auction, tight full-table preflop,
   short-handed range relief, and a new small value-oriented postflop policy.

Auction: enumerate ten boards as before, select the same best board and total
budget, then weight each selected card by the average quality lost when that
card is replaced by either unselected card. Zero-loss fillers get no bid when
another card matters. If all three losses are zero, retain equal allocation
to defend interchangeable combinations. Integer remainder goes to greatest
fractional entitlement. This is marginal hand quality, not knowledge of hidden
opponent bids or an optimal sealed-auction solution.

Betting: avoid pure bluffs and expensive speculative calls; value-bet genuine
private two-pair/trips or better on every street when estimated equity supports
it; distinguish board-only made hands; top pair/overpair can take small value
when few opponents remain. Strong hands raise against ordinary bets and use
remaining-street/effective-stack sizing to build the pot. Equity uses 32
deterministic random-opponent samples, with auction dead cards excluded.
No claim of optimal play or action-conditioned opponent ranges.

Full tables use the original tight chart and cheap-call caps. With <=3 players,
allow a wider explicitly defined range, including any pair or ace, and use it
for short-stack opens. This avoids the previous fixed 8% short-stack opening
range in heads-up. Large contested preflop commitments remain restricted.

## Evaluation

Screen control, keycard-only and patient-value on 512 tables each, all with
fresh prefix `patient-value-v1-screen-20261110`, four frozen v1 panels and
eight rotations. This is a wiring/diagnostic comparison, not parameter search.
Run both declared candidates on complete unchanged 4,096-table v1 schedules
regardless of screen ranking, provided decisions are legal. Compare each with
retained simple_tuned_v2 (10.67139); report two comparisons with individual 95%
and Bonferroni-adjusted 97.5% intervals. Report effect and limitations rather
than selecting a winner from the screen. Preserve raw sources, games and
elapsed times. No further tuning in this cycle.

Tests must include pocket 77 with one auction 7 (focus bids on that card),
combo-card contributions, tiny balances, shared board strength, a private royal
flush that raises rather than folding/calling, weak hands facing large bets,
and a short-handed QJs opening that the previous policy folded. Test real SDK
actions plus final frozen-v1 audits. Separate any latency check from strength.
