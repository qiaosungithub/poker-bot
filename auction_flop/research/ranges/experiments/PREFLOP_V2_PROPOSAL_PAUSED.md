# Paused preflop v2 proposal

Status: paused by manager direction on 2026-11-09 after the user requested
separate auction and betting modules. No v2 strategy code, simulation, or new
candidate has been produced. Frozen v1 remains commit ef66f6d. Do not interpret
this note as an executed experiment or a claimed improvement.

## Proposed isolated experiment

Keep auction and postflop decisions byte-for-byte unchanged. Evaluate an optional
preflop-only strength correction from event-time deterministic simulation, with
zero neural/CFR training. Its purpose would be to test, rather than assume,
changes in the value of pairs, suitedness and connectedness under auction-flop.

For each of the 169 starting-hand classes, simulate blocker-valid opponent
private cards, five public candidates, and turn/river from the remaining deck;
the two unselected candidates remain dead. Use one and three opponents as
interpolation anchors. Auction decisions must depend only on each bidder's own
hole cards and the five candidates, never sampled opponents' cards or future
turn/river. Future cards are used only to score completed simulated showdowns.

Use a zero-bid reference (first three candidates) and two simple non-oracle
bidder models: equal support for the bidder's favorite current three-card flop,
and support weighted by current marginal card preference. Apply relative own
bid capacities 0, 1 and 3 versus each opponent's capacity 1. Reuse identical
deals/runouts across regimes and collect per-deal equity differences and their
squared values, not only independent mean equities.

A small correction would shrink estimated differences toward zero using a fixed
prior scale, interpolate by live player count and own/public-opponent remaining
flop-chip ratio, and cap the change to the existing handwritten starting score.
All-zero balances imply no correction. The public balance ratio is only a proxy
for future bidding ability, not an observed bid or a causal guarantee of control.

Predeclare two correction scales before any strength evaluation; compare both
to unchanged v1 with fixed auction/postflop behavior, and use fresh confirmation
seeds. Never change starting_strength itself, because the v1 auction and
postflop opponent sampler uses it: that would confound the intended ablation.

## Limitations and next architecture

The proposed opponent auction models are intentionally crude. Small simulation
noise, ignored betting realization, uncertain continuation ranges, and future
flop-chip opportunity costs can overwhelm a minor hand-ordering correction.
No unsupported claim that any hand family is always stronger is warranted.

The user's new architecture prioritizes separate auction and betting policies.
A future betting module should use the actual two dead cards and each seat's
public current-hand paid flop-chips as conservative range evidence, and obtain
that information from a shared, reviewed state/memory contract. This proposal
is retained for possible later isolated testing, not active implementation.
