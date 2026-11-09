# Variant-first strategy direction

Updated after user review, 2026-11-09T23:21Z. Keep runnable v1 candidates as
controls. All improvements are handwritten rules or bounded simulations, no
neural training. Optimize expected tournament payout, not side-prize metrics.

## Decision objective

For public history H, own hand, normal chips S and auction balance B, compare
legal bid vectors b by approximate expected continuation value minus the future
opportunity cost of consumed flop-chips. The realized flop F is the top three
aggregate bids from us and opponents; cost is sum of our bids on F, not all bids.
An improvement must change outcome probabilities or reduce cost relative to
zero/alternative bids, not merely favor a board we like.

- Evaluate all ten possible three-card flops by relative equity/continuation
  value against common opponent scenarios. Raw evaluator rank is not equity.
  Remove the two known dead candidates from every downstream deck.
- Bid influence is marginal: opponents may provide useful votes for free;
  supporting one card can be useless without its complementary cards. Use
  a small set of bid vectors and plausible aggregate opponent-bid scenarios.
- Auction currency does not enter pots or ICM chip stacks. Its shadow cost
  depends on useful auctions remaining, survival probability, future normal
  stakes and opportunities. Do not spread 1000 mechanically over 250 hands.
- Spend where selection has meaningful effect on accessible-pot/tournament
  utility, including all-in survival opportunities. Respect side-pot eligibility.
- Preflop hand values change under five candidates and imperfect board control;
  own and opposing auction budgets affect realization. Test altered ranges,
  rather than asserting pairs/suited/connectors always improve.
- Postauction boards are selected using private preferences. Infer opponent
  ranges conservatively from preflop actions, chosen candidates and public total
  spend. Hidden per-card bids and holes remain unavailable.

## Branch responsibilities after v1

| Branch | Main mechanism to test |
| --- | --- |
| tournament | Dynamic scarce-currency value, survival/ICM/remaining-hand urgency |
| equity | Ten-flop relative equity with matched opponent/runout samples |
| variant | Bid influence, free-riding, minimal pivotal spending, robust scenarios |
| exploit | Within-table auction behavior and conservative postauction inference |
| ranges | Auction-aware preflop participation and range-conditioned continuation |

## Comparison discipline

First compare all five frozen v1 families under identical official rules and
rotated seed blocks. Then hold betting fixed while changing auction policy,
and auction fixed while changing betting. Include zero-bid and adaptation-off
controls, source hashes, official payout, errors, spending/remaining-budget
statistics and independent confirmation seeds. Do not attribute a whole-policy
score improvement to auction logic without the controlled comparison.

Latency remains a hard selection concern: prune optional search before spending
the official 1 ms margin. Pure policy timings cannot certify SDK/IPC or microVM
acceptance. Validate source-only packaged candidates on the official build box.
