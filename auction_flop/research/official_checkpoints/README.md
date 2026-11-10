# Official-round checkpoint diagnosis

This analysis covers 40,000 scored hero tables from P4, S1 and P5. All three
reconstructed mean payouts and all eight finish-place counts exactly match
the official results. It motivates broad policy experiments; it does not
establish any new strategy's strength or promise a two-point gain.

The reproducible compact result is [macro-diagnosis.json](evidence/macro-diagnosis.json).
Full checkpoint summaries and reduced per-table records are under
`poker-bot/.runtime/official-checkpoint-results/{P4,S1,P5}/`:
`summary.json` and `table-checkpoints.jsonl.gz`. The compact artifact records
absolute paths, hashes, all denominators and intervals. No hole cards are read
by the analysis. Raw private archives stay under `.runtime/private-web/`.

## Exact version scope

| Round | Tables | Actual submission / source | Points | Win rate | Top-three rate |
| --- | ---: | --- | ---: | ---: | ---: |
| P4 | 10,000 | `fabb5704…`, old hybrid `7e7aa88` | 13.745 | 12.73% | 41.79% |
| S1 | 20,000 | `d3e77750…`, corrected loose `2873210` | 13.168 | 10.615% | 41.96% |
| P5 | 10,000 | `d3e77750…`, corrected loose `2873210` | 13.081 | 10.40% | 41.87% |

Official health records establish these mappings. Neither P4 nor S1/P5 measures
the `hybrid_both` package activated at 00:57 UTC. Policy version, time and the
opponent population are confounded across rounds. Differences are not causal
estimates of a code change.

## Where points change

For every hand, subtract the hero's logged pre-hand ICM points from post-hand
points, grouping by the number alive **before** the hand. These changes telescope
exactly to final points minus the initial 12.5. They include equity changes
caused by other players; they are accounting, not isolated action EV.

| Players before hand | S1 change per initial table | P5 change per initial table |
| --- | ---: | ---: |
| 8 through 5 | +1.012 | +0.944 |
| 4 | +0.000 | +0.031 |
| 3 | −0.105 | −0.115 |
| 2 | −0.239 | −0.279 |
| Total | +0.668 | +0.581 |

Each stage's denominator is every initial table, including zeros for tables
that never reach it. The full JSON also reports the conditional denominator.
Combined-stage intervals use the per-table sums, preserving covariance.

Heads-up entry is the first live hand with two players, measured before its
cards or actions determine an outcome:

| Round | Entries | Mean entry chip share | Actual wins | Win minus chip-share reference, 95% interval |
| --- | ---: | ---: | ---: | --- |
| P4 | 2,805 | 48.68% | 45.06% | −3.62 pp [−5.29, −1.95] |
| S1 | 5,223 | 44.68% | 40.11% | −4.57 pp [−5.74, −3.40] |
| P5 | 2,574 | 45.09% | 39.67% | −5.43 pp [−7.07, −3.79] |

The repeated deficit is clearest at 10–40 effective BB. S1 has 2,504 entries
at 10–20 BB (−6.03 pp [−7.80, −4.27]) and 806 at 20–40 BB
(−6.97 pp [−10.19, −3.76]). P5 repeats the pattern: 1,250 entries
at 10–20 BB (−8.67 pp [−11.13, −6.20]) and 412 at 20–40 BB
(−5.95 pp [−10.47, −1.43]). The below-5-BB intervals include zero.

Matching the heads-up chip-share reference would add only 0.239/0.279 points
per initial S1/P5 table. This is **not an upper bound** on heads-up improvement,
nor an estimate of a causal treatment. At unchanged heads-up entry rates, a
two-point gain from heads-up alone requires roughly a 38–39 percentage-point
absolute win-rate increase. Broader opening, continuation and auction policies
are more plausible targets than repairing a few selected losses.

At the first three-player-or-fewer checkpoint, S1 leaders have 53.08% chip share
and win 48.74% (n=1,984); P5 leaders have 53.44% and win 48.56% (n=1,007).
Middle stacks' final-minus-checkpoint-ICM residuals are −1.109 points in S1
(n=3,165) and −1.409 in P5 (n=1,527). The JSON additionally separates exact
three-player and four-player states from the rare skipped stages.

Win rate alone is not the objective. Corrected loose earns above-symmetric
top-three rates but below-symmetric wins. In S1, the payout decomposition is
`+0.892 top3 + 0.153 top2 − 0.377 wins = +0.668 points`. Increasing wins while
losing enough paid finishes could make a candidate worse.

## Does the deployed source optimize ICM?

The audited package is `hybrid_both`, source `0c60de2`, ZIP SHA256
`3d5e6595a3c1f73f5a8ed051a25200609f19a3132487185739661b52a48520ef`.
It has no explicit ICM function calls or per-action payout-vector evaluation.
The benchmark optimizes selection by final tournament points and uses ICM at
the hand cap. That does not make the decision policy an ICM optimizer.

The active call path dispatches preflop to the loose threshold policy, which
has one fixed bubble call-threshold increment. Postflop uses the classic
hand-equity, pot-odds and commitment rules. Its sampled opponent cards are
uniform over unknown cards, not conditioned on the betting sequence.
`TournamentAgent` is invoked only for auctions; that branch returns before
its separate betting `risk_premium` code. Merely finding that helper in the
package would misidentify the executed policy.

## Three macro hypotheses

1. **Range-conditioned action values across streets.** Replace disconnected
   hand-class thresholds with fold/call/raise values against bounded ranges
   inferred from public actions, positions and bet sizes. The 10–40-BB conversion
   deficit motivates this, but coverage must extend beyond heads-up to target
   a two-point gain. Use fixed tables and a small action set within runtime limits.
2. **Tournament-point values at decision leaves.** Price win/lose/split stack
   vectors with actual payouts and coverage, instead of a blanket risk margin.
   Heads-up ICM is linear in chip share, so adding bubble aversion there cannot
   repair the observed deficit. Distinguish this objective layer from opponent
   range estimation; do not infer benefits solely from ICM residuals.
3. **Joint auction and continuation value.** Use the same downstream value model
   to score candidate flops and auction bids, including the future value of
   auction inventory. Low remaining inventory correlates with worse conversion
   in broad groups, but finer matched groups are sparse; evidence for a simple
   spend-more/save-more adjustment is weak. Ordinary and auction chips must
   retain separate units.

These are hypotheses for broad candidates, not implemented policy changes.
Evaluation should compare the exact candidate against Hboth on the fixed
regression benchmark and a separately named fresh field/seed holdout. No
benchmark, upload, activation or publication is performed by this diagnosis.

## Selection, uncertainty and validation

Checkpoints are fixed hand starts (1/21/41/61/81/101/121/161/201) and the first
live hand with at most 6/4/3/2 players. Strata use only contemporaneous stack
share, rank, ordinary-chip BB depth, players left and auction inventory.
Later checkpoints are survivor-conditioned; missing tables are separately
counted as already completed or hero already busted. No losing-hand selection
or future cards enter features.

Intervals are pointwise normal 95% intervals over iterations. Exactly one
non-filler hero table per iteration is checked. Thousands of exploratory
strata are not multiplicity-adjusted; sparse cells should not drive decisions.
ICM and chip proportionality are descriptive references, not calibrated truth.
Expected rank is independently checked via full ranking-permutation enumeration.

Official finish rank uses elimination order, starting-stack tie rules, and
chip rank for survivors at the hand cap. Thus cap-ranked outcomes are included,
not silently counted as completed winner-take-all contests. There are only
3 capped P4 tables, 0 S1 tables and 1 P5 table. The official place histograms
match exactly. Four tests cover ICM arithmetic, future-data exclusion with a
negative control, simultaneous bust/cap rules and the correct stage denominator.

## Reproduction

From this worktree, using Python 3.12:

```bash
python -m unittest auction_flop.research.official_checkpoints.test_diagnose -v
python -m auction_flop.research.official_checkpoints.diagnose \
  --name S1 --history PATH/scored1-hands-all.jsonl.gz \
  --results PATH/scored1-results.json --health PATH/health.json --out NEW_OUTPUT/S1
# Repeat for P4 and P5 with their exact archived inputs.
python -m auction_flop.research.official_checkpoints.report \
  --inputs NEW_OUTPUT --candidate auction_flop/candidates/hybrid_both \
  --out NEW_OUTPUT/macro-diagnosis.json
```

Raw archive identity and official round IDs are recorded in the output. The
reduced checkpoint records suffice to regenerate the compact report without
reading private raw histories again.
