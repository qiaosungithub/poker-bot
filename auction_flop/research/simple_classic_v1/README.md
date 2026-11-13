# Simple classic baseline

Completed result: **8.40237 points/table**, 95% interval **[8.03638, 8.76837]**,
4,096 tables, zero failed decisions. See [results](RESULTS.md).

This baseline implements the user's three-stage experiment independently of the
other candidates. The new decision rules occupy `simple_preflop.py` (68 lines)
and `simple_auction.py` (61 lines). The remaining policy and legality helpers
are retained classic code, with provenance recorded in `provenance.json`.

## Preflop

Blinds are posted by the engine. At our first betting decision, use these
opening ranges; `s` means suited, `o` means offsuit and `+` means stronger ranks.

| Position | Pairs | Other hands |
| --- | --- | --- |
| Early | 77+ | AJs+, KQs, AQo+ |
| Middle and blinds | 55+ | ATs+, KJs+, QJs, AJo+, KQo |
| Cutoff/button, or at most three players remain | 22+ | A2s+, K9s+, QTs+, JTs, T9s, 98s, ATo+, KJo+, QJo |

Open to 3 BB (2.5 BB late), plus 1 BB per limper. With at most 12 BB,
open-shove 66+, ATs+, AJo+ and KQs; check/fold other hands. Facing one raise,
three-bet QQ+/AK to three times the current bet; call 77+, ATs+, KQs, AJo+,
KQo only if the added cost is at most 4 BB and 15% of our stack. Facing
multiple raises or a call costing at least 30% of our stack, shove QQ+/AK;
at at most 20 BB, also include TT+/AQ. Everything else checks if free or folds.
All raises are clipped to the legal SDK limits.

## Auction: one tuning parameter

Enumerate the ten possible three-card flops. Give each a coarse private-hand
quality: weak pair 1, top pair/overpair 2, two pair 3, trips 4, straight 5,
flush 6, full house 7, quads 8, straight flush 9. Board-only pairs/trips have
no private value. Add 1 for a private flush draw and 0.5 for a private
four-of-five straight draw. This is a preference score, not estimated equity;
it does not model opponent cards or remove dead draw outs in this score.

Select the highest-quality flop, breaking ties by original card order:

```text
improvement = best quality - worst quality
total bid = min(balance, round(BASE_BID * improvement * 2 / live players))
```

Split this total equally across its three cards, with any extra chips assigned
in auction order. Count active and all-in players; exclude folded/busted seats.
The only tuning parameter is `BASE_BID = 20.0` in the candidate's `config.py`.
Zero disables bidding. Larger values spend the finite auction bankroll faster.
Normal chips, pot size, opponent bid histories and opponent identities do not
affect this policy. The balance only caps legal spending. Paying on the chosen
three cards does not guarantee that all three will win the sealed auction.

## Postflop

Reuse the exact `postflop`, `equity`, `hand_features`, RNG and action helper
functions from `classic_txnujk32_v1/classic_policy.py`; use the default unknown
opponent type and no profiling/memory. This retains 32 deterministic random
showdown samples, dead-card handling, shared-board/side-pot correctness helpers,
value betting, draw calls and pot control. No thresholds were tuned or repaired.
The known limitations of the donor also remain.

## Declared evaluation

Freeze BASE_BID=20 before reading results. Run a 64-table wiring check, then one
complete `auction-flop-v1` evaluation: four panels, 128 seed blocks per panel,
eight seat rotations, 4,096 tables. No clock penalties; this is a local fixed
field score, not live tournament standings. Compare against the retained
`classic_txnujk32_v1` control; other existing scores are descriptive context.
Do not select a parameter from this run and call it a fresh confirmation.

From the repository root:

```bash
.runtime/envs/auction-flop-benchmark312/bin/python -m unittest -v \
  auction_flop.research.simple_classic_v1.test_simple
.runtime/envs/auction-flop-benchmark312/bin/python -m auction_flop.benchmark.run evaluate \
  --candidate auction_flop/candidates/simple_classic_v1 --workers 16 \
  --out auction_flop/evidence/candidate-simple_classic_v1
.runtime/envs/auction-flop-benchmark312/bin/python -m auction_flop.benchmark.run audit \
  auction_flop/evidence/candidate-simple_classic_v1
```

Keep outputs outside the candidate directory. To test another BASE_BID, make
a separately named candidate snapshot and output directory; preserve this run.
