# Patient value policy: completed comparison

The requested mostly conservative, strong-value bot scored **12.02393** points
per table, versus **10.67139** for simple_tuned_v2, on the same full frozen
auction-flop-v1 schedule. The paired improvement is **+1.35254**, with a
multiplicity-adjusted interval excluding zero. It remains below the previously
measured classic_hybrid_v1 reference at 17.48047; this is not an optimal-policy
claim.

| Candidate | Mean points/table | 95% mean interval | Gain vs tuned simple | Family-adjusted interval for gain |
| --- | ---: | --- | ---: | --- |
| simple_tuned_v2, retained control | 10.67139 | [10.26390, 11.07887] | — | — |
| simple_keycard_v1, auction allocation only | 10.80311 | [10.38048, 11.22573] | +0.13172 | [-0.22622, +0.48966] |
| patient_value_v1, complete requested style | 12.02393 | [11.54958, 12.49827] | +1.35254 | [+0.74278, +1.96229] |

The adjusted intervals are Bonferroni 97.5% intervals for the two declared
comparisons; individual 95% paired intervals are in [comparison.json](comparison.json).
Each candidate played 4,096 tables, four fixed panels, 128 blocks per panel and
eight seat rotations. No opponent/rule/seed changes or clock penalties.
Both candidates and all opponents had zero failed decisions: patient made
590,220 decisions, keycard-only made 563,586.

## What the bot does

- At full tables, retain the original tight preflop chart, modest call caps and
  strong-hand raises. Against major preflop pressure, deep commitments require
  KK+; TT+/AQ can commit at <=20 BB.
- With <=3 players, explicitly widen opening eligibility to any pair or ace,
  K8+, JT+ broadways and suited connectors 87s+. Use that range for short-stack
  opens rather than waiting for the previous approximately 8% premium range.
  This is a small heuristic range, not a solved push/fold chart. Facing raises
  remains conservative.
- No pure postflop bluffs. Marginal hands and draws continue only for a modest
  price with an equity margin; checking is the default when weak.
- Genuine private two pair/trips or better value-bets on flop, turn and river
  when a bounded equity estimate supports it. Top pair/overpair can make small
  value bets against few opponents. Strong hands raise ordinary bets, and
  exceptionally strong hands can raise even against large bets.
- Value sizing considers effective stack and remaining streets, using bounded
  geometric pot growth. This aims to build a worthwhile pot while calls are
  available; it does not prove the bet size maximizes expected value.
- Preserve shared-board/side-pot guards and 32 deterministic equity samples.
  Draws and opponent cards are sampled from the legal deck, excluding auction
  dead cards. Opponent hands are uniform samples, not betting-conditioned ranges.

The actual decision source is [value_postflop.py](../../candidates/patient_value_v1/value_postflop.py)
and [simple_preflop.py](../../candidates/patient_value_v1/simple_preflop.py).
The patient bot won all chips at **175/4,096 tables (4.27%)**, compared with
40/4,096 (0.98%) for the retained tuned simple bot. Keycard-only won 51 (1.25%).
The patient version changes both preflop and postflop; their individual causal
contributions were not separated in this experiment.

## Key cards are a small, explicit calculation

Both new bots use the same [auction function](../../candidates/simple_keycard_v1/simple_auction.py).
Score all ten possible flops as before. For each card in the selected flop,
replace it with either outside card and measure the average lost hand quality.
Divide the existing total budget in proportion to those losses. Zero-loss
filler cards get no bid when another card matters. If all losses are zero, use
equal weights to protect interchangeable combinations.

Constructed test: hole cards **7s 7h**, candidates **7c Kd Qh 8s 2c**, two live
players, ample budget. The old bid is **[60, 60, 60, 0, 0]**. The new bid is
**[180, 0, 0, 0, 0]**. The 7 buys trips; the two selected high cards are replaceable.
The total budget and chosen target board are unchanged.

Keycard-only differs from simple_tuned_v2 in exactly one source file, with
1,000 random legal states confirming identical total bids. This isolates the
allocation change. Its full-score gain is statistically unresolved: recognizing
key cards is straightforward, but the evidence does not establish equal
allocation as the major cause of the earlier score gap. Marginal hand quality
also does not estimate hidden opponent bids or guarantee an auction win.

## Runtime and checks

| Work | Tables | Seconds |
| --- | ---: | ---: |
| Fresh diagnostic screen, all three arms | 1,536 | 23.32 |
| Patient full v1 evaluation | 4,096 | 62.60 |
| Keycard-only full v1 evaluation | 4,096 | 61.88 |
| Total strength evaluation | 9,728 | 147.80 |

Both full runs used 32 workers on a selected set of distinct physical core IDs;
shared host load remains uncontrolled. Source snapshots and runner audits are
included in these times; implementation and separate validations are excluded.
Both designs were declared in [PLAN.md](PLAN.md) before screen results and both
were confirmed regardless of screen rank. No parameter search followed.

The 512-table-per-arm screen used new seeds disjoint from permanent v1:

| Arm | Screen points/table | Failures |
| --- | ---: | ---: |
| patient | 11.42578 | 0 |
| keycard | 11.10975 | 0 |
| control | 10.68948 | 0 |

Eight behavioral tests passed, covering critical 7 allocation, three-card
synergy, tiny balances, genuine versus shared strength, private royal-flush
raising, strong flush betting on all three streets, weak-hand restraint and
short-handed QJs participation. Static SDK packaging passed. Two eight-seat,
20-hand official SDK subprocess tables had zero failed decisions for all bots
at a generous 1,000 ms deadline. No 1 ms production qualification is claimed.

All source snapshots, ZIP contents and candidate directories match by hash.
Complete v1 source/schedule/independent-payout audits passed, and direct raw
means independently reproduced the reported values. No old candidate, frozen
benchmark file or other agent's source was changed. No upload, commit or push.

## Files

- [Patient bot](../../candidates/patient_value_v1/agent.py) and [ZIP](patient_value_v1.zip)
- [Keycard-only bot](../../candidates/simple_keycard_v1/agent.py) and [ZIP](simple_keycard_v1.zip)
- [Full patient summary](../../evidence/candidate-patient_value_v1/summary.json)
- [Full keycard summary](../../evidence/candidate-simple_keycard_v1/summary.json)
- [Paired comparisons](comparison.json), [validation](validation.json), [SDK check](sdk-wiring.json)
- [Concrete auction example](key-card-example.json), [candidate catalog](../../candidate_catalog_patient_value_v1.json)

To rerun either candidate, use the unchanged benchmark command with a new
output directory:

```bash
.runtime/envs/auction-flop-benchmark312/bin/python -m auction_flop.benchmark.run evaluate \
  --candidate auction_flop/candidates/patient_value_v1 --workers 32 \
  --out .runtime/my-new-patient-run
```
