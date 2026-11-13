# Patient policy improvement: cycle 1

The simple-policy champion is now `patient_defend_v2`: **13.134765625** points
on the unchanged 4,096-table `auction-flop-v1`, versus **12.02392578125** for
`patient_value_v1`. The paired improvement is **+1.11083984375**, 95% interval
**[+0.9021495364, +1.3195301511]**. A fresh 1,024-table-per-arm holdout also
improved: **12.666015625 vs 11.689453125**, paired **+0.9765625**,
95% interval **[+0.6459295484, +1.3071954516]**.

This closes the current cycle following the parent threads' relay of the user's
final competition wrap-up. It is a local simple-policy improvement, below the
parent's approximately 20.3-point submission threshold. No submission or portal
write was made, and no further directions were launched.

## What was weak and what changed

The clearest measured weakness was excessive folding late in the tournament.
The original policy opened a wider range with two or three players left, but
still required near-premiums when facing a large fraction of its remaining
stack. It often surrendered reasonably priced calls and let blinds consume its
stack. It also opened too few heads-up hands.

Two small changes were retained, both confined to `simple_preflop.py` plus a
small existing event-time reference table:

1. With at most three tournament players and at most 12 big blinds, reconsider
   an otherwise folded hand using eligible pot odds and conservative equity
   margins. This only changes folds to calls, with price capped at 40%.
2. Heads-up only, open/shove more kings, suited queens and jacks, and selected
   connected hands. The ordinary full-table chart remains the baseline chart.

The equity reference contains 169 uniform-opponent hand classes. Raising that
equity to the live-opponent count and subtracting a margin is a readable
heuristic; it is not an opponent-conditioned range model or a solved push/fold
strategy. The auction and postflop files are byte-identical to patient v1.

Matched-seed 64-table diagnostic samples show short-handed preflop folds moving
from **589/776 (75.90%)** to **414/811 (51.05%)**. These are descriptive action
rates, not independent causal estimates: the changed policy visits different
states. The full benchmark's sole-chip wins rose from **175/4096 (4.27%)** to
**355/4096 (8.67%)**.

## All tested hypotheses

Fresh screening seed prefix: `patient-loop-c1-screen-20261110`.
Each arm played 512 tables against the same frozen panels and rotations.

| Candidate | Screen points | Difference from control | Disposition |
|---|---:|---:|---|
| patient_value_v1 | 12.97305 | 0 | Control |
| patient_price_v2 | 12.97305 | 0.00000 | Not retained; score benefit unestablished |
| patient_loose_v2 | 12.94922 | -0.02383 | Not retained; score benefit unestablished |
| patient_huopen_v2 | 13.49609 | +0.52305 | Qualified for declared combination |
| patient_shortcall_v2 | 13.80859 | +0.83555 | Qualified for declared combination |
| patient_defend_v2 | 14.51172 | +1.53867 | Selected combination; fully confirmed |

The combination followed the predeclared >=0.50 screen-gain rule. These small
screen means are selection results; the full and independent holdout results
above are the evidence for promotion. The fresh holdout seed prefix was
`patient-loop-c1-holdout-20261110`, frozen before reading the full result.
The reused scheduling program labels every output a screen; for this final
two-arm run, `cycle1-holdout-plan.md` and `cycle1-comparison.json` document its
prespecified independent confirmation role. Its score is not the permanent v1
score.

## Remaining weaknesses

- Strong single pairs can still be folded at favorable prices. One retained
  example is Ah Ad on 4d Qc 2h: calling costs 502 into a reachable final pot of
  2,528 (19.86% required equity), but the original generic large-bet rule folds.
  The isolated exception fixes that decision, yet its screen did not establish
  a net score gain. It was therefore not silently merged into the champion.
- Auction utility remains a coarse hand-category/draw score. It distinguishes
  whether a card changes the category, but treats different kickers or the
  quality of flush draws very similarly. Critical-card allocation was retained
  from v1; this cycle provides no new causal evidence about auction quality.
- Short-handed calling still approximates opponent strength with a uniform
  reference and a fixed margin. It does not distinguish a tight raise from a
  wide shove. This is a known simplicity tradeoff, not a measured explanation
  of the entire gap to other branches' stronger candidates.

## Validation, runtime and exact artifacts

- 35 behavioral/regression tests passed, including the observed AA decision,
  cheap short-handed all-in defense, full-table scope, private/shared nuts,
  weak-hand restraint and auction-budget invariants.
- Full benchmark: **611,634 candidate decisions, zero failures**; all opponent
  decisions also had zero failures. Full wall time: **63.466695 seconds**, 32
  workers on selected distinct physical cores.
- Five-arm screen: 2,560 tables in 36.664969 seconds. Combination screen: 512
  tables in 10.915867 seconds. Fresh holdout: 2,048 total tables; see its retained
  summary for wall time.
- Both screens and the holdout passed independent source, schedule, raw-payout
  and negative-control audits. Both full runs were audited before pairing.
- SDK pack/static checks passed. Two eight-seat, 20-hand SDK subprocess tables
  at a 1,000 ms deadline produced zero failures. This checks wiring and does
  **not** certify the competition's 1 ms production deadline. All strength
  numbers here use the untimed direct engine.
- ZIP content hashes exactly equal the full benchmark's measured source.
  ZIP SHA256: `c271c708d90b19be2439dd1f0746479b50bf09cc714573f92cf8561a48f2b834`.

Artifacts:

- Candidate: `../../candidates/patient_defend_v2/`
- ZIP: `patient_defend_v2.zip`
- Full raw evidence: `../../evidence/candidate-patient_defend_v2/`
- All comparisons, first-place counts and source hashes: `cycle1-comparison.json`
- Predeclared plan: `cycle1.md`; fresh confirmation plan: `cycle1-holdout-plan.md`
- Decision examples: `diagnosis-v1/` and `diagnosis-v2/`
- Reproduction: `build_cycle1.py`, `test_cycle1.py`, `analyze_cycle1.py`, retained
  arm manifests, immutable candidate snapshots and the shared frozen runner.

Do not rebuild into these existing candidate/output directories. Use new names
and fresh screening seeds for any resumed cycle. No frozen benchmark, peer
candidate, deployment or unrelated tracked repository file was edited.
