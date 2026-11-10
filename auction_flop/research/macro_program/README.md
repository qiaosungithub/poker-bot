# Macro policy experiments: results and limits

The frozen reference is `hybrid_both`, scoring **17.734375** on
`auction-flop-v1`. The program's declared target was a mean of at least
19.734375, with fresh-seed and style transfer checks. The prior-plus-Hboth
composition reaches **19.80713** on v1, but a robust two-point gain across
opponent styles is **not established**. No frozen v1 opponent, rule, seed,
weight, engine or runner changed.

## Four complete macro hypotheses

Every candidate and retained control has 4,096 tables, 512 paired seed blocks,
eight cyclic seats and zero failures. Sources, schedules and raw payouts were
independently audited. Intervals below adjust for these four comparisons only,
not the whole adaptive strategy search.

| Candidate | Mean points/table | Change versus Hboth | Simultaneous 95% interval |
| --- | ---: | ---: | --- |
| Frozen public opening prior + Hboth | 19.80713 | +2.07275 | [+1.23806, +2.90745] |
| Tournament-utility preflop | 18.98438 | +1.25000 | [+0.04748, +2.45252] |
| Counterfactual-value auction | 15.47607 | -2.25830 | [-3.19924, -1.31736] |
| Action-value postflop | 10.14160 | -7.59277 | [-8.51901, -6.66654] |

The auction and postflop replacements are rejected in this field; no subsequent
threshold sweep was run. The strongest composition changes opening behavior
while retaining Hboth postflop and tournament auction. It is event source
`13be90cc13dbde5ad1c5bfcd36a4f5de49cb3a3c`, ZIP SHA256
`0c62ece718f9387587ca0f7289932b71eeb17969f3617f316381ad598a6c1273`.
Its individual paired interval versus Hboth is [+1.41775,+2.72775]; an observed
mean gain above two is not a confidence bound proving at least two.

See [macro catalog](../../candidate_catalog_macro_v1.json),
[composition catalog](../../candidate_catalog_prior_hybrid_both_v1.json) and
[independently audited comparisons](../../evidence/macro-program-comparison.json).

## Prize objective versus chip objective

The tournament-utility model was separately compared with a linear objective,
`100 * final hero normal chips / total final normal chips`. Auction currency
never enters that objective. All ranges, action candidates, settlement,
continuation scenarios and other source files are identical; only
`tournament_utility.py` changes.

| Objective in the shared model | Mean points/table |
| --- | ---: |
| Prize ICM utility, optimized v2 | 18.98438 |
| Linear normal-chip utility | 16.86279 |

ICM minus linear chips is **+2.12158**, paired 95% interval
**[+1.42967,+2.81350]**. This supports a prize-objective benefit within this
particular approximate model and field. It does not establish universal ICM
optimality or a two-point gain over Hboth: Hboth remains the 17.73438 reference.

The original ICM v1 was slower. A separate 4,096-table validation confirms that
performance-optimized v2 produces exactly the same retained table records as
v1, excluding wall-clock time: zero changed records, including scores, stacks,
bust order, decision counts, phases and auction outcomes. The original full v1
did not retain literal action streams, so this is not a claim about every
individual action. The objective contrast itself uses measured v2 against the
chip control, whose other 13 files are byte-identical. See
[source/objective validation](../macro_utility_ablation/chip_control_validation.json)
and the full comparison JSON above.

## Fresh transfer checks

The finalist was frozen before the separately named protocol was written and
before any validation games. The protocol was locked at **01:17:01 UTC** with
SHA256 `0363d25ed333b373aefe4d535d489642ef84bd494686e3ad3b41a770a9a5eaec`.
Every recorded run starts after that timestamp. Each row below is a separate
paired field, with 4,096 tables per policy and zero failures.

| Field | Hboth | Prior + Hboth | Paired gain | Two-field simultaneous 95% interval |
| --- | ---: | ---: | ---: | --- |
| V1 opponents, new seeds | 18.03467 | 19.64844 | +1.61377 | [+0.84115, +2.38639] |
| Separate four-style field, new seeds | 20.76172 | 20.96680 | +0.20508 | [-0.61194, +1.02210] |

Absolute scores across these fields must not be compared. The style field
equally exposes exploit, equity, classic txnujk and TournamentSpecialist
opponents excluded from v1 evaluation, with no easy checkfold/minraise fillers.
It is a transfer check, not a claim to reproduce live competitor behavior.
The gain survives new seeds in the v1 field; the different-style result is
inconclusive. These inspected seeds will not be reused to tune the finalist.
[Locked protocol](../../evidence/macro-transfer-finalist1/protocol.json),
[paired results](../../evidence/macro-transfer-finalist1/paired-transfer-comparison.json),
and all four raw runs are retained together.

## Diagnostic and deployment scope

The [separate diagnostic harness](../../diagnostics/README.md) reproduced
256 balanced known-seed tables with exact retained-result and full-action
parity. It identifies frequent opening, turn/river free-value and expensive-call
populations for investigation, with denominators and explicit selection limits.
Selected losses are not counted as causal regret. Older official-round
[checkpoint evidence](../official_checkpoints/README.md) is version-attributed
and kept separate from these simulated fields.

Untimed strength is separate from production qualification. The prior-plus-Hboth
and auction packages passed SDK/IPC checks, with rare responses over 1 ms within
the 2 ms delivery grace; this is not a hard per-call 1 ms guarantee. The ICM and
action-value experiments are not deployment recommendations: their original
runtime checks exceeded the desired budget, and the objective control is a
research artifact. Upload and activation require their own recorded receipt.

```bash
python -m auction_flop.benchmark.analyze_macro
python -m auction_flop.diagnostics.audit_transfer auction_flop/evidence/macro-transfer-finalist1
```

## Formal transfer gap and activation

The completed official S4 uses **Hboth**, submission
`9bfee42d-d5b5-4f6d-87c0-e45d56aeef43`: **10.2705** points/table,
**143/194**, 20,000 tables, 2,715,368 decisions and zero misses. S3 classic hybrid
scored 11.2245; S2 corrected loose scored 12.3995. The policies and field change
between rounds, so these are descriptive comparisons, not causal estimates.
The [sanitized round audit](../official_checkpoints/evidence/round-transfer-audit.json)
preserves exact submission mappings and public results.

The [S4 prefix analysis](../official_checkpoints/evidence/S4-first1001-compact-summary.json)
covers **1,001** scored tables (iterations 0–1000, 34 filler entries excluded),
not all 20,000. Its mean is 10.12987. Settled stage ICM changes per initial table
are -1.00576 with 8–5 players, -0.64068 with four, -0.31373 with three and
-0.40995 heads-up. These account for broad losses across stages; they are not
individual-action regret or isolated strategy effects. The prefix was selected
by iteration, not outcome, and is not a random holdout.

The designated submission writer activated the **original `practice_prior_full_v1`**
at 01:33:13 UTC: source `7dfcd05f07c11c1cd93e2d1782ac1b71a72c73f8`,
submission `62e67012-528e-4510-9fba-0c522262f3fe`, v1 mean **17.65625**.
This formal experiment uses the prior with the older corrected postflop core.
The [receipt](../../evidence/submission-confirmation.json) confirms Ready/active;
[S5 health](../../evidence/submission-s5-snapshot.json), observed 01:35:16 UTC,
confirms that round `b009537f-a41d-433c-b6b0-f23f68db21d4` captured this source.
The snapshot contains no performance result for it yet.

The **19.80713 prior + Hboth remains the highest local v1 candidate and has not
been uploaded**. Its same-field fresh-seed gain and inconclusive style transfer
do not establish formal improvement. Frozen v1 remains unchanged so historical
comparisons remain reproducible.
