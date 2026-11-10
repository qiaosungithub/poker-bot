# Auction Flop fixed benchmark v1

## Selection on fresh seeds

Five original event policies and the existing hybrid were compared with two released examples in one eight-seat field. The predeclared selection used 512 independent seeds and all eight seat rotations, for 4,096 tables. Every policy had zero failed decisions. Scores are mean official table points; each table distributes 100 points.

| Policy | Points/table | 95% interval | Frozen opponent |
| --- | ---: | --- | --- |
| hybrid_tournament_auction | 21.30859 | [20.65054, 21.96665] | Yes |
| variant | 18.70361 | [18.04019, 19.36704] | Yes |
| tournament | 16.44287 | [15.87260, 17.01314] | Yes |
| ranges | 13.17383 | [12.66284, 13.68482] | Yes |
| exploit | 12.82471 | [12.25284, 13.39658] |  |
| equity | 10.62988 | [10.08498, 11.17479] |  |
| checkfold | 6.89209 | [6.54603, 7.23815] |  |
| minraise | 0.02441 | [0.00008, 0.04875] |  |

The hybrid scored 2.60498 more points than variant in this field. The paired simultaneous interval for this comparison is recorded in the raw summary. Selection is based on the prespecified mean-score rule, not significance. Ranges exceeded exploit by only 0.34912 points; the simultaneous 95% interval is [-0.89190, 1.59014], so their relative strength is unresolved. Ranges nevertheless occupies the fourth permanent slot under the predeclared rule.

Selection sources: [manifest](evidence/selection-v1/manifest.json), [full summary and pairwise intervals](evidence/selection-v1/summary.json), [raw tables](evidence/selection-v1/tables.jsonl.gz), [frozen opponents](benchmark/v1.json).

## Fixed-field calibration

These runs use the permanent four-panel opponent field, not the selection field above. Each candidate played 4,096 tables on the fixed evaluation seeds; all runs had zero failed decisions. Scores across the two fields are not directly comparable.

| Candidate | Points/table | 95% interval |
| --- | ---: | --- |
| hybrid_tournament_auction | 15.08789 | [14.46030, 15.71548] |
| variant | 13.24707 | [12.67121, 13.82293] |
| tournament | 11.79932 | [11.23357, 12.36506] |
| ranges | 9.38232 | [8.93861, 9.82604] |

Hybrid minus variant: **+1.84082 points/table**, paired 95% interval [+1.14046, +2.54118]. See [paired result](evidence/hybrid-vs-variant.json).

## Reproducibility and scope

The released Auction Flop engine, SDK, rules, seven original/example source folders and hybrid composition are pinned by hashes. Four panels balance exposure to the four selected opponents; each candidate plays 128 seed blocks per panel with eight rotations, totaling 4,096 tables. Evaluation seeds are separate from selection and identical across future candidate evaluations. Opponents, weights, seeds and rules will not be refreshed in v1.

Intervals use independent seed-block means, with equal panel weights and panel-stratified standard errors in the fixed-field results. All rotations within a block are correlated; they are not treated as independent observations. Paired comparisons are left minus right on the same panel/seed/rotation schedule. Normal intervals are approximate; one paired comparison does not correct a larger exploratory search.

The fixed payout schedule is 50/30/20 with ICM at the hand cap. This is a reproducible research environment, not a claim about arbitrary live round payouts, a global best policy, or the production microVM. Direct strength evaluation applies no clock penalties. The hybrid runs two separate donors and is not by itself a production packaging or latency recommendation. Public fixed seeds are for regression; fresh holdouts remain necessary to assess overfitting.

All 15,360 retained historical/selection tables passed independent payout recomputation, complete schedule checks, original raw-record hashes, original runner hashes, and source hashes. See [historical audit](evidence/historical-audit.json). Earlier field scores remain separate; they are not pooled into selection or fixed-field calibration.

The clean Python 3.12 environment uses only the four pinned packages in [requirements.txt](requirements.txt), plus the vendored released SDK. [Validation](evidence/validation.txt) exercises corrupted scores, missing/duplicate/wrongly seated tables, auction currency, changed source, deterministic repeats, exact identical-policy symmetry, and paired statistics. The external candidate-folder interface and relocation audit were also exercised.

Committed research snapshots preserve the other agent's implementation work through the revisions in [research/provenance.json](research/provenance.json). They are reviewable source, not newly ranked benchmark opponents. Newer active branch work can be published later as candidates without changing v1.

## New candidate batch

Nine later modular candidate arms were evaluated against the unchanged v1 opponents and seeds. Each played 4,096 tables, all with zero failed decisions. Every source snapshot matches [the published catalog](candidate_catalog.json).

| Candidate | Points/table | 95% interval |
| --- | ---: | --- |
| preflop_loose | 16.56250 | [15.97532, 17.14968] |
| preflop_tight | 15.66162 | [15.08891, 16.23433] |
| joint_on | 15.20264 | [14.55748, 15.84780] |
| joint_off | 15.08789 | [14.46030, 15.71548] |
| strong_modular_baseline | 15.08789 | [14.46030, 15.71548] |
| history0_inference1 | 12.70752 | [12.14171, 13.27333] |
| history0_inference0 | 12.53174 | [11.96461, 13.09887] |
| history1_inference1 | 12.51953 | [11.94526, 13.09380] |
| history1_inference0 | 12.32178 | [11.75357, 12.88998] |

The modular baseline and joint-OFF control both exactly reproduce the frozen hybrid score, 15.08789. The highest measured score is **preflop_loose, 16.56250**. The eight prespecified paired contrasts use simultaneous 95% normal intervals with Bonferroni correction; see [complete comparisons](evidence/candidate-comparison.json).

These are untimed strategy scores. They do not determine submission eligibility or SDK/IPC runtime acceptance.

## Known shared betting-core defects

A separate source audit reproduced three defects in the original variant betting core shared by these candidates: folding a guaranteed tie on a public royal flush; overrating a weak private kicker on public quads; and counting inaccessible side-pot money in short-stack call odds. The audit does not estimate their tournament frequency. Preserve the measured original opponents and v1 hashes; corrected implementations must be new candidates. These results establish comparative scores in this fixed field, not correctness of every betting decision.

Original audit evidence is retained under [research/frozen-core-audit](research/frozen-core-audit/FINDINGS.md).
