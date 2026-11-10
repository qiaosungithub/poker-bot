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

## Comparison with the teammate's prior named policy

The shared build box contains byte-identical `icm_v2/agent.py` copies in its v2 and v3 directories (SHA256 `73ed48b9128862704c8123b0e9c582ab366c4fa3b84206998874154fc4b17f56`). This source scored **10.96680 points/table** on the same 4,096-table v1 schedule, with zero failed decisions. The old portal submission was named `icm_v2.zip`; its downloadable bytes/hash are unavailable, so exact identity with the uploaded package is not verified.

| Our candidate | Points/table | Difference vs shared icm_v2 | Simultaneous 95% interval |
| --- | ---: | ---: | --- |
| submitted_hybrid | 15.08789 | +4.12109 | [+3.22643, +5.01576] |
| preflop_loose | 16.56250 | +5.59570 | [+4.76064, +6.43077] |

The two paired contrasts use a Bonferroni correction. Full local source snapshots and raw records remain in the shared workspace path recorded in [the comparison artifact](evidence/teammate-comparison.json); teammate source is not republished here. This comparison was completed after the user-requested immediate submission.

## Submission observed at 2026-11-10 00:06 UTC

`timothy-hybrid-7e7aa88.zip`, submission `fabb5704-315f-44f6-99b7-2a2df61992f0`, was uploaded at 00:03:07 UTC. The authenticated API subsequently reported both `ready` and a matching team `active_submission_id`. This is the qualified modular hybrid scoring 15.08789 in v1. `preflop_loose` has a higher measured score; its production qualification/upgrade is separate from the completed strength comparison. Later portal changes do not alter this dated observation.

## Corrected v2 candidates

The three documented shared-core defects were repaired in new source versions. Both corrected candidates ran the complete unchanged 4,096-table v1 schedule with zero failed decisions; their file hashes exactly match the separately qualified package bytes. The original benchmark opponents remain untouched.

| Corrected candidate | Points/table | Change from its original | Paired simultaneous 95% interval |
| --- | ---: | ---: | --- |
| corrected_preflop_loose_v2 | 16.62598 | +0.06348 | [+0.01357, +0.11338] |
| corrected_baseline_v2 | 15.16113 | +0.07324 | [-0.00552, +0.15200] |

The corrected loose candidate has the highest completed mean: **16.62598**. Its improvement over the original loose version is small, +0.06348 points, with a positive paired normal interval after correcting for these two contrasts. This is evidence within the fixed public seed set, not a new broad population guarantee. Source revisions, package hashes and composition manifests are in [candidate_catalog_corrected_v2.json](candidate_catalog_corrected_v2.json); [raw comparison](evidence/corrected-v2-comparison.json).

## Conditional history and inference contrasts

The four history/inference arms also change auction/value selection relative to the separate hybrid baseline. Their lower complete-policy scores do not isolate the effect of history. Four exploratory paired contrasts within the controlled 2x2, corrected jointly, all include zero:

| Contrast | Difference | Simultaneous 95% interval |
| --- | ---: | --- |
| history effect at inference=0 | -0.20996 | [-0.69197, +0.27204] |
| history effect at inference=1 | -0.18799 | [-0.66749, +0.29152] |
| inference effect at history=0 | +0.17578 | [-0.02911, +0.38067] |
| inference effect at history=1 | +0.19775 | [-0.01707, +0.41257] |

See [factorial contrasts](evidence/factorial-comparison.json). No additional games were used for these contrasts.

## Submission update observed at 2026-11-10 00:10 UTC

`timothy-preflop-loose-c5ec82a.zip`, submission `7798fc5a-a796-4483-8164-b0cba3c7465e`, was uploaded at 00:09:44 UTC and independently read back as Ready and active. This is the **original** loose candidate scoring 16.56250, not the separately measured corrected v2. Package SHA256: `053507112b6d266c0c2f8c18c6e2f0ef8af9970f609c6bb2bdda776f63844795`.
