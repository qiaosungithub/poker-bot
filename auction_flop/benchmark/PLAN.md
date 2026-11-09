# Auction Flop fixed opponents v1

Declared before the new selection run on 2026-11-09. Preserve all earlier runs.

Compare the five frozen event v1 policies (tournament, equity, ranges, exploit,
variant), the already confirmed tournament-auction/variant-betting composition,
and the two released examples in one eight-seat field. Use 512 independent
seed blocks, eight cyclic seat rotations per block, seed prefix
`auction-flop-v1-selection-20261109-independent`: 4,096 tables. Rank the six
non-example policies by mean official table payout; require zero failed
decisions and choose four. Break exact ties lexicographically. This is a
practical selection rule, not a claim of statistically distinct ranks.

Freeze sources, SDK, rules, selection evidence, and seed schedule in v1.
Eight seats; 2,000 normal chips; 1,000 auction chips; the released blind
progression every 20 hands; hand cap 250; fixed payouts 50/30/20, including
official bust/tie handling and ICM at the cap. No luck correction or resampling.
Strength runs use real public GameState views and direct calls without clock
penalties. Decision failures are counted and disqualify a candidate's run.
This benchmark does not certify production timing or a changing live tournament.

Future evaluations put one candidate against seven fixed opponents. Four panels
give each selected bot one copy in its own panel and two copies in the other
three panels; exposure is exactly equal overall. Every panel uses 128 seed
blocks and eight rotations: 4,096 tables total per candidate. All candidates
receive the same seeds and slot identities. Opponent instances and memory reset
per table. Neither candidate names nor strategy IDs enter public observations.

Primary metric: candidate mean points per table (higher is better, table total
100). Report normal-approximation 95% intervals from independent block means,
with panel-stratified standard errors. Paired comparisons use identical panels,
seeds and rotations. Also retain chips, failures, auction spend and remaining
budget. Raw records are losslessly compressed. Independent score recomputation,
full schedule validation, source hashes and corrupted-record negative controls
must pass before publishing a score.

Selection seeds are disjoint from fixed evaluation seeds. Repeated evaluation
on public fixed seeds is a regression benchmark, not fresh evidence against
overfitting. For new scientific claims add a separately named holdout study.
Never refresh v1 opponents, weights, seeds or rules. Improvements are candidates;
any changed benchmark gets a new version and leaves v1 available.
