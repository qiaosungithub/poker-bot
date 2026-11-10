# First nine modular candidates on unchanged v1

All nine source-only arms in candidate_catalog.json passed a 32-table interface
smoke test. Evaluate each for the complete 4,096-table auction-flop-v1 schedule;
do not change sources, opponents, seeds, sample size or payout rules mid-run.

Rank all nine by mean points. Compare the eight alternatives with
strong_modular_baseline using paired panel/seed block differences. Report
simultaneous normal 95% intervals with a Bonferroni correction for eight
comparisons. Treat source-equivalent controls as checks on the composition,
not as independent strategy discoveries. The four history/inference arms form
a controlled 2x2; any further factor contrasts are exploratory and must be
identified as such. Decisions with errors or illegal actions invalidate an arm.

The initial upload publishes the policies and frozen benchmark immediately;
completed candidate results are appended separately. V1 stays unchanged.
