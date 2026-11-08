# Frozen comparison plan (written before screening)

Use the pinned Bay Area starter-kit engine with 8 players, 2,000 chips each,
250-hand cap, blinds increasing every 20 hands, and fixed 50/30/20/0/0/0/0/0
payouts. Survivors at the cap receive exact ICM points. No resampling or luck
correction is enabled; special-prize counts do not enter the main score.

Source pool: 16 variant-aware implementations from three public repositories,
identified individually in `candidates.json`. Repository groups are not
necessarily three independent strategies: versions within a group are related,
and authors may share ideas. These are not 16 separate entrants or a recovered
official tournament field.

1. Verify game invariants and SDK fallback/memory behavior. Smoke-test one
   balanced eight-bot block through the actual host/guest SDK protocol.
2. Screen 16 candidates on 64 independent blocks. Randomly partition the pool
   into two tables of eight in each block; play all eight cyclic seat rotations
   at each table with a shared deal seed. Each candidate plays 512 tables.
3. Select eight finalists by screening mean points: reserve the highest-scoring
   candidate from each repository, then fill by score with at most four from
   any repository. Do not silently remove runtime failures after seeing scores;
   list any import/compatibility exclusions and preserve the failed run.
4. Confirm all eight together on **512 new blocks / 4,096 tables**, using a seed
   prefix disjoint from screening. Each table has exactly one of each finalist.
   Report the highest mean, confidence intervals and unresolved near ties.
5. If the leading difference remains unclear, report that uncertainty; do not
   sample repeatedly until a preferred candidate becomes significant. Further
   experiments need a separately recorded plan and independent seeds.

The primary benchmark uses the real SDK process deadline: 1 ms plus the kit's
2 ms delivery allowance, and a 100 ms per-table bank. These are local Linux
processes, not official microVMs or verified tournament hardware. Submissions
run unchanged; a guest wrapper seeds their PRNGs per request. Bots' own wall-time
guards remain, so exact replay is not guaranteed under different system load.

Average the eight rotations within a seed block before computing standard
errors. Use paired block differences for comparisons in the final fixed field,
and simultaneous Bonferroni 95% intervals over all 28 finalist pairs to avoid
declaring a winner merely because the highest sample mean was selected. Keep
raw table records, code/asset hashes, seeds, process settings and failures.
