# ICM texture audit candidate

Based on the event-time selected agent with source SHA256 `73ed48b9128862704c8123b0e9c582ab366c4fa3b84206998874154fc4b17f56`. The selected source was preserved.

## Verified corrections

- Board quads use kicker equity against random opponent holdings. Previously 4c5d on 2c2d2h2s3c received 0.985 quality. The exact heads-up kicker calculation gives about 0.02576.
- Two-pair private contribution counts the highest two pairs in the evaluated five-card hand. A lower excluded pair cannot elevate the result to two privately made pairs.
- Straight-draw bonuses require a private rank contributing to the straight. A four-straight entirely on the board adds no private draw bonus. Known dead auction cards reduce straight and flush outs.
- Guaranteed shared royal flushes, dry broadway straights, and board quads with the highest possible kicker always check or call. The previous policy folded these guaranteed ties against large river bets.
- For quality of at least 0.90, multiway adjustment uses a power of that quality. The earlier constant 0.89 penalty excessively reduced even quads and straight flushes with many opponents.

The first four corrections have constructed regression coverage in `bots/dianne/tools/test_icm_texture_v3.py`. The multiway adjustment is a conservative probabilistic approximation requiring comparison, rather than a full opponent-range calculation.

Official static submission validation passed. Parent benchmarking determines whether the candidate improves the selected policy. All timing results exclude production transport unless explicitly described as SDK measurements.
