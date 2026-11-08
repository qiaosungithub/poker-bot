# Bay Area public strategy comparison — 2026-11-08

The highest confirmation mean was **NC codex_nc_v7_adapted**. This experiment
**does not establish a unique strongest bot**: the first four sample means are
within 0.135 points/table, and no adjusted pairwise comparison separates the
leader from the other six non-tree finalists. James's `jj_tree` is substantially
weaker in this field. These are local matches, not official contest standings.

## Final eight bots, all playing together

One copy of each finalist at every table, 4,096 tables / 512 independent seed
blocks, all eight seat rotations per block. Payouts: 50/30/20/0/0/0/0/0, so
**12.5 points/table is the equal-share reference**, and a table victory pays 50.
Mean points are not win percentages. Each row uses the same field and schedule.

| Rank by sample mean | Strategy | Points/table | 95% CI | Source act P99 (ms) | SDK failed decisions |
|---|---|---:|---|---:|---:|
| 1 | [NC: codex_nc_v7_adapted](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots/codex_nc_v7_adapted/agent.py) | 13.782 | [13.234, 14.330] | 0.176 | 0 |
| 2 | [NC: gto_final_reference](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots/gto_final_reference/agent.py) | 13.706 | [13.160, 14.252] | 0.176 | 0 |
| 3 | [James: jamtop3_fix2](https://github.com/JamesGongYC/nolimit-pokerbot/blob/c91408d08d9946d33088ea6a6e5e477659c3df6b/bots/jj_jamtop3_fix2/agent.py) | 13.682 | [13.096, 14.267] | 0.033 | 0 |
| 4 | [James: jamtop_tuned_live](https://github.com/JamesGongYC/nolimit-pokerbot/blob/c91408d08d9946d33088ea6a6e5e477659c3df6b/bots/jj_jamtop_tuned_live/agent.py) | 13.647 | [13.015, 14.280] | 0.032 | 0 |
| 5 | [NC: codex_gto_refined](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots/codex_gto_refined/agent.py) | 13.025 | [12.490, 13.560] | 0.180 | 0 |
| 6 | [Denfujita: v7](https://github.com/denfujita/poker/blob/fc661ce9f505dd00f23fe656a5a4289fbc964b27/agent.py) | 12.822 | [12.286, 13.359] | 0.471 | 0 |
| 7 | [NC: gto_supplied](https://github.com/nnicholas-c/poker_sequoia/blob/e6e68a1aa7c71f78fec17bd7ead813687689fc61/bots/gto_supplied/agent.py) | 12.722 | [12.166, 13.278] | 0.174 | 0 |
| 8 | [James: jj_tree](https://github.com/JamesGongYC/nolimit-pokerbot/blob/c91408d08d9946d33088ea6a6e5e477659c3df6b/bots/jj_tree/agent.py) | 6.614 | [6.193, 7.034] | 0.496 | 0 |

There were **545,734 hands**, **3,312,165 decisions**,
and no SDK-reported failed decisions. The confirmation took **133.75 seconds**
with 16 host workers on this server. The preceding 1,024-table screening took
39.96 seconds. This speed is not a claim about other machines.

The leader's paired advantage over `gto_final_reference` was only
**+0.0757 points/table**, ordinary 95% CI **[-0.7486, +0.9000]**. Against
`jamtop3_fix2`, it was **+0.1001**, CI **[-0.7718, +0.9720]**. Ranking by the
sample mean is descriptive; it does not justify declaring these differences
real. The machine-readable report includes Bonferroni simultaneous 95%
intervals over all 28 comparisons. The leader-versus-tree interval remains
positive after that correction.

## Practical interpretation

`jamtop3_fix2` is a useful small baseline: it primarily chooses push/fold from
three-card strength and position/stack-dependent ranges, then handles selection
and any subsequent betting. Its observed score was close to the leading bots
while its source-function P99 was about 0.033 ms. That makes it a good comparison
point for a more complex policy, not a proven equilibrium strategy or universal
winner. A tree model is not automatically better; the tested tree lost points
in this particular field.

The 16 source candidates came from three public repository groups, with many
related versions. We retained the best screen scorer per group and filled the
final field by score, capped at four per group, as declared before screening.
This preserves source diversity; the finalists are not simply the global top
eight screen means. Some omitted NC variants screened above the admitted tree.
No result proves that the repositories represent three independent official
teams, or that any selected source was the final uploaded regional submission.

## Timing interpretation

Scores use the actual kit SDK process path, with 1 ms plus its 2 ms delivery
grace and 100 ms per-table bank. Thus zero failures does not, by itself, prove
a hard 1 ms service deadline. Local processes are not the production microVM.

The separate timing column measures the original `Agent.act` only on one pinned
CPU after eight warmup tables, with GC disabled. It excludes PRNG seeding, SDK
conversion, IPC and startup. There were 9,976–15,473 measurements per finalist
from 128 played tables; none exceeded 1 ms in that sample. Raw wall/thread-CPU
times and phase labels are retained. These measurements do not establish a
universal worst-case bound. Existing strategy time guards can also make replay
hardware-dependent.

## Evidence and reproduction

- [Predeclared protocol](../../bay_area/PLAN.md), [setup and commands](../../bay_area/README.md).
- [Candidate sources and prefilters](../../bay_area/candidates.json).
- [Screening table](screen/table.csv), [selection decision](selection.json).
- [Confirmation table](confirmation/table.csv), [full summary/pairwise intervals](confirmation/summary.json).
- [Raw confirmation tables](confirmation/tables.jsonl), [configuration and hashes](confirmation/manifest.json).
- [Latency summary](latency/result.json), [raw samples](latency/timings.npz).
- [Rule tests](validation.txt), [independent artifact audit](audit.json).

The raw table files are retained byte-for-byte. Each final score was independently
recomputed from final chips, elimination groups, payouts and an ICM oracle.
Confidence intervals treat each complete seed block as one independent unit;
the correlated rotations are not counted as eight independent trials.
