# Simple policy tuning v2

The selected simple bot scored **10.67139 points/table** on
4,096 unchanged auction-flop-v1 tables. Compared with simple_classic_v1's
8.40237, the paired improvement is **2.26901**,
95% interval **[1.85609, 2.68194]**.
The new mean's 95% interval is [10.26390, 11.07887].
All **568,847 candidate decisions** and all opponent decisions had zero failures.
This remains below the retained classic control's 13.61816; the result supports
an improvement of this simple implementation, not a competitive-best claim.

## Selected settings

| Parameter | Original | Selected | Human meaning |
| --- | --- | --- | --- |
| BASE_BID | 20 | 60 | Triple the auction budget per hand-quality improvement |
| OPEN_LOOSER | false | true | Early position uses middle chart; middle uses late chart |
| CALL_SCALE | 1.0 | 1.7 | Cheap-call caps change from 4 BB / 15% of stack to 6.8 BB / 25.5% |
| AUCTION_HORIZON | 0 | 0 | Keep constant base; no remaining-balance duration scaling |
| SHOVE_BB | 12 | 12 | Keep original short-stack opening shove threshold |
| BET_SCALE | 1.0 | 1.0 | Keep postflop bet/raise sizes |
| BLUFF_FREQUENCY | 0.5 | 0.5 | Keep original late flop air-stab probability |
| EQUITY_SAMPLES | 32 | 32 | Keep postflop estimate budget |

The opening-size position label is unchanged even when its hand chart widens.
All other preflop commitment thresholds and postflop decision rules are retained.
Mean auction spending rose from 187.21 to **637.00**
chips/table. The three changes were confirmed together; this does not isolate
how much improvement each contributed.

The editable values are in [config.py](../../candidates/simple_tuned_v2/config.py).
The [bot entry point](../../candidates/simple_tuned_v2/agent.py) and
[SDK-checked ZIP](simple_tuned_v2.zip) are ready for local use. No upload was made.
The original candidate, other agents' candidates and frozen benchmark files
were preserved. New source and results remain local, without a commit or push.

## Time cost

| Work | Tables | Wall time |
| --- | ---: | ---: |
| Twelve single-change screening arms together | 6,144 | 83.32 s |
| Combined arm screen | 512 | 18.28 s |
| Complete fixed v1 confirmation | 4,096 | 61.27 s |
| Total strength evaluation | 10,752 | 162.88 s |

These times include runner snapshots/audits but exclude implementation, the
19,754-decision continuity check and separate timing/report work. Runs used
32 sampled quiet, distinct physical cores on the local AMD EPYC 7B12 host.
The full run achieved 66.85 tables/second. The earlier 16-worker
run took 161.56 seconds, but that is not a controlled scaling experiment:
candidate trajectories and shared host load also changed. Current observed
planning estimates are about 18 seconds for 512-table screening of one arm
and about one minute for a complete 4,096-table confirmation.

## Search evidence

[PLAN.md](PLAN.md) was written before results. Each screening arm used the same
512 tables, four frozen v1 opponent panels and new seed prefix
`simple-tuning-v2-screen-20261110`, disjoint from permanent evaluation seeds.
All games retained the released blind schedule, 250-hand cap and 50/30/20
payouts. We did not shorten games to make scores faster.

| Screen arm | Points/table | Unadjusted 95% interval | Auction chips spent/table | Failures |
| --- | ---: | --- | ---: | ---: |
| call_wider | 8.94531 | [7.73145, 10.15918] | 192.36 | 0 |
| loose_open | 8.94531 | [7.77873, 10.11190] | 196.43 | 0 |
| samples_16 | 8.69141 | [7.59917, 9.78364] | 184.54 | 0 |
| bid_60 | 8.63281 | [7.52499, 9.74063] | 564.62 | 0 |
| bid_120 | 8.55469 | [7.49059, 9.61879] | 805.86 | 0 |
| control | 8.41797 | [7.33300, 9.50293] | 182.63 | 0 |
| duration_10 | 8.39844 | [7.31631, 9.48056] | 607.11 | 0 |
| duration_25 | 8.39844 | [7.29335, 9.50352] | 309.68 | 0 |
| bet_larger | 8.33984 | [7.32427, 9.35542] | 185.42 | 0 |
| shove_later | 8.32031 | [7.20196, 9.43867] | 187.73 | 0 |
| no_air_bluff | 7.92969 | [6.96549, 8.89389] | 180.38 | 0 |
| bid_zero | 6.83594 | [5.81456, 7.85731] | 0.00 | 0 |
| combined | 10.27344 | [8.99673, 11.55014] | 620.21 | 0 |

The combination followed the declared rule: take the best auction setting and
other individual changes with at least +0.5 screen points. It combined bid_60,
loose_open and call_wider. It had the highest screen mean and was frozen before
confirmation. Screen comparisons are exploratory, with multiple comparisons
and overlapping intervals; they do not establish a unique best parameter.
The full paired result uses different seeds from screening. Those final seeds
are the public regression benchmark, not a newly hidden tournament holdout.

AUCTION_HORIZON=10/25 substitutes remaining balance divided by that horizon for
BASE_BID; it is a spending scale, not an exact lifetime. Neither duration arm
beat control in this screen, and neither was selected. Larger postflop bets,
removing air bluffs and delaying short-stack shoves were also not selected.

## Decision runtime

A separate sequential test compared identical public states after strength runs
finished: 951 retained states, five repetitions per arm. It measures warm
Agent.act only, excluding SDK IPC and production scheduling. Full raw timings
and maxima are retained in [latency.json](latency.json) and
[latency-samples.jsonl.gz](latency-samples.jsonl.gz).

Reducing sampling from 32 to 16 cut flop median decision time from
0.222 to
0.143 ms on this corpus. It
was not selected by the declared strength rule. The selected 32-sample bot's
flop P99 was 0.303 ms,
with zero calls over 1 ms among 4,755 measured calls. This is not production
qualification or a worst-case bound. The original 32-sample arm had one river
wall-time outlier over 1 ms; all raw samples remain available.

## Validation and reproduction

The default parameterized bot matched original actions and byte memory on all
19,754 decisions from 16 native tables. Every screen was audited for schedule,
source hashes, currency conservation, independent payouts and zero failures;
a corrupt-score negative control was rejected for each arm. The final source,
selection snapshot and ZIP have identical file hashes. The full fixed-v1 audit
and separate direct sum of retained candidate points both passed.

```bash
# Every --out must be a new directory.
.runtime/envs/auction-flop-benchmark312/bin/python -m auction_flop.research.simple_tuning_v2.screen \
  --arms auction_flop/research/simple_tuning_v2/arms.json --workers 32 \
  --out .runtime/my-simple-screen
.runtime/envs/auction-flop-benchmark312/bin/python -m auction_flop.benchmark.run evaluate \
  --candidate auction_flop/candidates/simple_tuned_v2 --workers 32 \
  --out .runtime/my-simple-confirmation
```

The screen's `--cpus` takes a JSON list of allowed local CPU IDs; the default
`.runtime/simple-tuning-cpus.json` is host-local. The recorded measured CPU list
is in the screen manifest. Resample available physical cores on another host.

Evidence: [full summary](../../evidence/candidate-simple_tuned_v2/summary.json),
[full manifest](../../evidence/candidate-simple_tuned_v2/manifest.json),
[paired comparison](comparison.json), [selection](selection.json),
[validation](validation.json), [screen summary](../../evidence/simple-tuning-v2-screen/summary.json),
[combination summary](../../evidence/simple-tuning-v2-combined/summary.json).
