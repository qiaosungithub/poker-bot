# Simple classic v1 result

On 2026-11-10, the fixed BASE_BID=20 baseline scored **8.40237 points/table**
on the unchanged `auction-flop-v1` benchmark. Its 95% interval is
**[8.03638, 8.76837]**. This first simple implementation underperforms the
retained classic control; it does not establish that simple policies in general
are weak or identify which stage causes the difference.

| Candidate | Mean points/table | 95% interval | Failed decisions |
| --- | ---: | --- | ---: |
| simple_classic_v1, BASE_BID=20 | 8.40237 | [8.03638, 8.76837] | 0 |
| classic_txnujk32_v1, retained control | 13.61816 | [13.06578, 14.17055] | 0 |
| classic_hybrid_v1, descriptive context | 17.48047 | [16.86017, 18.10076] | 0 |

The predeclared paired comparison against classic_txnujk32_v1 is
**-5.21579 points/table**, 95% interval **[-5.82710, -4.60448]**. Both candidates
use the same postflop function bodies, but their preflop, auction and opponent
profiling differ. This is not an isolated test of auction spending.

The protocol is eight seats, four frozen opponent panels, 128 independent seed
blocks per panel and eight cyclic seat rotations: **4,096 tables** and 512
independent blocks. Starting ordinary/auction balances are 2,000/1,000; payouts
are 50/30/20, with official ICM at the hand cap. Each table distributes 100
points, so 12.5 is the equal-share reference, not a separately measured bot.
These are local fixed-field scores, not live tournament standings. Public seeds
are regression evidence rather than an unseen holdout.

The new bot made **536,181 decisions**, including **31,386 auctions**, with zero
exceptions or illegal actions. All opponents also had zero failed decisions.
The independent score/schedule audit passed, and a separate raw-record sum
reproduced the candidate mean. Mean auction spending was **187.21 of 1,000**
per table, **24.43 per participated auction**, with **7.66 auctions per table**.
It retained **812.79** auction chips on average. Low spending alone does not
prove that raising BASE_BID will fix the score: elimination and entering fewer
hands also limit the number of auctions. No parameter sweep was performed.

## Implementation and validation

The human-readable strategy and exact commands are in [README.md](README.md).
The entry point is [agent.py](../../candidates/simple_classic_v1/agent.py), and
the only auction parameter is [config.py](../../candidates/simple_classic_v1/config.py).
The source is independent of other active agents' work. No existing candidate,
frozen benchmark definition, opponent source or live submission was changed.

- Seven behavioral/continuity tests passed, including 1,000 random auction
  states with balances 0, 1, 2, 3, 7 and 1,000.
- All retained postflop function source segments and both correctness-helper
  files match the classic donor exactly. No opponent profiling is invoked;
  postflop uses the default `unknown` opponent type and empty byte memory.
- The 64-table wiring run had 8,506 candidate decisions and zero total failures.
- The 9,589-byte [package](simple_classic_v1.zip) passed official SDK pack/check;
  every packaged source hash matches the measured benchmark snapshot.
- Two eight-seat SDK subprocess tables of 20 hands passed with zero failures
  for all bots, using a generous 1,000 ms decision limit. The initial attempt in
  the benchmark-only environment failed because its subprocesses could not
  import the SDK; the successful rerun used the existing event environment,
  whose 38 released SDK files match the frozen benchmark. Both attempts are
  retained. These checks do not certify a production 1 ms deadline.

## CPU throughput observation

The complete run used 16 workers and took **161.56 seconds** including source
snapshotting and score audit, or **25.35 tables/second**. On this shared AMD
EPYC 7B12 host, a two-second sample during the last panel measured approximately
93-100% of one logical CPU per worker. This supports CPU-bound simulation.
There are 240 visible logical CPUs; logical CPUs 48/168 and 63/183 are examples
of sibling threads sharing physical cores. Other active evaluations used some
of those siblings, so logical-CPU affinity alone does not give physical-core
isolation.

The visible running ordinary GCE inventory in `he-vision-group` contained only
`kmh-ac-dispatcher` (`e2-standard-8`, `us-central1-a`, AMD Rome). This is not a
survey of TPU host availability, and no remote job or VM was launched. Moving
to GCloud is not by itself a speedup. A next throughput test can use 32 or 64
workers on distinct, available physical cores; scaling has not been measured.
A remote comparison needs identical source, seeds and dependencies and should
report tables/second separately from single-decision latency.

## Retained evidence

- [Audited summary](../../evidence/candidate-simple_classic_v1/summary.json)
- [Run manifest and source hashes](../../evidence/candidate-simple_classic_v1/manifest.json)
- [Compressed raw tables](../../evidence/candidate-simple_classic_v1/tables.jsonl.gz)
- [Paired comparison](comparison.json)
- [Validation and timing scope](validation.json)
- [Donor provenance](provenance.json)
- [Candidate catalog](../../candidate_catalog_simple_v1.json)
