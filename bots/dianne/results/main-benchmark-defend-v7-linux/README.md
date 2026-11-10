# SQA auction-flop-v1: Linux full evaluation

`icm_defend_v7` scored **12.8466796875** points per table, 95% interval
**[12.2826289925, 13.4107303825]**, across all 4,096 scheduled tables and 512
independent blocks. All 587,991 candidate decisions succeeded; total failures
were zero. The unmodified runner's independent audit passed.

Use the SQA `preflop_loose` candidate (published mean **16.5625**) instead of
replacing it with V7. Earlier Dianne-field scores use different opponents and
are not comparable to this benchmark.

The run used the frozen main snapshot `481d1f0`, benchmark SHA256
`dc6c9f772eabdd0db0b31d011e548d40b8119e872773d58ecd9c077eaaa5eba6`,
on the team's Linux build machine. It completed at
`2026-11-10T00:24:37.108585+00:00`. No runner, source lock, game rule, opponent,
seed, or scoring changes were made. Evaluation uses direct calls; SDK latency
qualification is separately recorded in `../linux-sdk-v4`.

Candidate SHA256:
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`.

```sh
cd ~/work/dianne-main-benchmark
~/work/starter-kit/.venv/bin/python -m auction_flop.benchmark.run evaluate \
  --candidate /home/ubuntu/work/dianne-boston-test/v4/bots/dianne/icm_defend_v7 \
  --workers 2 --out /home/ubuntu/work/dianne-main-benchmark/result-defend-v7
~/work/starter-kit/.venv/bin/python -m auction_flop.benchmark.run audit result-defend-v7
~/work/starter-kit/.venv/bin/python -m auction_flop.benchmark.run compare \
  result-defend-v7 auction_flop/evidence/calibration-hybrid_tournament_auction
```

The evaluation output directory must not already exist when reproducing a run.

- `summary.json`: complete score, uncertainty and decision counts.
- `manifest.json`, `sources/`, `tables.jsonl.gz`: locked provenance and raw results.
- `audit.json`: independent audit of all 4,096 tables.
- `compare-hybrid.json`: paired difference versus the frozen hybrid reference,
  **−2.2412109375**, 95% interval **[−3.0533876650, −1.4290342100]**.
- `cross-platform.json`: all 4,096 rows equal to the Mac replication, excluding
  only elapsed seconds; both raw gzip hashes verified against their manifests.

No paired interval against the newer `preflop_loose` candidate was calculated.
Its published artifact is
[`candidate-preflop_loose/summary.json`](https://github.com/qiaosungithub/poker-bot/blob/c5ec82ae4b1ba0abc35395cfc2127996b25be69b/auction_flop/evidence/candidate-preflop_loose/summary.json).
