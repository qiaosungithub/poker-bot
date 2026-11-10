# Linux SDK process validation

This is a runtime/interface check, not an `auction-flop-v1` ranking run.
Do not compare the points in `summary.json` with the unified benchmark scores.

The official SDK completed 64 eight-seat tables with zero failed decisions:

- `sqa_prior_classic_v1`: 10,845 decisions, zero failures.
- `sqa_prior_fragile_v1`: 9,084 decisions, zero failures.
- All eight seats combined: 68,635 decisions, zero failures.

The run used a 1 ms decision allowance, zero table time bank, and the released
SDK's 2 ms delivery grace. It tests the SDK host/guest process path on the team
Linux build machine; it is not a portal submission or remote microVM acceptance.

`full-source-verification.json` verifies every candidate Python module against
the composition manifest. `manifest.json` retains the runner, seed, settings,
and entry-point hashes. Raw table records are in `tables.jsonl`.

The runner is the existing `bots/dianne/tools/benchmark.py`, unchanged, with
`--sdk --blocks 8 --time-bank 0 --seed sqa-composition-sdk-v1`.
Remote output: `~/work/dianne-boston-test/v5/sdk-result`.
