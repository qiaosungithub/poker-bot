# SQA prior + fragile: full Mac replication

Mean points: **17.90283203125**, 4,096 tables, 512 independent
seed blocks, 646,011 candidate decisions, zero failures.
Independent score/schedule/source audit passed.

This is the unchanged `auction-flop-v1` with the documented macOS CPU-affinity
adapter. It is not a full Linux evaluation. The generated `official_score` field
marks a complete scoring schedule; it does not certify the platform.

See `platform-adapter.json` for platform, command, source revision and adapter
SHA256. Adapter source: `../sqa-compositions-v1/run_macos.py`.
The complete comparison and uncertainty are in
`../sqa-compositions-v1/comparison.json`; see `../../SQA_COMPOSITION_RESULTS.md`.

`manifest.json`, `sources/`, `tables.jsonl.gz`, `summary.json` and `audit.json`
retain the full schedule and immutable source evidence.
Linux SDK process validation is separate, in `../sqa-compositions-v1/linux-sdk`.
