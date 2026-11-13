# Repository and artifact boundaries

Initialized locally on 2026-11-08, branch `main`. The GitHub destination is
[qiaosungithub/poker-bot](https://github.com/qiaosungithub/poker-bot), configured
as `origin` after the user explicitly requested the first push. The initial commit uses the automation identity
`Codex <codex@localhost>` because no Git author identity was configured; no global
Git settings were changed.

The parent repository tracks documentation, our training/benchmark code,
configuration, result JSON/JSONL, the latency state corpus and raw timings, and
frozen source archives. The two original files in
`benchmarks/six_player_2026-11-06/` remain unchanged.

The three original upstream repositories are pinned submodules. Their histories,
remotes and working files were preserved:

| Path | Upstream | Pinned commit |
|---|---|---|
| `latency-survey/fullhouse-bot` | `advitrocks9/fullhouse-bot` | `e504793d480b1b975f25258d25939b45c6dbd5a4` |
| `holdem-ppo` | `JishnuJetwani/HoldemBot` | `0b39291ff8e971041594ce351dd21c263c024f50` |
| `txnujk-bot` | `txnujk09/fullhouse-poker-bot` | `4fe154318949aa60430682416f148733c6f0d564` |

Three more pinned submodules support the Bay Area environment and benchmark:

| Path | Upstream | Pinned commit |
|---|---|---|
| `external/bay-area-entry` | `nnicholas-c/poker_sequoia` | `e6e68a1aa7c71f78fec17bd7ead813687689fc61` |
| `external/denfujita-poker` | `denfujita/poker` | `fc661ce9f505dd00f23fe656a5a4289fbc964b27` |
| `external/james-nolimit` | `JamesGongYC/nolimit-pokerbot` | `c91408d08d9946d33088ea6a6e5e477659c3df6b` |

The raw variant benchmark, latency samples and runner snapshot are versioned
under `benchmarks/bay_area_2026-11-08/`. Its executable setup and source API are
described in [the environment guide](../bay_area/README.md).

Three additional pinned submodules support the standard Hold'em handwritten
strategy survey (nine submodules in total):

| Path | Upstream | Pinned commit |
|---|---|---|
| `external/fullhouse-vikram` | `Vikknows/Fullhouse-Hackathon` | `cb16737fea04805f2ab1bd7f6a6de6f9bb0ea974` |
| `external/fullhouse-lucas` | `lucashsu007-create/fullhouse-hackathon-write-up-repo` | `8b98ee3efce3b224177609352258dc86db901cce` |
| `external/fold-equity` | `HighRiskLowYield/fold-equity` | `f427cb5cd429123e14da3626d8f440f7f9ffd4dd` |

The [handwritten-policy report](../benchmarks/heuristic_holdem_2026-11-09/REPORT.md)
records candidate selection, latency failures and the independent confirmation.
`heuristic_holdem/policies.py` applies documented sampling caps at load time;
upstream submodules remain unmodified. Raw timings, per-table results and the
as-run source archive are retained under `benchmarks/heuristic_holdem_2026-11-09/`.

`fullhouse-repro/source/` is an ignored linked worktree of the Fullhouse
submodule at the same upstream commit. Its intentional `training/train.py`
modification is preserved as `fullhouse-repro/training_changes.patch`; the patch
was compared byte-for-byte with its Git diff at initialization. New changes in
that worktree require updating the tracked patch.

## Reconstruct source after a fresh clone

Run from the new repository root. The worktree path must not already exist;
inspect an existing path instead of replacing it.

```bash
git submodule update --init --recursive
git -C latency-survey/fullhouse-bot worktree add --detach \
  ../../fullhouse-repro/source e504793d480b1b975f25258d25939b45c6dbd5a4
git -C fullhouse-repro/source apply --check ../training_changes.patch
git -C fullhouse-repro/source apply ../training_changes.patch
```

These commands restore source, not a runnable environment or trained models.
Read [the reproduction report](research/fullhouse.md) for dependencies, artifact
paths, hashes and validation commands. Some historical launchers use this
server's absolute paths and need adjustment on a different machine.

## Files intentionally outside the parent repository

- `.runtime/`: environments, caches, temporary downloads and portal credentials.
- `fullhouse-repro/runs/`: training checkpoints, exported parameter arrays and
  replay reservoirs (including the approximately 9.2 GB final checkpoint).
- Model binaries (`*.pt`, `*.onnx`, `*.npz`, etc.), except the retained raw latency
  measurements. Existing binary files tracked by upstream submodules remain
  governed by those submodules.
- Generated logs and Python caches.

As of the November 13 closeout, selected iteration-210 exports, the iteration-250
comparison export and the released ONNX reference have public copies in
[`archive/2026-11-13/models/`](../archive/2026-11-13/models/), with original paths
and SHA-256 hashes in its [manifest](../archive/2026-11-13/model-manifest.json).
The original [model inventory](research/model-artifacts.json) remains dated
historical evidence. Full training/resume state and private worktree artifacts
are covered by the [encrypted recovery archive](../archive/2026-11-13/README.md).
Environments and caches can be recreated; they are excluded from that archive.

The ignored private browser helpers are also server-local. Their absence after
a clone does not prevent ordinary browser login or the official CLI setup.

## Cleanup on 2026-11-08

Removed 131 ignored files (21,330,982 bytes): regenerable pip download caches,
project Python/pytest caches, and three logs verified byte-for-byte identical
to retained records. The canonical records are
`fullhouse-repro/storage_relocation_validation.json`,
`fullhouse-repro/results.jsonl` and `hu-benchmark/benchmark_report.json`.
The server-local path, size, hash and recovery manifest is
`.runtime/maintenance/cleanup-2026-11-08.json`.

No tracked experiment files were removed or moved. Raw benchmark records,
frozen source archives, historical validation results, all checkpoints and
installed environments were retained. The four selected/reference model
artifacts in the model manifest were hash-checked before and after cleanup.
