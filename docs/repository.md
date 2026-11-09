# Repository and artifact boundaries

Initialized locally on 2026-11-08, branch `main`. The GitHub destination is
[qiaosungithub/poker-bot](https://github.com/qiaosungithub/poker-bot), configured
as `origin` after the user explicitly requested the first push. The initial commit uses the automation identity
`Codex <codex@localhost>` because no Git author identity was configured; no global
Git settings were changed.

## Branch `gpu-training`

Created on 2026-11-08 from `main` at `7356b42` and pushed to
`origin/gpu-training`, because the Fullhouse training code became GPU code. On
this branch:

- `fullhouse-gpu/` holds the 4x A100 trainer (traversal, fit, trainer, eval,
  tests), its three engineering reports and the measured records of runs 1 and
  2 ([summary](research/fullhouse-gpu.md)). Its outputs (`fullhouse-gpu/trainer/runs/`)
  and caches / CUDA-extension build (`fullhouse-gpu/.runtime/`,
  `fullhouse-gpu/fit/.runtime/`) are ignored; checkpoints of runs 1 and 2 stay on
  the VM `qiaos-4a100-2` and are identified by SHA256 in
  `fullhouse-gpu/results/runs/checkpoints_*.sha256`.
- The original TPU training and launch code was removed from `fullhouse-repro/`
  (JAX backend, `launch.py`, `supervise.py`, TPU configs, launch manifest and
  source ZIP, process / backend / resume validation records, environment lists,
  `validate_storage.py`). The TPU run's results, evaluation outputs, latency
  records and the evaluation / export / latency scripts are kept as reference.
  Use `main` for the complete TPU provenance.
- Commits are authored with the automation identity
  `Amply agent <amply@localhost>`, passed per command
  (`git -c user.name=... -c user.email=... commit`); no global Git settings were
  changed.

The GPU code needs `fullhouse-repro/source` (below) or `FULLHOUSE_SOURCE`; the
submodules are needed only for the six-player evaluation
(`latency-survey/fullhouse-bot`) and the Bay Area work.

## What the repository tracks

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
paths, hashes and validation commands. The GPU trainer's paths are relative to
the repository (see `fullhouse-gpu/README.md` §Environment).

## Files intentionally outside the parent repository

- `.runtime/`: environments, caches, temporary downloads and portal credentials.
- `fullhouse-repro/runs/`: training checkpoints, exported parameter arrays and
  replay reservoirs (including the approximately 9.2 GB final checkpoint).
- Model binaries (`*.pt`, `*.onnx`, `*.npz`, etc.), except the retained raw latency
  measurements. Existing binary files tracked by upstream submodules remain
  governed by those submodules.
- Generated logs and Python caches.

Models, environments and private session files remain on disk. Git is **not**
their backup. In particular, cloning only this repository does not transfer our selected
iteration-210 weights. The selected model and released reference are identified
in [the artifact manifest](research/model-artifacts.json).

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
