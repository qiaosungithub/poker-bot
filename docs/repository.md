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

All these existing files remain on disk. Git is **not** their backup. In
particular, copying/cloning only this repository does not transfer our selected
iteration-210 weights. The selected model and released reference are identified
in [the artifact manifest](research/model-artifacts.json).

The ignored private browser helpers are also server-local. Their absence after
a clone does not prevent ordinary browser login or the official CLI setup.
