# Poker workspace handoff

Read [README.md](README.md) first. It routes to competition rules, runtime and
infrastructure documentation, and the Fullhouse reproduction reports. Consult
[docs/sources.json](docs/sources.json) for observation dates and provenance.

**Branch `gpu-training` — Fullhouse training is GPU code here.** For anything
about training, reproducing or timing Fullhouse, read
[docs/research/fullhouse-gpu.md](docs/research/fullhouse-gpu.md) and
[fullhouse-gpu/README.md](fullhouse-gpu/README.md) first. The trainer runs on a
4x A100 machine (measured on GCP VM `qiaos-4a100-2`); its checkpoints stay there,
not in Git, identified by `fullhouse-gpu/results/runs/checkpoints_*.sha256`.
Do not edit `fullhouse-gpu/results/` or `fullhouse-gpu/fit/` (byte-identical to
the code that trained run 2, see `fit/MANIFEST.sha256`); write new runs and
evaluations to new paths. `fullhouse-repro/` keeps only the original TPU run's
results and the shared evaluation scripts.

- The user operates from a server terminal; prefer CLI/SSH workflows. A ChatGPT
  desktop app is not required.
- Official pages disagree on some defaults. Preserve the distinction between
  documented defaults, observed live configuration and unresolved rules.
- The organizer's November 8 email forbids pre-written code/bots for the event.
  Keep preparation/research separate from claims of submission eligibility.
- GPU reproduction: run 2 (10.3 min) selected iteration 40, paired +4.76
  [+3.88, +5.65] vs released; run 1 (16.4 min) selected iteration 20, +2.04
  [+1.19, +2.90]. Their final iterations are worse. Status records:
  `fullhouse-gpu/results/eval/{full,full_v2}/status.json`.
- Original TPU reproduction: iteration 210 is its selected checkpoint; final
  iteration 250 is worse. Read `fullhouse-repro/status.json`, not the historical
  status in `objective.json`. Its TPU launcher (`launch.py`, `supervise.py`,
  configs) is removed on branch `gpu-training`; it remains on `main` @ `7356b42`.
- Portal cookies and browser state under `.runtime/private-web/` are credentials.
  Do not print them, copy them into documentation or include them in submissions.
- The root is a Git repository. Read `docs/repository.md` for pinned submodules,
  ignored model files and restoring the modified Fullhouse worktree. Its local
  edits are tracked through `fullhouse-repro/training_changes.patch`, not directly.
- `docs/research/bay-area-2026.md` describes a participant-published Bay Area kit.
  It is not the confirmed Boston variant or evidence of a champion checkpoint.
- `bay_area/README.md` documents the runnable variant environment. The frozen
  benchmark is `benchmarks/bay_area_2026-11-08/REPORT.md`: three repository groups,
  not sixteen independent entrants. Preserve raw records and source commits;
  use a new output directory for additional experiments. No unique winner was
  established by the 4,096-table confirmation.

## Storage

Keep this project's checkpoints, downloaded models, Python environments, caches,
and temporary data under `/kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot`.
The user explicitly requested this because the local root filesystem is small.

- Fullhouse checkpoints and results: `fullhouse-repro/runs/` and `fullhouse-repro/eval/`
  (original TPU run); GPU runs write to `fullhouse-gpu/trainer/runs/` (ignored by Git).
- Environments, Python installation, caches, logs, and temporary files: `.runtime/`.
- `fullhouse-repro/runtime_paths.py` configures work-local runtime paths.
- Historical `/tmp/poker-*` symlinks are compatibility aliases only. Do not replace
  them with real directories or place new model/environment data on the root disk.
- Preserve the frozen benchmark records and their historical provenance.
