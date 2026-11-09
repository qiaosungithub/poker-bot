# Poker workspace handoff

Read [README.md](README.md) first. It routes to competition rules, runtime and
infrastructure documentation, and the Fullhouse reproduction report. Consult
[docs/sources.json](docs/sources.json) for observation dates and provenance.

- The user operates from a server terminal; prefer CLI/SSH workflows. A ChatGPT
  desktop app is not required.
- Official pages disagree on some defaults. Preserve the distinction between
  documented defaults, observed live configuration and unresolved rules.
- The organizer's November 8 email forbids pre-written code/bots for the event.
  Keep preparation/research separate from claims of submission eligibility.
- Use iteration 210 for the completed Fullhouse reproduction. Final iteration
  250 is worse. Read `fullhouse-repro/status.json`, not the historical status in
  `objective.json`; do not launch `supervise.py` merely to inspect progress.
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
- `heuristic_holdem/README.md` and
  `benchmarks/heuristic_holdem_2026-11-09/REPORT.md` cover the standard Hold'em
  handwritten-policy comparison. Preserve its raw timings, table records,
  frozen selection and source snapshot. Sampling-budget variants are distinct
  from upstream originals; passing the statistical timing filter does not
  guarantee every call finishes within 1 ms. Scores depend on the six-player
  field and custom ICM payout schedule, not official contest scoring.

## Storage

Keep this project's checkpoints, downloaded models, Python environments, caches,
and temporary data under `/kmh-nfs-ssd-us-mount/code/qiao/work/poker-bot`.
The user explicitly requested this because the local root filesystem is small.

- Fullhouse checkpoints and results: `fullhouse-repro/runs/` and `fullhouse-repro/eval/`.
- Environments, Python installation, caches, logs, and temporary files: `.runtime/`.
- `fullhouse-repro/runtime_paths.py` configures work-local runtime paths.
- Historical `/tmp/poker-*` symlinks are compatibility aliases only. Do not replace
  them with real directories or place new model/environment data on the root disk.
- Preserve the frozen benchmark records and their historical provenance.
