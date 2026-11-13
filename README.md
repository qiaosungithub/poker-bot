# Poker bot workspace

Start here when taking over without conversation history.
The November 9–10 Boston Auction Flop work is archived as of **2026-11-13**.
Start with [Auction Flop](auction_flop/README.md), the
[macro experiment results](auction_flop/research/macro_program/README.md), and
the [archive and recovery guide](archive/2026-11-13/README.md).
Earlier standard Hold'em and infrastructure reports retain their observation dates.

## Read in this order

1. [Competition rules and unresolved differences](docs/competition/rules.md).
2. [Official bot interface, sandbox and timing](docs/competition/runtime.md).
3. [Infrastructure: portal, SSH build box and local tools](docs/competition/infrastructure.md).
4. [Fullhouse reproduction: results, speed and exact artifacts](docs/research/fullhouse.md).
5. [Git layout, submodules and restoring source](docs/repository.md).
6. [Bay Area public resources and variant](docs/research/bay-area-2026.md).
7. [Runnable Bay Area environment and strategy comparison](bay_area/README.md).
8. [Standard Hold'em handwritten strategies: CPU timing and six-player results](benchmarks/heuristic_holdem_2026-11-09/REPORT.md).

[Source provenance](docs/sources.json) records the official pages and capture
locations. [human.md](human.md) and [scoring.md](scoring.md) are short Chinese notes.

## Current state

- Boston competition: `abb5539f-60ce-48b6-8925-c0c56a3102e3`.
- The released variant is **Auction Flop**. Its released SDK, frozen local
  benchmark and event-time policies are in `auction_flop/`.
- The strongest retained local v1 result is **19.80713 points/table** for
  prior + Hboth. Transfer to a different opponent field remains inconclusive;
  this is not an official score or a production qualification.
- The last recorded active submission is `practice_prior_full_v1`, a different
  policy with local v1 score 17.65625. See the dated
  [submission receipt](auction_flop/evidence/submission-confirmation.json).
  No current portal status or final standings were queried for this archive.
- The official build box is configured and SSH login was verified on November 9:
  `ssh no-limit-2026-build` from this server reaches `boston-team-95` as `ubuntu`.
  The setup skill is installed, the competition API key is saved privately, and
  our SSH public key is registered. Remote Codex login is optional and unverified.
  The November 9 pre-reveal 404 recorded in the infrastructure guide is
  historical; the starter kit was subsequently released and used.
- Fullhouse self-play training finished: 250 iterations, 7.5 million traversals,
  2 h 5 min 23 s. Use **iteration 210**, not final iteration 250.
- On the independent six-player comparison, iteration 210 scored **60.94** versus
  **58.59** for the released checkpoint; paired difference **+2.35**, 95% CI
  **[+1.51, +3.19]**. These are custom research points, not official contest points.
- A fresh CPU timing run measured iteration 210 at **0.331 ms P99** for the full
  research decision function. It excludes the official SDK and IPC; one of 30,000
  calls exceeded 1 ms. This is not an official deadline guarantee.
- The [November 9 build-box test](fullhouse-repro/latency/2026-11-09-buildbox-comparison/REPORT.md)
  measured iteration 210 at **0.196 ms P99**, **0.446 ms maximum** and zero
  overruns in 30,000 calls on the supplied Intel Xeon 8488C EC2 `m7i.large`.
  The matching research-server run had P99 0.363 ms. Both exclude official SDK
  and IPC; Python builds and load differ. The remote timing environment is ready
  at `~/work/fullhouse-latency`; that test preceded the starter-kit release.
- The November 8 organizer email prohibits pre-written code/bots and requires
  event-time coding. Our existing research bot is not a ready-to-submit entry.
  The email does not separately resolve the status of previously trained weights.
- The [Bay Area environment and public-bot benchmark](benchmarks/bay_area_2026-11-08/REPORT.md)
  are runnable. Sixteen versions from three public repositories were screened;
  eight finalists played 4,096 tables together. NC `codex_nc_v7_adapted` had the
  highest mean (13.782 points/table), but the experiment did not identify a
  unique best strategy. See the report for uncertainty and timing scope.
- The [November 9 standard Hold'em comparison](benchmarks/heuristic_holdem_2026-11-09/REPORT.md)
  screened nine handwritten families. In 12,288 six-player confirmation tables,
  TournamentSpecialist scored 45.13 and txnujk-32 scored 44.46 custom ICM points;
  their difference remains statistically unresolved. TournamentSpecialist had
  actual-game P99 0.084 ms and zero overruns in the two final timing tests.
  Released Fullhouse had the highest mean final chips in that same field.
  Follow-up source review reproduced a TournamentSpecialist bug that can fold
  trips/quads because of mismatched evaluator labels. Preserve the measured
  original as a baseline; the score does not justify using it unchanged.

## Directory map

| Path | Purpose |
|---|---|
| `auction_flop/` | Released Boston SDK, immutable v1 benchmark, candidate source and raw evidence |
| `archive/2026-11-13/` | Closeout index, event Git history, artifact recovery and cleanup provenance |
| `docs/competition/` | Official rules, runtime and infrastructure handoff |
| `docs/research/fullhouse.md` | Reproduction method, result provenance, commands and limitations |
| `bay_area/`, `external/` | Bay Area environment, SDK arena and pinned public strategy repositories |
| `benchmarks/bay_area_2026-11-08/` | Variant benchmark protocol, original table records and uncertainty |
| `heuristic_holdem/` | Standard Hold'em handwritten-policy loaders, timing, arena and independent score audit |
| `benchmarks/heuristic_holdem_2026-11-09/` | Candidate survey, frozen selection, six-player results, raw timings and source snapshot |
| `fullhouse-repro/` | Training implementation, checkpoints, evaluations and new latency results |
| `benchmarks/six_player_2026-11-06/` | Frozen comparison table and exact original protocol; preserve both files |
| `hu-benchmark/` | Frozen benchmark code, raw games and retained latency state corpus |
| `latency-survey/` | Original Fullhouse repository, released weights and ONNX adapter |
| `holdem-ppo/`, `txnujk-bot/` | Other existing research checkouts; not the Fullhouse reproduction |
| `.runtime/` | Work-local Python environments, caches, temporary files and private portal session |

The top-level directory is a Git repository on `main`. Nine upstream
checkouts are pinned submodules; the modified training worktree is reconstructed
from a tracked patch. Models, environments and private login state remain
outside ordinary Git source; see [repository boundaries](docs/repository.md)
and the [encrypted recovery archive](archive/2026-11-13/README.md).
Do not move large artifacts to tidy the layout: existing scripts and frozen
provenance use these paths.

## Resuming this project

Clone the repository and follow `auction_flop/README.md` to recreate the local
benchmark environment. Use new output directories and preserve v1 unchanged.
The archive guide covers models, unpublished worktree files and private data.
Recheck the portal and any remote machine before treating historical deployment
notes as live state. No further experiment or competition submission is part of
the November 13 archive operation.
