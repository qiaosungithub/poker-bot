# Poker bot workspace

Start here when taking over without conversation history. Documentation was
verified on **2026-11-08**; competition defaults may change at the variant reveal.

## Read in this order

1. [Competition rules and unresolved differences](docs/competition/rules.md).
2. [Official bot interface, sandbox and timing](docs/competition/runtime.md).
3. [Infrastructure: portal, SSH build box and local tools](docs/competition/infrastructure.md).
4. [Fullhouse reproduction: results, speed and exact artifacts](docs/research/fullhouse.md).
5. [Git layout, submodules and restoring source](docs/repository.md).
6. [Bay Area public resources and variant](docs/research/bay-area-2026.md).
7. [Runnable Bay Area environment and strategy comparison](bay_area/README.md).

[Source provenance](docs/sources.json) records the official pages and capture
locations. [human.md](human.md) and [scoring.md](scoring.md) are short Chinese notes.

## Current state

- Boston competition: `abb5539f-60ce-48b6-8925-c0c56a3102e3`.
- Portal login succeeded and Getting started / Format were read. The recorded
  competition response had no announced variant. Recheck live state before use.
- The official build box, credits and starter kit have **not been set up or
  claimed by this agent**. Existing account setup has not been audited.
- Fullhouse self-play training finished: 250 iterations, 7.5 million traversals,
  2 h 5 min 23 s. Use **iteration 210**, not final iteration 250.
- On the independent six-player comparison, iteration 210 scored **60.94** versus
  **58.59** for the released checkpoint; paired difference **+2.35**, 95% CI
  **[+1.51, +3.19]**. These are custom research points, not official contest points.
- A fresh CPU timing run measured iteration 210 at **0.331 ms P99** for the full
  research decision function. It excludes the official SDK and IPC; one of 30,000
  calls exceeded 1 ms. This is not an official deadline guarantee.
- The November 8 organizer email prohibits pre-written code/bots and requires
  event-time coding. Our existing research bot is not a ready-to-submit entry.
  The email does not separately resolve the status of previously trained weights.
- The [Bay Area environment and public-bot benchmark](benchmarks/bay_area_2026-11-08/REPORT.md)
  are runnable. Sixteen versions from three public repositories were screened;
  eight finalists played 4,096 tables together. NC `codex_nc_v7_adapted` had the
  highest mean (13.782 points/table), but the experiment did not identify a
  unique best strategy. See the report for uncertainty and timing scope.

## Directory map

| Path | Purpose |
|---|---|
| `docs/competition/` | Official rules, runtime and infrastructure handoff |
| `docs/research/fullhouse.md` | Reproduction method, result provenance, commands and limitations |
| `bay_area/`, `external/` | Bay Area environment, SDK arena and pinned public strategy repositories |
| `benchmarks/bay_area_2026-11-08/` | Variant benchmark protocol, original table records and uncertainty |
| `fullhouse-repro/` | Training implementation, checkpoints, evaluations and new latency results |
| `benchmarks/six_player_2026-11-06/` | Frozen comparison table and exact original protocol; preserve both files |
| `hu-benchmark/` | Frozen benchmark code, raw games and retained latency state corpus |
| `latency-survey/` | Original Fullhouse repository, released weights and ONNX adapter |
| `holdem-ppo/`, `txnujk-bot/` | Other existing research checkouts; not the Fullhouse reproduction |
| `.runtime/` | Work-local Python environments, caches, temporary files and private portal session |

The top-level directory is a Git repository on `main`. Six upstream
checkouts are pinned submodules; the modified training worktree is reconstructed
from a tracked patch. Models, environments and private login state remain
outside the parent repository; see [repository boundaries](docs/repository.md).
Do not move large artifacts to tidy the layout: existing scripts and frozen
provenance use these paths.

## Next useful work

Prepare the official SSH build box when requested. At reveal, read the kit's
`AGENTS.md`, `README.md`, `RULES.md` and pinned requirements, reconcile the rule
differences, and implement within the event's coding rules. Validate eight-seat
behavior, exact legal actions, per-table memory and the no-bank upload deadline
on the official environment. Do not present six-player research results as an
official eight-player evaluation.
