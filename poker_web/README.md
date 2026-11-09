# RIVER — human vs. bot practice tables

A Chinese-language poker website using the repository's actual engines and
policies. Python and model inference run on this research server. Visitors need
only a browser; no account or local installation is required.

## Public link

Open **https://portland-bids-classics-anyway.trycloudflare.com** to play or share
with teammates. Each browser has an independent human-vs-bots table. Both modes
and all individually selectable strategies are available.

The public instance runs in tmux `poker-web-public`, with HTTPS through a separate
Cloudflare Quick Tunnel. Its current URL is also in `.runtime/web/public-url.txt`.
The address changes when the tunnel is recreated, and the link requires the
server and tunnel to remain running. See [deployment and recovery](DEPLOYMENT.md).
The original SSH instance below remains available independently.

## Open from your laptop

The intended server listens only on `127.0.0.1:8788`. In a terminal on **your
laptop**, add forwarding to your normal SSH connection:

```bash
ssh -N -L 8788:127.0.0.1:8788 YOUR_EXISTING_SSH_TARGET
```

Preserve any `-J`, `-p`, or key options you normally need. Keep that terminal
open, then visit **http://127.0.0.1:8788** in your laptop browser. If local port
8788 is occupied, use `-L 8789:127.0.0.1:8788` and open port 8789 instead.
VS Code users can alternatively forward remote port 8788 in the Ports panel.

## Start and inspect

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m poker_web.server --port 8788
```

The HTTP server uses only Python's standard library. It launches each table in
its own process using the existing evaluation environment for that mode:

- Classic: `.runtime/envs/poker-hu-validation/bin/python`.
- Bay Area: `.runtime/envs/poker-bay-area312/bin/python`.

On the current server the service was started in tmux session **`poker-web`**:

```bash
tmux attach -t poker-web
curl --fail http://127.0.0.1:8788/health
```

Server logs are `.runtime/web/server.log`; per-table diagnostic logs are in
`.runtime/web/logs/`. Stop through Ctrl-C in the owned tmux pane. Restarting the
service loses its in-memory tables. Never stop unrelated tmux sessions or kill
processes by a broad pattern. Do not start a second server on the same port.

## Playing

1. Choose classic Hold'em or the Bay Area three-card discard/reveal variant.
2. Choose your nickname and table size: **2–6** in classic, **2–8** in Bay Area.
3. Select each opponent independently. Repeated copies of a strategy are allowed.
4. Start the table. The human occupies seat 1; the dealer rotates normally.
5. Fold, check/call, or enter a legal raise total. The slider and minimum,
   half-pot, pot and all-in presets obey the engine's legal bounds. Raise
   amounts mean the total committed on the current street.
6. In Bay Area selection, choose one discard and a different card to expose,
   then lock the choice. Exposures appear only after all players have chosen.
7. Each hand stops at settlement. Inspect chip changes, points and public
   history before starting the next hand.

Playback can be comfortable, fast or immediate; it can also be paused. Fast
forward stops at the next human decision or the end of the hand. Keyboard
shortcuts are F/C/R when no input/control is focused. A busted human may keep
watching or start a new table. Starting a new table replaces the previous one.

Refresh resumes the current table while the server and browser session remain
alive. Lineup and playback preferences are stored in browser local storage;
private game state stays in the server process. This is one human per browser
session, not a network multiplayer lobby. The public server automatically reaps
tables idle for 30 minutes; the default local timeout is three hours. At most
eight sessions per server are retained. There is
no disk persistence or full replay export.

## Modes and policies

Classic offers txnujk-32, both real Fullhouse FP32 models, TournamentSpecialist,
solver_like, CFR-Lite-32, Hydra-32, Calling Station and Maniac. Fullhouse's
reproduced control is iteration 210. Model files stay at their existing paths;
missing ONNX files disable the relevant menu entry. A fresh clone also needs
the original Fullhouse package/model data described in
[the model guide](../docs/research/fullhouse.md).

TournamentSpecialist deliberately retains the benchmark source, including its
known trips/quads classification bug. The menu marks that limitation. There is
no silent repair or claim that a repaired version was benchmarked. Sampling
caps remain those in `heuristic_holdem/policies.py`.

Bay Area offers the eight finalists from the retained public-bot comparison:
NC v7 adapted, Den v7, James Tree, NC Final Reference, NC Refined, NC Supplied,
James Jam Top 3 and James Jam Tuned. These are versions from three source
projects, not eight independently ranked competitors. Classic policies are not
offered in the variant because their input/selection interfaces differ.

Classic uses the frozen `hu-benchmark/multiway_benchmark.py` Hand adapter and
PokerKit 0.7.7: 10,000 chips, fixed 50/100 blinds and a 200-hand cap. Its payout
vector is the first N entries of `[100,60,40,25,10,0]` for N starting seats.
Eliminations receive placement awards and surviving stacks split the remaining
prizes by exact ICM.

Bay Area uses the pinned participant-published engine with PokerKit 0.7.6:
2,000 chips, blinds beginning at 10/20 and rising every 20 hands, 250-hand cap,
50/30/20 payouts (places beyond the starting field do not pay). The human and
all-in bots still select after the flop. Discards stay private and dead.
This is the researched **Bay Area** ruleset, not a claim about another event's
unannounced variant. See [the environment guide](../bay_area/README.md).

Human thinking time is unlimited. Bot actions use real policies but this
practice UI does **not** enforce the competition's 1 ms timeout/bank. A generous
30-second process-response timeout prevents a stuck table from blocking the
server. This website is not a replacement for the frozen latency benchmarks.

## Implementation boundaries

| File | Responsibility |
|---|---|
| `catalog.py` | Allowlisted policy IDs, labels, descriptions and configuration validation |
| `server.py` | Loopback HTTP/static server, browser sessions, table subprocesses and stale-action checks |
| `worker.py` | One game per process; versioned commands and sanitized snapshots |
| `classic.py` | Human decisions and per-hand pauses around the unchanged classic hand engine |
| `bay.py` | Human/SDK decisions and per-hand pauses around the unchanged Bay Area engine |
| `static/` | Buildless HTML/CSS/JavaScript interface; no third-party requests |
| `tests/` | Legal actions, private information, settlement, sessions and actual browser gameplay |

Bay Area uses a narrow private-engine boundary: after the variant hand reaches
`actor() is None`, call the pinned table's `_end_hand()` and pause before its
next `decision()`. Do not replace it with an eager table step that silently deals
the next hand. Snapshot creation reads a view without calling table `_state()`;
the latter would mutate the SDK's per-seat event delivery bookkeeping.

Only human cards and publicly shown bot cards are serialized. The engine's
organizer hand log, deck seed, bots' private observations and memory are never
sent to the browser. Invalid human actions are rejected before engine mutation;
bot normalization/fallback follows the underlying adapter and is surfaced when
reported by the wrapper. A random HttpOnly SameSite session cookie isolates
browser sessions. Mutations require JSON, same-origin requests, the owned table
ID and current state version. The public instance additionally requires Secure
cookies and has a shorter idle timeout. Its HTTP listener stays on loopback;
only the Cloudflare tunnel exposes it. This is a small team practice deployment,
with no login or persistent game storage. Anyone with the URL can open a table.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-hu-validation/bin/python \
  -m unittest -v poker_web.tests.test_classic poker_web.tests.test_server
PYTHONDONTWRITEBYTECODE=1 .runtime/envs/poker-bay-area312/bin/python \
  -m unittest -v poker_web.tests.test_bay
node --check poker_web/static/app.js
```

Browser integration uses an installed Playwright and Chromium:

```bash
PLAYWRIGHT_MODULE=/absolute/path/to/playwright \
CHROMIUM_PATH=/absolute/path/to/chromium \
node poker_web/tests/browser.cjs
```

`POKER_WEB_URL` optionally selects another running practice server. The script
creates and closes its own browser tables. Screenshots and test output go to
ignored `.runtime/web/qa/`. If browser dependencies are missing, install tooling
under `.runtime/`, not the small root filesystem.

Verified on November 9, 2026: desktop (1500×1000) and mobile (390×844) gameplay,
classic raising/settlement/refresh, Bay Area selection/settlement, private-card
filtering, legal bounds, all-in selection and every selectable policy in
heads-up play. The mobile check asserts both no horizontal clipping and no
overlap between seat labels/cards and the board. These are interaction and
correctness checks, not new policy-strength measurements.
