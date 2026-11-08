# Official runtime and bot contract

Source: authenticated [Getting started](https://pokerbot.sequoiacap.com/tournament/competitions/abb5539f-60ce-48b6-8925-c0c56a3102e3/start),
read 2026-11-08. The actual variant and starter kit were not yet released. Treat
the following as the documented interface, not a completed integration test.

## Interface and legal actions

An `Agent` subclass implements `act(self, state, memory)` and returns an Action
plus bytes to carry to the next call at that table. Standard betting supports
fold, check, call and `raise_to(total)`. The total is the entire amount committed
on the current street, not an increment. A variant can add phases/action kinds.

- Read `state.legal_actions`; `state.can(kind)` returns the legal descriptor or
  `None`. A descriptor contains bounds such as `min_amount` and `max_amount`.
- Check and call are not simultaneously offered. Reference Hold'em does not
  offer fold when checking is free.
- Out-of-bounds raises are failures, **not automatically clamped**. Floats are
  floored, so rounding can produce a raise one chip below the legal minimum.
- In an added phase without check/fold, the documented fallback takes the first
  offered action kind at its minimum amount.
- Showdown needs no bot action. Every showdown participant reveals all cards,
  including losing hands. Folding reveals no additional cards; an uncontested
  winner need not show.

The research benchmark uses a different dict-based interface and normalizes
some invalid raises. That adapter must not be mistaken for the official API.

## Available information

| Area | Fields / meaning |
|---|---|
| Current hand | `seat`, `cards`, `board`, `pot`, `to_call`, `legal_actions`, `history`, `phase`, `variant` |
| Table | `seats` with stack/bet/status/shown cards, and `button` |
| Tournament | `blind_level`, `blinds`, `hands_until_next_level`, `hand_number`, `players_remaining`, `bust_order` |
| Fixed table configuration | `table_id`, `seat`, `variant`, `players`, `starting_stack`, blind schedule, `hands_per_level`, `hand_cap`, `payouts`, `decision_ms`, `time_bank_ms`, `memory_bytes` |
| Incremental events | `state.events`: public events since this bot's last turn at the table; the whole public table history on its first turn |
| Remaining timing reserve | `state.time_bank_ms` |

`history` covers this hand only. Preserve longer-range information in `memory`.
Events identify kind, seat, amount and cards. Reference kinds include hand,
ante, blind, fold, check, call, raise, board, show, win and stack. A raise event
contains the street total; call contains the added chips; board contains newly
dealt cards. Ignore unknown event kinds unless the variant requires them.

## Time limits: three distinct cases

| Context | Decision allowance | Per-table reserve |
|---|---|---|
| Scored round default | 1 ms | 100 ms |
| Upload acceptance match | 1 ms | **0 ms**, independent of round settings |
| Local `poker-agent play` as documented | See kit configuration | Always supplies 100 ms |

The host starts timing when it sends the request, with a small delivery
allowance. Excess consumes the table's reserve. A timeout empties the reserve;
the next table starts with a fresh one. Serialization and memory transport are
inside the decision path. A timeout/exception/illegal action uses check when
available, otherwise fold, or the variant's automatic fallback.

Passing local play does not prove upload acceptance. Measure the complete
official decision path, including memory parsing; report P99, maximum and the
number over the threshold. Our local Fullhouse measurements omit SDK/IPC and
cannot certify this limit.

## Model loading, processes and memory

- A team has a pool of identical worker copies; any copy can answer any table.
  `self` and globals are not dependable per-table state.
- Production imports `agent.py` once, forks copies, then constructs each
  `Agent()`. Load immutable model data and perform a warm-up prediction during
  import, within the separate startup allowance. Its exact duration was not
  stated; upload rejection reports the applied limit.
- Only the importing thread survives a fork. Do not depend on a background
  thread/thread pool created before fork. Verify the chosen inference runtime
  with the actual production process model.
- `memory` starts at `b""` per table. Default maximum: 64 KB, with truncation
  beyond the configured limit. The page says roughly 16 KB adds no measurable
  overhead in their environment; that is not a portable timing guarantee.
- After timeout, exception or crash, old memory remains and the missed turn's
  events are delivered again. After an illegal action, returned memory is kept.
  Handle empty/truncated/old data defensively and avoid double-counting events.
- Workers may restart after crashes, stalls or protocol failures, with limited
  starts per round. An `act` exception only fails that decision. Exhausted worker
  restarts cause fallback play for remaining decisions on that machine.

## Libraries and sandbox

Python **3.12**, standard library, `poker_agent_sdk`, `poker_agent_protocol`, and
the pinned kit versions of NumPy, SciPy, pandas, scikit-learn, CPU PyTorch,
LightGBM, XGBoost, ONNX Runtime, Numba and eval7 are documented. Exact versions
must be read from the released `requirements.txt`. `poker_engine` and `pokerkit`
are local simulator dependencies and forbidden in uploaded source.

There is **no GPU and no network**. Uploaded files are read-only and on a
non-executable mount: uploaded native extensions/executables cannot run. The
sandbox's `/tmp` is 64 MB, writable but non-executable, shared by the team's
copies. No persistent cross-table files or computation between decisions.
This official sandbox rule does not change our local workspace rule: keep local
models/caches/temporary data under `poker-bot/.runtime`, not local `/tmp`.

Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` to `1` before
numeric imports. ONNX Runtime additionally needs an explicit single-thread
session; our research adapter sets intra-op and inter-op threads to one.
Set `NUMBA_NUM_THREADS=1` before importing Numba. The page notes macOS Accelerate
uses `VECLIB_MAXIMUM_THREADS`, which is not relevant to the Linux build box.

## Packaging and acceptance

Submit `agent.py`, or a ZIP with `agent.py` at its root plus data files:

| Limit | Portal value |
|---|---:|
| Uploaded package | 50 MB |
| Unpacked package | 200 MB |
| Archive entries, including directories | 1,000 |
| `agent.py` source | 1 MB |
| Path length / nesting | 512 characters / 16 levels |

`poker-agent check FILE` statically checks a file/ZIP without running it. Every
Python file in a ZIP is checked for forbidden engine imports, even unused ones.
`pack DIR OUT.zip` excludes hidden paths and bytecode caches, rejects symlinked
files, skips symlinked directories and validates the archive. Package only the
bot directory, not the entire kit or this research workspace.

The acceptance match is heads-up against check/fold, at most twenty hands in the
announced variant. More than 10% failed decisions, startup failure, crash or
unresponsiveness rejects the upload; a crashed acceptance bot is not restarted.
Team pages show rejection reasons, validation logs and round errors. Local play
exposes the last 64 KB of worker stderr; the production page does not expose
arbitrary bot print output. Verify the actual active submission after acceptance,
because the two portal pages disagree on automatic activation.
