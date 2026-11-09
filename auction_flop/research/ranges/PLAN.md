# Ranges branch: event-time implementation plan

Owner: ranges_manager. Branch: strategy/ranges. Initial inspection: 2026-11-09.
The shared competition snapshot has no announced variant and RELEASE.json is
absent. This commit contains design only. Implementation starts after reveal.

## Hypothesis and scope

A small position-aware range policy with selective semi-bluffs and disciplined
showdown calls can improve on coarse hand-strength thresholds while keeping a
large latency margin. It is a handwritten heuristic, not a solved equilibrium,
trained CFR strategy, or neural policy. Existing solver_like research informs
method choice only; event code will be written afresh.

The previous standard-Hold'em solver_like result (34.88 custom points in its
six-player field) is not evidence of strength for the forthcoming variant.
Our differentiator is explicit own-card contribution, opponent action context,
and consistent ranges across betting streets, rather than copying that policy.

## First runnable version

1. Read the official kit instructions, variant rules, legal actions and state
   structures. Implement its native Agent.act(state, memory) contract directly.
   Write agent.py in a bot/ directory, without engine imports or runtime files.
2. Preflop: score explicit rank classes (pairs, suitedness, connectedness,
   domination risk), then apply position/opening/three-bet/call thresholds.
   Determine position from active physical seats and button, including heads-up.
   Effective stack and number of opponents modify thresholds and raise sizing.
3. Postflop: identify hand category, relative rank/kicker, contribution from
   private cards, draw strength, and board texture (paired, connected, monotone).
   Favor integer category values over evaluator display strings. Treat board-only
   hands as shared strength, not automatically as strong private holdings.
4. Use an action-context range estimate: a previous raise narrows opponents;
   repeated raises narrow further; a checked street may cap a range only weakly.
   v1 uses scalar strength thresholds and bounded recent-history parsing, not
   an expensive full combinatorial distribution or online optimization.
5. Value bet strong private holdings; protect medium hands at small sizes on
   appropriate boards; check marginal hands and avoid thin multiway bluffs.
   Draws with useful nut/blocker properties are preferred bluff candidates.
   River bluffs require relevant blockers and weak showdown value. A blocker
   is a modifier, never sufficient reason to call or bluff by itself.
6. Evaluate calls against pot odds with a conservative range-adjusted strength
   estimate and reverse-implied-odds penalty. Remove extra risk penalties on
   truly forced/free actions. No minimum-defense-frequency formula applied
   blindly to multiway hands or arbitrary opponent populations.
7. Mix selected actions using a deterministic per-table private-card/state hash
   and a compact counter in memory if needed. Never depend on global RNG or
   self for persistent table state. Keep memory below 1 KB in v1.
8. Generate actions only from state.legal_actions. Clamp integer raise-to totals
   to the descriptor bounds; if raising is closed, use legal check/call/fold.
   Variant-only phases have an explicit documented decision branch, with a
   legal fallback for unexpected states.

## Initial adjustable parameters

| Parameter | Initial family / limited sweep |
| --- | --- |
| Opening range width | tight / medium / loose by position |
| Opening raise | 2.0 / 2.4 / 2.8 big blinds, adjusted for limpers |
| Small / large postflop size | 0.33 / 0.65 pot, legal bounds always dominate |
| Semi-bluff frequency | 0.15 / 0.30 / 0.45 for qualifying draws |
| River bluff frequency | 0.00 / 0.10 / 0.20 for qualifying blockers |
| Multiway tightening | increase threshold per additional live opponent |
| Marginal-call safety margin | 0.02 / 0.05 / 0.08 estimated equity |

These are starting hypotheses, not estimates of optimal play. Freeze a small
configuration set before screening and use fresh confirmation seeds for selection.
Avoid large searches under the event budget.

## Variant extension points

- Rebuild rank/equity assumptions for changed decks, private-card counts,
  hand-ranking rules, multiple boards or split pots.
- Price auctions, discards, redraws or information purchases by explicit
  incremental benefit versus chip cost; do not silently map them to bets.
- Read actual table payout, stack, blind and hand-cap settings. Coordinate with
  the tournament branch for payout-risk ideas while retaining an independent
  range policy. Do not import another branch implementation wholesale.
- If the new mechanic dominates EV, prioritize a correct variant decision
  before elaborate range modeling or additional parameters.

## Acceptance and evidence

Before benchmarking: static official check; native simulator smoke games;
zero exceptions/illegal actions in eight-seat play; empty/truncated memory;
raise-not-reopened, min/max raise, all-in call and no-check/no-fold phase cases.
Exercise trips, quads, board-only two pair/straight/flush, weak kickers, paired
boards and private-card draws. Validate head-up button order and changing seats.

Retain the source commit/hash with each benchmark. Compare official points on
matched seeds and seat rotations; report illegal, exception and timeout counts
separately. Root owns the common arena. Run timing separately from parallel
strength work, including long histories and variant phases. Target ample margin
below 1 ms for the full act path; report measured P99/max/overruns and distinguish
SDK/IPC-inclusive timing from a pure-function benchmark.

At reveal, target a runnable v1 within 10–15 minutes, hand it to the manager,
then apply one evidence-backed refinement per subsequent cycle.
