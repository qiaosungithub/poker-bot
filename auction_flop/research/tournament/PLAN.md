# Tournament branch plan

Owner: tournament_manager. Branch: strategy/tournament.
Prepared 2026-11-09 before reveal; this file contains design only, no competing
implementation. Read the released kit before choosing any concrete variant rule.

## Hypothesis and first version

Build a fresh, low-latency rule policy whose main difference is tournament risk.
Use the existing TournamentSpecialist research as method-level evidence, not
source to copy. Its old six-player custom-score results do not establish strength
in this event. In particular, do not reproduce its evaluator-label mistakes or
its treatment of board-only made hands.

V1 sequence after reveal:

1. Implement official Agent.act(state, memory), all variant phases, and one
   legal-action projection shared by every decision path. Integer raise-to
   totals are clamped against the actual descriptor. No stale action rules.
2. Derive position among active seats, effective stack in big blinds, amount
   already invested, number of live opponents, stack coverage, remaining paid
   places, and hand/blind urgency from state.config and current state.
3. Preflop: fresh handwritten opening, defending, and short-stack push ranges;
   tighten early position and multiway calls, widen late unopened pots, separate
   opening from calling large raises. Stack bands initially 12 / 25 BB.
4. Postflop: deterministic made-hand, kicker, private-card contribution, board
   texture and draw classification. Prefer thin value/pot control with modest
   pairs; build pots with private-card strong hands; reduce speculative calls
   against large bets and multiple opponents. Strong draws get bounded
   semi-bluff frequencies. Avoid Monte Carlo and unbounded enumeration in V1.
5. Risk control: a bounded extra calling-equity margin when a covered medium
   stack risks elimination near a payout jump; smaller margin for the shortest
   stack or a covering stack. Opening pressure remains distinct from calling
   risk. Scale with actual payouts; remove the premium for winner-take-all.
   This is an ICM-inspired heuristic, not an exact ICM calculation.
6. Deterministic mixing based only on public state plus own cards. V1 needs no
   persistent opponent memory. Any later memory is explicit per-table bytes.

## Variant integration

- Read RULES.md, protocol models, engine transition rules and example agent.
- Add an explicit handler for each non-betting phase before ordinary betting.
- Re-derive card strength and draws if deck, hole-card count, showdown selection,
  board progression or hand ranking changes; do not silently use Hold'em ranks.
- Price any resource, exchange, bidding or reveal action against its actual
  chip cost and tournament risk. First version may use a conservative legal
  choice, but must implement the phase deliberately and record that limitation.
- Re-read true hand cap, blind schedule, payout schedule, and round weighting.
  The live portal currently says fibonacci; older linear weights are stale.

## Small refinement space

| Parameter | Initial candidate values | Purpose |
| --- | --- | --- |
| Short-stack boundary | 10, 12, 15 BB | Push versus raise/fold transition |
| Medium-stack boundary | 22, 25, 30 BB | Restrict speculative calls |
| Risk-premium scale | 0, 0.5, 1.0 | Test value of survival control |
| Small value size | 0.33, 0.50 pot | Charge worse hands without overcommitting |
| Strong value size | 0.66, 0.85 pot | Extract from strong private hands |
| Bluff frequency multiplier | 0, 0.5, 1.0 | Bound risk against a weak/passive field |

Use a small paired screen, not the Cartesian product. Start with risk-premium
ablation while holding the rest fixed; reserve disjoint confirmation seeds.

V2 if V1 is valid and time remains: replace a narrow class of heads-up all-in
calls with exact payout utility. For properly constructed fold/win/lose chip
vectors, the required equity is (U_fold - U_lose)/(U_win - U_lose). This requires
correct pot ownership, side-pot eligibility, ties and elimination payouts;
otherwise retain V1. Top-three-payout ICM can be calculated with bounded loops,
but profile the entire decision path before enabling it. Never label a risk
proxy or chip-EV estimate as exact ICM.

## Acceptance checks and reporting

- Official check and play with eight seats and actual variant config; report
  failed decisions rather than hiding them behind fallback success.
- Classifier regressions: pocket aces making trips; pocket nines making quads;
  board-only two pair; board-only straight/flush; paired-board full house;
  kicker comparisons and a private-card improvement over the board.
- Legal edges: no reopening, minimum/maximum raise, short all-in, free check,
  all-in call, each added phase, and changing surviving-seat counts.
- Tournament edges: heads-up and winner-take-all risk behavior, zero chips,
  payout overrides, impending blind increase, final hand and unequal stacks.
- Official decision path timing: P99, maximum, count over 1 ms; separate timing
  from parallel benchmark load. Report explicitly whether SDK/IPC is included.
- Retain source commit/hash, exact parameters, config, independent seed blocks,
  seat rotations, raw scores, latency and legal/failure counts for every version.
- First runnable version target: within 15 minutes of confirmed release; do not
  delay valid baseline for optional exact utility or opponent modeling.

## Initial release check

At plan creation, shared RELEASE.json and starter-kit directory were absent.
The watcher directory contained competition-latest.json, watch.jsonl and an
unreleased starter-kit.download response. Parent manager owns continued polling.
