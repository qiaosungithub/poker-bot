# Variant-specialist strategy plan

Prepared before reveal on 2026-11-09. This is a design document, not competing
implementation. Write all submitted code fresh after the official reveal. No
neural training, trained strategy tables, or copied historical bot source.

## Hypothesis and scope

A variant changes the value of actions before it changes optimal bet sizes.
Model its extra choice explicitly, then use simple value-oriented betting.
Historical Bay Area discard/reveal rules are a method example only: Boston may
have a different deck, hand evaluator, action vocabulary, or settlement rule.

## First five minutes after release

1. Read official AGENTS, README, RULES, pinned packages and protocol. Compare the
   rule text with the actual local engine for phase ordering and public fields.
2. Record cards dealt/retained, valid combinations, dead cards, public revelation
   timing, simultaneous versus sequential choices, actions required of all-in
   players, tie/side-pot treatment, and every new action's legal descriptor.
3. Identify the scoring objective and actual configuration. Side-prize statistics
   are not pot or tournament rewards unless the released rules explicitly say so.
4. Make a rule-to-state/action map with concrete fixtures from official examples.
   Report disagreements to the root manager immediately; do not silently guess.

## V1 decision decomposition

- **Small finite choice** (discard, swap, reveal, select): enumerate legal choices.
  Score retained made-hand strength, private contribution beyond the board,
  best draw potential and domination risk. Start with bounded structural scores;
  add small common-random-number rollouts only if timing leaves substantial room.
- **Public information**: remove known dead/exposed cards from candidate decks.
  Condition opponent possibilities on their shown cards. Publicly exposed cards
  remain unavailable even after their owner folds when rules say so. Never use
  simulator-private cards or future outcomes.
- **Information disclosure**: among near-equal retained hands, hide the card that
  reveals more about our strongest made hand/draw. Keep this adjustment smaller
  than a meaningful strength difference until an ablation supports it. Treat
  simultaneous choices as simultaneous; never assume opponents already acted.
- **Priced choice** (auction, insurance, extra card): estimate incremental value
  against the free/default option, subtract the real marginal price and apply a
  conservative stack-risk cap. Bid/spend only through legal bounds. Do not
  confuse committed chips with extra cost or assume auction payment semantics.
- **Changed evaluation/deck**: derive categories and combinations from official
  semantics. Standard eval7 is useful only after compatibility is proved.
- **Betting**: use position, effective stack, live opponents, board texture, private
  contribution, pot odds and stack-to-pot ratio. Prefer value bets and selected
  strong draws; avoid stacking off weak one-pair hands into several opponents.
  Restrict bluffing when exposed cards make our claimed range implausible.

V1 must remain runnable without any expensive exact equity path. Exact
enumeration is for genuinely small decision spaces, never an unrestricted
river opponent-hand enumeration inside a 1 ms request.

## Parameters and refinement

Keep a small explicit parameter record: choice-strength weights, draw weight,
information-disclosure weight (start at zero for a clean ablation), optional
rollout cap (0, 8, 16), price/value safety factor (0.5, 0.7), maximum stack
fraction spent on a priced mechanic, value threshold, and bluff frequency.
Only instantiate parameters relevant to the actual revealed variant.

First refinement compares the same betting policy with and without the special
choice logic. Second compares information conditioning and disclosure terms.
Choose on tuning seed blocks; freeze the candidate before fresh confirmation.
Do not infer general strength from a heads-up check/fold acceptance match.

## Validation and budget

- Aim for a legal, runnable first version within 12 minutes of usable kit access.
- Exercise every phase and action bound, including all-in mandatory choices,
  minimum/maximum indices/amounts, short-stack and no-raise-reopening cases.
- Construct examples where a choice is unambiguously dominated and require the
  scorer to reject it; flip a relevant card to require the preference to change.
- Test board-only pairs/two-pair, Trips/Quads labels, known dead/public cards,
  ties, side pots and duplicated/replayed events if memory is used.
- Use bytes memory only for per-table state. Start V1 stateless where possible;
  no self/global opponent state, files, network or between-decision computation.
- Run official static check, smoke tables, eight-seat play and short tables;
  count exceptions, illegal actions and timeouts, not just final score.
- Measure each new phase separately, then actual-game P99/max and counts over
  1 ms. Include memory parsing and report whether SDK/IPC is included. Validate
  the complete path on the organizer build box before claiming upload readiness.
- Preserve source commit, actual config, raw table outcomes, seeds and timing
  samples. Coordinate paired seat-rotation benchmarks through the root manager.

## Current evidence

At preparation, no RELEASE.json or extracted starter kit exists. The shared
watcher has observed null variant and HTTP 404 for the kit. No Boston-specific
strategy or runtime guarantee is claimed before inspection of the released kit.

## Post-reveal implementation checkpoint

The user photographs establish auction-flop Hold'em. `auction.py` now scores
all ten candidate flops and chooses among bounded minimum-forcing bid vectors
under seven anonymous opponent-budget scenarios. Selected-card-only payment
and currency opportunity cost enter the objective. Normal chips and auction
currency remain separate. The tie order is an explicit adapter argument.

Remaining rules await official kit: initial/replenished/transferred budget,
final-flop card order, exact auction action representation, flop betting, and
whether unused candidates remain dead. No API adapter is claimed yet.

### Bounded-latency revision

The richer implementation is frozen in `experiments/rich_kernel_v1.py`.
The active kernel scores the same ten boards using equivalent rank bit masks,
then evaluates at most seven analytic vectors against three opposition cases.
On 100 varied synthetic states / 2,000 local calls: P99 0.291 ms, maximum
0.516 ms, zero calls above 1 ms. These exclude SDK/IPC and are development
measurements, not submission guarantees. Raw states and per-call nanoseconds
are retained in `evidence/fast_kernel_timing.json`.
