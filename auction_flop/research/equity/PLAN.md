# Equity branch plan

Prepared at 2026-11-09T23:04Z. This file contains design only; no competing
source was written before the official reveal. The latest observed watcher
cycle (23:03:54Z) had no announced variant and starter HTTP 404.

## Hypothesis and boundary

A bounded equity/pot-odds policy can improve calls and value bets without a
neural network or an offline learned strategy. The conceptual reference is
`txnujk-32` in the existing handwritten-policy report: its six-player standard
Hold'em result was statistically unresolved against TournamentSpecialist, but
its observed timing tails require a simpler event implementation. That result
is neither evidence for the unreleased variant nor an official score.

All source will be newly implemented in this worktree after reveal. The bot
will only depend on the official SDK and allowed evaluator/library versions.
No research bot source or pre-event weights will enter the submission.

## First version, target <=12 minutes after kit access

1. Read released AGENTS, README, RULES, legal descriptors and card/phase types.
   Verify deck, ranking, public information, added actions and exact scoring.
2. Implement a legal-action boundary: every requested raise is an integer
   street-total within the offered bounds; no raise if action is not reopened;
   check/call/fold selected only from actual legal descriptors. Added phases
   get their own deliberate legal policy.
3. Implement cheap preflop ordering from pair/rank/suit/connectivity features,
   with position, effective-stack and raise-size adjustments. Start tighter
   against multiple raises and large cold calls. This ordering is a heuristic,
   not a claimed exact equity table.
4. Postflop: evaluate actual hole-card contribution and board texture. Detect
   board-only strong hands, paired boards, four-flush/straight boards, vulnerable
   one-pair hands and clean versus discounted draw outs. Verify actual evaluator
   rank/category semantics instead of comparing assumed text labels.
5. Estimate showdown share with a small, fixed upper bound on compatible deals.
   Remove all known cards, include split pots and active opponent count. Use
   phase/range-conditioned sampling only if its cost fits the measured margin;
   no unbounded rejection sampling or exact river enumeration.
6. Compare estimated call value against the actual call price. For a simple
   uncapped pot, price is call/(pot+call); inspect side-pot eligibility and
   outstanding opponents before applying it. Apply a modest realization/risk
   haircut to marginal multiway hands and large stack commitments.
7. Value-bet strong hands; use small pot-control sizes for medium strength;
   choose rare semi-bluffs only with plausible folds and useful outs. Avoid
   unsupported pure bluffs into several opponents. Keep a cheap deterministic
   path for obvious folds/checks and near-certain strong hands.

## Bounded work and table isolation

Initial sample budget candidates: 12, 20 and 28 complete runouts, with total
opponent evaluations capped (initial target <=96 per decision). First ship the
smallest timing-safe candidate. Prefer immutable card/evaluator data created at
import; no mutable opponent knowledge in globals or `self`. Deterministic mixing
uses legal state inputs or a small PRNG state transported in `memory`. No work
between decisions and no cross-table files or learning. V1 may use empty memory
if public history suffices. Unknown/truncated memory must remain safe.

Target policy-compute P99 <=0.35 ms on the build box, leaving margin for official
SDK/IPC. This is a development target, not a guaranteed production limit.
Measure full official worker decisions separately: upload acceptance has no
reserve, unlike ordinary local play. Do not inflate sampling to spend time-bank.

## Variant extension points

Rebuild card distribution/ranking and payout accounting from revealed rules.
If the variant adds auction, discard, insurance, card choice or alternate
showdown rules, price each action by bounded incremental expected share minus
its cost. Use an explicitly conservative simple rule until verified. Never feed
nonstandard hands blindly to a standard Hold'em evaluator. Parent/variant branch
can share rule discoveries; this branch retains an independent betting policy.

## Hyperparameters and refinement order

Tune only a small declared set: sample count, multiway realization haircut,
large-call safety margin, preflop opening threshold by position, value threshold,
and bet fraction (initial candidates one-third and two-thirds pot). Preserve v1
as the control. First fix correctness/timing, then compare a few one-variable
variants on shared tuning seed blocks. Freeze a candidate before fresh-seed
confirmation; report official points, paired differences and uncertainty.

## Required evidence before candidate handoff

- Official static check and packaged artifact import; no forbidden imports.
- Full official match smoke test at eight seats and heads-up; zero illegal
  actions, exceptions and failed decisions in the observed run.
- Cases: minimum/maximum/all-in-only raise, raise not reopened, zero-cost check,
  multiway/side-pot all-in, trips/quads labels, board-only hand and split pot,
  short stack, final street, every added phase, empty/truncated memory.
- Sequential trajectory timing including longest observed public history,
  startup/fork behavior and full worker path. Retain per-call times, P99, max,
  over-1-ms count and whether SDK/IPC was included.
- Exact source commit/hash, parameters, seed blocks, table records and score
  protocol. Parent manager owns matched comparison and final activation.
