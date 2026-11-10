# Independent frozen-source audit — 2026-11-09 23:54–23:59 UTC

Scope: strongest modular hybrid `7e7aa88`, inference `6d12db2`, joint
`3a74011`. All three reuse the variant-v1 betting core. The three failures
below originate in that original core, not in the later inference integration.
No shared or production source was changed, and no strength benchmark ran.

Reproduce with the shared official Python 3.12:

```
python audit/reproduce_frozen.py
```

The script extracts exact Git objects into ignored `.audit_sources/` and runs
each snapshot in a separate process. It asserts the observed defects and saves
`audit/frozen_findings.json`. It never imports a changing peer worktree.

## 1. Unbeatable board is incorrectly treated as worthless private contribution

Hero `2c 3d`, board `As Ks Qs Js Ts`, heads-up, pot 300 before our call,
to-call 100, own stack 1000, legal fold/call/raise. Every possible opponent has
exactly the same best hand. Actual pot share is exactly 0.5; calling pays 100
for a 200 return (net +100).

All three frozen versions assign quality 0.12, category 8, and **fold**.
The exact-board-equality branch returns a low score solely because the hole
cards do not improve the board. Missing private improvement does not imply
poor showdown share when the board itself is unbeatable.

Required repair: separate guaranteed ties/nut boards from weak shared boards;
use tie share and reachable pot eligibility. Do not simply set every board-only
hand to high quality: board-only two pair can still be badly beaten.

## 2. Public quads hide a weak private kicker

Hero `3c 4d`, board `As Ah Ad Ac 2s`, public dead candidates `8c 9d`, heads-up,
pot 300, to-call 100. All three assign quality **0.97** and **raise to 400**.
Exact offline enumeration of all 903 possible opponent two-card combinations
(excluding hero, board and known dead cards) gives hero only **0.0282392** of
the pot under a uniform opponent distribution.

The hole cards technically improve the board kicker (2 to 4), so the board-only
check does not fire. The generic `category >= 6` rule then values all quads at
0.97 without checking whether the quads are public and the private kicker is
weak. That category is not a relative-strength estimate. This is a material
hand-contribution defect, though this audit does not estimate its tournament
frequency or prove a general expected-value ranking of alternate bet sizes.

Required repair: classify public quads by the best private kicker and available
higher kickers; likewise inspect low one-card flush/straight contribution on
four-card boards before treating category alone as a strong value hand.

## 3. Inaccessible side-pot chips distort a short-stack call price

Three players on `Ks Qd 9c 7h 4d`. Hero `Kc 2d` has 100 chips, no street bet;
the other two players are all-in with 1000 each committed on this street.
The public pot is 2000 and `state.to_call` is 1000 (the native engine reports
the largest bet minus hero's street bet). Legal actions are fold/call.

The policies cap call cost at 100 but use the **entire** pot: 100 / 2100 =
**4.76%**. Hero can only contest 100 from each opponent: after calling the main
pot is 300, so the correct main-pot threshold is 100 / 300 = **33.33%**. The
remaining 1800 is a side pot hero cannot win. All three policies **call** using
the full pot; a diagnostic control changing only the policy's pot denominator
to the accessible opponent contribution (200) makes them **fold**. The control
is a measurement of the predicate, not a claim that its altered state is a
real engine snapshot.

Required repair: reconstruct normal-chip per-hand contributions from public
history and sum the layers eligible after this action. Count already committed
hero chips, folded contributions and prior streets correctly. Test heads-up
uncalled excess and several side-pot layers. Do not count flop currency as any
normal-chip contribution. Auction importance should also use an accessible-pot
quantity when it values all-in survival opportunities.

## Checked boundaries and remaining uncertainty

- Numeric eval7 category checks recognize Trips and Quads; the historical string
  label mismatch was not reintroduced. The contribution problems above remain.
- `public_unseen_cards` in the inference snapshot excludes hero cards, current
  board, all five known auction candidates and explicit dead cards. No simple
  candidate-dead-card reinsertion was found in this bounded review.
- Auction adapters keep flop-chip currency separate from betting pots/stacks.
  Their approximate auction importance can still overweight inaccessible pots.
- New inference attempts to remove the current paid observation from historical
  means; joint `_heavy_competing_spend` instead uses its supplied mean directly.
  That is a possible attenuation/double-use issue, not established here as a
  rule error or a wrong action. No broad claim of memory/feature correctness.
- Inference and joint probes use their documented missing-features fallback;
  the hybrid probe executes the complete Agent. These are valid cold/reset
  cases and expose the shared core. Rich-feature paths were reviewed but not
  exhaustively enumerated.
- No latency or strength claim follows from these small correctness probes.
