# Standard Hold'em handwritten-policy search — 2026-11-09

This protocol is written before latency or strength measurements. Preserve the
November 6 six-player benchmark; all new results use a new directory.

## Scope

Find public, runnable 6-max no-limit Hold'em policies whose decisions are
handwritten. Equity simulation and static equity tables are allowed; trained
neural policies and offline CFR strategy tables are not handwritten candidates.
Released Fullhouse and our iteration-210 reproduction are learned controls.
This is a search over accessible candidates, not proof of global optimality.

Survey: pinned Fullhouse opponent policies, txnujk's submitted floorfix,
Vikknows' submitted v12 candidate, Lucas Hsu's final v16_squeeze_C_metacap,
and HighRiskLowYield's fold-equity. Other sources and exclusion reasons will be
recorded. Author-reported event results are not independently verified rankings.

## Timing and permitted variants

First measure original candidates on 400 retained legal six-seat states (100
per street). This is a rejection screen, not final deadline evidence. Budget
variants may only lower the number of Monte Carlo samples; no thresholds,
equity formulas, or preflop/river decision rules may be tuned against scores.
Test sample caps 32 and 64 where a bounded sampling change is possible. Exact
river enumeration remains exact; a slow original path may therefore still fail.
Original and modified policies must have distinct names and documented edits.

Use CPU 2, one thread, GC disabled during timing, warm imports and evaluator.
Record complete state-dict-to-action-dict wall and thread-CPU times, including
features, history processing and equity computation; exclude engine, RNG
isolation, JSON/IPC and startup. Qualification is overall AND per-street wall
P99 below 1 ms, with at most 0.1% of sampled wall calls above 1 ms. This is a
measured statistical filter, not a hard real-time guarantee. Finalists receive
30,000 calls on the existing 10,000-state corpus and additional sequential
real-table timing with up to 200 public historical actions. Do not time policies
while parallel strength matches are running.

## Strength screen and independent confirmation

Reuse the validated PokerKit 0.7.7 Hand adapter and ICM implementation without
editing the frozen source: six players, 10,000 chips each, 50/100 fixed blinds,
200-hand cap, payouts [100,60,40,25,10,0]. At the cap, survivors receive exact
ICM; eliminated seats receive their actual payouts. Equal share is 235/6 points.
Raise normalization is the existing shared adapter and is counted. No timed
action penalties are applied to parallel strength matches.

Each candidate plays the SAME five controls: released Fullhouse, solver_like,
Hydra-32, station, maniac. Duplicate policy instances have separate memory.
Screen 128 independent blocks, 6 cyclic seat rotations per block; deck and
policy seeds are shared across candidates. Screening seed start 31,100,000.
Public bot identifiers are anonymous physical-seat IDs, not policy names.

Rank eligible handwritten candidates by screening mean; keep the best variant
per base-policy family, then select the four highest means. Freeze the selection
before confirmation. Confirm those four plus both Fullhouse controls playing
together: 2,048 new blocks, 6 rotations = 12,288 tables, seed start 31,200,000.
No strategy edits after screening. Means/CIs use block means, not hands or
correlated rotations. Report paired differences and simultaneous 95% intervals
for all final pairs, using a Bonferroni normal critical value.

If a finalist fails the larger timing filter, exclude it from the low-latency
recommendation and replace it with the next eligible screened family BEFORE
confirmation. Preserve its failed timing. Do not choose sample size by looking
for a statistically significant result.

## Validation and retained evidence

Reuse existing rule/side-pot/ICM tests. Add checks for independent bot state,
reproducible seeds, anonymous input, sample-cap edits, exception diagnostics,
and identical-policy controls. Preserve original source commits, modified-code
hashes, environment, raw timing samples, original per-table records, source
snapshot and independently recomputed scores. Include negative audit controls.
