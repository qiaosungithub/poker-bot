# Betting with public auction-payment information

Event-time module, 2026-11-09. This module implements stage-three postflop
inference while preserving stage-one preflop behavior. It does not choose bids.

## Integration contract

Copy `betting_baseline.py` and `betting_inference_core.py` beside one selected
wrapper, installed as `betting_policy.py`:

- `betting_policy.py`: observed-payment inference with table-history calibration.
- `betting_policy_off.py`: exact frozen variant-v1 betting control.
- `betting_policy_nohistory.py`: current payment cue against a fixed prior.

Every wrapper exports `choose_bet(state, features)`, `choose_preflop`, and
`choose_postflop`. The compatibility `choose_bet` dispatches by board presence.
The shared observer owns all memory and updates it once. No mutable table state
lives in this policy. It depends only on the SDK, eval7 and standard library.

`betting_baseline.py` is a semantics-preserving betting-only extraction of fresh
event code from variant-v1 commit `6af47e4833ba09e91830b4f0064bc026ba08e1f0`.
The only extension is an explicit `quality_penalty` parameter. No pre-event
strategy implementation was imported.

## Hypothesis and precise limits

An opponent's unusually large observed payment on the selected flop can tilt
its hand range toward hands that benefit from that flop. This implementation
represents the tilt as a small penalty to our existing hand-strength proxy,
not as exact equity or a reconstructed posterior over hands. It is a hypothesis
for controlled testing; no strength improvement is yet established.

- Only active/all-in opponents with known current-hand paid amounts contribute.
- Unknown and zero payments produce no inference. Zero may reflect losing-card
  bids or free-riding; it does not establish disinterest or a weak hand.
- Payment is normalized to available auction budget and shrunk historical mean
  and fraction spent. Eight prior auctions regularize each history estimate.
- The observer's history includes the current settled auction. When identified
  by its last-payment marker, this observation is removed from its own history
  calibration so it cannot manufacture historical evidence for itself.
- Small payments and sparse history have small effects. The total quality-proxy
  penalty is capped at 0.055 across all opponents, rather than adding an
  unbounded per-opponent penalty. Preflop aggression modestly strengthens a cue.
- Draw-heavy flops attenuate the made-hand interpretation. An uncompleted draw
  on the river attenuates it again; a completed public draw restores caution.
  The policy never declares that a high spender necessarily has a made hand.
- Known private/public/candidate/dead cards are excluded from compatible-card
  support. For example, an opponent cannot privately match a paired rank when
  both remaining copies are known to be ours or dead.
- Strong private made hands remain robust to this noisy cue. No auction payment
  changes normal-chip pot odds, stack vectors or ICM prize calculations.

The normal betting policy remains a coarse deterministic heuristic. This first
module does not enumerate full opponent ranges, infer hidden per-card bids,
recover card scores, or use showdown-private engine data. Its cost and value
must be tested with the exact composed observer and auction policy.

## Evidence

Nine unit/regression checks pass, including 2,000 frozen-baseline action parity
states, an actual call-to-fold decision crossing, unknown/zero censoring,
sparse-history caps, draw/made distinctions, dead-card exclusion, rejection of
hidden fields, folded-player exclusion and current-observation removal.

`integration-smoke.json`: initial composed official SDK process run, eight
8-seat tables, all bots zero failed decisions. `integration-final-smoke.json`
records the final source layout with ON/OFF/no-history policies at the same
8-seat tables. Exact composition hashes are in `integration-final-source-hashes.json`.
These small matches validate compatibility only; they are not strength evidence.
No latency bound or production microVM acceptance is claimed. Root owns paired
large-seed comparison and independent remote qualification.
