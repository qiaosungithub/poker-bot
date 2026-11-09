# Frozen modular baseline validation

The package uses tournament-v1 auction decisions and variant-v1 betting decisions
through independently replaceable functions. The common observer records only
public information and is unused by both baseline policies, preserving actions.
Frozen source provenance is in VENDORED_SOURCES.json (only import names changed).

- Six focused observer tests pass: exact current selected payments, zero payment
  censoring, both dead cards, repeated/timeout replay, preflop-folded player's next
  hand observations, multi-auction aggregate censoring, malformed/missing memory,
  truncated-memory midauction recovery, table isolation and currency separation.
- Eight official-engine eight-seat trajectories: 9,739 decisions, zero illegal
  actions; every action equals its original frozen phase policy. Every per-seat
  cumulative observed payment and current-hand payment matches independent
  engine settlement values in the development-only audit.
- Memory maximum 656 bytes in that audit. Observer-only shared-host wall P99
  0.126 ms, maximum 0.984 ms; concurrent host load uncontrolled, no SDK/IPC.
- Official BotPool/guest process validation: six 20-hand heads-up zero-bank tables
  plus three 100-hand eight-seat normal-bank tables. All 495 hero decisions and
  all opponents have zero failures. One river transport measurement exceeds
  1 ms (1.254 ms); the official local process test uses 2 ms delivery grace.
  Auction transport P99 0.656 ms, maximum 0.719 ms over 44 auction decisions.
  This is local official SDK/IPC validation, not production microVM acceptance.

Exact reports: evidence/trajectory-parity-v1.json and
 evidence/process-baseline-v1.json. Root owns independent remote qualification.

CONTRACT_V2.md describes the next three-stage extension; it is not implemented
in the frozen v1 baseline and does not change these measurements.
