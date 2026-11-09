# Equity v1

Official variant integration completed 2026-11-09T23:24Z, after live reveal.
The submission directory is `bot/`; both `agent.py` and `equity_core.py` are
required. `equity-v1.zip` was made by the official packer.

## Decisions

- Betting: handwritten position/stack/preflop rules and bounded complete-deal
  equity with dead auction cards excluded, pot-odds margin, conservative large
  calls, simple side-pot eligibility accounting and legal raise clamping.
- Auction: ten joint three-card boards share opponent hands and runouts. Compare
  relative equity changes under candidate bids and broad public-balance bid
  priors. Choose zero when no intervention clears future-currency opportunity
  cost. Candidate bids include concentrated one-card changes and preferred
  three-card sets; never spend flop-chips as if they were normal chips.
- No neural inference/training, no imported pre-event policy and no mutable
  cross-table state. Memory is empty in v1.
- Auction models at most three random opponents and extrapolates additional
  opposition. This is a bounded heuristic, not exact multiway equilibrium.

## Validation

Official pinned local Python 3.12 environment and SDK: static check and pack
passed. Five focused kernel tests passed. Eight three-seat official matches
(seed equity-v1-smoke, hand cap 100) and eight eight-seat self-play matches
(seed equity-v1-eight, hand cap 100) reported zero failed decisions for every
seat. `smoke-eight.json` records the latter. These smoke results do not establish
comparative strength or bank-free upload acceptance.

`trajectory-v1.json` records full Agent-call timing and legality on four exact
engine trajectories, excluding process IPC and state construction. The root
manager owns matched comparison, remote full-path timing and activation.
