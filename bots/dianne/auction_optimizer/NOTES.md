# Auction optimizer

Written during the Boston reveal on November 9, 2026. The implementation uses the released Boston rules and SDK dataclasses. It has no third-party strategy source or model dependency.

The auction evaluates all ten possible three-card flops with a fast made-hand and draw estimate. It assigns each preferred card a marginal value equal to the best-flop value lost when that card is unavailable. Available spending is divided according to these marginal values. Unhelpful auctions receive a zero bid.

The spending horizon decreases with remaining hands and stack depth. Large pots receive more budget, and an all-in hand can spend substantially more because surviving that hand matters immediately. Each bid is an integer, and total bids are bounded by the legal auction balance. Betting uses position, stack depth, pot odds, made hands, draws, and the number of opponents. The agent returns empty memory and carries no mutable state between calls.

Validation used the official SDK source directly while dependency installation was in progress.

- The SDK submission checker accepted `agent.py`.
- 3,000 randomized auctions produced valid five-element bids within budgets from 0 through 1,000.
- 1,200 randomized preflop, flop, turn, and river states produced legal betting actions.
- The auction decision function measured 0.0406 ms median, 0.0830 ms P99, and 0.1871 ms maximum over those 3,000 auctions on the laptop. This measurement excludes SDK serialization and process transport.

The board estimate is heuristic and does not integrate opponent ranges or exact equity. The initial implementation has no learned opponent bidding model. Tournament performance and full SDK timing require the parent comparison run.
