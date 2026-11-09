# Auction shadow price

Original Boston implementation written during the November 9 build. The betting
and hand-texture helpers reuse our team's event-time `value_heuristic` candidate.

The auction evaluates all ten three-card flops, estimates three broad opponent
demand scenarios, and constructs the minimum targeted bid packet that selects
each desired flop in each scenario. Exact auction index tie ordering is included.
A packet can bid on only one card if the remaining desired cards already win.
Expected improvement is discounted by expected payment and a shadow price for
retaining chips for future useful auctions. Large pots and all-ins increase the
value of changing today's flop. Opponents' remaining public flop-chip balances
scale the demand prior. Individual hidden current bids are never assumed known.

The broad conceptual inspiration is using a cheap equity-related value proxy
for an auction when full subgame solving is too expensive. The implementation
uses a made-hand/texture heuristic and an explicit lifetime budget cost. The
source discussed a different auction with a third hole card and second-price
payment. No source code or bid multiplier was copied.
[GTO Wizard's MIT Auction Hold'em methodology](https://github.com/gtowizard-ai/mitpoker-2024#the-auction)

Validation used the installed official SDK and Python 3.12. The official
`poker-agent check` passed. In 2,000 randomized two-to-ten-player auction states,
every bid was legal. Direct auction-call median was 0.1774 ms, P99 0.2015 ms,
maximum 0.6470 ms, with zero calls over 1 ms. These measurements exclude SDK
transport. Match strength remains for central benchmarking.

The opponent bid scenarios and future-auction horizon are heuristic. They have
no calibrated posterior and do not infer opponent private cards. This candidate
keeps no per-table memory.
