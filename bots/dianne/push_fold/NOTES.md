# Boston push/fold pressure candidate

Implemented during the Boston build on November 9, 2026.

The candidate uses a stack-depth and position dependent shove range below 14 big blinds. It opens smaller pots with deeper stacks, tightens against raises, and applies a simple bubble premium when another live stack is much shorter. Calls for a large fraction of the stack require a stronger hand than opening shoves.

Postflop decisions use fast made-hand and draw estimates, a multiway discount, pot odds and stack-to-pot ratio. Strong hands commit at low SPR. Modest made hands can make small heads-up bets when checking is free.

Auction bidding scores all ten possible flops. Spending grows with marginal improvement, pot leverage and short-stack urgency. A preflop all-in spends more of the remaining auction balance because the current hand can decide survival. Bids concentrate on cards whose omission would most weaken the preferred flop.

The standalone file uses only the standard library and official SDK. Its made-hand scoring component was reused from our team's `auction_optimizer` implementation written during this event. It keeps no mutable per-table state and returns empty memory.

## Validation

The official `check_submission` accepted the file. A deterministic randomized test exercised 2,000 auctions and 4,000 betting states, including zero auction balances, all-in auction states and arbitrary incoming memory. Every action passed protocol legality checks. All bid vectors contained five nonnegative integers with total at most the available balance.

On this laptop, across these 6,000 direct calls, median latency was 0.010916 ms, P99 was 0.065292 ms, maximum was 0.22525 ms, and no call exceeded 1 ms. These timings exclude SDK serialization and transport. Official table simulation and competitive evaluation are handled by the shared benchmark.

The ranges and hand-strength scores are heuristics. In particular, no equilibrium solution or empirical tournament strength is claimed from the legality test.
