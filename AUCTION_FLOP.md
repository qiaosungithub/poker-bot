# Auction Flop strategies and fixed benchmark

The released Boston variant and event-time strategies are published in
[auction_flop/](auction_flop/README.md). Start there for source code, the
fixed opponent benchmark, reproduction commands, and measured results.

Benchmark **auction-flop-v1** freezes four selected opponents, their source,
the released SDK/rules, equal opponent exposure, 512 independent evaluation
seed blocks, and all eight seat rotations: **4,096 tables per candidate**.
New strategies are evaluated against that unchanged version.
