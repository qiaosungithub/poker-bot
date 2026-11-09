# Five-stage policy contract v3

Extends CONTRACT.md and CONTRACT_V2.md with explicit independent entry points:

| Decision stage | Native state | Module / function |
| --- | --- | --- |
| Preflop betting | phase=betting, board length0 | preflop_policy.choose_preflop |
| Flop auction | phase=auction | auction_policy.choose_bid |
| Flop betting | phase=betting, board length3 | flop_policy.choose_flop |
| Turn betting | phase=betting, board length4 | turn_policy.choose_turn |
| River betting | phase=betting, board length5 | river_policy.choose_river |

Every function receives `(state, features)`. Bid returns five integer amounts;
all betting stages return native Action. Existing choose_postflop and choose_bet
modules remain compatible through thin adapters. Policies may share code while
remaining separately replaceable. Forced blinds, dealing, and showdown do not
ask for decisions. An all-in player still participates in the auction but may
skip subsequent betting decisions.

The same shared observer runs once per request. Public earlier actions remain
in state.history/state.events; they are not copied into an unbounded second log.
Own private bid context and the bounded preflop trace follow CONTRACT_V2.md.
Additional Features attributes `own_flop_action`, `own_turn_action`, and
`own_river_action` hold the latest OwnDecision at those streets, or None. They
are reset each hand and reconciled against later public execution events.

Each betting street must reevaluate the auction evidence on its CURRENT board:
a large earlier auction payment is neither a permanent strength label nor proof
of a made hand. Newly dealt cards, dead candidates, betting actions and opponent
participation change its interpretation. Every stage has the preceding public
history and own submitted/reconciled private bid available for joint strategies.

The current implementation preserves the exact baseline decisions by adapting
frozen variant-v1 betting at all four betting streets and tournament-v1 at the
auction. The frozen two-module baseline remains at tag modules-baseline-v1.
