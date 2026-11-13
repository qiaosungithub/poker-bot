# Five actual decision stages: latest user steering

Supersedes the grouping terminology in THREE_STAGE_PLAN.md. Actual order is
preflop betting -> sealed flop auction -> flop betting -> turn betting -> river
betting -> automatic showdown. Turn/river each add a new public card. Each
betting street may call a bot several times because of raises; these are not
five single-action calls. Forced blinds, dealing and showdown have no strategy
callback. All-in can skip betting streets while auction remains required.

Independent policy exports: choose_preflop, choose_bid, choose_flop, choose_turn,
choose_river. Shared implementations/choose_postflop/choose_bet compatibility
wrappers are fine; actual board length selects the betting street. Shared
observer and own context update once per actual decision. Public prior actions
already exist in state.history. Own private five-element bid travels in memory;
opponent per-card bids never do. Preserve stage source versions and hashes.

Both separately designed and jointly coordinated five-stage policies are to
be compared. Same public/own information must be available to both. Joint ON
and coupling OFF are needed to separate coordination from extra features.

Auction spending information must be reinterpreted as turn/river arrive.
High flop spending can favor a draw rather than a made hand; a completed draw
and a missed river need different posterior treatment, with later betting as
additional evidence. Do not permanently add 'paid a lot = strong' per street,
and do not count own spending as new strength evidence or a reason to chase.

Current branch tasks: modules manager publishes backward-compatible contract
V3 and five-stage dispatch/context; betting inference manager refines draw
completion/miss logic; joint manager develops bounded coordination ON/OFF.
Auction-history v1 (1e737eb) is frozen and ready for controlled comparison.
Frozen modules-baseline-v1 (7847c69) retains exact validated hybrid decisions;
it is undergoing independent remote qualification and200-tableSDK test.

The 4096-table hybrid confirmation supports tournament auction + variant
preflop/flop/turn/river in its tested field. It is the initial baseline, not a
claim that each street is separately optimal. No upload/activation has occurred.
