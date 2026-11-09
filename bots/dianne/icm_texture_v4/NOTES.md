# Paired-board underpair correction

Extends texture v3. The v3 source and selected agent remain preserved.

An observed diagnostic hand held TdTc on QcKdJsQs, called 2,990 on the turn, then called 1,200 on a 4c river. V3 still assigned 0.704 on the turn because it classified the shared queens and pocket tens as a strong two-pair hand.

V4 discounts a pocket pair according to the number of higher board ranks. It also applies a narrow range adjustment when opponents have bet on at least two distinct postflop streets. This targets the pocket-underpair bluff-catcher, while keeping overpairs valuable. The turn quality becomes 0.3525 before the betting-range adjustment and the river quality becomes 0.30.

Replaying both recorded public decision states produces folds. AA on 772K retains at least 0.75 quality. Three regression tests passed in `bots/dianne/tools/test_icm_texture_v4.py`, and official static submission checking passed.

The guaranteed shared-nuts check/call rule is retained. With a guaranteed shared best hand, every remaining player ties, so showdown cannot eliminate another player by beating their hand. Calling returns the hero's eligible contribution and share of dead money. The policy avoids raising in these cases.

This is a targeted exploitative range heuristic. The numerical discounts require parent-run comparison against independent opponents and seeds. The diagnostic uses only the public state at decision time.

Frozen agent SHA256 is `b42fa580803d7e9c3ce4a829948ddb85b8c49f79f2dcdeaa4234bb5353d77f91`.
