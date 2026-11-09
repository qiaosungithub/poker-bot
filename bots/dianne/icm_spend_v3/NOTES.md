# Auction spending candidate

This candidate retains the frozen selected strategy's betting behavior. An AST comparison verified that only `auction_bid` changed. The baseline source hash is `73ed48b9128862704c8123b0e9c582ab366c4fa3b84206998874154fc4b17f56`.

The baseline diagnostic left more than 500 flop-chips in 55 of 105 eliminations. This candidate increases ordinary auction spending by at most 90 percent when stack depth and remaining hands imply fewer future opportunities. Large board improvements receive more of that increase. Public opponent balances modulate the increase. When every opponent still in the hand has exhausted their auction currency, at most three chips secure the preferred three cards. Existing all-in spending floors remain.

The SDK checker accepted the candidate, and 800 randomized auctions passed legality checks, including depleted opposing budgets and all-in hands.

A 128-table, 16-block development screen placed this candidate and the frozen selected policy in the same eight-player field. The candidate scored 16.40625 points/table against selected's 13.671875. Both had zero failures. The small sample requires independent confirmation before selection. Candidate auction P99 was 0.194333 ms and maximum was 0.665334 ms for direct function calls, excluding SDK transport.

The raw development run is stored locally under `.runtime/boston/diagnostics/spend-v3-screen128/`.
