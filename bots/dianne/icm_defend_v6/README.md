# Default flop defense

This isolated auction candidate starts from `icm_texture_price_v5`, SHA256
`abc0595874221a0fa95f466fea82370e305765a4762f6843a1896ffec78c01b9`.
Only `auction_decision` changes, verified by AST comparison.

The free default flop may already be desirable, but opponents can change it.
When the baseline gain is below 0.06 and an active opponent retains auction
currency, the gain becomes the larger of its original value and 0.6 times the
difference between the best of ten flops and their mean. The 0.06 decision
threshold and baseline spending schedule remain. Already-positive baseline
bids are unchanged, and exhausted opponents trigger no new defense spending.

The SDK checker passed. Four hundred randomized auction cases verified legal
budgets and unchanged positive baseline bids and exhausted-opponent decisions.
Independent evaluation determines whether defensive bids improve scores.

Frozen SHA256:
`cd06e1dc6fb4d5b7c17b83758051a0c167eefe216cecee7a53ff672f36b5479f`.
