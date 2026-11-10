# Board-quads kicker correction

Built from texture-price v5 phase modules, with `modular_v4/common_texture_v7.py` replacing the common module. The existing source files are preserved.

Both `hand_quality` and `postflop_hand_quality` now distinguish a private kicker from the shared board kicker. When the best kicker is already on the board, the hero can only tie opponents whose best kicker is no higher. Board-only quads also retain this calculated equity through the generic board-only-hand check.

Exact enumeration over all legal opposing two-card holdings confirms the corrected values for three cases.

- 2d2h on 3c3d3h3s4c has about 0.00505 equity.
- 2d2h on 3c3d3h3sAc has exactly 0.5 equity.
- Ad2h on 3c3d3h3s4c has about 0.93485 equity, with remaining aces causing ties.

The dedicated regression script is `bots/dianne/tools/test_icm_texture_price_v7.py`. Both helper functions also matched v5 exactly on 1,000 random unaffected states. Official static submission checking passed before the source was rebuilt with updated component provenance comments.
