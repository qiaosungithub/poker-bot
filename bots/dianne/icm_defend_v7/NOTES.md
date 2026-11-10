# Auction defense with board-quads correction

Assembled using `tools/build_modular.py` with the following sources.

- Common helpers from `modular_v4/common_texture_v7.py`.
- Price-aware preflop from `modular_v4/preflop.py`.
- Auction defense from `modular_v4/auction_defend_v6.py`.
- Texture-aware postflop from `modular_v4/postflop.py`.
- Dispatcher from `modular_v4/dispatcher.py`.

The only behavioral change from defense v6 is the narrow shared-board-quads kicker equity fix described in `../icm_texture_price_v7/NOTES.md`. Source hashes in the generated file identify the actual component files.

Official static checking passed. Four eight-seat official-engine tables produced 1,376 decisions with no action failures and identical decisions to defense v6 on every encountered state unaffected by the quads correction.

Frozen agent SHA256 is `30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`.
