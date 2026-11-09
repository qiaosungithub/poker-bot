# Texture plus price v5

Preflop uses `icm_price_v4`. Auction and postflop match `icm_texture_v4`,
including dead-card-aware auction evaluation and the original ICM spending
schedule. Phase sources live in `modular_v4/`.

Rebuild from the repository root.

```sh
.runtime/envs/boston-test/bin/python bots/dianne/tools/build_modular.py \
  --common bots/dianne/modular_v4/common_texture.py \
  --preflop bots/dianne/modular_v4/preflop.py \
  --auction bots/dianne/modular_v4/auction_texture.py \
  --postflop bots/dianne/modular_v4/postflop.py \
  --output bots/dianne/icm_texture_price_v5/agent.py
```

SDK static validation passed. `tools/test_texture_price_v5.py` verifies exact
phase equivalence against price v4 and texture v4 on official-engine states.
The result is recorded in `equivalence.json`. Holdout strength evaluation is pending.
