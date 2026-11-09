# Independent phase alternatives

The original `modular/` sources remain the ICM v2 reference.

- `preflop.py` uses the cheap-continuation correction from `icm_price_v4`.
- `auction.py` uses spending urgency from `icm_spend_v3`.
- `postflop.py` uses the texture and dead-card corrections from `icm_texture_v4`.
- `common.py` retains the original `hand_quality` for bidding and adds a separate
  `postflop_hand_quality`. Postflop texture changes therefore affect only postflop.

Generated standalone candidates use these combinations.

| Candidate | Preflop | Auction | Postflop |
|---|---|---|---|
| `icm_postonly_v4` | ICM v2 | ICM v2 | Texture v4 |
| `icm_price_spend_v4` | Price v4 | Spend v3 | ICM v2 |
| `icm_combined_v4` | Price v4 | Spend v3 | Texture v4 |

Use `tools/build_modular.py` with `--common`, `--preflop`, `--auction`,
`--postflop` and `--output` to reproduce a combination. Defaults select the
original `modular/` phases. Every combination here uses `modular_v4/common.py`.

`tools/test_modular_v4.py` compares isolated price and spend assemblies with
their original candidates, and checks all three combinations phase by phase
on real official-engine states. Its report is `equivalence.json`.
