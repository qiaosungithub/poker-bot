# Reproducible defense-v6 assembly

This submission reproduces the behavior of the frozen `icm_defend_v6` candidate. Its auction phase is maintained in `modular_v4/auction_defend_v6.py`; each generated source header records the correct phase-file hash. The frozen benchmark candidate retains its original bytes and hash.

Build from the repository root.

```sh
.runtime/envs/boston-test/bin/python bots/dianne/tools/build_modular.py \
  --common bots/dianne/modular_v4/common_texture.py \
  --preflop bots/dianne/modular_v4/preflop.py \
  --auction bots/dianne/modular_v4/auction_defend_v6.py \
  --postflop bots/dianne/modular_v4/postflop.py \
  --dispatcher bots/dianne/modular_v4/dispatcher.py \
  --output bots/dianne/icm_defend_modular_v6/agent.py
.runtime/envs/boston-test/bin/python bots/dianne/tools/test_defend_modular_v6.py
.runtime/envs/boston-test/bin/poker-agent check bots/dianne/icm_defend_modular_v6/agent.py
```

The equivalence test verifies exact rebuild bytes, identical function and class ASTs, and matching actions and memory on real official-engine states from four complete eight-player tables. Its counts and hashes are in `equivalence.json`. The SDK checker accepted this generated submission.

Generated SHA256 is `c55dee481b8c57f9c479cbb3bbba9fa181a6ccd9946a60cb775892769bc2b810`.
