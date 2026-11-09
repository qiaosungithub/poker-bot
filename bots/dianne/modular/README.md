# Independently editable Boston phases

This package preserves the frozen `icm_v2` policy and its empty returned memory.
The implementation is split into shared helpers and three decision phases.

| File | Responsibility |
|---|---|
| `common.py` | Card quality, ICM, legal raise bounds and contestable-pot arithmetic |
| `preflop.py` | `preflop_decision(state, memory)` |
| `auction.py` | `auction_decision(state, memory)` |
| `postflop.py` | `postflop_decision(state, memory)` |
| `dispatcher.py` | Route the current state to one phase |
| `agent.py` | Generated standalone submission |

Each phase returns `(Action, bytes)`. Native Python imports can use
`bots.dianne.modular.dispatcher.Agent`. The generated submission contains the
same helpers and phases in a single file, with source hashes in its header.

From the repository root, rebuild after editing a phase.

```sh
.runtime/envs/boston-test/bin/python bots/dianne/tools/build_modular.py
.runtime/envs/boston-test/bin/poker-agent check bots/dianne/modular/agent.py
```

A comparison can substitute a compatible phase file and write to another bot
directory. Every replacement defines the same phase function signature.

```sh
.runtime/envs/boston-test/bin/python bots/dianne/tools/build_modular.py \
  --auction path/to/alternative_auction.py \
  --output bots/dianne/alternative_modular/agent.py
```

The builder also accepts `--preflop`, `--postflop`, `--common` and `--dispatcher`.
Package the generated `agent.py` when preparing a single-file submission.

`tools/test_modular.py` compares both native modules and the generated agent
against frozen ICM v2 on real Boston-engine states from eight varied opponents.
Its baseline equivalence result is recorded in `equivalence.json`. This verifies
behavioral preservation and reports phase coverage, without claiming strength
improvement from the refactor.
