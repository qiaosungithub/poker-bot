"""Rebuild defense-v6 and compare its decisions with the frozen candidate."""

import ast
from collections import Counter
import hashlib
import json

from build_modular import build
from test_texture_price_v5 import BASE, BayAreaEnv, load


def main():
    fragments = BASE / "modular_v4"
    assembled = build(fragments / "common_texture.py", fragments / "preflop.py",
                      fragments / "auction_defend_v6.py", fragments / "postflop.py",
                      fragments / "dispatcher.py")
    target = BASE / "icm_defend_modular_v6" / "agent.py"
    assert assembled == target.read_text()
    frozen_source = (BASE / "icm_defend_v6" / "agent.py").read_text()
    original_definitions = [ast.dump(node) for node in ast.parse(frozen_source).body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    generated_definitions = [ast.dump(node) for node in ast.parse(assembled).body
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    assert original_definitions == generated_definitions
    names = ("icm_v2", "icm_price_v4", "icm_spend_v3", "icm_texture_v4",
             "push_fold", "opponent_adaptive", "auction_shadow_price", "range_pressure")
    opponents = [load(name) for name in names]
    frozen, generated = load("icm_defend_v6"), load("icm_defend_modular_v6")
    counts = Counter()
    for number in range(4):
        env = BayAreaEnv(f"defend-v6-equivalence:{number}", variant="flop_auction_holdem")
        while env.state is not None:
            state, memory = env.state, env.memory[env.state.seat]
            phase = "auction" if state.phase == "auction" else "preflop" if not state.board else "postflop"
            assert frozen.act(state, memory) == generated.act(state, memory), (number, state.hand_number, phase)
            counts[phase] += 1
            _, reason = env.step(opponents[(state.seat + number) % 8].act(state, memory))
            assert reason is None, reason
    result = {"tables": 4, "states": sum(counts.values()), "phase_states": dict(counts),
              "mismatches": 0, "variant": "flop_auction_holdem",
              "definition_ast_identical": True, "builder_reproduces_bytes": True,
              "frozen_sha256": hashlib.sha256(frozen_source.encode()).hexdigest(),
              "generated_sha256": hashlib.sha256(assembled.encode()).hexdigest()}
    text = json.dumps(result, indent=2) + "\n"
    (BASE / "icm_defend_modular_v6" / "equivalence.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
