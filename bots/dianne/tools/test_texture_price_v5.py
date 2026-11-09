"""Verify the combined agent against the intended original phase policies."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bay_area.environment import BayAreaEnv

BASE = ROOT / "bots" / "dianne"


def load(name):
    spec = importlib.util.spec_from_file_location("verify_" + name, BASE / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent()


def main():
    names = ("icm_v2", "icm_price_v4", "icm_spend_v3", "icm_texture_v4", "push_fold",
             "opponent_adaptive", "auction_shadow_price", "range_pressure")
    opponents = [load(name) for name in names]
    price, texture, combined = load("icm_price_v4"), load("icm_texture_v4"), load("icm_texture_price_v5")
    counts = Counter()
    for number in range(16):
        env = BayAreaEnv(f"texture-price-v5-equivalence:{number}", variant="flop_auction_holdem")
        while env.state is not None:
            state, memory = env.state, env.memory[env.state.seat]
            phase = "auction" if state.phase == "auction" else "preflop" if not state.board else "postflop"
            expected = (price if phase == "preflop" else texture).act(state, memory)
            assert combined.act(state, memory) == expected, (number, state.hand_number, phase)
            counts[phase] += 1
            _, reason = env.step(opponents[(state.seat + number) % 8].act(state, memory))
            assert reason is None, reason
    result = {"tables": 16, "states": sum(counts.values()), "phase_states": dict(counts),
              "mismatches": 0, "variant": "flop_auction_holdem",
              "expected_preflop": "icm_price_v4", "expected_auction_postflop": "icm_texture_v4",
              "sources": {name: hashlib.sha256((BASE / name / "agent.py").read_bytes()).hexdigest()
                          for name in ("icm_price_v4", "icm_texture_v4", "icm_texture_price_v5")}}
    text = json.dumps(result, indent=2) + "\n"
    (BASE / "icm_texture_price_v5" / "equivalence.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
