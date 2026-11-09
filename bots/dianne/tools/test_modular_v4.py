"""Small official-engine check of independently assembled phase alternatives."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bay_area.environment import BayAreaEnv
from bots.dianne.modular_v4.dispatcher import Agent as NativeAgent
from build_modular import build

BASE = ROOT / "bots" / "dianne"
OLD = BASE / "modular"
NEW = BASE / "modular_v4"


def load(name):
    spec = importlib.util.spec_from_file_location("check_" + name, BASE / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent()


def isolated(preflop, auction):
    source = build(NEW / "common.py", preflop, auction, OLD / "postflop.py", OLD / "dispatcher.py")
    module = ModuleType("isolated_phase")
    exec(compile(source, "isolated_agent.py", "exec"), module.__dict__)
    return module.Agent()


def main():
    baseline, price, spend, texture = (load(name) for name in
                                     ("icm_v2", "icm_price_v4", "icm_spend_v3", "icm_texture_v4"))
    isolated_price = isolated(NEW / "preflop.py", OLD / "auction.py")
    isolated_spend = isolated(OLD / "preflop.py", NEW / "auction.py")
    postonly, price_spend, combined = (load(name) for name in
                                     ("icm_postonly_v4", "icm_price_spend_v4", "icm_combined_v4"))
    native = NativeAgent()
    opponents = [baseline, price, spend, texture, load("push_fold"), load("opponent_adaptive"),
                 load("auction_shadow_price"), load("range_pressure")]
    counts = Counter()
    for number in range(16):
        env = BayAreaEnv(f"modular-v4-phase-equivalence:{number}", variant="flop_auction_holdem")
        while env.state is not None:
            state, memory = env.state, env.memory[env.state.seat]
            original = baseline.act(state, memory)
            price_answer, spend_answer = price.act(state, memory), spend.act(state, memory)
            assert isolated_price.act(state, memory) == price_answer
            assert isolated_spend.act(state, memory) == spend_answer
            phase = "auction" if state.phase == "auction" else "preflop" if not state.board else "postflop"
            expected = {"auction": spend_answer, "preflop": price_answer,
                        "postflop": texture.act(state, memory)}[phase]
            assert combined.act(state, memory) == expected
            assert native.act(state, memory) == expected
            assert postonly.act(state, memory) == (expected if phase == "postflop" else original)
            assert price_spend.act(state, memory) == (original if phase == "postflop" else expected)
            counts[phase] += 1
            _, reason = env.step(opponents[(state.seat + number) % 8].act(state, memory))
            assert reason is None, reason
    names = ("icm_postonly_v4", "icm_price_spend_v4", "icm_combined_v4")
    result = {"tables": 16, "states": sum(counts.values()), "phase_states": dict(counts),
              "mismatches": 0, "variant": "flop_auction_holdem",
              "checks": ["isolated_price_equals_price_v4", "isolated_spend_equals_spend_v3",
                         "three_generated_combinations_match_intended_phase_sources",
                         "native_combination_equals_generated"],
              "sources": {name: hashlib.sha256((BASE / name / "agent.py").read_bytes()).hexdigest()
                          for name in names}}
    output = json.dumps(result, indent=2) + "\n"
    (NEW / "equivalence.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
