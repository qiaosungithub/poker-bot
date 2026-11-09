"""Compare original, native modules and generated bot on official Boston states."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bay_area.environment import BayAreaEnv
from bots.dianne.modular.dispatcher import Agent as NativeAgent

FOLDERS = ROOT / "bots" / "dianne"
NAMES = ("icm_v2", "value_heuristic", "push_fold", "opponent_adaptive", "range_pressure",
         "auction_optimizer", "auction_shadow_price", "equity_sampling")


def load(name):
    spec = importlib.util.spec_from_file_location("test_" + name, FOLDERS / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent()


def one_table(number):
    reference, generated, native = load("icm_v2"), load("modular"), NativeAgent()
    agents = [load(name) for name in NAMES]
    agents = agents[number % 8:] + agents[:number % 8]
    env = BayAreaEnv(f"boston-modular-equivalence:{number}", variant="flop_auction_holdem")
    counts = Counter()
    while env.state is not None:
        state = env.state
        memory = env.memory[state.seat]
        expected = reference.act(state, memory)
        assert generated.act(state, memory) == expected, (number, state.hand_number, state.phase)
        assert native.act(state, memory) == expected, ("native", number, state.hand_number, state.phase)
        phase = "auction" if state.phase == "auction" else "preflop" if not state.board else "postflop"
        counts[phase] += 1
        _, reason = env.step(agents[state.seat].act(state, memory))
        assert reason is None, (number, reason)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one_table, range(args.tables)):
            counts.update(result)
    report = {"tables": args.tables, "states": sum(counts.values()), "phase_states": dict(counts),
              "mismatches": 0, "checked": ["generated_single_file", "native_phase_modules"],
              "variant": "flop_auction_holdem", "seed_prefix": "boston-modular-equivalence:",
              "opponents": NAMES, "seconds": time.perf_counter() - started,
              "sources": {name: hashlib.sha256((FOLDERS / name / "agent.py").read_bytes()).hexdigest()
                          for name in ("icm_v2", "modular")}}
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
