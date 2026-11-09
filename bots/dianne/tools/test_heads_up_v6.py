"""Official-state regression and non-heads-up equivalence for BB defense."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bay_area.environment import BayAreaEnv
from poker_agent_protocol.game import Action
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

BASE = ROOT / "bots" / "dianne"


def load(name):
    spec = importlib.util.spec_from_file_location("test_" + name, BASE / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent()


def main():
    old, new = load("icm_texture_price_v5"), load("icm_heads_up_v6")
    for amount, expected in ((44, "call"), (1000, "fold"), (2000, "fold")):
        table = SitAndGo(rules=TableRules(players=2, variant="flop_auction_holdem"),
                         variant=load_variant("flop_auction_holdem"), table_id="bb", seed="0")
        state = table.decision()
        assert state.seat == 1
        table.apply(Action.raise_to(amount), answered=True)
        state = table.decision()
        assert state.cards == ("Jd", "8c")
        assert old.act(state, b"")[0].kind == "fold"
        assert new.act(state, b"")[0].kind == expected
    opponents = [load(name) for name in ("icm_v2", "push_fold", "value_heuristic", "opponent_adaptive",
                                        "range_pressure", "auction_optimizer", "equity_sampling", "auction_shadow_price")]
    counts = Counter()
    for number in range(8):
        env = BayAreaEnv(f"hu-defense-equivalence:{number}", variant="flop_auction_holdem")
        while env.state is not None:
            state, memory = env.state, env.memory[env.state.seat]
            if state.players_remaining != 2 or state.phase != "betting" or state.board:
                assert new.act(state, memory) == old.act(state, memory)
                counts["unchanged_states"] += 1
            _, reason = env.step(opponents[(number + state.seat) % 8].act(state, memory))
            assert reason is None
    result = {"tables": 8, "regressions": {"J8_vs_44": "call", "J8_vs_1000": "fold", "J8_vs_2000_allin": "fold"},
              **counts, "mismatches": 0}
    output = json.dumps(result, indent=2) + "\n"
    (BASE / "icm_heads_up_v6" / "equivalence.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
