"""Official-engine regression for a nearly completed preflop blind call."""

import importlib.util
from pathlib import Path
import unittest

from poker_agent_protocol.game import Action
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommittedPreflopTests(unittest.TestCase):
    def state_after_raise(self, hero_stack):
        table = SitAndGo(rules=TableRules(players=2, variant="flop_auction_holdem"),
                         variant=load_variant("flop_auction_holdem"), table_id="regression", seed="0")
        table.stacks = [hero_stack, 4000 - hero_stack]
        state = table.decision()
        self.assertEqual(state.seat, 1)
        table.apply(Action.raise_to(100), answered=True)
        return table, table.decision()

    def test_one_chip_to_call_is_retained(self):
        table, state = self.state_after_raise(21)
        self.assertEqual(state.cards, ("Jd", "8c"))
        self.assertEqual(state.seats[0].stack, 1)
        old = load("icm_v2")
        new = load("icm_v3")
        self.assertEqual(old.Agent().act(state, b"")[0].kind, "fold")
        action = new.Agent().act(state, b"")[0]
        self.assertEqual(action.kind, "call")
        self.assertTrue(table.apply(action, answered=True))

    def test_normal_expensive_call_stays_selective(self):
        _, state = self.state_after_raise(2000)
        self.assertEqual(load("icm_v3").Agent().act(state, b"")[0].kind, "fold")


if __name__ == "__main__":
    unittest.main()
