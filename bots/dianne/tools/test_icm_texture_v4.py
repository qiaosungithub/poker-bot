"""Replay the observed TT turn/river overcalls without using hidden cards."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bay_area.environment import BayAreaEnv
from poker_agent_protocol.game import BlindLevel, Event, LegalAction, Seat

spec = importlib.util.spec_from_file_location(
    "texture_v4", ROOT / "bots/dianne/icm_texture_v4/agent.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class PocketPairRegression(unittest.TestCase):
    def state(self, river=False):
        env = BayAreaEnv("texture-v4-regression", players=8, variant="flop_auction_holdem")
        board = ("Qc", "Kd", "Js", "Qs")
        seats = [Seat(index, 0, 0, "busted") for index in range(8)]
        seats[1] = Seat(1, 1283, 0, "folded")
        seats[3] = Seat(3, 1200, 2990, "active")
        seats[4] = Seat(4, 5927, 0, "active")
        history = [Event("board", cards=board[:3]), Event("raise", 3, 1300),
                   Event("call", 4, 1300), Event("board", cards=("Qs",)),
                   Event("raise", 3, 2990)]
        pot, call = 7590, 2990
        if river:
            board += ("4c",)
            seats[3] = Seat(3, 0, 1200, "all_in")
            seats[4] = Seat(4, 2937, 0, "active")
            history += [Event("call", 4, 2990), Event("board", cards=("4c",)),
                        Event("raise", 3, 1200)]
            pot, call = 11780, 1200
        return replace(env.state, seat=4, hand_number=121, cards=("Td", "Tc"),
                       board=board, pot=pot, to_call=call, seats=tuple(seats),
                       players_remaining=3, blinds=BlindLevel(200, 400),
                       legal_actions=(LegalAction("fold"), LegalAction("call")),
                       history=tuple(history), events=(), phase="betting",
                       variant={"auction_cards": ["Ad", "Qc", "Kd", "Js", "5c"]})

    def test_observed_turn_call_is_fold(self):
        action, _ = candidate.Agent().act(self.state(), b"")
        self.assertEqual(action.kind, "fold")

    def test_observed_river_call_is_fold(self):
        action, _ = candidate.Agent().act(self.state(river=True), b"")
        self.assertEqual(action.kind, "fold")

    def test_overpair_keeps_value(self):
        quality, category = candidate.hand_quality(("Ac", "Ah"), ("7c", "7h", "2s", "Kd"))
        self.assertEqual(category, 2)
        self.assertGreaterEqual(quality, 0.75)


if __name__ == "__main__":
    unittest.main()
