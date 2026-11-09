"""Constructed evaluator regressions for the texture-v3 candidate."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bay_area.environment import BayAreaEnv
from poker_agent_protocol.game import LegalAction, Seat

spec = importlib.util.spec_from_file_location(
    "texture_v3", ROOT / "bots/dianne/icm_texture_v3/agent.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class TextureRegression(unittest.TestCase):
    def test_board_quads_with_weak_kicker(self):
        quality, category = candidate.hand_quality(("4c", "5d"), ("2c", "2d", "2h", "2s", "3c"))
        self.assertEqual(category, 7)
        self.assertLess(quality, 0.10)

    def test_excluded_low_pair(self):
        quality, category = candidate.hand_quality(("2c", "3c"), ("2d", "3d", "Kh", "Ks", "Ac"))
        self.assertEqual(category, 2)
        self.assertAlmostEqual(quality, 0.68)

    def test_board_only_straight_draw(self):
        quality, _ = candidate.hand_quality(("2d", "3d"), ("Tc", "Jh", "Qs", "Kd"))
        self.assertAlmostEqual(quality, 0.22)

    def test_dead_gutshot_outs_reduce_draw_value(self):
        hole, board = ("As", "2d"), ("3c", "4s", "Kh")
        live, _ = candidate.hand_quality(hole, board)
        reduced, _ = candidate.hand_quality(hole, board, ("5c", "5h"))
        self.assertLess(reduced, live)

    def test_guaranteed_shared_nuts_call(self):
        env = BayAreaEnv("texture-v3-regression", players=2, hand_cap=20,
                         variant="flop_auction_holdem")
        state = replace(env.state, seat=0, cards=("4h", "5d"), pot=1020, to_call=1000,
                        seats=(Seat(0, 1000, 0, "active"), Seat(1, 0, 1000, "all_in")),
                        legal_actions=(LegalAction("call"), LegalAction("fold")),
                        history=(), events=(), phase="betting", variant={})
        boards = [("Tc", "Jc", "Qc", "Kc", "Ac"), ("Tc", "Jd", "Qh", "Ks", "Ac"),
                  ("2c", "2h", "2s", "2d", "Ac"), ("Ac", "Ah", "As", "Ad", "Kc")]
        for board in boards:
            with self.subTest(board=board):
                action, _ = candidate.Agent().act(replace(state, board=board), b"")
                self.assertEqual(action.kind, "call")


if __name__ == "__main__":
    unittest.main()
