"""Targeted checks for the shadow-price candidate's specific arithmetic regressions."""

import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from poker_agent_protocol.game import Action, BlindLevel
from poker_engine.holdem import HoldemHand

SOURCE = Path(__file__).resolve().parents[1] / "shadow_price_v2" / "agent.py"
SPEC = importlib.util.spec_from_file_location("shadow_price_v2_agent", SOURCE)
AGENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGENT)


class SelectedRegressionTests(unittest.TestCase):
    def test_official_engine_overshove_pot(self):
        hand = HoldemHand(stacks=(10000, 1000), blinds=BlindLevel(10, 20), rng=random.Random(10))
        self.assertEqual(hand.actor(), 1)
        hand.apply(Action.call())
        self.assertEqual(hand.actor(), 0)
        maximum = next(action.max_amount for action in hand.legal_actions() if action.kind == "raise")
        hand.apply(Action.raise_to(maximum))
        hero = hand.actor()
        view = hand.view(hero)
        state = SimpleNamespace(seat=hero, pot=view.pot,
                                seats=[SimpleNamespace(bet=bet) for bet in view.bets])
        cost = min(view.to_call, view.stacks[hero])
        pot = AGENT.contestable_pot(state, cost)
        self.assertEqual(pot, 1020)
        self.assertEqual(cost / (pot + cost), 0.49)

    def test_multiway_current_street_sidepots(self):
        state = SimpleNamespace(seat=0, pot=15000,
                                seats=[SimpleNamespace(bet=20), SimpleNamespace(bet=10000),
                                       SimpleNamespace(bet=4000), SimpleNamespace(bet=980)])
        # After calling 980, hero can contest 1,000 from each higher bettor.
        self.assertEqual(AGENT.contestable_pot(state, 980), 3000)

    def test_normal_call_keeps_pot(self):
        state = SimpleNamespace(seat=0, pot=150,
                                seats=[SimpleNamespace(bet=50), SimpleNamespace(bet=100)])
        self.assertEqual(AGENT.contestable_pot(state, 50), 150)

    def test_private_flush_rank_on_four_flush_board(self):
        board = ("Ah", "Kh", "8h", "4h", "9d")
        low, _ = AGENT.hand_quality(("2h", "3c"), board)
        nuts, _ = AGENT.hand_quality(("Qh", "3c"), board)
        self.assertLess(low, 0.50)
        self.assertGreater(nuts, 0.90)

    def test_fullhouse_pair_rank_on_board_trips(self):
        board = ("Kh", "Ks", "Kd", "Qc", "8s")
        low, _ = AGENT.hand_quality(("2c", "2d"), board)
        high, _ = AGENT.hand_quality(("Ac", "As"), board)
        self.assertLess(low, 0.60)
        self.assertGreater(high, 0.90)


if __name__ == "__main__":
    unittest.main()
