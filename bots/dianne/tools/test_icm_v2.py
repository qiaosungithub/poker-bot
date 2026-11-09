"""Regressions for ICM call outcome conservation and contested chips."""

import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from poker_agent_protocol.game import Action, BlindLevel
from poker_engine.holdem import HoldemHand

SOURCE = Path(__file__).resolve().parents[1] / "icm_v2" / "agent.py"
SPEC = importlib.util.spec_from_file_location("icm_v2_agent", SOURCE)
AGENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGENT)


class ICMRegressionTests(unittest.TestCase):
    def test_overshove_outcomes_conserve_chips(self):
        hand = HoldemHand(stacks=(10000, 1000), blinds=BlindLevel(10, 20), rng=random.Random(10))
        hand.actor()
        hand.apply(Action.call())
        hand.actor()
        maximum = next(action.max_amount for action in hand.legal_actions() if action.kind == "raise")
        hand.apply(Action.raise_to(maximum))
        hero = hand.actor()
        view = hand.view(hero)
        seats = [SimpleNamespace(seat=index, stack=stack, bet=bet)
                 for index, (stack, bet) in enumerate(zip(view.stacks, view.bets))]
        state = SimpleNamespace(seat=hero, seats=seats, pot=view.pot, to_call=view.to_call,
                                config=SimpleNamespace(payouts=(50, 30, 20)))
        cost = min(state.to_call, seats[hero].stack)
        self.assertEqual(AGENT.contestable_pot(state, cost), 1020)
        snapshots = []
        original = AGENT.own_icm
        def capture(stacks, seat, payouts):
            snapshots.append(tuple(stacks))
            return original(stacks, seat, payouts)
        AGENT.own_icm = capture
        try:
            threshold = AGENT.call_threshold(state, [seats[0]], cost, 0)
        finally:
            AGENT.own_icm = original
        self.assertEqual(snapshots, [(10020, 980), (9000, 2000), (11000, 0)])
        self.assertTrue(all(sum(stacks) == 11000 for stacks in snapshots))
        self.assertGreaterEqual(threshold, 0.55 - 1e-12)

    def test_multiway_sidepot_and_regular_pot(self):
        state = SimpleNamespace(seat=0, pot=15000,
                                seats=[SimpleNamespace(bet=value) for value in (20, 10000, 4000, 980)])
        self.assertEqual(AGENT.contestable_pot(state, 980), 3000)
        state = SimpleNamespace(seat=0, pot=150,
                                seats=[SimpleNamespace(bet=50), SimpleNamespace(bet=100)])
        self.assertEqual(AGENT.contestable_pot(state, 50), 150)

    def test_hand_rank_corrections(self):
        board = ("Ah", "Kh", "8h", "4h", "9d")
        self.assertLess(AGENT.hand_quality(("2h", "3c"), board)[0], 0.50)
        self.assertGreater(AGENT.hand_quality(("Qh", "3c"), board)[0], 0.90)
        board = ("Kh", "Ks", "Kd", "Qc", "8s")
        self.assertLess(AGENT.hand_quality(("2c", "2d"), board)[0], 0.60)
        self.assertGreater(AGENT.hand_quality(("Ac", "As"), board)[0], 0.90)


if __name__ == "__main__":
    unittest.main()
