"""Exact board-quads kicker regressions and unaffected-state equivalence."""

import importlib.util
from itertools import combinations
from pathlib import Path
import random
import unittest

import eval7

ROOT = Path(__file__).resolve().parents[1]


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, ROOT / source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load("base_v5", "icm_texture_price_v5/agent.py")
CANDIDATE = load("candidate_v7", "icm_texture_price_v7/agent.py")
MODULAR = load("modular_v7", "modular_v4/common_texture_v7.py")


def exact_equity(hole, board):
    unavailable = set(hole + board)
    deck = [card for card in CANDIDATE.CARDS if card not in unavailable]
    own_rank = eval7.evaluate([eval7.Card(card) for card in hole + board])
    wins, total = 0.0, 0
    for other in combinations(deck, 2):
        other_rank = eval7.evaluate([eval7.Card(card) for card in other + board])
        wins += (own_rank > other_rank) + 0.5 * (own_rank == other_rank)
        total += 1
    return wins / total


class BoardQuadsRegression(unittest.TestCase):
    def test_exact_kicker_equities(self):
        cases = [(("2d", "2h"), ("3c", "3d", "3h", "3s", "4c")),
                 (("2d", "2h"), ("3c", "3d", "3h", "3s", "Ac")),
                 (("Ad", "2h"), ("3c", "3d", "3h", "3s", "4c"))]
        for hole, board in cases:
            expected = exact_equity(hole, board)
            for module in (CANDIDATE, MODULAR):
                for function in (module.hand_quality, module.postflop_hand_quality):
                    with self.subTest(hole=hole, board=board, function=function.__name__):
                        quality, category = function(hole, board)
                        self.assertEqual(category, 7)
                        self.assertAlmostEqual(quality, expected, places=12)
        self.assertLess(CANDIDATE.hand_quality(*cases[0])[0], 0.01)
        self.assertEqual(CANDIDATE.hand_quality(*cases[1])[0], 0.5)
        self.assertGreater(CANDIDATE.hand_quality(*cases[2])[0], 0.93)

    def test_unaffected_states_identical(self):
        rng = random.Random(247)
        deck = list(CANDIDATE.CARDS)
        for index in range(1000):
            cards = rng.sample(deck, 7)
            hole, board = tuple(cards[:2]), tuple(cards[2:5 + index % 3])
            if any(sum(card[0] == rank for card in board) == 4 for rank in "23456789TJQKA"):
                continue
            for name in ("hand_quality", "postflop_hand_quality"):
                expected = getattr(BASE, name)(hole, board)
                self.assertEqual(getattr(CANDIDATE, name)(hole, board), expected)
                self.assertEqual(getattr(MODULAR, name)(hole, board), expected)


if __name__ == "__main__":
    unittest.main()
