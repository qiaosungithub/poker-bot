"""Replay observed cheap-fold diagnostics through the candidate policy."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

from poker_agent_protocol.game import BlindLevel, Event, GameState, LegalAction, Seat, TableConfig

ROOT = Path(__file__).resolve().parents[1]
CASES = [{'hand': 184,
  'cards': ['5s', 'Jc'],
  'pot': 15680,
  'stack': 320,
  'bet': 3520,
  'to_call': 8640,
  'players_remaining': 2,
  'big_blind': 1600,
  'legal_actions': [['fold', 0, 0], ['call', 0, 0]],
  'seats': [[0, 0, 0, 'busted', []],
            [1, 0, 0, 'busted', []],
            [2, 0, 0, 'busted', []],
            [3, 0, 12160, 'all_in', []],
            [4, 320, 3520, 'active', []],
            [5, 0, 0, 'busted', []],
            [6, 0, 0, 'busted', []],
            [7, 0, 0, 'busted', []]],
  'recent_public_events': [['blind', 3, 1600, []],
                           ['blind', 4, 800, []],
                           ['raise', 4, 3520, []],
                           ['raise', 3, 12160, []]]},
 {'hand': 209,
  'cards': ['Ac', '5h'],
  'pot': 11288,
  'stack': 4712,
  'bet': 5280,
  'to_call': 728,
  'players_remaining': 2,
  'big_blind': 2400,
  'legal_actions': [['fold', 0, 0], ['call', 0, 0]],
  'seats': [[0, 0, 0, 'busted', []],
            [1, 4712, 5280, 'active', []],
            [2, 0, 0, 'busted', []],
            [3, 0, 0, 'busted', []],
            [4, 0, 6008, 'all_in', []],
            [5, 0, 0, 'busted', []],
            [6, 0, 0, 'busted', []],
            [7, 0, 0, 'busted', []]],
  'recent_public_events': [['blind', 4, 2400, []],
                           ['blind', 1, 1200, []],
                           ['raise', 1, 5280, []],
                           ['raise', 4, 6008, []]]},
 {'hand': 149,
  'cards': ['8s', 'Kd'],
  'pot': 6381,
  'stack': 300,
  'bet': 1320,
  'to_call': 3441,
  'players_remaining': 3,
  'big_blind': 600,
  'legal_actions': [['fold', 0, 0], ['call', 0, 0]],
  'seats': [[0, 0, 0, 'busted', []],
            [1, 0, 0, 'busted', []],
            [2, 9319, 300, 'folded', []],
            [3, 0, 0, 'busted', []],
            [4, 0, 4761, 'all_in', []],
            [5, 0, 0, 'busted', []],
            [6, 0, 0, 'busted', []],
            [7, 300, 1320, 'active', []]],
  'recent_public_events': [['blind', 2, 300, []],
                           ['blind', 4, 600, []],
                           ['raise', 7, 1320, []],
                           ['fold', 2, 0, []],
                           ['raise', 4, 4761, []]]},
 {'hand': 121,
  'cards': ['Jc', '9c'],
  'pot': 2024,
  'stack': 2071,
  'bet': 880,
  'to_call': 64,
  'players_remaining': 4,
  'big_blind': 400,
  'legal_actions': [['fold', 0, 0], ['call', 0, 0]],
  'seats': [[0, 0, 0, 'busted', []],
            [1, 0, 0, 'busted', []],
            [2, 0, 0, 'busted', []],
            [3, 9446, 0, 'folded', []],
            [4, 2459, 200, 'folded', []],
            [5, 0, 944, 'all_in', []],
            [6, 2071, 880, 'active', []],
            [7, 0, 0, 'busted', []]],
  'recent_public_events': [['blind', 4, 200, []],
                           ['blind', 5, 400, []],
                           ['raise', 6, 880, []],
                           ['fold', 3, 0, []],
                           ['fold', 4, 0, []],
                           ['raise', 5, 944, []]]}]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore(data):
    hero = next(seat[0] for seat in data["seats"]
                if seat[1] == data["stack"] and seat[2] == data["bet"] and seat[3] == "active")
    blinds = BlindLevel(data["big_blind"] // 2, data["big_blind"])
    config = TableConfig("diagnostic", hero, 8, "boston", 2000, (blinds,), 20, 250,
                         (50, 30, 20), 1, 65536)
    events = tuple(Event(kind, seat, amount, tuple(cards))
                   for kind, seat, amount, cards in data["recent_public_events"])
    seats = tuple(Seat(seat, stack, bet, status, tuple(cards))
                  for seat, stack, bet, status, cards in data["seats"])
    return GameState(events, data["hand"], hero, tuple(data["cards"]), (), data["pot"],
                     data["to_call"], tuple(LegalAction(*action) for action in data["legal_actions"]),
                     events, seats, hero, 0, blinds, 1, data["players_remaining"], (), config)


class CheapRaiseContinuationTests(unittest.TestCase):
    def test_four_observed_folds_become_calls(self):
        old, new = load("icm_v2"), load("icm_price_v4")
        for data in CASES:
            with self.subTest(cards=data["cards"]):
                state = restore(data)
                self.assertEqual(old.Agent().act(state, b"")[0].kind, "fold")
                self.assertEqual(new.Agent().act(state, b"")[0].kind, "call")

    def test_own_raise_is_required(self):
        new = load("icm_price_v4")
        state = restore(CASES[1])
        state = replace(state, history=tuple(event for event in state.history
                                             if event.seat != state.seat or event.kind != "raise"))
        self.assertEqual(new.Agent().act(state, b"")[0].kind, "fold")

    def test_expensive_price_and_large_icm_risk_are_rejected(self):
        new = load("icm_price_v4")
        state = restore(CASES[1])
        opponents = [seat for seat in state.seats if seat.status == "all_in"]
        self.assertFalse(new.cheap_preflop_continue(state, opponents, 3000, 5000, 0.2))
        self.assertFalse(new.cheap_preflop_continue(state, opponents, 728, 11288, 0.30))


if __name__ == "__main__":
    unittest.main()
