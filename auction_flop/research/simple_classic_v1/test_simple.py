"""Behavioral contracts and exact continuity of the reused postflop rules."""
import ast
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

from auction_flop.benchmark import run

BOT = run.ROOT / "candidates/simple_classic_v1"
sys.path.insert(0, str(BOT))
import simple_auction
import simple_preflop
from agent import Agent
from poker_agent_protocol.game import LegalAction, Seat


class SimpleTests(unittest.TestCase):
    def setUp(self):
        table = run.arena.SitAndGo(
            rules=run.arena.TableRules(variant=run.arena.VARIANT, players=8),
            variant=run.arena.load_variant(run.arena.VARIANT),
            table_id="simple-tests", seed="simple-tests")
        self.state = table.decision()

    def auction_state(self, players=2, balance=1000):
        seats = tuple(Seat(i, 2000, 0, "active" if i < players else "folded")
                      for i in range(8))
        return replace(self.state, seat=0, phase="auction", cards=("As", "Kh"),
                       seats=seats, board=(),
                       variant={"auction_cards": ("Ah", "Ac", "Kd", "2s", "7c")},
                       legal_actions=(LegalAction("bid", 0, balance),))

    def test_hand_value_requires_private_contribution(self):
        quality = simple_auction.flop_quality
        self.assertEqual(quality(("7s", "2h"), ("Ac", "Ad", "Ah")), 0)
        self.assertGreater(quality(("As", "2h"), ("Ac", "Ad", "7h")), 0)
        self.assertGreater(quality(("As", "Kh"), ("Ah", "Ac", "Kd")),
                           quality(("As", "Kh"), ("2s", "7c", "9d")))

    def test_zero_factor_and_no_preference_spend_nothing(self):
        with patch.object(simple_auction, "BASE_BID", 0):
            self.assertEqual(sum(simple_auction.auction(self.auction_state()).cards), 0)
        # Every candidate flop is dry air for 72o; no reason to influence it.
        state = replace(self.auction_state(), cards=("7s", "2h"),
                        variant={"auction_cards": ("Ac", "Kd", "9h", "4c", "Js")})
        self.assertEqual(sum(simple_auction.auction(state).cards), 0)

    def test_multiway_discount_counts_all_in_but_not_folds(self):
        two = self.auction_state(2)
        four = self.auction_state(4)
        self.assertEqual(sum(simple_auction.auction(two).cards),
                         2 * sum(simple_auction.auction(four).cards))
        seats = list(four.seats)
        seats[3] = replace(seats[3], status="all_in", stack=0)
        self.assertEqual(simple_auction.auction(four),
                         simple_auction.auction(replace(four, seats=tuple(seats))))

    def test_auction_ignores_other_information(self):
        state = self.auction_state()
        changed = replace(state, pot=99999, to_call=9999, history=(), events=(),
                          hand_number=200, button=6,
                          seats=tuple(replace(s, stack=11, shown_cards=("Qc", "Qh"))
                                      for s in state.seats),
                          variant={**state.variant, "flop_chips_by_seat": [3]*8})
        self.assertEqual(simple_auction.auction(state), simple_auction.auction(changed))

    def test_random_auction_bids_are_deterministic_and_within_balance(self):
        rng = random.Random(70931)
        deck = list(simple_auction.CARDS)
        for _ in range(1000):
            hole_and_public = rng.sample(deck, 7)
            balance = rng.choice((0, 1, 2, 3, 7, 1000))
            state = replace(self.auction_state(rng.randrange(2, 9), balance),
                            cards=tuple(hole_and_public[:2]),
                            variant={"auction_cards": tuple(hole_and_public[2:])})
            action, memory = Agent().act(state, b"ignored")
            self.assertEqual((action, memory), Agent().act(state, b""))
            self.assertEqual(len(action.cards), 5)
            self.assertTrue(all(type(v) is int and v >= 0 for v in action.cards))
            self.assertLessEqual(sum(action.cards), balance)
            self.assertEqual(action.amount, 0)

    def test_open_chart_has_a_real_fold_and_raise(self):
        state = replace(self.state, cards=("7s", "2h"))
        self.assertEqual(Agent().act(state, b"")[0].kind, "fold")
        state = replace(state, cards=("As", "Ah"))
        self.assertEqual(Agent().act(state, b"")[0].kind, "raise")
        self.assertEqual(simple_preflop.hand_key(("2h", "As")), "A2o")

    def test_postflop_functions_and_dependencies_unchanged(self):
        donor = run.ROOT / "candidates/classic_txnujk32_v1"
        def functions(path):
            source = path.read_text()
            return {n.name: ast.get_source_segment(source, n)
                    for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        old = functions(donor / "classic_policy.py")
        new = functions(BOT / "classic_postflop.py")
        self.assertIn("postflop", new)
        for name, source in new.items():
            self.assertEqual(source, old[name], name)
        for name in ("hand_contribution.py", "pot_accounting.py"):
            self.assertEqual((BOT/name).read_bytes(), (donor/name).read_bytes())


if __name__ == "__main__":
    unittest.main()
