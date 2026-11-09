import copy
import unittest
from poker_agent_sdk import Action
from poker_web.bay import BayGame
from poker_web.catalog import BAY


class Passive:
    def act(self, state, memory):
        if state.phase == "select":
            return Action("select", cards=(2, 0)), memory
        return (Action.check() if state.can("check") else Action.call()), memory


class BayTests(unittest.TestCase):
    def game(self):
        g = BayGame(dict(mode="bay_area", bots=["nc_codex_nc_v7_adapted"] * 3, name="Human"), seed="web-select-tests")
        g.agents = [None] + [Passive() for _ in range(3)]
        return g

    def advance(self, g):
        s = g.snapshot()
        if s["actor"] != 0:
            g.step()
        elif s["phase"] == "select":
            g.act(dict(kind="select", cards=[2, 0]))
        else:
            kind = next(a["kind"] for a in s["legal"] if a["kind"] in ("check", "call"))
            g.act(dict(kind=kind))

    def test_sealed_selection_and_hidden_cards(self):
        g = self.game()
        seen_selection = False
        seen_reveals = False
        for _ in range(200):
            s = g.snapshot()
            self.assertEqual(sum(p["stack"] for p in s["seats"]) + s["pot"], 8000)
            for seat in s["seats"][1:]:
                self.assertEqual(seat["cards"], seat["shown"])
            if s["phase"] == "select":
                seen_selection = True
                self.assertTrue(all(not p["shown"] for p in s["seats"]))
                self.assertFalse(any(e["cards"] for e in s["events"] if e["kind"] == "select"))
            if not s["done"] and any(p["shown"] for p in s["seats"]):
                seen_reveals = True
                self.assertTrue(all(len(p["shown"]) == 1 for p in s["seats"]))
                self.assertEqual(len(s["seats"][0]["cards"]), 2)
                self.assertEqual(len(s["discarded"]), 1)
            if s["done"]:
                break
            self.advance(g)
        self.assertTrue(s["done"])
        self.assertTrue(seen_selection and seen_reveals)
        self.assertAlmostEqual(sum(s["points"]), 100)
        self.assertEqual(g.snapshot(), s)
        g.next_hand()
        self.assertEqual(g.snapshot()["hand"], 2)

    def test_invalid_selection_and_all_in_selection(self):
        g = self.game()
        while not (g.snapshot()["actor"] == 0 and g.snapshot()["phase"] == "betting"):
            self.advance(g)
        g.act(dict(kind="raise", amount=next(a["max"] for a in g.snapshot()["legal"] if a["kind"] == "raise")))
        while not (g.snapshot()["actor"] == 0 and g.snapshot()["phase"] == "select"):
            self.advance(g)
        before = copy.deepcopy(g.snapshot())
        self.assertEqual(before["seats"][0]["stack"], 0)
        for cards in ([0, 0], [2, 3], [True, 1]):
            with self.assertRaises(ValueError):
                g.act(dict(kind="select", cards=cards))
            self.assertEqual(g.snapshot(), before)
        for _ in range(100):
            if g.snapshot()["done"]:
                break
            self.advance(g)
        self.assertTrue(g.snapshot()["done"])

    def test_every_selectable_policy_in_heads_up_play(self):
        for name, *_ in BAY:
            with self.subTest(policy=name):
                g = BayGame(dict(mode="bay_area", bots=[name], name="Human"), seed="web-policy-" + name)
                for _ in range(200):
                    s = g.snapshot()
                    if s["done"]:
                        break
                    self.advance(g)
                    self.assertFalse(g.warning, g.warning)
                self.assertTrue(g.snapshot()["done"])


if __name__ == "__main__":
    unittest.main()
