import copy
import unittest

from poker_web.classic import ClassicGame
from poker_web.catalog import CLASSIC


class ClassicTests(unittest.TestCase):
    def game(self, bots=None, seed=771):
        return ClassicGame(dict(mode="classic", bots=bots or ["station"] * 5, name="Human"), seed=seed)

    def check_snapshot(self, game):
        s = game.snapshot()
        self.assertEqual(sum(p["stack"] for p in s["seats"]) + s["pot"], game.n * 10000)
        for seat in s["seats"][1:]:
            self.assertEqual(seat["cards"], seat["shown"])
            if not s["done"]:
                self.assertEqual(seat["cards"], [])
        return s

    def test_live_privacy_settlement_and_next_hand(self):
        g = self.game()
        for _ in range(2):
            for step in range(300):
                s = self.check_snapshot(g)
                if s["done"]:
                    break
                if s["actor"] == 0:
                    kind = next(a["kind"] for a in s["legal"] if a["kind"] in ("check", "call"))
                    g.act(dict(kind=kind))
                else:
                    g.step()
            else:
                self.fail("Hand failed to settle")
            self.assertAlmostEqual(sum(s["points"]), 235)
            self.assertEqual(g.snapshot(), s)  # Reading does not deal another hand.
            before = s["hand"]
            g.next_hand()
            self.assertEqual(g.snapshot()["hand"], before + 1)

    def test_invalid_human_action_does_not_change_game(self):
        g = self.game(["station"])
        while g.snapshot()["actor"] != 0:
            g.step()
        before = copy.deepcopy(g.snapshot())
        for action in [dict(kind="select", cards=[0, 1]), dict(kind="raise", amount=-1), dict(kind="raise", amount=True)]:
            with self.assertRaises(ValueError):
                g.act(action)
            self.assertEqual(g.snapshot(), before)
        with self.assertRaises(ValueError):
            g.next_hand()

    def test_duplicate_policies_have_private_state(self):
        g = self.game(["txnujk_32", "txnujk_32"])
        self.assertIsNot(g.policies[1].namespace, g.policies[2].namespace)
        self.assertIsNot(g.policies[1].namespace["_RNG"], g.policies[2].namespace["_RNG"])

    def test_every_selectable_policy_in_heads_up_play(self):
        for index, (name, *_) in enumerate(CLASSIC):
            with self.subTest(policy=name):
                g = self.game([name], seed=9200 + index)
                for _ in range(200):
                    s = self.check_snapshot(g)
                    if s["done"]:
                        break
                    if s["actor"] == 0:
                        a = next(a for a in s["legal"] if a["kind"] in ("check", "call"))
                        g.act(dict(kind=a["kind"]))
                    else:
                        g.step()
                    self.assertFalse(g.warning, g.warning)
                self.assertTrue(g.snapshot()["done"])


if __name__ == "__main__":
    unittest.main()
