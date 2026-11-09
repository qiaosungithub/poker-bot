"""Human interface around the exact retained standard Hold'em hand adapter."""
import copy
import secrets

from heuristic_holdem import policies as P
from .catalog import MODES, labels


class ClassicGame:
    def __init__(self, config, seed=None):
        self.config = config
        self.names = labels(config)
        self.n = len(self.names)
        self.seed = seed if seed is not None else secrets.randbits(48)
        self.policies = [None] + [P.Policy(b, self.seed + i) for i, b in enumerate(config["bots"])]
        self.stacks = [10000] * self.n
        self.locked = {}
        self.completed = 0
        self.dealer = self.seed % self.n
        self.history = []
        self.recent = []
        self.done = True
        self.finished = False
        self.warning = ""
        self.next_hand()

    def next_hand(self):
        if not self.done or self.finished:
            raise ValueError("当前还不能开始下一手。")
        if self.completed:
            self.dealer = (self.dealer + 1) % self.n
        while not self.stacks[self.dealer]:
            self.dealer = (self.dealer + 1) % self.n
        self.alive = [s for s, x in enumerate(self.stacks) if x > 0]
        self.before = list(self.stacks)
        self.hand = P.engine.Hand([f"seat_{i}" for i in self.alive], [self.stacks[i] for i in self.alive],
                                  self.alive.index(self.dealer), self.seed * 1000003 + self.completed,
                                  f"practice_{self.completed}")
        self.events = [dict(kind=e["action"], seat=self.alive[e["seat"]], amount=e["amount"], cards=[]) for e in self.hand.log]
        self.shown = {}
        self.done = False
        self.warning = ""
        self.delta = [0] * self.n
        self._refresh()

    def _refresh(self):
        self.state = self.hand.state()
        if self.state is None:
            self._settle()

    def _settle(self):
        if self.done:
            return
        for s, chips in zip(self.alive, self.hand.final_stacks()):
            self.stacks[s] = chips
        self.delta = [a - b for a, b in zip(self.stacks, self.before)]
        folds = {e["seat"] for e in self.events if e["kind"] == "fold"}
        contenders = [s for s in self.alive if s not in folds]
        if len(contenders) > 1:
            self.shown = {s: self.hand.holes[self.alive.index(s)] for s in contenders}
        payouts = MODES["classic"]["payouts"][:self.n]
        dead = [s for s in self.alive if self.stacks[s] == 0]
        bottom = len(self.alive)
        for size in sorted({self.before[s] for s in dead}):
            tied = [s for s in dead if self.before[s] == size]
            prize = sum(payouts[bottom-len(tied):bottom]) / len(tied)
            self.locked.update({s: prize for s in tied})
            bottom -= len(tied)
        self.completed += 1
        self.done = True
        self.finished = sum(s > 0 for s in self.stacks) < 2 or self.completed >= 200
        self.recent.insert(0, {"hand": self.completed, "delta": self.delta[0], "board": list(self.hand.board)})
        self.recent = self.recent[:30]
        assert sum(self.stacks) == self.n * 10000

    def _legal(self):
        if self.state is None:
            return []
        k = self.hand.k
        result = []
        if k.can_fold():
            result.append(dict(kind="fold"))
        result.append(dict(kind="check" if self.state["can_check"] else "call", amount=min(self.state["amount_owed"], self.state["your_stack"])))
        if self.state["can_raise"]:
            result.append(dict(kind="raise", min=k.min_completion_betting_or_raising_to_amount,
                               max=k.max_completion_betting_or_raising_to_amount))
        return result

    def _apply(self, action):
        seat = self.alive[self.state["seat_to_act"]]
        street = self.state["street"]
        previous_board = list(self.hand.board)
        entry = self.hand.act(action)
        self.history.append(dict(entry, bot_id=f"seat_{seat}", hand_num=self.completed))
        self.events.append(dict(kind=entry["action"], seat=seat, amount=entry["amount"], cards=[], street=street))
        if previous_board != self.hand.board:
            self.events.append(dict(kind="board", seat=None, amount=0, cards=self.hand.board[len(previous_board):]))
        self._refresh()

    def act(self, data):
        if self.state is None or self.alive[self.state["seat_to_act"]] != 0:
            raise ValueError("现在不是你的回合。")
        action = data.get("kind")
        legal = {a["kind"]: a for a in self._legal()}
        if action not in legal:
            raise ValueError("这个动作当前不可用。")
        raw = {"action": action}
        if action == "raise":
            amount = data.get("amount")
            if type(amount) is not int or not legal[action]["min"] <= amount <= legal[action]["max"]:
                raise ValueError("下注金额必须在合法范围内，金额表示本轮累计下注。")
            raw["amount"] = amount
        self._apply(raw)

    def step(self):
        if self.state is None:
            raise ValueError("本手已经结束。")
        seat = self.alive[self.state["seat_to_act"]]
        if seat == 0:
            raise ValueError("请先选择你的动作。")
        p = self.policies[seat]
        state = copy.deepcopy(self.state)
        state["match_action_log"] = self.history[-200:]
        p.enter()
        try:
            action = p.act(state)
        except Exception:
            action = {"action": "check" if state["can_check"] else "fold"}
            self.warning = f"{self.names[seat]} 本次决策失败，已执行过牌或弃牌。"
        finally:
            p.leave()
        if p.exceptions:
            self.warning = f"{self.names[seat]} 本次使用了策略内部的异常回退。"
            p.exceptions.clear()
        self._apply(action)

    def snapshot(self):
        actor = None if self.state is None else self.alive[self.state["seat_to_act"]]
        folded = {e["seat"] for e in self.events if e["kind"] == "fold"}
        seats = []
        for seat in range(self.n):
            chips = self.stacks[seat]
            bet = 0
            if not self.done and seat in self.alive:
                ki = self.hand.inverse[self.alive.index(seat)]
                chips = self.hand.k.stacks[ki]
                bet = self.hand.k.bets[ki]
            status = "busted" if seat not in self.alive or (self.done and chips == 0) else "folded" if seat in folded else "all_in" if chips == 0 else "active"
            cards = list(self.hand.holes[self.alive.index(seat)]) if seat == 0 and seat in self.alive else list(self.shown.get(seat, []))
            seats.append(dict(seat=seat, name=self.names[seat], stack=chips, bet=bet, status=status,
                              cards=cards, shown=list(self.shown.get(seat, [])), card_count=2 if seat in self.alive else 0,
                              delta=self.delta[seat], human=seat == 0))
        points = None
        if self.done:
            points = [self.locked.get(s, 0) for s in range(self.n)]
            living = [s for s in range(self.n) if self.stacks[s] > 0]
            for s, point in zip(living, P.engine.icm([self.stacks[s] for s in living], MODES["classic"]["payouts"][:len(living)])):
                points[s] = point
        return dict(mode="classic", hand=self.completed if self.done else self.completed + 1, hand_cap=200,
                    phase="finished" if self.done else "betting", street="showdown" if self.done else self.state["street"],
                    board=list(self.hand.board), pot=0 if self.done else self.state["pot"],
                    actor=actor, seats=seats, button=self.dealer, blinds=[50, 100], blind_level=1,
                    legal=self._legal() if actor == 0 else [], events=self.events, done=self.done,
                    finished=self.finished, points=points, recent=self.recent, warning=self.warning, discarded=[])
