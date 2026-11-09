"""Interactive Bay Area tables, with the pinned engine owning all poker rules."""
import importlib.util
import json
from pathlib import Path
import secrets
import sys

from poker_agent_protocol.game import Action, action_from_dict, is_legal
from poker_agent_sdk.guest import reply
from poker_engine.scoring import table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

from bay_area.seeded_guest import SeededAgent
from .catalog import ROOT, labels


class BayGame:
    def __init__(self, config, seed=None):
        self.config = config
        self.names = labels(config)
        self.n = len(self.names)
        self.seed = str(seed if seed is not None else secrets.token_hex(16))
        self.table = SitAndGo(rules=TableRules(variant="flop_discard_reveal", players=self.n),
                             variant=load_variant("flop_discard_reveal"), table_id="practice", seed=self.seed)
        registry = json.loads((ROOT / "bay_area/candidates.json").read_text())["candidates"]
        self.agents = [None]
        for i, name in enumerate(config["bots"]):
            folder = ROOT / registry[name]["path"]
            spec = importlib.util.spec_from_file_location(f"practice_bot_{i}_{name}", folder / "agent.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            sys.path.insert(0, str(folder))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            self.agents.append(SeededAgent(module, folder))
        self.memory = [b""] * self.n
        self.failures = [0] * self.n
        self.decisions = [0] * self.n
        self.done = True
        self.finished = False
        self.recent = []
        self.next_hand()

    def next_hand(self):
        if not self.done or self.finished:
            raise ValueError("当前还不能开始下一手。")
        self.before = list(self.table.stacks)
        self.state = self.table.decision()
        self.hand = self.table._hand
        self.positions = list(self.table._positions)
        self.hero_cards = []
        self.delta = [0] * self.n
        self.done = False
        self.warning = ""
        self._view()

    def _view(self):
        hero_position = self.positions.index(0) if 0 in self.positions else None
        view = self.hand.view(hero_position if hero_position is not None else 0)
        if hero_position is not None:
            # Mucked cards can be cleared by PokerKit. Preserve only the human's own cards.
            if view.cards:
                self.hero_cards = list(view.cards)
            elif not self.hero_cards:
                self.hero_cards = list(self.hand.dealt()[hero_position])
        return view

    def _refresh(self):
        actor = self.hand.actor()
        self._view()
        if actor is not None:
            self.state = self.table.decision()
            return
        self.table._end_hand()
        self.state = None
        self.done = True
        self.delta = [a - b for a, b in zip(self.table.stacks, self.before)]
        self.finished = sum(x > 0 for x in self.table.stacks) < 2 or self.table.hands_played >= self.table.rules.hand_cap
        self.recent.insert(0, dict(hand=self.table.hands_played, delta=self.delta[0], board=list(self._view().board)))
        self.recent = self.recent[:30]
        assert sum(self.table.stacks) == self.n * 2000

    def _apply(self, action, memory=None, answered=True):
        seat = self.state.seat
        self.decisions[seat] += 1
        if memory is not None:
            self.memory[seat] = memory[:self.table.rules.memory_bytes]
        legal = self.table.apply(action, answered=answered)
        if not legal:
            self.failures[seat] += 1
            self.warning = f"{self.names[seat]} 的动作无效，已按游戏规则执行默认动作。"
        self._refresh()

    def act(self, data):
        if self.state is None or self.state.seat != 0:
            raise ValueError("现在不是你的回合。")
        kind = data.get("kind")
        if kind == "select":
            cards = data.get("cards")
            if self.state.phase != "select" or not isinstance(cards, list) or len(cards) != 2 or any(type(c) is not int or c not in range(3) for c in cards) or cards[0] == cards[1]:
                raise ValueError("请选择一张弃牌，再选择另一张保留牌亮出。")
            action = Action("select", cards=tuple(cards))
        elif kind in ("fold", "check", "call"):
            action = Action(kind)
        elif kind == "raise" and type(data.get("amount")) is int:
            action = Action.raise_to(data["amount"])
        else:
            raise ValueError("无效的动作。")
        if not is_legal(action, self.state.legal_actions):
            raise ValueError("该动作或下注金额不在当前合法范围内。")
        self._apply(action)

    def step(self):
        if self.state is None or self.state.seat == 0:
            raise ValueError("当前没有等待行动的机器人。")
        seat = self.state.seat
        try:
            raw, memory = reply(self.agents[seat].act(self.state, self.memory[seat]))
            action = action_from_dict(raw)
        except Exception:
            self._apply(None, answered=False)
            return
        self._apply(action, memory)

    def snapshot(self):
        view = self._view()
        events = [dict(kind=e.kind, seat=None if e.player is None else self.positions[e.player],
                       amount=e.amount, cards=list(e.cards)) for e in self.hand.events()]
        folded = {e["seat"] for e in events if e["kind"] == "fold"}
        seats = []
        for seat in range(self.n):
            pos = self.positions.index(seat) if seat in self.positions else None
            stack = self.table.stacks[seat] if self.done or pos is None else view.stacks[pos]
            shown = list(view.shown[pos]) if pos is not None else []
            status = "busted" if pos is None or (self.done and stack == 0) else "folded" if seat in folded else "all_in" if stack == 0 else "active"
            count = 0 if pos is None else 2 if shown or len(view.board) >= 3 and self.hand._selected and not self.hand._selecting else 3
            seats.append(dict(seat=seat, name=self.names[seat], stack=stack,
                              bet=0 if self.done or pos is None else view.bets[pos], status=status,
                              cards=list(self.hero_cards) if seat == 0 else shown, shown=shown,
                              card_count=count, delta=self.delta[seat], human=seat == 0))
        legal = []
        if self.state and self.state.seat == 0:
            for a in self.state.legal_actions:
                legal.append(dict(kind=a.kind, min=a.min_amount, max=a.max_amount,
                                  amount=min(self.state.to_call, self.state.seats[0].stack) if a.kind == "call" else 0))
        level = (self.table.hands_played - (1 if self.done else 0)) // self.table.rules.hands_per_level
        blinds = self.table.rules.blinds[min(level, len(self.table.rules.blinds) - 1)]
        points = None
        if self.done:
            points = table_points(self.table.result(tuple(self.failures), tuple(self.decisions)), self.table.payouts)
        return dict(mode="bay_area", hand=self.table.hands_played if self.done else self.table.hands_played + 1,
                    hand_cap=self.table.rules.hand_cap, phase="finished" if self.done else self.state.phase,
                    street="showdown" if self.done else {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(view.board)],
                    board=list(view.board), pot=0 if self.done else view.pot,
                    actor=None if self.done else self.state.seat, seats=seats, button=self.table._button,
                    blinds=[blinds.small_blind, blinds.big_blind], blind_level=level + 1,
                    legal=legal, events=events, done=self.done, finished=self.finished,
                    points=points, recent=self.recent, warning=self.warning,
                    discarded=list(view.variant.get("discarded", [])) if 0 in self.positions else [])
