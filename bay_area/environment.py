"""Small step API around the pinned participant-published competition engine.

No poker rules are reimplemented: the upstream hand and table engine owns them.
This wrapper owns per-seat memory, SDK reply validation and local time accounting.
"""
from __future__ import annotations

from poker_agent_protocol.game import action_from_dict
from poker_agent_sdk.guest import reply
from poker_engine.scoring import table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

VARIANT = "flop_discard_reveal"


class BayAreaEnv:
    def __init__(self, seed: str, *, players=8, hand_cap=250, time_bank_ms=100, variant=VARIANT):
        self.rules = TableRules(variant=variant, players=players, hand_cap=hand_cap,
                                time_bank_ms=time_bank_ms)
        self.table = SitAndGo(rules=self.rules, variant=load_variant(variant),
                             table_id=seed, seed=seed)
        self.memory = [b""] * players
        self.failures = [0] * players
        self.decisions = [0] * players
        self.state = self.table.decision()

    def step(self, answer=None, *, elapsed_ms=0.0, crashed=False):
        """Apply an (Action, bytes) answer; return next state and failure reason.

        elapsed_ms is optional local decision time, with no delivery allowance.
        Actual benchmark deadlines use the SDK process runner instead.
        """
        if self.state is None:
            raise RuntimeError("The table is finished.")
        state = self.state
        seat = state.seat
        self.decisions[seat] += 1
        action, answered, reason = None, False, None
        debit = min(state.time_bank_ms, max(0.0, elapsed_ms - state.config.decision_ms))
        if elapsed_ms > state.config.decision_ms + state.time_bank_ms:
            reason = "timeout"
        elif crashed:
            reason = "exception"
        else:
            try:
                raw, memory = reply(answer)
                action = action_from_dict(raw)
                self.memory[seat] = memory[:self.rules.memory_bytes]
                answered = True
            except (TypeError, ValueError, OverflowError):
                reason = "protocol"
        legal = self.table.apply(action, answered=answered, bank_ms=debit)
        if not legal:
            self.failures[seat] += 1
            reason = reason or "illegal"
        self.state = self.table.decision()
        return self.state, reason

    def result(self):
        if self.state is not None:
            raise RuntimeError("Play to termination before requesting a result.")
        result = self.table.result(tuple(self.failures), tuple(self.decisions))
        return result, table_points(result, self.table.payouts)
