"""Stateless phase routing for the Boston agent."""

from .common import Action, poker_agent_sdk
from .auction import auction_decision
from .preflop import preflop_decision
from .postflop import postflop_decision


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            return auction_decision(state, memory)
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), b""
        if not state.board:
            return preflop_decision(state, memory)
        return postflop_decision(state, memory)
