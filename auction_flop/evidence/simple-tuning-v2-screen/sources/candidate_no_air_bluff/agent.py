"""Readable preflop ranges, one-parameter auction, unchanged classic postflop."""

import poker_agent_sdk
from simple_preflop import preflop
from simple_auction import auction
from classic_postflop import postflop


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            action = auction(state)
        elif not state.board:
            action = preflop(state)
        else:
            action = postflop(state)
        return action, b""
