"""Patient with marginal hands; buy key cards and value-bet genuine strength."""
import poker_agent_sdk
from simple_preflop import preflop
from simple_auction import auction
from value_postflop import postflop


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        if state.phase == "auction":
            action = auction(state)
        elif not state.board:
            action = preflop(state)
        else:
            action = postflop(state)
        return action, b""
