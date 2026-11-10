"""Event-time TournamentSpecialist continuation with the frozen auction."""
import poker_agent_sdk
from public_features import prepare, finish
from tournament_agent import Agent as AuctionAgent
from specialist_policy import preflop, postflop


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        _, prepared = prepare(state, memory)
        if state.phase == 'auction':
            action, _ = AuctionAgent().act(state, b'')
        else:
            action = postflop(state) if state.board else preflop(state)
        return action, finish(state, action, prepared)
