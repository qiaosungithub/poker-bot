"""Classic txnujk rule continuation with the frozen event tournament auction."""
import poker_agent_sdk
from public_features import prepare, finish
from tournament_agent import Agent as AuctionAgent
from classic_policy import update_profiles, preflop, postflop

class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, prepared = prepare(state, memory)
        archetype = update_profiles(state, prepared)
        if state.phase == 'auction':
            action, _ = AuctionAgent().act(state, b'')
        else:
            action = preflop(state, archetype) if not state.board else postflop(state, archetype)
        return action, finish(state, action, prepared)
