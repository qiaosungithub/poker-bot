"""Active loose preflop plus complete classic postflop and frozen auction."""
import poker_agent_sdk
from public_features import prepare, finish
from tournament_agent import Agent as AuctionAgent
from classic_policy import update_profiles, postflop
from independent_preflop import choose_preflop_loose

class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, prepared = prepare(state, memory)
        archetype = update_profiles(state, prepared)
        if state.phase == 'auction':
            action, _ = AuctionAgent().act(state, b'')
        else:
            action = choose_preflop_loose(state, features) if not state.board else postflop(state, archetype, price_veto=True, contribution_veto=False)
        return action, finish(state, action, prepared)
