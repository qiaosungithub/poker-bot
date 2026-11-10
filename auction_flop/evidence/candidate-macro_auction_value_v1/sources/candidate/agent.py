"""Active loose preflop plus complete classic postflop and frozen auction."""
import poker_agent_sdk
from public_features import prepare, finish
from marginal_auction import auction_action
from classic_policy import update_profiles, postflop
from independent_preflop import choose_preflop_loose

class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, prepared = prepare(state, memory)
        archetype = update_profiles(state, prepared)
        if state.phase == 'auction':
            action = auction_action(state, features)
        else:
            action = choose_preflop_loose(state, features) if not state.board else postflop(state, archetype, price_veto=True, contribution_veto=True)
        return action, finish(state, action, prepared)
