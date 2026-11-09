"""Independent auction and betting modules sharing one public observer."""
import poker_agent_sdk
from poker_agent_sdk import Action
from public_features import observe
from auction_policy import choose_bid
from betting_policy import choose_bet


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, updated = observe(state, memory)
        if state.phase == 'auction':
            bid = tuple(choose_bid(state, features))
            legal = state.can('bid')
            if len(bid) != 5 or any(type(v) is not int or v < 0 for v in bid):
                raise ValueError('Auction policy must return five nonnegative integers')
            if legal is None or sum(bid) > legal.max_amount:
                raise ValueError('Auction policy exceeded legal budget')
            return Action('bid', cards=bid), updated
        action = choose_bet(state, features)
        legal = state.can(action.kind)
        if legal is None:
            raise ValueError('Betting policy returned a missing action kind')
        if action.kind == 'raise':
            action = Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(action.amount))))
        return action, updated
