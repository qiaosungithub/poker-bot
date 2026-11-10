"""Five-stage joint candidate with explicit table memory."""
import poker_agent_sdk
from poker_agent_sdk import Action
from public_features import prepare, finish
from joint_off import choose_preflop, choose_bid, choose_flop, choose_turn, choose_river


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, prepared = prepare(state, memory)
        if state.phase == 'auction':
            bids = tuple(choose_bid(state, features))
            legal = state.can('bid')
            if legal is None or len(bids) != 5 or any(type(v) is not int or v < 0 for v in bids) or sum(bids) > legal.max_amount:
                raise ValueError('Invalid bid from joint policy')
            action = Action('bid', cards=bids)
        else:
            count = len(state.board)
            chooser = choose_preflop if count == 0 else choose_flop if count == 3 else choose_turn if count == 4 else choose_river if count == 5 else None
            if chooser is None:
                raise ValueError('Unexpected betting board')
            action = chooser(state, features)
            legal = state.can(action.kind)
            if legal is None:
                raise ValueError('Illegal joint betting action')
            if action.kind == 'raise':
                action = Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(action.amount))))
        return action, finish(state, action, prepared)
