"""Five decision stages sharing public and own-hand context."""
import poker_agent_sdk
from poker_agent_sdk import Action
from public_features import prepare, finish
from auction_policy import choose_bid
from preflop_policy import choose_preflop
from flop_policy import choose_flop
from turn_policy import choose_turn
from river_policy import choose_river


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        features, prepared = prepare(state, memory)
        if state.phase == 'auction':
            bid = tuple(choose_bid(state, features))
            legal = state.can('bid')
            if len(bid) != 5 or any(type(v) is not int or v < 0 for v in bid):
                raise ValueError('Auction policy must return five nonnegative integers')
            if legal is None or sum(bid) > legal.max_amount:
                raise ValueError('Auction policy exceeded legal budget')
            action = Action('bid', cards=bid)
            return action, finish(state, action, prepared)
        count = len(state.board)
        if count == 0:
            action = choose_preflop(state, features)
        elif count == 3:
            action = choose_flop(state, features)
        elif count == 4:
            action = choose_turn(state, features)
        elif count == 5:
            action = choose_river(state, features)
        else:
            raise ValueError('Unexpected board-card count during betting')
        legal = state.can(action.kind)
        if legal is None:
            raise ValueError('Betting policy returned a missing action kind')
        if action.kind == 'raise':
            action = Action.raise_to(max(legal.min_amount, min(legal.max_amount, int(action.amount))))
        return action, finish(state, action, prepared)
