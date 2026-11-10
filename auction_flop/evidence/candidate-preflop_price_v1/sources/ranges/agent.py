"""Boston auction-flop range policy, independently written during the event."""
import poker_agent_sdk
from poker_agent_sdk import Action, GameState
from range_core import BettingView, auction_bids, betting_intent, legalize

class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        mine = next(s for s in state.seats if s.seat == state.seat)
        # Only this street's public aggression; no unseen cards or cross-table state.
        raises = 0
        for event in reversed(state.history):
            if event.kind == 'board':
                break
            if event.kind == 'raise' and event.seat != state.seat:
                raises += 1
        bb = max(1, state.blinds.big_blind)
        if state.phase == 'auction':
            descriptor = state.can('bid')
            if descriptor is None:
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), b''
            candidates = tuple(state.variant.get('auction_cards', ()))
            budget = min(descriptor.max_amount, int(state.variant.get('flop_chips',0)))
            if len(candidates) != 5 or budget <= 0:
                return Action('bid', cards=(0,0,0,0,0)), b''
            bids = auction_bids(state.cards, candidates, budget,
                max(1,state.config.hand_cap-state.hand_number+1), mine.stack/bb,
                raises, state.pot/max(1,mine.stack+state.pot))
            return Action('bid',cards=bids), b''
        if state.phase != 'betting':
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), b''
        in_hand = [s for s in state.seats if s.status in ('active','all_in')]
        active = [s for s in state.seats if s.status == 'active']
        seats_alive = sum(s.status != 'busted' for s in state.seats)
        shift = (0 if seats_alive == 2 else 3) if not state.board else 1
        active.sort(key=lambda s:(s.seat-state.button-shift)%state.config.players)
        position = next((i for i,s in enumerate(active) if s.seat==state.seat),0)/max(1,len(active)-1)
        # Every unselected public candidate is permanently dead in the released rules.
        dead = tuple(c for c in state.variant.get('auction_cards',()) if c not in state.board)
        view = BettingView(tuple(state.cards),tuple(state.board),state.pot,state.to_call,
            mine.stack,mine.bet,bb,max(1,len(in_hand)-1),position,raises,state.hand_number,dead)
        legal = {a.kind:(a.min_amount,a.max_amount) for a in state.legal_actions}
        kind, amount = legalize(betting_intent(view),legal)
        return Action(kind,amount), b''
