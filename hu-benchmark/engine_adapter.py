"""Standard heads-up forced-action handling around the vendored engine.

The upstream engine requests redundant actions when only one player has chips
after a matched all-in, and can request action from an all-in blind poster.
Neither condition should query a policy in standard heads-up poker.
"""
from engine.game import PokerEngine as UpstreamPokerEngine


class PokerEngine(UpstreamPokerEngine):
    def _advance_if_street_over(self, last_seat):
        active = [p for p in self.players if p.is_active]
        if not active or (len(active) == 1 and active[0].bet_this_street >= self.current_bet):
            return self._run_it_out()
        return super()._advance_if_street_over(last_seat)

    def _handle_aggression(self, seat, raise_size):
        super()._handle_aggression(seat, raise_size)
        # A short all-in does not reopen raising, but a player who already
        # acted still has to call the increment or fold. In heads-up the
        # aggressor is all-in, so there is no further live raising opponent.
        if raise_size > 0:
            self._needs_to_act.update(p.seat for p in self.players
                                      if p.is_active and p.seat != seat
                                      and p.bet_this_street < self.current_bet)

    def start_hand(self):
        state = super().start_hand()
        active = [p for p in self.players if p.is_active]
        if not active or (len(active) == 1 and active[0].bet_this_street >= self.current_bet):
            return self._run_it_out()
        if not self.players[state['seat_to_act']].is_active:
            return self._build_state(active[0].seat)
        return state

    def _advance_street(self):
        if sum(p.is_active for p in self.players) <= 1:
            return self._run_it_out()
        return super()._advance_street()
