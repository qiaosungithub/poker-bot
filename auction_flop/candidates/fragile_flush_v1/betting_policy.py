"""Corrected shared betting core; auction and other policy rules unchanged."""
from corrected_betting_baseline import choose_baseline_bet


def choose_bet(state, features):
    return choose_baseline_bet(state)


choose_preflop = choose_bet
choose_postflop = choose_bet
choose_flop = choose_bet
choose_turn = choose_bet
choose_river = choose_bet
