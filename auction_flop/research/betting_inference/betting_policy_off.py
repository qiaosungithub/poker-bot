"""Exact same betting baseline with auction inference disabled."""
from betting_baseline import choose_baseline_bet

def choose_bet(state, features):
    return choose_baseline_bet(state)

choose_preflop = choose_bet
choose_postflop = choose_bet
