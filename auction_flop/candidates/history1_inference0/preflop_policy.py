"""Backward-compatible preflop adapter for existing betting modules."""
from betting_policy import choose_bet


def choose_preflop(state, features):
    return choose_bet(state, features)
