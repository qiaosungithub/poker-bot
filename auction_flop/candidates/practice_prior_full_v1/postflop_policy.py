"""Backward-compatible postflop adapter for existing betting modules."""
from betting_policy import choose_bet


def choose_postflop(state, features):
    return choose_bet(state, features)
