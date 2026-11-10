"""Flop adapter; receives the current board and complete shared context."""
from postflop_policy import choose_postflop


def choose_flop(state, features):
    return choose_postflop(state, features)
