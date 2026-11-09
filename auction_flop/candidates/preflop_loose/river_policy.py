"""River adapter; no future-card draws are assumed at this stage."""
from postflop_policy import choose_postflop


def choose_river(state, features):
    return choose_postflop(state, features)
