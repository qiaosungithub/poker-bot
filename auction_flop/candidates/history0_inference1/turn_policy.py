"""Turn adapter; downstream policy must reevaluate on the newly known card."""
from postflop_policy import choose_postflop


def choose_turn(state, features):
    return choose_postflop(state, features)
