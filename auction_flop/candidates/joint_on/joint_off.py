"""Exact component control: identical observations, all coordination disabled."""
import joint_policy as policy


def choose_preflop(state, features):
    return policy.choose_preflop(state, features, coupling=False)


def choose_bid(state, features):
    return policy.choose_bid(state, features, coupling=False)


def choose_flop(state, features):
    return policy.choose_flop(state, features, coupling=False)


def choose_turn(state, features):
    return policy.choose_turn(state, features, coupling=False)


def choose_river(state, features):
    return policy.choose_river(state, features, coupling=False)


def choose_postflop(state, features):
    return policy.choose_postflop(state, features, coupling=False)


def choose_bet(state, features):
    return policy.choose_bet(state, features, coupling=False)
