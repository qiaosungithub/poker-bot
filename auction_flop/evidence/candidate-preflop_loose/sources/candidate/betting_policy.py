"""Frozen event-time variant betting adapter; public features unused control."""
from variant_agent import Agent


def choose_bet(state, features):
    action, _ = Agent().act(state, b'')
    return action
