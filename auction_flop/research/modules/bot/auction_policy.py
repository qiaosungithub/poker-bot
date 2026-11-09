"""Frozen event-time tournament auction adapter; public features unused control."""
from tournament_agent import Agent


def choose_bid(state, features):
    action, _ = Agent().act(state, b'')
    return action.cards
