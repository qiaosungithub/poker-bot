import itertools
import random
from dataclasses import replace

import pytest

from bay_area.environment import BayAreaEnv
from poker_agent_sdk import Action
from poker_agent_protocol.game import passive_action
from poker_engine.scoring import table_points
from poker_engine.table import TableResult
from poker_engine.variant import load_variant
from poker_agent_protocol.game import BlindLevel
import eval7


def call_or_check(state):
    return Action.check() if state.can("check") else Action.call()


def reach_selection(env):
    while env.state and env.state.phase != "select":
        env.step((call_or_check(env.state), b""))
    assert env.state is not None
    return env.state


def test_every_selection_is_legal_and_private_until_last_choice():
    for discard, reveal in itertools.permutations(range(3), 2):
        env = BayAreaEnv("selection-six-choices", hand_cap=1)
        reach_selection(env)
        before = None
        choices = {}
        while env.state and env.state.phase == "select":
            state = env.state
            assert len(state.board) == 3 and len(state.cards) == 3
            assert all(not s.shown_cards for s in state.seats)
            assert all(not e.cards for e in state.history if e.kind == "select")
            choices[state.seat] = tuple(state.cards)
            if before is None:
                before = [s.stack for s in state.seats]
            _, reason = env.step((Action("select", cards=(discard, reveal)), b""))
            assert reason is None
        state = env.state
        assert state is not None
        assert [s.stack for s in state.seats] == before
        assert len(state.cards) == 2
        assert state.variant["discarded"] == [choices[state.seat][discard]]
        assert tuple(state.cards) == tuple(c for i,c in enumerate(choices[state.seat]) if i!=discard)
        for s in state.seats:
            assert tuple(s.shown_cards) == (choices[s.seat][reveal],)
        dead = {cards[discard] for cards in choices.values()}
        while env.state:
            assert not dead.intersection(env.state.board)
            env.step((call_or_check(env.state), b""))
        for hand in env.table.log:
            for event in hand["events"]:
                if event["kind"] in ("show", "reveal"):
                    assert not dead.intersection(event["cards"])


def test_failed_selection_defaults_and_does_not_fold():
    env = BayAreaEnv("bad-selection", hand_cap=1)
    state = reach_selection(env)
    seat, cards = state.seat, state.cards
    _, reason = env.step((Action("select", cards=(0, 0)), b"new-memory"))
    assert reason == "illegal" and env.memory[seat] == b"new-memory"
    while env.state.phase == "select":
        env.step((Action("select", cards=(2, 0)), b""))
    assert env.state.seats[seat].status != "folded"
    assert tuple(env.state.seats[seat].shown_cards) == (cards[0],)


def test_all_in_players_select_before_any_hidden_cards_are_revealed():
    env = BayAreaEnv("allin-selection", hand_cap=1)
    while env.state and env.state.phase != "select":
        state = env.state
        legal = state.can("raise")
        action = Action.raise_to(legal.max_amount) if legal else call_or_check(state)
        env.step((action, b""))
    count = 0
    while env.state and env.state.phase == "select":
        assert all(not s.shown_cards for s in env.state.seats)
        assert not any(e.kind == "show" for e in env.state.history)
        count += 1
        env.step((Action("select", cards=(2, 0)), b""))
    assert count == 8
    assert env.state is None
    result, points = env.result()
    assert sum(result.final_stacks) == 16000 and sum(points) == pytest.approx(100)


def test_public_card_remains_visible_after_fold():
    env = BayAreaEnv("folded-upcard", hand_cap=1)
    reach_selection(env)
    while env.state.phase == "select":
        env.step((Action("select",cards=(2,0)), b""))
    raise_action = env.state.can("raise")
    env.step((Action.raise_to(raise_action.min_amount), b""))
    state=env.state
    assert state.can("fold")
    seat, exposed = state.seat, tuple(state.seats[state.seat].shown_cards)
    env.step((Action.fold(), b""))
    assert env.state.seats[seat].status == "folded"
    assert tuple(env.state.seats[seat].shown_cards) == exposed
    assert len(exposed) == 1


def test_uncontested_preflop_pot_skips_selection():
    env=BayAreaEnv("all-fold", hand_cap=1)
    while env.state:
        assert env.state.phase == "betting" and not env.state.board
        env.step((Action.fold(),b""))
    result,points=env.result()
    assert result.hands_played == 1
    assert sum(result.final_stacks) == 16000 and sum(points) == pytest.approx(100)


def test_timeout_preserves_memory_and_replays_events():
    env = BayAreaEnv("timeout", time_bank_ms=2)
    state = env.state
    seat = state.seat
    env.memory[seat] = b"old"
    _, reason = env.step((Action.call(), b"new"), elapsed_ms=4)
    assert reason == "timeout"
    assert env.memory[seat] == b"old" and env.failures[seat] == 1
    assert env.table._bank[seat] == 0
    assert env.table._seen[seat] == 0


def test_illegal_raise_is_not_clamped_and_memory_is_truncated():
    env = BayAreaEnv("illegal-raise")
    state = env.state
    seat = state.seat
    _, reason = env.step((Action.raise_to(10**9), b"x" * 70000))
    assert reason == "illegal"
    assert env.memory[seat] == b"x" * 65536


def test_icm_numeric_oracle_and_simultaneous_equal_stack_busts():
    result = TableResult(250, (2000,) * 8, (), (0,) * 8)
    assert table_points(result, (50, 30, 20)) == pytest.approx([12.5] * 8)
    result = TableResult(250, (12000, 4000), (), (0, 0))
    assert table_points(result, (50, 30)) == pytest.approx([45, 35])
    # Seats 2/3 bust together on equal start stacks, sharing 3rd/4th payout.
    result = TableResult(1, (8000, 8000, 0, 0), (2, 3), (0,) * 4, ties=(1,))
    assert table_points(result, (50, 30, 20, 0)) == pytest.approx([40, 40, 10, 10])


def test_full_table_negative_control_and_reproducibility():
    def play(seed):
        env = BayAreaEnv(seed, hand_cap=12)
        while env.state:
            env.step((passive_action(env.state.legal_actions), b""))
        return env.result()
    a, b = play("repeat"), play("repeat")
    assert a == b
    assert sum(a[0].final_stacks) == 16000
    assert sum(a[1]) == pytest.approx(100)
    # An intentionally corrupted chip vector must fail the conservation predicate.
    broken = replace(a[0], final_stacks=(1,) + a[0].final_stacks[1:])
    assert sum(broken.final_stacks) != 16000
    with pytest.raises(RuntimeError):
        BayAreaEnv("unfinished").result()


def test_unequal_stack_allins_match_independent_sidepot_oracle():
    # These stacks make every possible split divisible by the winner count,
    # avoiding dependence on the engine's separate odd-chip ordering rule.
    initial = (120, 240, 600)
    for seed in range(30):
        hand = load_variant("flop_discard_reveal").deal(
            stacks=initial, blinds=BlindLevel(10,20), rng=random.Random(seed))
        retained, board, selected = {}, None, 0
        while (player := hand.actor()) is not None:
            state = hand.view(player)
            if state.phase == "select":
                assert not any(state.shown)
                retained[player] = tuple(state.cards[:2])
                selected += 1
                action = Action("select",cards=(2,0))
            else:
                legal = {a.kind:a for a in hand.legal_actions()}
                action = Action.raise_to(legal["raise"].max_amount) if "raise" in legal else (
                    Action.check() if "check" in legal else Action.call())
            hand.apply(action)
        assert selected == 3
        board = hand.view(0).board
        ranks = {i:eval7.evaluate([eval7.Card(c) for c in (*retained[i],*board)]) for i in range(3)}
        expected, previous = [0]*3, 0
        for level in sorted(initial):
            eligible = [i for i in range(3) if initial[i]>=level]
            pot = (level-previous)*len(eligible)
            winners = [i for i in eligible if ranks[i]==max(ranks[j] for j in eligible)]
            for i in winners:
                expected[i] += pot//len(winners)
            previous = level
        assert hand.stacks() == tuple(expected)
