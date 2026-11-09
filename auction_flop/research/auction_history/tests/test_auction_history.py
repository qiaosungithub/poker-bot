import sys
from pathlib import Path
from types import SimpleNamespace as NS
import random
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auction_value import chosen_indices, forcing_bid, TRIPLES
from auction_policy import build_scenarios, choose_bid, choose_bid_no_history, pressure_estimates, eligible_auction_pot
from poker_agent_protocol.game import GameState, TableConfig, Seat, BlindLevel, LegalAction, Event


def fixture(balance=1000, mean=0., observations=0, positive=0, hand=10):
    seats = tuple(Seat(i, 2000, 0, 'active') for i in range(8))
    cfg = TableConfig('test', 0, 8, 'flop_auction_holdem', 2000,
                      (BlindLevel(10, 20),), 20, 250, (50, 30, 20, 0, 0, 0, 0, 0), 1, 65536, 100)
    state = GameState((), hand, 0, ('As', 'Ks'), (), 400, 0,
                      (LegalAction('bid', 0, balance),), (), seats, 0, 0,
                      BlindLevel(10, 20), 10, 8, (), cfg,
                      {'flop_chips': balance, 'flop_chips_by_seat': [balance] * 8,
                       'auction_cards': ['2d', '7c', 'Qs', 'Js', 'Ts']}, 'auction', 100)
    stats = tuple(NS(fraction_observations=observations, fraction_mean=mean,
                     positive_observations=observations, positive_auctions=positive)
                  for _ in range(8))
    return state, NS(balances=(balance,) * 8, history=stats)


def test_scenarios_legal():
    rng = random.Random(928217)
    for _ in range(400):
        state, features = fixture(balance=rng.randrange(1001), mean=rng.random(),
                                  observations=rng.randrange(80), positive=0)
        scenarios, _ = build_scenarios(state, features)
        assert abs(sum(p for _, p in scenarios) - 1) < 1e-12
        assert all(type(x) is int and x >= 0 for v, _ in scenarios for x in v)
        assert all(sum(v) <= sum(features.balances[1:]) for v, _ in scenarios)
        bid = choose_bid(state, features)
        assert len(bid) == 5 and all(type(x) is int and x >= 0 for x in bid)
        assert sum(bid) <= features.balances[0]


def test_no_history_is_exact_ablation():
    state, blank = fixture()
    _, rich = fixture(mean=.30, observations=50, positive=45)
    assert build_scenarios(state, blank, False) == build_scenarios(state, rich, False)
    assert choose_bid_no_history(state, blank) == choose_bid_no_history(state, rich)
    assert build_scenarios(state, blank) != build_scenarios(state, rich)
    assert choose_bid(state, blank) != choose_bid(state, rich)


def test_observed_pressure_changes_prior():
    state, quiet = fixture(mean=.001, observations=50, positive=2)
    _, active = fixture(mean=.20, observations=50, positive=45)
    low, _ = pressure_estimates(state, quiet)
    high, _ = pressure_estimates(state, active)
    assert sum(x[2] for x in high) > sum(x[2] for x in low)
    assert all(x[2] > 0 for x in low), 'censored zeros must not imply zero willingness'


def test_zero_balance_and_scarce_balance():
    state, features = fixture(balance=0)
    assert choose_bid(state, features) == (0, 0, 0, 0, 0)
    state, features = fixture(balance=1)
    assert sum(choose_bid(state, features)) <= 1


def test_no_mutation_or_global_adaptation():
    state, features = fixture(mean=.1, observations=12, positive=8)
    original = repr(features)
    first = choose_bid(state, features)
    alien, alien_features = fixture(mean=.8, observations=100, positive=100, hand=123)
    choose_bid(alien, alien_features)
    assert choose_bid(state, features) == first
    assert repr(features) == original


def test_no_opposition_uses_exact_tie_rule():
    state, features = fixture()
    features.balances = (1000, 0, 0, 0, 0, 0, 0, 0)
    bid = choose_bid(state, features)
    # AK of spades selects QJT: each later candidate needs a single vote.
    assert bid == (0, 0, 1, 1, 1)


def test_allin_sidepot_cap():
    state, _ = fixture()
    seats = (replace(state.seats[0], stack=0, status='all_in'),) + state.seats[1:]
    state = replace(state, seats=seats, pot=2100,
                    history=(Event('raise', 0, 100), Event('raise', 1, 1000),
                             Event('call', 2, 1000)))
    assert eligible_auction_pot(state) == 300
    state = replace(state, seats=(replace(seats[0], stack=1000, status='active'),) + seats[1:])
    assert eligible_auction_pot(state) == 2100


def test_fast_tie_helpers_match_generic():
    rng = random.Random(83912)
    for _ in range(1000):
        totals = tuple(rng.randrange(100) for _ in range(5))
        target = rng.choice(TRIPLES)
        # list equality deliberately selects the retained slow generic branch.
        assert chosen_indices(totals) == chosen_indices(totals, [0, 1, 2, 3, 4])
        assert forcing_bid(target, totals) == forcing_bid(target, totals, [0, 1, 2, 3, 4])


if __name__ == '__main__':
    for name, fn in sorted(globals().copy().items()):
        if name.startswith('test_'):
            fn()
            print(name, 'ok')
