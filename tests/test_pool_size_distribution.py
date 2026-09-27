"""Tests for the exact pool-size / phase distribution (Markov chain)."""

from __future__ import annotations

import math
import random
import statistics
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from tayan_lab.pool_size_distribution import pool_size_distribution
from tayan_lab.reference import simulate_rounds


def test_hand_worked_example_two_players_start_1_limit_3():
    """2 players, 1 starting card, elimination at 3 cards: worked by hand in the M2 plan."""
    d = pool_size_distribution(players=2, starting_cards=1, elimination_limit=3)
    assert d.by_pool_size == {2: Fraction(1), 3: Fraction(1), 4: Fraction(1, 2)}
    assert d.expected_rounds == Fraction(5, 2)


@pytest.mark.parametrize(
    "players,starting_cards,elimination_limit",
    [(p, sc, el) for p in range(2, 8) for sc in (1, 2) for el in (5, 6)],
)
def test_probabilities_sum_to_one_and_rounds_are_positive(players, starting_cards, elimination_limit):
    if starting_cards >= elimination_limit:
        pytest.skip("invalid combination")
    d = pool_size_distribution(players, starting_cards, elimination_limit)
    assert sum(d.by_pool_size.values()) == d.expected_rounds
    assert d.expected_rounds > 0
    # every phase from `players` active down to 2 active must occur, for exactly 1+ rounds
    for phase in range(2, players + 1):
        assert phase in d.by_phase
        assert sum(d.by_phase[phase].values()) >= 1


@pytest.mark.parametrize("players,expected", [(3, Fraction(1009, 100)), (6, Fraction(2233, 100))])
def test_control_values_from_the_brief(players, expected):
    """2 starting cards, elimination limit 6, uniform loser model. Brief: 3p=10.09, 6p=22.33 rounds."""
    d = pool_size_distribution(players, starting_cards=2, elimination_limit=6)
    assert round(float(d.expected_rounds), 2) == round(float(expected), 2)


@pytest.mark.parametrize(
    "players,starting_cards,elimination_limit,loser_model",
    [(4, 2, 6, "uniform"), (5, 1, 5, "uniform"), (4, 2, 6, "inverse_cards"), (4, 2, 6, "fewest_cards")],
)
def test_matches_monte_carlo_within_4_standard_errors(players, starting_cards, elimination_limit, loser_model):
    """Rounds within one game are correlated, so we average per game and treat games (not
    individual rounds) as the independent samples for the standard error."""
    d = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    games = 4_000
    rng = random.Random(20260927)
    game_mean_pool: list[float] = []
    game_rounds: list[int] = []
    for _ in range(games):
        pool_sizes, _ = simulate_rounds(players, starting_cards, elimination_limit, 1, rng, loser_model)
        game_mean_pool.append(statistics.fmean(pool_sizes))
        game_rounds.append(len(pool_sizes))

    # A tiny floor guards against se == 0 when a model is (near-)deterministic by symmetry,
    # e.g. "fewest_cards" with all players starting equal.
    exact_mean_pool = float(sum(n * p for n, p in d.by_pool_size.items()) / d.expected_rounds)
    se_pool = max(statistics.pstdev(game_mean_pool) / math.sqrt(games), 1e-9)
    assert abs(statistics.fmean(game_mean_pool) - exact_mean_pool) < 4 * se_pool

    se_rounds = max(statistics.pstdev(game_rounds) / math.sqrt(games), 1e-9)
    assert abs(statistics.fmean(game_rounds) - float(d.expected_rounds)) < 4 * se_rounds


@given(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=1, max_value=2),
    st.integers(min_value=3, max_value=6),
)
@settings(max_examples=25, deadline=None)
def test_by_phase_sums_match_by_pool_size_total(players, starting_cards, elimination_limit):
    if starting_cards >= elimination_limit:
        return
    d = pool_size_distribution(players, starting_cards, elimination_limit)
    total_by_phase = sum((sum(v.values()) for v in d.by_phase.values()), Fraction(0))
    assert total_by_phase == d.expected_rounds


def test_unknown_loser_model_is_rejected():
    with pytest.raises(KeyError):
        pool_size_distribution(3, 2, 6, loser_model="does-not-exist")


def test_starting_cards_must_be_below_elimination_limit():
    with pytest.raises(ValueError):
        pool_size_distribution(3, 6, 6)


def test_needs_at_least_two_players():
    with pytest.raises(ValueError):
        pool_size_distribution(1, 2, 6)
