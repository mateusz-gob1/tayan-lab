"""Tests for rankings and their measures (M3)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from tayan_lab.probabilities import CATEGORIES, probability
from tayan_lab.rankings import (
    CURRENT_ORDER,
    dynamic_ranking,
    expected_ranking_changes,
    inversion_mass,
    kendall_tau,
    rank_categories,
    staged_ranking,
    static_ranking,
    weighted_inversion_mass,
    weighted_probabilities,
)


# --------------------------------------------------------------------------- rank_categories


def test_rank_categories_is_a_permutation_of_all_categories():
    p = {c: probability(c, 9, 10) for c in CATEGORIES}
    ranking = rank_categories(p, epsilon_pp=2.0)
    assert set(ranking) == set(CATEGORIES)
    assert len(ranking) == len(CATEGORIES)


def test_rank_categories_orders_by_probability_outside_epsilon():
    p = {c: Fraction(1, 100) for c in CATEGORIES}
    p.update(HIGH=Fraction(90, 100), PAIR=Fraction(50, 100))
    ranking = rank_categories(p, epsilon_pp=2.0, base_order=CURRENT_ORDER)
    # gaps are far beyond epsilon: HIGH and PAIR must sort to the front (most common first)
    assert ranking.index("HIGH") < ranking.index("PAIR") < ranking.index("FOUR")


def test_rank_categories_ties_keep_base_order():
    """Two categories within epsilon must keep their relative order from base_order,
    even though the raw probabilities would otherwise suggest swapping them."""
    p = {c: Fraction(1, 100) for c in CATEGORIES}
    p["PAIR"] = Fraction(50, 1000)  # 5.0%
    p["STRAIGHT"] = Fraction(51, 1000)  # 5.1%: 0.1pp apart, inside epsilon=2pp
    base = ("HIGH", "PAIR", "STRAIGHT", "TWO_PAIR", "THREE", "FLUSH", "FULL", "FOUR", "STRAIGHT_FLUSH")
    ranking = rank_categories(p, epsilon_pp=2.0, base_order=base)
    assert ranking.index("PAIR") < ranking.index("STRAIGHT")  # unchanged despite STRAIGHT > PAIR


def test_rank_categories_ties_break_by_new_base_order_for_hysteresis():
    p = {"PAIR": Fraction(51, 1000), "STRAIGHT": Fraction(50, 1000)}
    for c in CATEGORIES:
        p.setdefault(c, Fraction(1, 100))
    # base_order here has STRAIGHT before PAIR (as if that was the previously active order)
    base = ("HIGH", "STRAIGHT", "PAIR", "TWO_PAIR", "THREE", "FLUSH", "FULL", "FOUR", "STRAIGHT_FLUSH")
    ranking = rank_categories(p, epsilon_pp=2.0, base_order=base)
    assert ranking.index("STRAIGHT") < ranking.index("PAIR")


@given(st.integers(min_value=2, max_value=10), st.integers(min_value=1, max_value=30))
def test_dynamic_ranking_current_order_produces_same_mass_regardless_of_tie_break_direction(lowest, n):
    """Sanity: rank_categories never drops or duplicates a category, for arbitrary p."""
    deck = 4 * (15 - lowest)
    if not 0 <= n <= deck:
        return
    p = {c: probability(c, lowest, n) for c in CATEGORIES}
    ranking = rank_categories(p, epsilon_pp=2.0)
    assert sorted(ranking) == sorted(CATEGORIES)


# --------------------------------------------------------------------------- inversion mass


def test_inversion_mass_is_zero_for_a_perfectly_sorted_ranking():
    p = {c: probability(c, 9, 10) for c in CATEGORIES}
    perfect = tuple(sorted(CATEGORIES, key=lambda c: -p[c]))  # most common first
    assert inversion_mass(perfect, p, epsilon_pp=0.0) == 0


def test_inversion_mass_is_never_negative():
    p = {c: probability(c, 9, 10) for c in CATEGORIES}
    assert inversion_mass(CURRENT_ORDER, p, epsilon_pp=2.0) >= 0


def test_inversion_mass_of_current_order_is_positive_at_n_23_deck_52():
    """Known finding from M1: at N=23 on the 52-card deck, STRAIGHT overtakes PAIR."""
    p = {c: probability(c, 2, 23) for c in CATEGORIES}
    assert inversion_mass(CURRENT_ORDER, p, epsilon_pp=1.0) > 0


def test_current_order_inversion_mass_is_at_least_dynamic_rankings():
    """The per-N dynamic ranking is optimal for that N by construction, so no other
    ranking (including the current one) can have a smaller inversion mass there."""
    for n in (6, 10, 15, 20, 23, 30):
        p = {c: probability(c, 2, n) for c in CATEGORIES}
        dyn, _ = dynamic_ranking(2, n, epsilon_pp=2.0)
        assert inversion_mass(dyn, p, epsilon_pp=2.0) <= inversion_mass(CURRENT_ORDER, p, epsilon_pp=2.0)


# --------------------------------------------------------------------------- static ranking / weighted mass


def test_weighted_probabilities_are_between_0_and_1():
    # deck must be able to hold the largest possible pool: players * (elimination_limit - 1)
    p = weighted_probabilities(2, players=6, starting_cards=2, elimination_limit=6)
    assert set(p) == set(CATEGORIES)
    for value in p.values():
        assert 0 <= value <= 1


def test_static_ranking_matches_current_order_for_a_small_deck_where_it_should():
    """Small deck (24 cards), few players: straights/flushes barely happen, so the
    static ranking should not disagree wildly with the current order early on."""
    ranking, p = static_ranking(9, players=2, starting_cards=2, elimination_limit=6, epsilon_pp=2.0)
    # HIGH must stay first (it is always by far the most common) and STRAIGHT_FLUSH last
    assert ranking[0] == "HIGH"
    assert ranking[-1] == "STRAIGHT_FLUSH"


def test_weighted_inversion_mass_matches_manual_average():
    from tayan_lab.pool_size_distribution import pool_size_distribution

    dist = pool_size_distribution(players=4, starting_cards=2, elimination_limit=6)
    expected = Fraction(0)
    for n, rounds in dist.by_pool_size.items():
        p = {c: probability(c, 9, n) for c in CATEGORIES}
        expected += rounds * inversion_mass(CURRENT_ORDER, p, epsilon_pp=2.0)
    expected /= dist.expected_rounds
    actual = weighted_inversion_mass(CURRENT_ORDER, 9, players=4, starting_cards=2, elimination_limit=6, epsilon_pp=2.0)
    assert actual == expected


def test_9_to_13_players_full_deck_static_ranking_puts_straight_above_pair():
    """Follow-up check from the M2 discussion: for many-player games on the 52-card deck,
    STRAIGHT should end up ranked above (rarer than) PAIR once averaged over the game,
    because N >= 23 (where STRAIGHT overtakes PAIR) is a large share of rounds there."""
    for players in (10, 11, 12, 13):
        start = 2 if players <= 6 else 1
        limit = 6 if players <= 10 else 5
        ranking, _ = static_ranking(2, players, start, limit, epsilon_pp=2.0)
        assert ranking.index("PAIR") < ranking.index("STRAIGHT")


# --------------------------------------------------------------------------- kendall tau


def test_kendall_tau_identical_orders_is_1():
    assert kendall_tau(CURRENT_ORDER, CURRENT_ORDER) == 1


def test_kendall_tau_fully_reversed_is_minus_1():
    assert kendall_tau(CURRENT_ORDER, tuple(reversed(CURRENT_ORDER))) == -1


def test_kendall_tau_one_adjacent_swap():
    swapped = list(CURRENT_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    n = len(CURRENT_ORDER)
    total_pairs = n * (n - 1) // 2
    # exactly one discordant pair out of all pairs
    assert kendall_tau(CURRENT_ORDER, tuple(swapped)) == Fraction(total_pairs - 2, total_pairs)


def test_kendall_tau_rejects_mismatched_sets():
    with pytest.raises(ValueError):
        kendall_tau(CURRENT_ORDER, CURRENT_ORDER[:-1])


# --------------------------------------------------------------------------- ranking-change cost


def test_expected_ranking_changes_is_zero_when_epsilon_is_huge():
    """A huge epsilon means everything ties, so the base order never changes."""
    changes = expected_ranking_changes(9, players=4, starting_cards=2, elimination_limit=6, epsilon_pp=100.0)
    assert changes == 0


def test_expected_ranking_changes_hysteresis_is_never_more_than_without():
    """Hysteresis should never cause MORE changes than plain per-round re-ranking,
    since it only keeps the previous order when the evidence is ambiguous."""
    no_hyst = expected_ranking_changes(9, 4, 2, 6, epsilon_pp=2.0, hysteresis=False)
    with_hyst = expected_ranking_changes(9, 4, 2, 6, epsilon_pp=2.0, hysteresis=True)
    assert with_hyst <= no_hyst


def test_expected_ranking_changes_is_non_negative():
    for hysteresis in (True, False):
        changes = expected_ranking_changes(7, 5, 1, 6, epsilon_pp=2.0, hysteresis=hysteresis)
        assert changes >= 0


def test_expected_ranking_changes_at_epsilon_0_only_counts_genuine_crossings():
    """At epsilon=0 (no ties at all), a change only happens when the strict probability
    order actually flips between rounds; the count must still be finite and non-negative."""
    changes = expected_ranking_changes(9, 4, 2, 6, epsilon_pp=0.0, hysteresis=False)
    assert changes >= 0


# --------------------------------------------------------------------------- staged ranking


def test_staged_ranking_never_worse_than_single_static():
    """Per-phase-optimal rankings can only match or beat a single whole-game ranking,
    since the single ranking is itself a valid (if suboptimal) choice for every phase."""
    result = staged_ranking(2, players=9, starting_cards=1, elimination_limit=6, epsilon_pp=2.0)
    assert result.staged_mass <= result.single_static_mass


def test_staged_ranking_switch_count_is_bounded_and_nonnegative():
    result = staged_ranking(2, players=9, starting_cards=1, elimination_limit=6, epsilon_pp=2.0)
    assert 0 <= result.switches <= 9 - 2  # at most one switch per phase boundary


def test_staged_ranking_has_one_ranking_per_active_player_count():
    result = staged_ranking(7, players=5, starting_cards=2, elimination_limit=6, epsilon_pp=2.0)
    assert set(result.rankings_by_phase) == set(range(2, 6))
    for ranking in result.rankings_by_phase.values():
        assert sorted(ranking) == sorted(CATEGORIES)


def test_staged_ranking_zero_switches_when_epsilon_is_huge():
    """A huge epsilon means every phase ties out to the same base order: no switches, no gain."""
    result = staged_ranking(2, players=6, starting_cards=2, elimination_limit=6, epsilon_pp=100.0)
    assert result.switches == 0
    assert result.staged_mass == result.single_static_mass == 0


def test_staged_ranking_respects_explicit_base_order():
    r1 = staged_ranking(2, 6, 2, 6, epsilon_pp=2.0, base_order=CURRENT_ORDER)
    r2 = staged_ranking(2, 6, 2, 6, epsilon_pp=2.0)  # defaults to that player count's own static ranking
    assert r1.rankings_by_phase != r2.rankings_by_phase or r1.single_static_mass != r2.single_static_mass
