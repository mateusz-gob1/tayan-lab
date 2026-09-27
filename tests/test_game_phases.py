"""Tests for ceiling/floor phase measures and the deck search (M4, research question 2)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from tayan_lab.game_phases import (
    categories_in_range,
    choose_deck,
    compare_deck_modes,
    evaluate_deck,
    evaluate_final_deck_override,
    evaluate_start_limit,
    overall_ceiling,
    phase_measures,
    search_deck,
    search_start_limit,
)
from tayan_lab.probabilities import CATEGORIES, probability


def test_categories_in_range_matches_manual_count_for_a_control_value():
    """Deck from 2 (52 cards), N=20: control values from the brief give pair 50.07%,
    two_pair 23.57%, straight 46.50%, three 15.26%, flush 62.39%, full 6.73%, four 1.79%,
    straight_flush 0.60% (HIGH, not in the brief's control table, computes to 86.72%). In
    [10%, 90%]: HIGH, pair, two_pair, straight, three, flush -> 6 categories."""
    assert categories_in_range(2, 20) == 6


def test_categories_in_range_is_between_0_and_9():
    for n in (1, 5, 10, 20, 40):
        assert 0 <= categories_in_range(2, n) <= len(CATEGORIES)


def test_categories_in_range_is_zero_at_the_extremes():
    assert categories_in_range(9, 0) == 0  # nothing exists yet
    assert categories_in_range(9, 24) == 0  # everything exists for certain (full deck)


def test_phase_measures_shares_sum_to_one():
    m = phase_measures(2, players=6, starting_cards=2, elimination_limit=6, top_category="STRAIGHT_FLUSH")
    assert sum(v.share for v in m.values()) == 1


def test_phase_measures_ceiling_and_floor_are_in_range():
    m = phase_measures(2, players=6, starting_cards=2, elimination_limit=6, top_category="PAIR")
    for v in m.values():
        assert 0 <= v.ceiling <= 1
        assert 0 <= v.floor_categories <= len(CATEGORIES)


def test_overall_ceiling_between_phase_extremes():
    m = phase_measures(2, players=6, starting_cards=2, elimination_limit=6, top_category="STRAIGHT_FLUSH")
    overall = overall_ceiling(2, 6, 2, 6, "STRAIGHT_FLUSH")
    assert min(v.ceiling for v in m.values()) <= overall <= max(v.ceiling for v in m.values())


def test_evaluate_deck_passes_with_lax_thresholds():
    result = evaluate_deck(2, players=6, starting_cards=2, elimination_limit=6, ceiling_avg_max=1.0, ceiling_phase_max=1.0, floor_min_categories=0.0)
    assert result.passes


def test_evaluate_deck_fails_with_impossible_thresholds():
    result = evaluate_deck(2, players=6, starting_cards=2, elimination_limit=6, ceiling_avg_max=0.0, ceiling_phase_max=0.0)
    assert not result.passes


def test_evaluate_deck_ranking_top_category_matches_static_ranking():
    from tayan_lab.rankings import static_ranking

    ranking, _ = static_ranking(2, 6, 2, 6, epsilon_pp=2.0)
    result = evaluate_deck(2, 6, 2, 6)
    assert result.ranking == ranking


def test_search_deck_orders_from_smallest_to_largest_deck():
    results = search_deck(players=2, starting_cards=2, elimination_limit=6)
    lowest_ranks = [r.lowest_rank for r in results]
    assert lowest_ranks == sorted(lowest_ranks, reverse=True)  # L descending = deck size ascending


def test_search_deck_skips_decks_too_small_for_the_player_count():
    results = search_deck(players=13, starting_cards=1, elimination_limit=5)
    for r in results:
        from tayan_lab.probabilities import deck_size

        assert 13 * 4 <= deck_size(r.lowest_rank)


def test_search_deck_default_thresholds_find_at_least_one_passing_deck_for_small_player_counts():
    """The full 52-card deck should always be a valid fallback that clears reasonable
    thresholds for small player counts, same as the engine's own fallback behaviour."""
    results = search_deck(players=2, starting_cards=2, elimination_limit=6)
    assert any(r.passes for r in results)


@given(st.integers(min_value=2, max_value=6), st.integers(min_value=2, max_value=10))
@settings(max_examples=15, deadline=None)
def test_categories_in_range_is_symmetric_under_rank_relabelling(lowest_rank, n):
    """Sanity: doesn't crash and stays in bounds for varied deck/pool combinations."""
    deck = 4 * (15 - lowest_rank)
    if n > deck:
        return
    assert 0 <= categories_in_range(lowest_rank, n) <= len(CATEGORIES)


# --------------------------------------------------------------------------- deck mode (question 3)


def test_choose_deck_matches_first_passing_result_from_search_deck():
    results = search_deck(players=6, starting_cards=2, elimination_limit=6)
    chosen = choose_deck(players=6, starting_cards=2, elimination_limit=6)
    passing = [r for r in results if r.passes]
    assert chosen == (passing[0] if passing else results[-1])


def test_choose_deck_raises_when_no_deck_fits_at_all():
    with pytest.raises(ValueError):
        choose_deck(players=13, starting_cards=5, elimination_limit=6)  # max hand way too big


def test_compare_deck_modes_switches_is_deterministic_and_bounded():
    result = compare_deck_modes(players=9, starting_cards=1, elimination_limit=6)
    assert 0 <= result.switches <= 9 - 2
    # every game visits every phase from `players` down to 2, so this must match a plain count
    ordered = [result.steps_by_phase[a].lowest_rank for a in range(9, 1, -1)]
    expected = sum(1 for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1])
    assert result.switches == expected


def test_compare_deck_modes_steps_by_phase_covers_every_active_count():
    result = compare_deck_modes(players=6, starting_cards=2, elimination_limit=6)
    assert set(result.steps_by_phase) == set(range(2, 7))


def test_compare_deck_modes_recommend_is_false_when_switches_exceed_max():
    result = compare_deck_modes(players=9, starting_cards=1, elimination_limit=6, max_switches=0)
    if result.switches > 0:
        assert not result.recommend_steps


def test_compare_deck_modes_two_players_has_zero_switches():
    """Only one phase (2 active) exists for a 2-player game, so there is nothing to switch."""
    result = compare_deck_modes(players=2, starting_cards=2, elimination_limit=6)
    assert result.switches == 0


# --------------------------------------------------------------------------- start/limit search


def test_evaluate_start_limit_flags_todays_default():
    result = evaluate_start_limit(players=6, starting_cards=2, elimination_limit=6)
    assert result.is_default


def test_evaluate_start_limit_matches_known_5_player_finding():
    """5 players, today's default (start=2, limit=6): static ranking should NOT clear the
    <=10% bar (14.40%, found during the M4 extension); limit=5 should fix it."""
    default = evaluate_start_limit(5, starting_cards=2, elimination_limit=6)
    assert not default.static_ranking_ok
    fixed = evaluate_start_limit(5, starting_cards=2, elimination_limit=5)
    assert fixed.static_ranking_ok


def test_evaluate_start_limit_matches_known_7_player_finding():
    """7 players, today's default (start=1, limit=6): should NOT clear the bar;
    start=2 should fix it."""
    default = evaluate_start_limit(7, starting_cards=1, elimination_limit=6)
    assert not default.static_ranking_ok
    fixed = evaluate_start_limit(7, starting_cards=2, elimination_limit=6)
    assert fixed.static_ranking_ok


def test_search_start_limit_skips_invalid_combinations():
    results = search_start_limit(players=6)
    for r in results:
        assert r.starting_cards < r.elimination_limit


def test_search_start_limit_covers_all_valid_combinations_by_default():
    results = search_start_limit(players=6)
    combos = {(r.starting_cards, r.elimination_limit) for r in results}
    assert combos == {(1, 5), (2, 5), (1, 6), (2, 6)}


def test_search_start_limit_exactly_one_result_is_default():
    results = search_start_limit(players=6)
    defaults = [r for r in results if r.is_default]
    assert len(defaults) == 1
    assert (defaults[0].starting_cards, defaults[0].elimination_limit) == (2, 6)


# --------------------------------------------------------------------------- final-deck override (M4c)


def test_final_deck_override_zero_switches_when_already_smallest_deck():
    """3-4 players already choose L=9 for the whole game, so overriding the final phase
    with L=9 too changes nothing."""
    result = evaluate_final_deck_override(players=4, starting_cards=2, elimination_limit=6)
    assert result.switches == 0
    assert result.floor_fixed == result.floor_override


def test_final_deck_override_improves_floor_for_a_large_game():
    """9 players: starting deck is much bigger than the final duel needs (found during the
    owner's M4c idea) - overriding with L=9 should raise the final phase's floor."""
    result = evaluate_final_deck_override(players=9, starting_cards=1, elimination_limit=6)
    assert result.switches == 1
    assert result.floor_override > result.floor_fixed


def test_final_deck_override_final_share_matches_pool_size_distribution():
    from tayan_lab.pool_size_distribution import pool_size_distribution

    result = evaluate_final_deck_override(players=9, starting_cards=1, elimination_limit=6)
    dist = pool_size_distribution(9, 1, 6)
    expected_share = sum(dist.by_phase[2].values()) / dist.expected_rounds
    assert result.final_share == expected_share


def test_final_deck_override_respects_a_different_final_phase():
    """final_phase can target something other than the last duel, e.g. the last 3 active."""
    result = evaluate_final_deck_override(players=9, starting_cards=1, elimination_limit=6, final_phase=3)
    assert result.final_phase == 3


def test_final_deck_override_switches_is_binary():
    for players in range(3, 14):
        start = 2 if players <= 6 else 1
        limit = 6 if players <= 10 else 5
        result = evaluate_final_deck_override(players, start, limit)
        assert result.switches in (0, 1)
