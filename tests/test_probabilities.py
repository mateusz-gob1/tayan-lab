"""Tests for the exact probabilities. Written before the formulas (test-first)."""

from __future__ import annotations

import math
import random
from fractions import Fraction
from functools import lru_cache

import pytest
from hypothesis import given, settings, strategies as st

from tayan_lab.probabilities import (
    CATEGORIES,
    deck_size,
    favorable,
    probability,
)
from tayan_lab.reference import _exists, _summarise, build_deck, count_pools, representative


# --------------------------------------------------------------------------- exhaustive


@lru_cache(maxsize=None)
def _enumerated(lowest_rank: int, n: int):
    return count_pools(lowest_rank, n)


# (lowest rank, max pool size). 16-, 20- and 24-card decks; the 24-card one stops at 5
# to keep the run short (its N = 6 case is covered by the control values and Monte Carlo).
EXHAUSTIVE = [(11, 6), (10, 6), (9, 5)]


@pytest.mark.parametrize("lowest_rank,max_n", EXHAUSTIVE)
def test_formulas_match_full_enumeration_of_all_pools(lowest_rank, max_n):
    """Every declaration, every pool: formula numerator == number of pools found by brute force."""
    for n in range(0, max_n + 1):
        total, hits = _enumerated(lowest_rank, n)
        assert total == math.comb(deck_size(lowest_rank), n)
        for decl, count in hits.items():
            assert favorable(decl.category, lowest_rank, n) == count, (decl, n)


@pytest.mark.parametrize("lowest_rank,max_n", EXHAUSTIVE)
def test_every_declaration_of_a_category_has_the_same_count(lowest_rank, max_n):
    """Symmetry of ranks and suits: within a category all declarations are equally likely."""
    for n in range(0, max_n + 1):
        _, hits = _enumerated(lowest_rank, n)
        per_category: dict[str, set[int]] = {}
        for decl, count in hits.items():
            per_category.setdefault(decl.category, set()).add(count)
        for category, counts in per_category.items():
            assert len(counts) == 1, (category, n, counts)


# --------------------------------------------------------------------------- Monte Carlo


@pytest.mark.parametrize(
    "lowest_rank,n",
    [(9, 6), (9, 10), (2, 20), (7, 12)],
)
def test_formulas_match_monte_carlo_within_4_standard_errors(lowest_rank, n):
    samples = 60_000
    rng = random.Random(20260927)
    deck = build_deck(lowest_rank)
    decls = {c: representative(c, lowest_rank) for c in CATEGORIES}
    hits = {c: 0 for c in CATEGORIES}
    for _ in range(samples):
        pool = rng.sample(deck, n)
        summary = _summarise(pool)
        for c, d in decls.items():
            if _exists(d, summary):
                hits[c] += 1
    for c in CATEGORIES:
        p = float(probability(c, lowest_rank, n))
        se = math.sqrt(max(p * (1 - p), 1e-12) / samples)
        assert abs(hits[c] / samples - p) < 4 * se, (c, p, hits[c] / samples)


# --------------------------------------------------------------------------- control values

# Precomputed reference values from the research brief (percent, two decimals).
CONTROL = [
    (9, 6, dict(PAIR=25.13, TWO_PAIR=3.80, STRAIGHT=8.75, THREE=3.53, FLUSH=0.08, FULL=0.30, FOUR=0.14, STRAIGHT_FLUSH=0.01)),
    (9, 10, dict(PAIR=56.32, TWO_PAIR=28.22, STRAIGHT=56.95, THREE=17.79, FLUSH=2.78, FULL=7.62, FOUR=1.98, STRAIGHT_FLUSH=0.59)),
    (2, 20, dict(PAIR=50.07, TWO_PAIR=23.57, STRAIGHT=46.50, THREE=15.26, FLUSH=62.39, FULL=6.73, FOUR=1.79, STRAIGHT_FLUSH=0.60)),
]


@pytest.mark.parametrize("lowest_rank,n,expected", CONTROL)
def test_control_values_from_the_brief(lowest_rank, n, expected):
    for category, percent in expected.items():
        assert round(float(probability(category, lowest_rank, n)) * 100, 2) == percent, category


# --------------------------------------------------------------------------- classical poker


def test_classical_five_card_poker_from_52_cards():
    """Known hand counts of 5-card poker. Our probabilities are for ONE declaration and
    count 'inclusively', so we multiply by the number of declarations and pick the
    categories whose classical count is not affected by overlaps at N = 5."""
    total = math.comb(52, 5)

    def hands(category, declarations):
        value = probability(category, 2, 5) * total * declarations
        assert value.denominator == 1
        return int(value)

    assert hands("FOUR", 13) == 624  # four of a kind
    assert hands("FULL", 13 * 12) == 3744  # full house
    # two pair; inclusive counting also lets every full house (3 + 2 cards) count as a two-pair
    assert hands("TWO_PAIR", 13 * 12 // 2) == 123552 + 3744
    assert hands("FLUSH", 4) == 5148  # flush, straight flushes included
    # Ace is high only in this game (no A-2-3-4-5), so 9 straights instead of the classical 10:
    # 36 straight flushes instead of 40, and 9 * 1024 straights.
    assert hands("STRAIGHT_FLUSH", 9 * 4) == 36
    assert hands("STRAIGHT", 9) == 9 * 4**5
    # 'three of a kind' here also covers hands with a full house or four of a kind inside.
    assert hands("THREE", 13) == 54912 + 3744 + 624

# --------------------------------------------------------------------------- properties

ranks = st.integers(min_value=2, max_value=10)  # decks with at least 5 ranks


@st.composite
def deck_and_pool(draw, min_n=0):
    lowest = draw(ranks)
    return lowest, draw(st.integers(min_value=min_n, max_value=deck_size(lowest)))


@given(deck_and_pool(), st.sampled_from(CATEGORIES))
def test_probability_is_between_0_and_1(dn, category):
    lowest, n = dn
    assert 0 <= probability(category, lowest, n) <= 1


@given(deck_and_pool(), st.sampled_from(CATEGORIES))
def test_probability_does_not_decrease_with_more_cards(dn, category):
    lowest, n = dn
    if n < deck_size(lowest):
        assert probability(category, lowest, n + 1) >= probability(category, lowest, n)


@given(deck_and_pool())
def test_stronger_hands_are_never_more_likely_than_the_hands_they_contain(dn):
    lowest, n = dn
    p = {c: probability(c, lowest, n) for c in CATEGORIES}
    assert p["FOUR"] <= p["THREE"] <= p["PAIR"] <= p["HIGH"]
    assert p["FULL"] <= p["THREE"]
    assert p["TWO_PAIR"] <= p["PAIR"]
    assert p["STRAIGHT_FLUSH"] <= p["FLUSH"]
    assert p["STRAIGHT_FLUSH"] <= p["STRAIGHT"]


@given(ranks, st.integers(min_value=0, max_value=4), st.sampled_from(["STRAIGHT", "FLUSH", "STRAIGHT_FLUSH"]))
def test_fewer_than_5_cards_cannot_make_straight_flush_or_straight(lowest, n, category):
    assert probability(category, lowest, n) == 0


@given(deck_and_pool())
def test_pool_of_the_whole_deck_contains_everything(dn):
    lowest, _ = dn
    for c in CATEGORIES:
        assert probability(c, lowest, deck_size(lowest)) == 1


def test_probabilities_are_exact_fractions():
    assert isinstance(probability("PAIR", 2, 10), Fraction)


def test_pool_larger_than_deck_is_rejected():
    with pytest.raises(ValueError):
        probability("PAIR", 9, 25)


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        probability("NOPE", 9, 5)

