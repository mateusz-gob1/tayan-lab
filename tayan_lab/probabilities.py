"""Exact probabilities that a declaration exists in a random pool.

Setting: a deck of `R = 15 - L` ranks (L = lowest rank, 2..10 for straights) times 4 suits,
`D = 4R` cards. A pool is `N` cards drawn without replacement. All results are exact
`Fraction`s built from integer binomial coefficients (no floating point, no simulation).

`probability(category, L, N)` is the chance that ONE declaration of that category exists.
By symmetry (ranks are interchangeable, suits are interchangeable) every declaration of a
category has the same chance, so one number per category is enough.

Counting is inclusive, like the game engine's `existsInPool`: "pair of aces" exists when the
pool holds at least two aces, even if it also holds three or four.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

CATEGORIES = (
    "HIGH",
    "PAIR",
    "TWO_PAIR",
    "STRAIGHT",
    "THREE",
    "FLUSH",
    "FULL",
    "FOUR",
    "STRAIGHT_FLUSH",
)

SUITS = 4
CARDS_PER_RANK = 4
STRAIGHT_LENGTH = 5
ACE = 14


def C(n: int, k: int) -> int:
    """Binomial coefficient that is 0 (instead of an error) outside 0 <= k <= n."""
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def deck_size(lowest_rank: int) -> int:
    return SUITS * (ACE + 1 - lowest_rank)


def _n_ranks(lowest_rank: int) -> int:
    return ACE + 1 - lowest_rank


def _same_rank_at_least(deck: int, n: int, at_least: int) -> int:
    """Pools in which >= `at_least` of the 4 cards of one given rank are present."""
    return sum(
        C(CARDS_PER_RANK, i) * C(deck - CARDS_PER_RANK, n - i)
        for i in range(at_least, CARDS_PER_RANK + 1)
    )


def _two_ranks(deck: int, n: int, at_least_a: int, at_least_b: int) -> int:
    """Pools with >= at_least_a cards of rank a AND >= at_least_b cards of rank b (a != b)."""
    rest = deck - 2 * CARDS_PER_RANK
    return sum(
        C(CARDS_PER_RANK, i) * C(CARDS_PER_RANK, j) * C(rest, n - i - j)
        for i in range(at_least_a, CARDS_PER_RANK + 1)
        for j in range(at_least_b, CARDS_PER_RANK + 1)
    )


def _straight(deck: int, n: int, ranks: int) -> int:
    """Pools with at least one card of each of 5 given ranks (inclusion-exclusion on missing ranks)."""
    if ranks < STRAIGHT_LENGTH:
        return 0
    return sum(
        (-1) ** j * comb(STRAIGHT_LENGTH, j) * C(deck - CARDS_PER_RANK * j, n)
        for j in range(STRAIGHT_LENGTH + 1)
    )


def _flush(deck: int, n: int, ranks: int) -> int:
    """Pools with >= 5 cards of one given suit (the suit has `ranks` cards)."""
    return sum(C(ranks, i) * C(deck - ranks, n - i) for i in range(STRAIGHT_LENGTH, ranks + 1))


def _straight_flush(deck: int, n: int, ranks: int) -> int:
    """Pools containing 5 given specific cards."""
    if ranks < STRAIGHT_LENGTH:
        return 0
    return C(deck - STRAIGHT_LENGTH, n - STRAIGHT_LENGTH)


def favorable(category: str, lowest_rank: int, n: int) -> int:
    """Number of N-card pools (out of C(D, N)) in which one declaration of `category` exists."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    ranks = _n_ranks(lowest_rank)
    deck = deck_size(lowest_rank)
    if not 0 <= n <= deck:
        raise ValueError(f"pool size {n} outside 0..{deck} for lowest rank {lowest_rank}")
    if category == "HIGH":
        return _same_rank_at_least(deck, n, 1)
    if category == "PAIR":
        return _same_rank_at_least(deck, n, 2)
    if category == "THREE":
        return _same_rank_at_least(deck, n, 3)
    if category == "FOUR":
        return _same_rank_at_least(deck, n, 4)
    if category == "TWO_PAIR":
        return _two_ranks(deck, n, 2, 2)
    if category == "FULL":
        return _two_ranks(deck, n, 3, 2)
    if category == "STRAIGHT":
        return _straight(deck, n, ranks)
    if category == "FLUSH":
        return _flush(deck, n, ranks)
    return _straight_flush(deck, n, ranks)


def probability(category: str, lowest_rank: int, n: int) -> Fraction:
    """Exact chance that one declaration of `category` exists in a random pool of `n` cards."""
    return Fraction(favorable(category, lowest_rank, n), comb(deck_size(lowest_rank), n))
