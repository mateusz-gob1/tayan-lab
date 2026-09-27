"""Slow, obviously-correct reference implementation of the game rules.

A direct Python port of `existsInPool` and `generateDeclarations` from the game engine
(packages/engine/src/pool.ts, declarations.ts). It exists only to check the closed-form
formulas in `probabilities.py`; it is never used to produce results.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Iterator, NamedTuple

SUITS = ("C", "D", "H", "S")
ACE = 14

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

Card = tuple[int, str]  # (rank 2..14, suit)


class Declaration(NamedTuple):
    category: str
    ranks: tuple[int, ...] = ()
    suit: str | None = None


    @property
    def id(self) -> str:
        """Engine-style id, e.g. 'FULL:12:9', 'FLUSH:H', 'STRAIGHT_FLUSH:14:S'."""
        parts = [self.category, *map(str, self.ranks)]
        if self.suit:
            parts.append(self.suit)
        return ":".join(parts)


def build_deck(lowest_rank: int) -> list[Card]:
    return [(r, s) for r in range(lowest_rank, ACE + 1) for s in SUITS]


def run_of(top: int) -> range:
    return range(top - 4, top + 1)


class _Pool(NamedTuple):
    by_rank: Counter
    by_suit: Counter
    has: frozenset


def _summarise(cards: Iterable[Card]) -> _Pool:
    cards = list(cards)
    return _Pool(
        Counter(r for r, _ in cards), Counter(s for _, s in cards), frozenset(cards)
    )


def _exists(decl: Declaration, pool: _Pool) -> bool:
    by_rank, by_suit, has = pool
    r0 = decl.ranks[0] if decl.ranks else None
    r1 = decl.ranks[1] if len(decl.ranks) > 1 else None
    c = decl.category
    if c == "HIGH":
        return by_rank[r0] >= 1
    if c == "PAIR":
        return by_rank[r0] >= 2
    if c == "THREE":
        return by_rank[r0] >= 3
    if c == "FOUR":
        return by_rank[r0] >= 4
    if c == "TWO_PAIR":
        return by_rank[r0] >= 2 and by_rank[r1] >= 2
    if c == "FULL":
        return by_rank[r0] >= 3 and by_rank[r1] >= 2
    if c == "STRAIGHT":
        return all(by_rank[r] >= 1 for r in run_of(r0))
    if c == "FLUSH":
        return by_suit[decl.suit] >= 5
    if c == "STRAIGHT_FLUSH":
        return all((r, decl.suit) in has for r in run_of(r0))
    raise ValueError(f"unknown category {c!r}")


def exists_in_pool(decl: Declaration, cards: Iterable[Card]) -> bool:
    """Does the declared hand occur in the pool? (inclusive counting, like the engine)"""
    return _exists(decl, _summarise(cards))


def declarations(lowest_rank: int) -> list[Declaration]:
    """All declarations for a deck, mirroring the engine's `draftsFor`."""
    ranks = list(range(lowest_rank, ACE + 1))
    out: list[Declaration] = []
    for r in ranks:
        out += [Declaration("HIGH", (r,)), Declaration("PAIR", (r,))]
        out += [Declaration("THREE", (r,)), Declaration("FOUR", (r,))]
    out += [Declaration("TWO_PAIR", (a, b)) for a in ranks for b in ranks if a > b]
    out += [Declaration("FULL", (a, b)) for a in ranks for b in ranks if a != b]
    out += [Declaration("STRAIGHT", (t,)) for t in ranks if t - 4 >= lowest_rank]
    out += [Declaration("FLUSH", (), s) for s in SUITS]
    out += [
        Declaration("STRAIGHT_FLUSH", (t,), s)
        for t in ranks
        if t - 4 >= lowest_rank
        for s in SUITS
    ]
    return out


def count_pools(lowest_rank: int, n: int) -> tuple[int, dict[Declaration, int]]:
    """Enumerate every pool of `n` cards; return (number of pools, hits per declaration)."""
    deck = build_deck(lowest_rank)
    decls = declarations(lowest_rank)
    hits: dict[Declaration, int] = {d: 0 for d in decls}
    total = 0
    for pool in combinations(deck, n):
        total += 1
        summary = _summarise(pool)
        for d in decls:
            if _exists(d, summary):
                hits[d] += 1
    return total, hits


def representative(category: str, lowest_rank: int) -> Declaration:
    """One declaration per category (all are equally likely by symmetry): top ranks, spades."""
    for d in reversed(declarations(lowest_rank)):
        if d.category == category:
            return d
    raise ValueError(f"no {category} declaration for lowest rank {lowest_rank}")


def iter_pools(lowest_rank: int, n: int) -> Iterator[tuple[Card, ...]]:
    return combinations(build_deck(lowest_rank), n)


def simulate_rounds(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    games: int,
    rng,
    loser_model: str = "uniform",
) -> tuple[list[int], list[int]]:
    """Monte Carlo port of the engine's `simulateRates` game loop (loser-picking only, no
    cards dealt). Returns (pool size N of every round played, active-player count of every
    round played), across `games` independent games."""
    pool_sizes: list[int] = []
    phases: list[int] = []
    for _ in range(games):
        counts = [starting_cards] * players
        alive = list(range(players))
        while len(alive) > 1:
            total = sum(counts[i] for i in alive)
            pool_sizes.append(total)
            phases.append(len(alive))
            if loser_model == "uniform":
                loser = alive[rng.randrange(len(alive))]
            elif loser_model == "inverse_cards":
                weights = [1 / counts[i] for i in alive]
                loser = rng.choices(alive, weights=weights, k=1)[0]
            elif loser_model == "fewest_cards":
                lowest = min(counts[i] for i in alive)
                holders = [i for i in alive if counts[i] == lowest]
                loser = holders[rng.randrange(len(holders))]
            else:
                raise ValueError(f"unknown loser model {loser_model!r}")
            counts[loser] += 1
            if counts[loser] >= elimination_limit:
                alive.remove(loser)
    return pool_sizes, phases

