"""Category rankings: current, static (once per game) and dynamic (once per round), plus
the measures used to compare them (weighted inversion mass, Kendall's tau, and the cost of
the dynamic ranking in ranking changes per game).
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import cmp_to_key
from typing import Mapping, NamedTuple

from .pool_size_distribution import LOSER_MODELS, LoserModel, State, _next_state, iter_game_transitions, pool_size_distribution
from .probabilities import CATEGORIES, deck_size, probability

#: The game's declared order today (`DEFAULT_CATEGORY_ORDER` in the engine), weakest first.
CURRENT_ORDER: tuple[str, ...] = (
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

ProbabilityMap = Mapping[str, Fraction]


def _check_deck_fits(lowest_rank: int, players: int, elimination_limit: int) -> None:
    """The largest possible pool (every player holding the max hand) must fit the deck,
    same guard as the engine's `chooseDeck` (`maxHand * players <= deckSize`)."""
    max_pool = players * (elimination_limit - 1)
    deck = deck_size(lowest_rank)
    if max_pool > deck:
        raise ValueError(
            f"deck of {deck} cards (lowest rank {lowest_rank}) is too small for {players} "
            f"players at up to {elimination_limit - 1} cards each (needs >= {max_pool})"
        )


def rank_categories(p: ProbabilityMap, epsilon_pp: float, base_order: tuple[str, ...] = CURRENT_ORDER) -> tuple[str, ...]:
    """Sort categories from most common (weakest, first) to rarest (strongest, last).

    Two categories within `epsilon_pp` percentage points of each other are a tie and keep
    the relative order they have in `base_order`, so the ranking does not reorder on noise.
    """
    eps = epsilon_pp / 100
    base_index = {c: i for i, c in enumerate(base_order)}

    def cmp(a: str, b: str) -> int:
        if abs(float(p[a]) - float(p[b])) <= eps:
            return base_index[a] - base_index[b]
        return -1 if p[a] > p[b] else 1

    return tuple(sorted(CATEGORIES, key=cmp_to_key(cmp)))


def dynamic_ranking(
    lowest_rank: int, n: int, epsilon_pp: float = 2.0, base_order: tuple[str, ...] = CURRENT_ORDER
) -> tuple[tuple[str, ...], dict[str, Fraction]]:
    """Ranking for a single round: `p_c(D, N)` for the deck's actual pool size N."""
    p = {c: probability(c, lowest_rank, n) for c in CATEGORIES}
    return rank_categories(p, epsilon_pp, base_order), p


def probabilities_weighted_by(lowest_rank: int, weights: Mapping[int, Fraction]) -> dict[str, Fraction]:
    """p_c averaged over an arbitrary N -> weight mapping (need not sum to 1; normalized here)."""
    total = sum(weights.values())
    return {
        c: sum((w * probability(c, lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total
        for c in CATEGORIES
    }


def weighted_probabilities(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    loser_model: str | LoserModel = "uniform",
) -> dict[str, Fraction]:
    """p_c averaged over a whole game, weighted by the pool-size distribution w(N) from M2."""
    _check_deck_fits(lowest_rank, players, elimination_limit)
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    return {
        c: sum(
            (rounds * probability(c, lowest_rank, n) for n, rounds in dist.by_pool_size.items()),
            Fraction(0),
        )
        / dist.expected_rounds
        for c in CATEGORIES
    }


def static_ranking(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
    base_order: tuple[str, ...] = CURRENT_ORDER,
) -> tuple[tuple[str, ...], dict[str, Fraction]]:
    """Ranking fixed once per game: `p_c` averaged over the whole game's pool sizes."""
    p = weighted_probabilities(lowest_rank, players, starting_cards, elimination_limit, loser_model)
    return rank_categories(p, epsilon_pp, base_order), p


def inversion_mass(ranking: tuple[str, ...], p: ProbabilityMap, epsilon_pp: float = 2.0) -> Fraction:
    """Sum of `(p_higher - p_lower)` over every pair of categories placed so that the one
    ranked higher is, in fact, more common by more than `epsilon_pp` (an inversion).
    """
    eps = epsilon_pp / 100
    total = Fraction(0)
    for i, low in enumerate(ranking):
        for high in ranking[i + 1 :]:
            diff = p[high] - p[low]
            if diff > eps:
                total += diff
    return total


def weighted_inversion_mass(
    ranking: tuple[str, ...],
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
) -> Fraction:
    """Inversion mass of `ranking`, averaged over the game's pool-size distribution w(N)."""
    _check_deck_fits(lowest_rank, players, elimination_limit)
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    total = Fraction(0)
    for n, rounds in dist.by_pool_size.items():
        p = {c: probability(c, lowest_rank, n) for c in CATEGORIES}
        total += rounds * inversion_mass(ranking, p, epsilon_pp)
    return total / dist.expected_rounds


def kendall_tau(a: tuple[str, ...], b: tuple[str, ...]) -> Fraction:
    """Kendall's tau between two total orders over the same categories: 1 = identical
    order, -1 = fully reversed, 0 = as many agreeing as disagreeing pairs."""
    if set(a) != set(b):
        raise ValueError("both rankings must contain the same categories")
    pos_a = {c: i for i, c in enumerate(a)}
    pos_b = {c: i for i, c in enumerate(b)}
    concordant = discordant = 0
    items = list(a)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            agree = (pos_a[x] - pos_a[y] > 0) == (pos_b[x] - pos_b[y] > 0)
            if agree:
                concordant += 1
            else:
                discordant += 1
    return Fraction(concordant - discordant, concordant + discordant)


def expected_ranking_changes(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
    hysteresis: bool = True,
) -> Fraction:
    """Expected number of times the dynamic ranking's order changes between consecutive
    rounds of a game (the per-game cost of the dynamic-ranking mode).

    Without hysteresis: ties always break by the fixed `CURRENT_ORDER`, so the ranking is a
    pure function of N; we compare the ranking of consecutive rounds along every path.

    With hysteresis: ties break by whatever order was already in force, so a category only
    overtakes another once the gap exceeds epsilon (not just crosses it by a hair). This
    needs the order-in-force as part of the state we propagate, since it depends on the
    whole path, not just the current N.
    """
    _check_deck_fits(lowest_rank, players, elimination_limit)
    if not hysteresis:
        cache: dict[int, tuple[str, ...]] = {}

        def order_at(n: int) -> tuple[str, ...]:
            if n not in cache:
                cache[n] = dynamic_ranking(lowest_rank, n, epsilon_pp)[0]
            return cache[n]

        total = Fraction(0)
        for state, prob, edges in iter_game_transitions(players, starting_cards, elimination_limit, loser_model):
            order_here = order_at(sum(state))
            for next_state, edge_prob in edges:
                if order_here != order_at(sum(next_state)):
                    total += prob * edge_prob
        return total

    weight_fn = LOSER_MODELS[loser_model] if isinstance(loser_model, str) else loser_model
    initial_state: State = tuple([starting_cards] * players)
    initial_order = dynamic_ranking(lowest_rank, sum(initial_state), epsilon_pp, base_order=CURRENT_ORDER)[0]

    total = Fraction(0)
    frontier: dict[tuple[State, tuple[str, ...]], Fraction] = {(initial_state, initial_order): Fraction(1)}
    while frontier:
        next_frontier: dict[tuple[State, tuple[str, ...]], Fraction] = defaultdict(Fraction)
        for (state, order), prob in frontier.items():
            for i, w in enumerate(weight_fn(state)):
                if w == 0:
                    continue
                new_state = _next_state(state, i, elimination_limit)
                if len(new_state) <= 1:
                    continue
                p2 = prob * w
                p_n = {c: probability(c, lowest_rank, sum(new_state)) for c in CATEGORIES}
                new_order = rank_categories(p_n, epsilon_pp, base_order=order)
                if new_order != order:
                    total += p2
                next_frontier[(new_state, new_order)] += p2
        frontier = dict(next_frontier)
    return total


class StagedRankingResult(NamedTuple):
    single_static_mass: Fraction
    """Weighted inversion mass of the single, whole-game static ranking (baseline)."""
    staged_mass: Fraction
    """Weighted inversion mass when each game phase (active-player count) uses its own
    phase-optimal ranking instead of the single whole-game one."""
    switches: int
    """How many times the ranking actually differs between two consecutive phases, going
    from the starting player count down to the final 2 (the real cost of staging: at most
    `players - 2`, often far fewer, since neighbouring phases frequently agree)."""
    rankings_by_phase: dict[int, tuple[str, ...]]
    """active players -> that phase's own locally-optimal ranking."""


def staged_ranking(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
    base_order: tuple[str, ...] | None = None,
) -> StagedRankingResult:
    """Decision-support analysis (not a recommended mode by itself): what if the ranking
    were fixed once per game PHASE (active-player count) instead of once per game? Ties
    within each phase break by `base_order` (default: the single whole-game static ranking),
    so a phase only diverges when its own evidence is strong enough - not by noise.

    This measures an upper bound on what a small number of mid-game "steps" could achieve,
    using thresholds (active-player count) the game already has, without committing to
    building that mode. Compare `staged_mass` to `single_static_mass` for the benefit, and
    `switches` to `expected_ranking_changes` (the fully dynamic mode's cost) for the price.
    """
    _check_deck_fits(lowest_rank, players, elimination_limit)
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    if base_order is None:
        base_order, _ = static_ranking(lowest_rank, players, starting_cards, elimination_limit, epsilon_pp, loser_model)

    single_mass = Fraction(0)
    for n, rounds in dist.by_pool_size.items():
        p = {c: probability(c, lowest_rank, n) for c in CATEGORIES}
        single_mass += rounds * inversion_mass(base_order, p, epsilon_pp)
    single_mass /= dist.expected_rounds

    rankings_by_phase: dict[int, tuple[str, ...]] = {}
    staged_mass = Fraction(0)
    prev_ranking: tuple[str, ...] | None = None
    switches = 0
    for phase in sorted(dist.by_phase, reverse=True):
        weights = dist.by_phase[phase]
        p_phase = probabilities_weighted_by(lowest_rank, weights)
        ranking = rank_categories(p_phase, epsilon_pp, base_order=base_order)
        rankings_by_phase[phase] = ranking
        if prev_ranking is not None and ranking != prev_ranking:
            switches += 1
        prev_ranking = ranking
        for n, rounds in weights.items():
            p = {c: probability(c, lowest_rank, n) for c in CATEGORIES}
            staged_mass += rounds * inversion_mass(ranking, p, epsilon_pp)
    staged_mass /= dist.expected_rounds

    return StagedRankingResult(single_mass, staged_mass, switches, rankings_by_phase)
