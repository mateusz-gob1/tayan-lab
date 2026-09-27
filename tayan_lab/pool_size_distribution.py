"""Exact distribution of the pool size N over the course of a game.

State: a sorted tuple of the active players' card counts (order doesn't matter, so we sort
to merge equivalent states). Each round the "loser" gains one card and, on reaching
`elimination_limit`, is removed. The total

    points(state) = sum(state) + elimination_limit * (players - len(state))

increases by exactly 1 every round and never decreases, so the chain has no cycles: states
reachable after `t` rounds all share the same `points`, and we can propagate probability
forward wave by wave (all arithmetic on exact `Fraction`s) instead of solving equations.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Callable, NamedTuple

State = tuple[int, ...]
LoserModel = Callable[[State], list[Fraction]]


def loser_weights_uniform(state: State) -> list[Fraction]:
    """Every active player is equally likely to lose the round."""
    n = len(state)
    return [Fraction(1, n)] * n


def loser_weights_inverse_cards(state: State) -> list[Fraction]:
    """A player with more cards loses less often: weight proportional to 1/cards."""
    inv = [Fraction(1, c) for c in state]
    total = sum(inv)
    return [w / total for w in inv]


def loser_weights_fewest_cards(state: State) -> list[Fraction]:
    """The player(s) with the fewest cards always lose (split evenly on a tie)."""
    lowest = min(state)
    holders = [i for i, c in enumerate(state) if c == lowest]
    weights = [Fraction(0)] * len(state)
    for i in holders:
        weights[i] = Fraction(1, len(holders))
    return weights


LOSER_MODELS: dict[str, LoserModel] = {
    "uniform": loser_weights_uniform,
    "inverse_cards": loser_weights_inverse_cards,
    "fewest_cards": loser_weights_fewest_cards,
}


class PoolDistribution(NamedTuple):
    expected_rounds: Fraction
    """Expected total number of rounds played in a game."""
    by_pool_size: dict[int, Fraction]
    """N -> expected number of rounds with that many cards in the pool."""
    by_phase: dict[int, dict[int, Fraction]]
    """active players -> {N -> expected number of rounds}."""


def _next_state(state: State, loser_index: int, elimination_limit: int) -> State:
    cards = list(state)
    cards[loser_index] += 1
    if cards[loser_index] >= elimination_limit:
        del cards[loser_index]
    return tuple(sorted(cards))


def _validate(players: int, starting_cards: int, elimination_limit: int) -> None:
    if players < 2:
        raise ValueError("need at least 2 players")
    if starting_cards >= elimination_limit:
        raise ValueError("starting_cards must be below elimination_limit")


def iter_game_transitions(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    loser_model: str | LoserModel = "uniform",
):
    """Yield `(state, prob, edges)` for every reachable non-terminal state, wave by wave
    (a round is played in every non-terminal state). `prob` is the probability of a game
    being in `state` at the start of that round; `edges` lists `(next_state, edge_prob)`
    for the transitions that keep the game going (the remaining probability mass ends the
    game there). Shared by `pool_size_distribution` and the ranking-cost measures in
    `rankings.py`, so both walk the exact same chain.
    """
    _validate(players, starting_cards, elimination_limit)
    weight_fn = LOSER_MODELS[loser_model] if isinstance(loser_model, str) else loser_model

    frontier: dict[State, Fraction] = {tuple([starting_cards] * players): Fraction(1)}
    while frontier:
        next_frontier: dict[State, Fraction] = defaultdict(Fraction)
        for state, prob in frontier.items():
            edges: list[tuple[State, Fraction]] = []
            for i, w in enumerate(weight_fn(state)):
                if w == 0:
                    continue
                new_state = _next_state(state, i, elimination_limit)
                if len(new_state) > 1:
                    edges.append((new_state, w))
                    next_frontier[new_state] += prob * w
            yield state, prob, edges
        frontier = dict(next_frontier)


def pool_size_distribution(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    loser_model: str | LoserModel = "uniform",
) -> PoolDistribution:
    """Exact distribution of the pool size N across a game, via forward propagation.

    `elimination_limit` is the card count that knocks a player out: they can hold at most
    `elimination_limit - 1` cards and stay in.
    """
    by_pool_size: dict[int, Fraction] = defaultdict(Fraction)
    by_phase: dict[int, dict[int, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
    finished = Fraction(0)

    for state, prob, edges in iter_game_transitions(players, starting_cards, elimination_limit, loser_model):
        n = sum(state)
        by_pool_size[n] += prob
        by_phase[len(state)][n] += prob
        finished += prob * (1 - sum((w for _, w in edges), Fraction(0)))

    assert finished == 1, f"probabilities must sum to 1, got {finished}"
    expected_rounds = sum(by_pool_size.values(), Fraction(0))
    return PoolDistribution(
        expected_rounds=expected_rounds,
        by_pool_size=dict(by_pool_size),
        by_phase={k: dict(v) for k, v in by_phase.items()},
    )
