"""Ceiling/floor measures per game phase (research question 2), and a deck search under this
study's own criteria (as opposed to the game engine's `DECK_TABLE`, chosen under a different
criterion - see `docs/decisions.md` D8's caveat and D9).
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

from .pool_size_distribution import LoserModel, pool_size_distribution
from .probabilities import CATEGORIES, deck_size, probability
from .rankings import CURRENT_ORDER, probabilities_weighted_by, rank_categories, static_ranking, weighted_inversion_mass

IN_RANGE_LOW = Fraction(1, 10)
IN_RANGE_HIGH = Fraction(9, 10)


def categories_in_range(lowest_rank: int, n: int) -> int:
    """How many of the 9 categories have p_c(D, N) in [10%, 90%] at this pool size."""
    return sum(1 for c in CATEGORIES if IN_RANGE_LOW <= probability(c, lowest_rank, n) <= IN_RANGE_HIGH)


class PhaseMeasures(NamedTuple):
    share: Fraction
    """Share of this game's rounds spent in this phase (active-player count)."""
    ceiling: Fraction
    """Weighted-average chance the top-ranked category exists, within this phase."""
    floor_categories: Fraction
    """Weighted-average count of categories with p in [10%, 90%], within this phase."""


def phase_measures(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    top_category: str,
    loser_model: str | LoserModel = "uniform",
) -> dict[int, PhaseMeasures]:
    """Ceiling and floor, broken down by game phase (active-player count)."""
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    out: dict[int, PhaseMeasures] = {}
    for phase, weights in dist.by_phase.items():
        total = sum(weights.values())
        ceiling = (
            sum((w * probability(top_category, lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total
        )
        floor = sum((w * categories_in_range(lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total
        out[phase] = PhaseMeasures(total / dist.expected_rounds, ceiling, floor)
    return out


def overall_ceiling(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    top_category: str,
    loser_model: str | LoserModel = "uniform",
) -> Fraction:
    """Chance the top-ranked category exists, averaged over the whole game (criterion 3's '<=10% average')."""
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    return (
        sum((r * probability(top_category, lowest_rank, n) for n, r in dist.by_pool_size.items()), Fraction(0))
        / dist.expected_rounds
    )


class DeckEvaluation(NamedTuple):
    lowest_rank: int
    ranking: tuple[str, ...]
    overall_ceiling: Fraction
    worst_phase_ceiling: Fraction
    dead_phases: tuple[int, ...]
    """Phases that are both >= min_phase_share of the game AND have < floor_min_categories
    categories in [10%, 90%] - the phases that fail the floor criterion."""
    passes: bool
    """True if this deck clears criteria 3 (ceiling) and 4 (dead phase) for this player count."""


def evaluate_deck(
    lowest_rank: int,
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
    ceiling_avg_max: float = 0.10,
    ceiling_phase_max: float = 0.20,
    floor_min_categories: float = 2.0,
    min_phase_share: float = 0.05,
) -> DeckEvaluation:
    """Check one candidate deck against this study's ceiling/floor criteria for one player count.

    The ranking used for "the top category" is this deck's own M3 static ranking (not the
    engine's `CURRENT_ORDER`), since M3 found the current order wrong in most cases.
    """
    ranking, _ = static_ranking(lowest_rank, players, starting_cards, elimination_limit, epsilon_pp, loser_model)
    top = ranking[-1]
    phases = phase_measures(lowest_rank, players, starting_cards, elimination_limit, top, loser_model)
    overall = overall_ceiling(lowest_rank, players, starting_cards, elimination_limit, top, loser_model)
    worst_phase = max((m.ceiling for m in phases.values()), default=Fraction(0))
    dead = tuple(
        phase
        for phase, m in phases.items()
        if float(m.share) >= min_phase_share and float(m.floor_categories) < floor_min_categories
    )
    passes = float(overall) <= ceiling_avg_max and float(worst_phase) <= ceiling_phase_max and not dead
    return DeckEvaluation(lowest_rank, ranking, overall, worst_phase, dead, passes)


def search_deck(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    candidate_lowest_ranks: list[int] | None = None,
    **kwargs,
) -> list[DeckEvaluation]:
    """Evaluate candidate decks from smallest (L=9, 24 cards) to largest (L=2, 52 cards),
    mirroring the engine's own `chooseDeck` preference for the smallest deck that clears its
    thresholds - except here the thresholds are this study's own (see `evaluate_deck`).
    Skips decks too small to hold every player at their max hand. Returns every evaluated
    candidate (not just the winner), so the caller can see the whole trade-off.
    """
    if candidate_lowest_ranks is None:
        candidate_lowest_ranks = list(range(9, 1, -1))
    max_pool = players * (elimination_limit - 1)
    return [
        evaluate_deck(L, players, starting_cards, elimination_limit, **kwargs)
        for L in candidate_lowest_ranks
        if max_pool <= deck_size(L)
    ]


def choose_deck(players: int, starting_cards: int, elimination_limit: int, **kwargs) -> DeckEvaluation:
    """The single deck this study's criteria would pick for `players` (smallest passing
    deck; falls back to the largest deck evaluated if none passes, same spirit as the
    engine's own fallback to the full 52-card deck)."""
    results = search_deck(players, starting_cards, elimination_limit, **kwargs)
    if not results:
        raise ValueError(f"no candidate deck fits {players} players at this elimination limit")
    return next((r for r in results if r.passes), results[-1])


class DeckModeComparison(NamedTuple):
    fixed: DeckEvaluation
    """The single deck chosen once, for the whole game (research question 3's 'stała')."""
    steps_by_phase: dict[int, DeckEvaluation]
    """active players -> the deck this study's criteria would choose if that were the
    starting player count ('stopnie': re-chosen at each elimination threshold)."""
    fixed_dead_phases: tuple[int, ...]
    steps_dead_phases: tuple[int, ...]
    floor_gain_lower_half: Fraction
    """Average (rounds-weighted) gain in categories-in-range from 'steps', across phases
    from ceil(players/2) active players down to the final 2 (criterion 5's 'phases from
    half of the players down to the final')."""
    switches: int
    """How many times the deck actually differs between two consecutive phases, going from
    the starting player count down to the final 2. Deterministic: every game visits every
    phase exactly once (players are eliminated one at a time), so this is not an expectation."""
    recommend_steps: bool
    """True if 'steps' clears criterion 5's bar over 'fixed': removes every dead phase, or
    gains >= 1 category on average in the lower half, at <= max_switches deck changes."""


def compare_deck_modes(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    max_switches: int = 2,
    loser_model: str | LoserModel = "uniform",
    **kwargs,
) -> DeckModeComparison:
    """Research question 3: deck held fixed for the whole game vs re-chosen at each phase.

    Pool-size distribution doesn't depend on deck under the uniform loser model (M2), so both
    modes share the same `pool_size_distribution(players, ...).by_phase`; only which deck (and
    hence which probabilities) applies within each phase differs.
    """
    fixed = choose_deck(players, starting_cards, elimination_limit, loser_model=loser_model, **kwargs)
    steps_by_phase = {
        a: choose_deck(a, starting_cards, elimination_limit, loser_model=loser_model, **kwargs)
        for a in range(2, players + 1)
    }

    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    min_phase_share = kwargs.get("min_phase_share", 0.05)
    floor_min_categories = kwargs.get("floor_min_categories", 2.0)

    fixed_top = fixed.ranking[-1]
    fixed_measures = phase_measures(fixed.lowest_rank, players, starting_cards, elimination_limit, fixed_top, loser_model)
    fixed_dead = tuple(
        a for a, m in fixed_measures.items() if float(m.share) >= min_phase_share and float(m.floor_categories) < floor_min_categories
    )

    steps_measures: dict[int, PhaseMeasures] = {}
    for a, weights in dist.by_phase.items():
        L, top = steps_by_phase[a].lowest_rank, steps_by_phase[a].ranking[-1]
        total = sum(weights.values())
        ceiling = sum((w * probability(top, L, n) for n, w in weights.items()), Fraction(0)) / total
        floor = sum((w * categories_in_range(L, n) for n, w in weights.items()), Fraction(0)) / total
        steps_measures[a] = PhaseMeasures(total / dist.expected_rounds, ceiling, floor)
    steps_dead = tuple(
        a for a, m in steps_measures.items() if float(m.share) >= min_phase_share and float(m.floor_categories) < floor_min_categories
    )

    half = -(-players // 2)  # ceil(players / 2)
    lower_half_phases = [a for a in steps_measures if a <= half]
    gain_weight = sum(steps_measures[a].share for a in lower_half_phases)
    floor_gain = (
        sum(
            (steps_measures[a].share) * (steps_measures[a].floor_categories - fixed_measures[a].floor_categories)
            for a in lower_half_phases
        )
        / gain_weight
        if gain_weight > 0
        else Fraction(0)
    )

    ordered = [steps_by_phase[a].lowest_rank for a in range(players, 1, -1)]
    switches = sum(1 for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1])

    removes_dead = len(fixed_dead) > 0 and len(steps_dead) == 0
    recommend = switches <= max_switches and (removes_dead or float(floor_gain) >= 1.0)

    return DeckModeComparison(fixed, steps_by_phase, fixed_dead, steps_dead, floor_gain, switches, recommend)


class StartLimitEvaluation(NamedTuple):
    starting_cards: int
    elimination_limit: int
    deck: DeckEvaluation
    """This combination's own deck, chosen by `choose_deck` under M4's criteria."""
    static_ranking_ok: bool
    """True if the static ranking (M3) leaves <= `mass_ratio_max` of the current order's
    weighted inversion mass for this exact (starting_cards, elimination_limit, deck)."""
    mass_ratio: Fraction
    """Static ranking's inversion mass as a fraction of the current order's (criterion 2)."""
    expected_rounds: Fraction
    """Expected game length (M2), for the owner to weigh against a ranking-quality gain."""
    is_default: bool
    """True if this is the game's actual default (starting_cards, elimination_limit) for
    this player count (2 cards up to 6 players, 1 from 7; limit 6 up to 10, 5 from 11)."""


def _is_default_start_limit(players: int, starting_cards: int, elimination_limit: int) -> bool:
    default_start = 2 if players <= 6 else 1
    default_limit = 6 if players <= 10 else 5
    return starting_cards == default_start and elimination_limit == default_limit


def evaluate_start_limit(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    epsilon_pp: float = 2.0,
    mass_ratio_max: float = 0.10,
    loser_model: str | LoserModel = "uniform",
    **deck_kwargs,
) -> StartLimitEvaluation:
    """Check one (starting_cards, elimination_limit) combination: does the resulting deck
    (chosen under M4's criteria) let the static ranking (M3) clear criterion 2's <= 10% bar?
    """
    deck = choose_deck(players, starting_cards, elimination_limit, epsilon_pp=epsilon_pp, loser_model=loser_model, **deck_kwargs)
    m_current = weighted_inversion_mass(CURRENT_ORDER, deck.lowest_rank, players, starting_cards, elimination_limit, epsilon_pp, loser_model)
    m_static = weighted_inversion_mass(deck.ranking, deck.lowest_rank, players, starting_cards, elimination_limit, epsilon_pp, loser_model)
    ratio = m_static / m_current if m_current > 0 else Fraction(0)
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    return StartLimitEvaluation(
        starting_cards, elimination_limit, deck, float(ratio) <= mass_ratio_max, ratio, dist.expected_rounds,
        _is_default_start_limit(players, starting_cards, elimination_limit),
    )


def search_start_limit(
    players: int,
    candidate_starts: tuple[int, ...] = (1, 2),
    candidate_limits: tuple[int, ...] = (5, 6),
    **kwargs,
) -> list[StartLimitEvaluation]:
    """Evaluate every valid (starting_cards, elimination_limit) combination for `players`
    (skips combinations where starting_cards >= elimination_limit, or no deck fits).
    """
    out = []
    for limit in candidate_limits:
        for start in candidate_starts:
            if start >= limit:
                continue
            if players * (limit - 1) > deck_size(2):
                continue
            out.append(evaluate_start_limit(players, start, limit, **kwargs))
    return out


class FinalDeckOverride(NamedTuple):
    fixed: DeckEvaluation
    """The whole-game deck (chosen once, for the full starting player count)."""
    final_phase: int
    """Which active-player-count phase counts as 'the final' (default: 2, the last duel)."""
    final_share: Fraction
    """Share of this game's rounds spent in the final phase."""
    floor_fixed: Fraction
    """Categories-in-range in the final phase, using the whole-game deck."""
    floor_override: Fraction
    """Categories-in-range in the final phase, using `final_lowest_rank` instead."""
    ceiling_fixed: Fraction
    """Top category's chance in the final phase, using the whole-game deck."""
    ceiling_override: Fraction
    """Top category's chance in the final phase, using `final_lowest_rank` instead."""
    switches: int
    """0 if the whole-game deck already equals the override deck, else 1 - this mode never
    needs more than a single switch, unlike full 'steps' (M4's compare_deck_modes)."""


def evaluate_final_deck_override(
    players: int,
    starting_cards: int,
    elimination_limit: int,
    final_lowest_rank: int = 9,
    final_phase: int = 2,
    epsilon_pp: float = 2.0,
    loser_model: str | LoserModel = "uniform",
    **deck_kwargs,
) -> FinalDeckOverride:
    """One-switch deck mode: play the whole game on the deck chosen for the starting player
    count, but always finish the `final_phase`-active-player phase (default: the last duel)
    on `final_lowest_rank` (default: 9, the smallest deck), regardless of how many players
    started. Owner's idea: large games leave few cards for a small final phase, so the
    starting deck is oversized there no matter which deck the full game began on.
    """
    fixed = choose_deck(players, starting_cards, elimination_limit, epsilon_pp=epsilon_pp, loser_model=loser_model, **deck_kwargs)
    dist = pool_size_distribution(players, starting_cards, elimination_limit, loser_model)
    weights = dist.by_phase[final_phase]
    total = sum(weights.values())
    share = total / dist.expected_rounds

    floor_fixed = sum((w * categories_in_range(fixed.lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total
    floor_override = sum((w * categories_in_range(final_lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total

    top_fixed = fixed.ranking[-1]
    top_override = rank_categories(probabilities_weighted_by(final_lowest_rank, weights), epsilon_pp, base_order=CURRENT_ORDER)[-1]
    ceiling_fixed = sum((w * probability(top_fixed, fixed.lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total
    ceiling_override = sum((w * probability(top_override, final_lowest_rank, n) for n, w in weights.items()), Fraction(0)) / total

    switches = 0 if fixed.lowest_rank == final_lowest_rank else 1
    return FinalDeckOverride(fixed, final_phase, share, floor_fixed, floor_override, ceiling_fixed, ceiling_override, switches)
