"""The Python rules must agree 100% with the game engine (TypeScript `existsInPool`).

Fixtures come from `pnpm engine:export-fixtures` in the game repo (branch
feat/engine-export-fixtures): random pools plus the ids of every declaration that exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tayan_lab.reference import declarations, exists_in_pool

FIXTURE = Path(__file__).parent / "fixtures" / "engine_pools.json"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="engine fixtures not exported yet")


@pytest.fixture(scope="module")
def data():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_declaration_lists_match_the_engine(data):
    for lowest_rank, engine_ids in data["declarations"].items():
        ours = [d.id for d in declarations(int(lowest_rank))]
        assert sorted(ours) == sorted(engine_ids), lowest_rank
        assert len(ours) == len(set(ours))


def test_exists_in_pool_agrees_with_the_engine_on_every_declaration(data):
    checked = 0
    for pool in data["pools"]:
        cards = [(rank, suit) for rank, suit in pool["cards"]]
        engine_exists = set(pool["exists"])
        for decl in declarations(pool["lowestRank"]):
            assert exists_in_pool(decl, cards) == (decl.id in engine_exists), (decl.id, cards)
            checked += 1
    assert checked == 82_600  # 500 pools x declarations per deck; guards against a truncated export


def test_fixtures_contain_a_useful_mix_of_outcomes(data):
    """Guard against a degenerate export (e.g. all pools empty)."""
    seen_categories = {i.split(":")[0] for pool in data["pools"] for i in pool["exists"]}
    assert {"HIGH", "PAIR", "TWO_PAIR", "STRAIGHT", "THREE", "FLUSH", "FULL", "FOUR"} <= seen_categories
