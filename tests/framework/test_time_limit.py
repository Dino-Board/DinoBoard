"""`get_ai_action(time_limit_ms=...)`: a wall-clock budget for one search.

0 keeps the fixed-count search byte-identical; a positive budget stops at the
first sim boundary past the deadline, `simulations` stays a ceiling, at least
one sim always runs, and `stats["simulations"]` reports the sims that ran.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from tests.conftest import get_test_model

GAME = "6nimmit"
# Generous: a loaded CI box can be late to the first sim boundary.
SLACK_MS = 150


@pytest.fixture(scope="module")
def model_path():
    if GAME not in engine.available_games():
        pytest.skip(f"{GAME} not registered in this build")
    return get_test_model(GAME)


def _search(model_path, **kwargs):
    gs = engine.GameSession(GAME, seed=5, model_path=model_path, use_filter=False)
    started = time.monotonic()
    result = gs.get_ai_action(temperature=0.0, **kwargs)
    return result, (time.monotonic() - started) * 1000.0


@pytest.mark.parametrize("world_cap", [0, 8])
def test_zero_budget_is_the_fixed_count_search(model_path, world_cap):
    plain, _ = _search(model_path, simulations=120, world_cap=world_cap)
    zero, _ = _search(model_path, simulations=120, world_cap=world_cap, time_limit_ms=0.0)
    assert plain["stats"]["simulations"] == 120
    assert zero["action"] == plain["action"]
    assert zero["stats"]["root_action_visits"] == plain["stats"]["root_action_visits"]
    assert zero["stats"]["best_value"] == plain["stats"]["best_value"]


@pytest.mark.parametrize("world_cap", [0, 8])
@pytest.mark.parametrize("limit_ms", [50.0, 300.0])
def test_budget_stops_the_search_on_time(model_path, world_cap, limit_ms):
    result, elapsed = _search(model_path, simulations=10_000_000, world_cap=world_cap,
                              time_limit_ms=limit_ms)
    stats = result["stats"]
    assert limit_ms <= elapsed <= limit_ms + SLACK_MS, f"took {elapsed:.0f} ms for {limit_ms} ms"
    assert 0 < stats["simulations"] < 10_000_000
    assert sum(stats["root_action_visits"]) == stats["simulations"]


def test_the_ceiling_still_applies(model_path):
    result, elapsed = _search(model_path, simulations=40, world_cap=8, time_limit_ms=60_000.0)
    assert result["stats"]["simulations"] == 40
    assert elapsed < 5_000


def test_a_spent_budget_still_runs_one_sim(model_path):
    result, _ = _search(model_path, simulations=1000, time_limit_ms=1e-6)
    assert result["stats"]["simulations"] >= 1
    assert sum(result["stats"]["root_action_visits"]) >= 1


def test_a_negative_budget_is_refused(model_path):
    gs = engine.GameSession(GAME, seed=5, model_path=model_path, use_filter=False)
    with pytest.raises(ValueError):
        gs.get_ai_action(simulations=10, time_limit_ms=-1.0)
