"""Unit tests for compute_schedule_ratio (three-segment schedule).

The schedule has three regions:
  step <= hold_steps           → initial_ratio
  step >= decay_end_step       → 0.0
  hold_steps < step < end      → linear interpolation

Used by:
  - heuristic_guidance ratio
  - training_filter ratio
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.cli import validate_guidance_until
from training.pipeline import adaptive_guidance_ratio, compute_schedule_ratio


def test_disabled_when_decay_end_zero():
    for step in [0, 1, 100]:
        assert compute_schedule_ratio(step, 0, 1.0) == 0.0


def test_disabled_when_decay_end_negative():
    assert compute_schedule_ratio(5, -1, 1.0) == 0.0


def test_no_hold_linear_decay():
    # hold=0 → step 1 already starts decaying.
    assert compute_schedule_ratio(0, 10, 1.0, hold_steps=0) == pytest.approx(1.0)
    assert compute_schedule_ratio(5, 10, 1.0, hold_steps=0) == pytest.approx(0.5)
    assert compute_schedule_ratio(9, 10, 1.0, hold_steps=0) == pytest.approx(0.1)
    assert compute_schedule_ratio(10, 10, 1.0, hold_steps=0) == 0.0
    assert compute_schedule_ratio(11, 10, 1.0, hold_steps=0) == 0.0


def test_hold_period_then_decay():
    # hold=2, decay_end=5, initial=1.0 → 1.0 / 1.0 / 1.0 / 0.667 / 0.333 / 0.0
    assert compute_schedule_ratio(0, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(1, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(2, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(3, 5, 1.0, hold_steps=2) == pytest.approx(2.0 / 3.0)
    assert compute_schedule_ratio(4, 5, 1.0, hold_steps=2) == pytest.approx(1.0 / 3.0)
    assert compute_schedule_ratio(5, 5, 1.0, hold_steps=2) == 0.0


def test_initial_ratio_below_one():
    # Hold period locks at initial_ratio (not 1.0).
    assert compute_schedule_ratio(0, 4, 0.4, hold_steps=1) == pytest.approx(0.4)
    assert compute_schedule_ratio(1, 4, 0.4, hold_steps=1) == pytest.approx(0.4)
    # Step 2: decay span 4-1=3, remaining 4-2=2 → 0.4 * 2/3
    assert compute_schedule_ratio(2, 4, 0.4, hold_steps=1) == pytest.approx(0.4 * 2 / 3)


def test_misconfigured_hold_too_large_raises():
    with pytest.raises(ValueError):
        compute_schedule_ratio(0, 5, 1.0, hold_steps=5)
    with pytest.raises(ValueError):
        compute_schedule_ratio(0, 5, 1.0, hold_steps=10)


def test_strictly_decreasing_in_decay_window():
    prev = compute_schedule_ratio(2, 10, 1.0, hold_steps=2)
    for step in range(3, 10):
        cur = compute_schedule_ratio(step, 10, 1.0, hold_steps=2)
        assert cur < prev, f"non-monotonic at step {step}: {cur} >= {prev}"
        prev = cur


def test_after_decay_end_is_exactly_zero():
    for step in range(5, 20):
        assert compute_schedule_ratio(step, 5, 1.0, hold_steps=2) == 0.0


def test_adaptive_guidance_holds_until_trigger_then_decays():
    for step in (1, 50, 5000):
        assert adaptive_guidance_ratio(step, None, 0.8, 10) == pytest.approx(0.8)
    assert adaptive_guidance_ratio(40, 40, 0.8, 10) == pytest.approx(0.8)
    assert adaptive_guidance_ratio(45, 40, 0.8, 10) == pytest.approx(0.4)
    assert adaptive_guidance_ratio(50, 40, 0.8, 10) == 0.0
    assert adaptive_guidance_ratio(99, 40, 0.8, 10) == 0.0


def _guidance(**until):
    base = dict(benchmark="heuristic_free", threshold=0.5, decay_steps=300, max_hold_steps=2000)
    return dict(name="rule", initial_ratio=1.0, hold_steps=50, steps=0, temperature=0.0,
                until_win_rate={**base, **until})


def test_validate_guidance_until_accepts_valid_and_absent():
    validate_guidance_until(_guidance(), "t.json")
    validate_guidance_until(dict(name="rule", hold_steps=1, steps=2), "t.json")


@pytest.mark.parametrize("gs, match", [
    (dict(_guidance(), steps=400), "steps must be 0"),
    (_guidance(threshold=1.5), "threshold"),
    (_guidance(decay_steps=0), "decay_steps"),
    (_guidance(max_hold_steps=10), "max_hold_steps"),
    (dict(_guidance(), until_win_rate=dict(benchmark="heuristic_free", threshold=0.5)), "exactly the keys"),
])
def test_validate_guidance_until_rejects(gs, match):
    with pytest.raises(ValueError, match=match):
        validate_guidance_until(gs, "t.json")
