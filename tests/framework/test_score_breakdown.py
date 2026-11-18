"""training.score_breakdown_head: per-seat final points per scoring category as an auxiliary target.

* score_breakdown_targets rotates whole seats (the mover's categories first) and scales them.
* A breakdown of the wrong length is rejected.
"""
from __future__ import annotations

import pytest

from training.pipeline import score_breakdown_targets


def test_targets_rotate_seats_and_scale():
    bd = [1, 2, 3, 10, 20, 30]  # seat 0: 1 2 3, seat 1: 10 20 30
    assert score_breakdown_targets(bd, 0, 2, 3, 10.0) == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    assert score_breakdown_targets(bd, 1, 2, 3, 10.0) == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    three = [1, 2, 3, 4, 5, 6]
    assert score_breakdown_targets(three, 2, 3, 2, 1.0) == [5, 6, 1, 2, 3, 4]


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        score_breakdown_targets([1, 2, 3], 0, 2, 2, 1.0)
