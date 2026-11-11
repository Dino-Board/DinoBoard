"""training.compact_replay: sparse policy rows and config validation."""
from __future__ import annotations

import numpy as np
import pytest

from training.pipeline import densify_policies, normalize_policy, parse_compact_replay, sparse_policy

CONCURRENT = {"selfplay_concurrency": {"enabled": True}}


@pytest.mark.parametrize("ids, visits", [
    ([3, 0, 7], [5, 0, 15]),
    ([1, 2], [0, 0]),
    ([], []),
    ([9, 4, 99], [2, -1, 3]),       # negative visits and out-of-range ids are dropped
])
def test_sparse_policy_round_trips_to_normalize_policy(ids, visits):
    dense = densify_policies([sparse_policy(ids, visits, 10)], 10)[0]
    np.testing.assert_allclose(dense, np.asarray(normalize_policy(ids, visits, 10), dtype=np.float32), rtol=1e-6)


def test_parse_compact_replay():
    assert parse_compact_replay({}) is None
    assert parse_compact_replay({"compact_replay": {"features_dtype": "float16"}, **CONCURRENT}) == np.float16
    assert parse_compact_replay({"compact_replay": {"features_dtype": "float32"}, **CONCURRENT}) == np.float32
    with pytest.raises(ValueError, match="features_dtype"):
        parse_compact_replay({"compact_replay": {"features_dtype": "int8"}, **CONCURRENT})
    with pytest.raises(ValueError, match="exactly"):
        parse_compact_replay({"compact_replay": {}, **CONCURRENT})
    with pytest.raises(ValueError, match="selfplay_concurrency"):
        parse_compact_replay({"compact_replay": {"features_dtype": "float16"}})
