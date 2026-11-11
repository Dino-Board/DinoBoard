"""Unit tests for Python-side training logic."""
import torch
import pytest

from conftest import FRAMEWORK_GAMES, load_game_config

from training.pipeline import normalize_policy, compute_schedule_ratio
from training.model import PVNet, create_model_from_config


# ---------------------------------------------------------------------------
# normalize_policy
# ---------------------------------------------------------------------------

def test_normalize_policy_valid_distribution():
    result = normalize_policy([0, 1, 2], [100, 200, 700], 5)
    assert len(result) == 5
    assert abs(sum(result) - 1.0) < 1e-6
    assert abs(result[0] - 0.1) < 1e-6
    assert abs(result[1] - 0.2) < 1e-6
    assert abs(result[2] - 0.7) < 1e-6
    assert result[3] == 0.0
    assert result[4] == 0.0


def test_normalize_policy_empty_visits():
    result = normalize_policy([], [], 5)
    assert result == [0.0] * 5


def test_normalize_policy_all_zero_visits():
    result = normalize_policy([0, 1], [0, 0], 5)
    assert result == [0.0] * 5


def test_normalize_policy_negative_visits_clamped():
    result = normalize_policy([0, 1], [-10, 100], 5)
    assert abs(result[0]) < 1e-9
    assert abs(result[1] - 1.0) < 1e-6


def test_normalize_policy_out_of_range_ids():
    # action_space=5 means valid range [0,5). -1 and 5 are out of range.
    # total = sum of ALL visits (300), but only action 2 is in-range.
    # So action 2 gets 100/300 = 0.333. Policy won't sum to 1.0.
    # (In practice MCTS never produces out-of-range ids.)
    result = normalize_policy([-1, 5, 2], [100, 100, 100], 5)
    assert len(result) == 5
    assert abs(result[2] - 1.0 / 3.0) < 1e-6
    assert result[0] == 0.0
    assert result[4] == 0.0


# ---------------------------------------------------------------------------
# compute_schedule_ratio
# ---------------------------------------------------------------------------

def test_compute_schedule_ratio_at_zero():
    assert compute_schedule_ratio(0, 100, 0.8) == 0.8


def test_compute_schedule_ratio_at_total():
    assert compute_schedule_ratio(100, 100, 0.8) == 0.0


def test_compute_schedule_ratio_linear():
    assert abs(compute_schedule_ratio(50, 100, 1.0) - 0.5) < 1e-9


def test_compute_schedule_ratio_past_total():
    assert compute_schedule_ratio(200, 100, 0.8) == 0.0


# ---------------------------------------------------------------------------
# Model creation and shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", FRAMEWORK_GAMES)
def test_create_model_output_shapes(game_id):
    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(game_id)
    cfg = dict(load_game_config(game_id))
    cfg["num_players"] = meta["num_players"]
    cfg["action_space"] = meta["action_space"]
    cfg["feature_dim"] = meta["feature_dim"]
    net = create_model_from_config(cfg)
    feature_dim = cfg["feature_dim"]
    action_space = cfg["action_space"]
    num_players = cfg["num_players"]

    dummy = torch.zeros(1, feature_dim)
    net.eval()
    with torch.no_grad():
        outputs = net(dummy)

    policy, value = outputs
    assert policy.shape == (1, action_space), f"policy shape: {policy.shape}"
    assert value.shape == (1, num_players), f"value shape: {value.shape}"
