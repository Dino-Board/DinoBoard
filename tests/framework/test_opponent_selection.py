"""ISMCTS opponent_selection mode regression tests.

`opponent_selection="prior"` mitigates strategy fusion / opponent
omniscience: at non-root opponent decision nodes, ISMCTS replaces the
PUCT bandit with a multinomial sample from the policy head's frozen
prior. Two invariants that must hold:

  1. **Root unchanged.** The acting player's own root decision must
     still go through PUCT — otherwise we're dropping bandit on the
     player who matters. This is enforced by the `cur_idx != 0` guard
     in `engine/search/net_mcts.cpp`.
  2. **Self-consistency / no crash.** With both modes available, all
     downstream wiring (selfplay, GameSession, API) must accept the
     kwarg, route it through to NetMctsConfig, and run a full episode
     without throwing. Bad pybind plumbing surfaces here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


# Deterministic + hidden-info coverage. TicTacToe is fully public so
# opponent_selection="prior" effectively still acts on observable state;
# it's mainly there to confirm the kwarg pipeline.
_GAMES = ["tictactoe", "loveletter"]


@pytest.mark.parametrize("game_id", _GAMES)
@pytest.mark.parametrize("opp_sel", ["puct", "prior"])
def test_selfplay_runs_with_opponent_selection(game_id, opp_sel):
    """run_selfplay_episode accepts opponent_selection and produces a
    well-formed episode for both modes."""
    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id,
        seed=42,
        model_path=model_path,
        simulations=20,
        max_game_plies=40,
        temperature=0.0,
        opponent_selection=opp_sel,
    )
    assert ep["total_plies"] > 0
    assert "samples" in ep


@pytest.mark.parametrize("game_id", _GAMES)
def test_invalid_opponent_selection_raises(game_id):
    """Unknown mode strings must raise — silent fallbacks would mask config typos."""
    model_path = get_test_model(game_id)
    with pytest.raises((ValueError, RuntimeError)):
        engine.run_selfplay_episode(
            game_id=game_id,
            seed=42,
            model_path=model_path,
            simulations=8,
            max_game_plies=8,
            temperature=0.0,
            opponent_selection="bogus_mode",
        )


@pytest.mark.parametrize("opp_sel", ["puct", "prior"])
def test_game_session_get_ai_action_accepts_kwarg(opp_sel):
    """GameSession.get_ai_action(opponent_selection=...) must run end-to-end."""
    game_id = "tictactoe"
    model_path = get_test_model(game_id)
    gs = engine.GameSession(game_id, seed=7, model_path=model_path, use_filter=False)
    res = gs.get_ai_action(20, 0.0, opponent_selection=opp_sel)
    assert "action" in res


def test_one_simulation_root_action_matches_prior_argmax():
    """With simulations=1, MCTS expands the root + does one descent. The
    root's own action selection is PUCT (acting player) regardless of the
    mode — that's the `cur_idx != 0` guard in net_mcts.cpp. The first sim
    therefore picks the prior-argmax root edge in both modes (PUCT with
    visits=0 reduces to prior selection). Backup of one rollout doesn't
    change the argmax visit count after sim 1 (the picked edge has visits=1,
    everyone else 0). Both modes must agree on the chosen action.

    A regression that drops the `cur_idx != 0` guard would make "prior"
    mode sample the root, possibly choosing a non-argmax edge → divergence.
    """
    game_id = "tictactoe"
    model_path = get_test_model(game_id)

    res_puct = engine.GameSession(
        game_id, seed=42, model_path=model_path, use_filter=False
    ).get_ai_action(1, 0.0, opponent_selection="puct")
    res_prior = engine.GameSession(
        game_id, seed=42, model_path=model_path, use_filter=False
    ).get_ai_action(1, 0.0, opponent_selection="prior")

    assert res_puct["action"] == res_prior["action"], (
        f"sim=1 root action diverged: puct={res_puct['action']} "
        f"prior={res_prior['action']}; the cur_idx==0 PUCT guard may be missing")
