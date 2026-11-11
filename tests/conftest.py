"""Shared fixtures and helpers for DinoBoard tests.

Architectural principle: this file knows ONLY about the framework matrix
carrier (quoridor + azul + loveletter). Per-game tests in tests/<game>/
are responsible for their own assertions and load their own config via
the load_game_config helper. The framework layer never embeds rules,
constants, or assertions for any specific game outside the matrix.

Two coexisting test-matrix mechanisms:

  1. **Manifest-driven helpers** (`enabled_games`, `games_with_capability`,
     `hidden_info_games`, `games_with_snapshot`, `games_with_tracker`,
     `framework_whitelist_games`) read `games/manifest.json`, so capability-
     keyed framework tests (`test_public_snapshot_round_trip`,
     `test_public_hash_excludes_internal_rng`, `test_encoder_respects_hash_scope`,
     etc.) self-extend when a game is enabled or a capability tag is added.
  2. **Fixed three-game carrier** `FRAMEWORK_GAMES = ["quoridor", "azul",
     "loveletter"]` (deterministic / public-random / asymmetric-hidden, one
     representative each) — used by tests that intentionally don't run on
     every enabled game (`test_full_game_via_api`, the default `game_id`
     fixture, etc.). Adding a game does NOT require touching this list.

Some tests further use per-game checker maps or hand-written narrow lists
(`test_tracker_consistent_with_truth._CHECKERS`,
`test_api_belief_matches_selfplay.GAMES_WITH_EVENT_PROTOCOL`,
`test_api_mcts_policy_invariance.LEAK_SENSITIVE_GAMES`) where capability-
based parametrization isn't yet appropriate. So "no hardcoded game lists"
is **not** a global property of framework tests — it's true of the
manifest-helper-keyed subset and false of the fixed-carrier subset.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import dinoboard_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = PROJECT_ROOT / "games" / "manifest.json"


@lru_cache(maxsize=1)
def _manifest() -> dict:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _registered() -> frozenset:
    return frozenset(dinoboard_engine.available_games())


def _entries() -> list:
    return _manifest()["games"]


def enabled_games() -> list:
    """Manifest-enabled AND engine-registered. Declaration order preserved."""
    reg = _registered()
    return [g["id"] for g in _entries()
            if g.get("enabled", True) and g["id"] in reg]


def games_with_capability(*caps: str) -> list:
    """Enabled games whose capability set contains every cap in `caps`."""
    reg = _registered()
    out = []
    for g in _entries():
        if not g.get("enabled", True):
            continue
        if g["id"] not in reg:
            continue
        gcaps = set(g.get("capabilities", []))
        if all(c in gcaps for c in caps):
            out.append(g["id"])
    return out


def framework_whitelist_games() -> list:
    """Enabled + framework_whitelist=true. The set of games the framework
    structurally guarantees core invariants for."""
    reg = _registered()
    return [g["id"] for g in _entries()
            if g.get("enabled", True) and g.get("framework_whitelist", False)
            and g["id"] in reg]


def hidden_info_games() -> list:
    return games_with_capability("hidden_info")


def games_with_snapshot() -> list:
    return games_with_capability("snapshot")


def games_with_tracker() -> list:
    return games_with_capability("tracker")

# Framework matrix — minimal carrier set used by tests/framework/. These
# three games together cover every structural feature the framework cares
# about (deterministic / fully-public-with-physical-randomness /
# asymmetric-hidden-with-tracker, 2p / 2-4p, tail solver, elimination).
# Per-game tests in tests/<game>/ test their own game in full regardless
# of whether it is in this list.
FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]
FRAMEWORK_HIDDEN_INFO_GAMES = ["loveletter"]
FRAMEWORK_MULTIPLAYER_GAMES = ["azul", "loveletter"]
FRAMEWORK_TAIL_SOLVER_GAMES = ["quoridor"]


def load_game_config(game_id: str) -> dict:
    """Load a game's game.json plus C++-side metadata. Generic helper —
    works for any registered game, framework or per-game test."""
    import re
    base = re.sub(r"(_v?tok)?(_\d+p)?$", "", game_id)
    config_path = PROJECT_ROOT / "games" / base / "config" / "game.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    meta = dinoboard_engine.game_metadata(game_id)
    cfg["action_space"] = meta["action_space"]
    cfg["feature_dim"] = meta["feature_dim"]
    cfg["num_players"] = meta["num_players"]
    return cfg


_MODEL_CACHE: dict[str, str] = {}


def get_test_model(game_id: str, tmp_path_factory=None) -> str:
    """Return path to a random ONNX model for the given game. Cached per session.

    Weights are initialized under a per-game deterministic seed (derived
    from game_id) so the same game always produces byte-equal ONNX
    regardless of cache fill order across pytest runs. Without this,
    selfplay trajectories driven by these random nets become nondeterministic
    across runs, surfacing as occasional flakes in tests that depend on
    specific action sequences (e.g. splendor's belief-equivalence test
    walking into a reserve+deck_flip edge case at ply ~5)."""
    if game_id in _MODEL_CACHE:
        return _MODEL_CACHE[game_id]
    from training.model import create_model_from_config, export_onnx
    meta = dinoboard_engine.game_metadata(game_id)
    feature_dim = meta["feature_dim"]
    action_space = meta["action_space"]
    num_players = meta["num_players"]
    cfg = {"feature_dim": feature_dim, "action_space": action_space, "num_players": num_players}
    import hashlib
    digest = hashlib.md5(f"dinoboard_test_model:{game_id}".encode()).digest()
    seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF or 1
    rng_state = torch.random.get_rng_state()
    layout = meta.get("token_layout")
    try:
        torch.manual_seed(seed)
        if layout and layout["variable_tokens"]:  # no flat input: needs the token transformer
            net_cfg = {"type": "token_transformer", "d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64}
            if layout["token_positions"]:  # the network then needs its relative position radius
                net_cfg["relative_position_radius"] = 4
            net = create_model_from_config(
                net_cfg,
                input_dim=feature_dim, policy_dim=action_space, num_players=num_players, token_layout=layout)
        else:
            net = create_model_from_config(cfg)
    finally:
        torch.random.set_rng_state(rng_state)
    model_dir = Path(os.environ.get(
        "DINOBOARD_TEST_MODEL_DIR", "/tmp/dinoboard_test_models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"test_{game_id}.onnx"
    export_onnx(net, path, feature_dim)
    _MODEL_CACHE[game_id] = str(path)
    return str(path)


def _public_equivalent(value):
    """Drop details explicitly marked hidden before comparing public state."""
    if isinstance(value, dict):
        if value.get("visible") is False:
            return {"visible": False}
        return {k: _public_equivalent(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_public_equivalent(v) for v in value]
    return value


@pytest.fixture
def model_path(game_id):
    """Fixture providing a random ONNX model path for the current game_id."""
    return get_test_model(game_id)


@pytest.fixture
def game_config(game_id):
    """Fixture providing the loaded config for the current game_id. Thin
    convenience wrapper around load_game_config — works for any game,
    framework or otherwise."""
    return load_game_config(game_id)


# `game_id` fixture defaults to FRAMEWORK_GAMES (the matrix carrier set).
# Tests that need exhaustive coverage (e.g. tests/<game>/test_checklist.py)
# hardcode their own game id and don't use this fixture; tests that need
# a different subset can use @pytest.mark.parametrize("game_id", [...]).
@pytest.fixture(params=FRAMEWORK_GAMES)
def game_id(request):
    return request.param


def run_short_selfplay(game_id: str, seed: int = 42, **kwargs) -> dict:
    defaults = dict(
        simulations=10,
        max_game_plies=50,
    )
    defaults.update(kwargs)
    if "model_path" not in defaults:
        defaults["model_path"] = get_test_model(game_id)
    return dinoboard_engine.run_selfplay_episode(
        game_id=game_id, seed=seed, **defaults,
    )


def run_short_heuristic(game_id: str, seed: int = 42, temperature: float = 1.0) -> dict:
    from training.cli import find_train_config
    selector = find_train_config(game_id)[0]["heuristic_guidance"]["name"]
    return dinoboard_engine.run_heuristic_episode(
        game_id=game_id, seed=seed, temperature=temperature, max_game_plies=50,
        heuristic_selector=selector,
    )


def run_random_episode_states(
    game_id: str,
    *,
    seed: int = 42,
    max_plies: int = 200,
):
    """Drive a game by uniform-random legal actions and yield the
    state_dict at every ply (including the initial state and the terminal
    state). Intended for rule-invariant tests in per-game checklists:

        for state in run_random_episode_states(GAME, seed=s):
            assert_invariants(state)

    Uses random actions instead of MCTS so the test is fast and exercises
    rare states (illegal-looking edge cases that an AI would avoid). The
    underlying engine is the C++ rules implementation, so any rule
    violation surfaces as a broken invariant.
    """
    import random as _random
    rng = _random.Random(seed)
    gs = dinoboard_engine.GameSession(
        game_id, seed=seed, model_path="", use_filter=False)
    yield gs.get_state_dict()
    for _ in range(max_plies):
        if gs.is_terminal:
            break
        legal = gs.get_legal_actions()
        if not legal:
            break
        gs.apply_action(rng.choice(legal))
        yield gs.get_state_dict()


def assert_api_belief_matches_selfplay(
    game_id: str,
    public_keys: list[str],
    *,
    perspective: int = 0,
    seed_gt: int = 42,
    seed_ai: int = 9999,
    plies: int = 80,
    simulations: int = 20,
) -> None:
    """Three-layer equivalence assertion between self-play (ground truth)
    and an independent-seed API session that only sees public events.

    Asserts:
      1. Belief tracker matches at every ply (initial + post-action)
      2. Public state fields (named in `public_keys`) match after replay
      3. Legal-action sets match on perspective-player turns

    Used by per-game checklists for any game with a public_event_extractor.
    The framework matrix carrier (azul, loveletter) calls this through its
    own framework test; other games call this directly from their checklist.
    """
    import sys as _sys
    _platform_path = str(PROJECT_ROOT / "platform")
    if _platform_path not in _sys.path:
        _sys.path.insert(0, _platform_path)

    model_path = get_test_model(game_id)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=game_id, seed=seed_gt, model_path=model_path,
        simulations=simulations, max_game_plies=plies,
        trace_perspective=perspective,
    )
    assert ep.get("observation_trace"), \
        f"[{game_id}] trace empty — extractor missing or episode terminated immediately"
    trace = ep["observation_trace"]

    api_gs = dinoboard_engine.GameSession(
        game_id, seed=seed_ai, model_path="", use_filter=False)
    api_gs.apply_initial_observation(perspective, ep["initial_observation"])
    assert api_gs.get_belief_snapshot() == ep["initial_belief_snapshot"], \
        f"[{game_id}] initial belief mismatch"

    for step in trace:
        api_gs.apply_observation(
            step["action"], events=step["events"],
            public_snapshot=step.get("public_snapshot", {}))
        assert api_gs.get_belief_snapshot() == step["belief_snapshot_after"], \
            f"[{game_id}] belief diverged at ply {step['ply']}"

    api_state = api_gs.get_state_dict()
    # Ground truth is replayed from the acting seats' own actions; the trace
    # only carries their public projection (IGameRules::public_action).
    real_actions = [s["action_id"] for s in ep["samples"]]
    assert len(real_actions) == len(trace)
    gt_gs = dinoboard_engine.GameSession(
        game_id, seed=seed_gt, model_path="", use_filter=False)
    for action in real_actions:
        gt_gs.apply_action(action)
    gt_state = gt_gs.get_state_dict()
    for key in public_keys:
        assert _public_equivalent(api_state[key]) == _public_equivalent(gt_state[key]), \
            f"[{game_id}] public field '{key}' diverged after trace replay"

    api2 = dinoboard_engine.GameSession(
        game_id, seed=seed_ai + 1, model_path="", use_filter=False)
    api2.apply_initial_observation(perspective, ep["initial_observation"])
    gt2 = dinoboard_engine.GameSession(
        game_id, seed=seed_gt, model_path="", use_filter=False)
    for step, action in zip(trace, real_actions):
        api2.apply_observation(
            step["action"], events=step["events"],
            public_snapshot=step.get("public_snapshot", {}))
        gt2.apply_action(action)
        if api2.is_terminal or gt2.is_terminal:
            continue
        if api2.current_player != perspective or gt2.current_player != perspective:
            continue
        assert sorted(api2.get_legal_actions()) == sorted(gt2.get_legal_actions()), \
            f"[{game_id}] perspective legal_actions diverged at ply {step['ply']}"
