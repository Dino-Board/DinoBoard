"""External-engine interop (GameBundle external_state_loader / external_option_lister /
external_rules_switch) with ark-nova-ai states, as scripts/arknova_board.py uses it.

data/ark_nova_ai_states.json holds two states exported from ark-nova-ai (games/arknova/tools/
diff_test.py export_state) with that engine's option keys: the opening setup keep, and a turn of
player 1 while player 0 holds hidden cards.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

import dinoboard_engine as engine
from training.model import create_model_from_config, export_onnx

GAME = "arknova"
STATES = json.loads((Path(__file__).parent / "data" / "ark_nova_ai_states.json").read_text())
NET_CFG = {"type": "token_transformer", "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64,
           "relative_position_radius": 4, "attention": "matmul"}


@pytest.fixture(autouse=True)
def ark_nova_ai_rules():
    engine.set_external_rules(GAME, True)
    yield
    engine.set_external_rules(GAME, False)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    meta = engine.game_metadata(GAME)
    torch.manual_seed(0)
    net = create_model_from_config(NET_CFG, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
                                   num_players=meta["num_players"], token_layout=meta["token_layout"]).eval()
    path = tmp_path_factory.mktemp("interop") / "ark.onnx"
    export_onnx(net, path, meta["feature_dim"])
    return str(path)


def _load(state, model_path=""):
    gs = engine.GameSession(GAME, 0, model_path)
    gs.load_external_state(json.dumps(state))
    return gs


def _keys(gs):
    return {json.dumps(json.loads(k), separators=(",", ":")) for k, _ in gs.external_options()}


@pytest.mark.parametrize("name", ["start", "midgame"])
def test_options_match_ark_nova_ai(name):
    gs = _load(STATES[name]["state"])
    assert _keys(gs) == set(STATES[name]["options"])
    assert gs.current_player == STATES[name]["state"]["stack"][-1]["player"]


@pytest.mark.parametrize("name", ["start", "midgame"])
def test_atoms_narrow_to_their_option(name):
    state = STATES[name]["state"]
    first = {}
    for key, atoms in _load(state).external_options():
        first.setdefault(key, atoms)
    for key, atoms in list(first.items())[:12]:
        gs = _load(state)
        for i, atom in enumerate(atoms[:-1]):
            assert atom in gs.get_legal_actions()
            gs.apply_action(atom)
            listed = gs.external_options()
            assert all(a[:i + 1] == atoms[:i + 1] for _, a in listed)
            assert key in {k for k, _ in listed}
        assert {k for k, a in gs.external_options() if a[len(atoms) - 1] == atoms[-1]} == {key}


def _swap_hand_card(state, player, rng):
    """A copy where one of `player`'s hand cards the other player has not seen trades places with a
    deck card, and the deck is reshuffled."""
    s = json.loads(json.dumps(state))
    pl = s["players"][player]
    hidden = [c for c in pl["hand"] if c not in pl["public_hand"]]
    card, other = hidden[0], s["deck"][0]
    pl["hand"] = sorted(other if c == card else c for c in pl["hand"])
    s["deck"][0] = card
    rng.shuffle(s["deck"])
    return s


def test_search_ignores_the_opponents_hidden_cards(model):
    state = STATES["midgame"]["state"]
    mover = state["stack"][-1]["player"]
    rng = random.Random(1)

    def search(s):
        st = _load(s, model).get_ai_action(simulations=48, world_cap=8, temperature=0.0)["stats"]
        return st["root_actions"], st["root_action_visits"]

    def seen(s):
        return _load(s).state_hash_for_perspective(mover)

    opponent = _swap_hand_card(state, 1 - mover, rng)
    assert seen(opponent) == seen(state)
    assert search(opponent) == search(state)
    assert seen(_swap_hand_card(state, mover, rng)) != seen(state)  # the loader does read the mover's hand


def test_misuse_raises():
    state = json.dumps(STATES["midgame"]["state"])
    gt = engine.GameSession(GAME, 0)
    view = engine.GameSession(GAME, 0)
    view.apply_initial_observation(0, gt.extract_initial_observation(0))
    with pytest.raises(RuntimeError, match="holds an AI view"):
        view.load_external_state(state)
    other = engine.GameSession("tictactoe", 0)
    with pytest.raises(RuntimeError, match="no external_state_loader"):
        other.load_external_state("{}")
    with pytest.raises(RuntimeError, match="no external_option_lister"):
        other.external_options()
    with pytest.raises(RuntimeError, match="no external_rules_switch"):
        engine.set_external_rules("tictactoe", True)
    with pytest.raises(RuntimeError):
        gt.load_external_state('{"players": []}')
