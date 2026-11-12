"""Ark Nova adapter: factored atoms, hidden cards in the UI payload, greedy heuristic, token net.

The rules core itself is checked against the Python reference by games/arknova/tools/diff_test.py
(needs the ark-nova-ai checkout); framework invariants (masking, tracker, viz) run in
tests/framework for every game.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import dinoboard_engine as engine

sys.path.insert(0, str(Path(__file__).parent))
from arknova_reference import decode  # noqa: E402
from training.model import (TokenTransformerPVNet, create_belief_model_from_config, create_model_from_config,
                            export_onnx)

GAME = "arknova"
NET_CFG = {"type": "token_transformer", "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64,
           "relative_position_radius": 4}


def _play(seed, seats, max_plies=3000):
    """seats: a registered heuristic name or 'random' per player. Returns the session at the end."""
    gs = engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    for _ in range(max_plies):
        if gs.is_terminal:
            return gs
        if seats[gs.current_player] == "random":
            action = rng.choice(gs.get_legal_actions())
        else:
            action = gs.get_heuristic_action(seats[gs.current_player])["action"]
        gs.apply_action(action)
    pytest.fail(f"seed {seed}: game did not end within {max_plies} plies")


def test_atom_names_are_distinct():
    meta = engine.game_metadata(GAME)
    gs = engine.GameSession(GAME, seed=1)
    names = [gs.get_action_info(a)["name"] for a in range(meta["action_space"])]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_random_game_ends_with_scores(seed):
    gs = _play(seed, ("random", "random"))
    st = gs.get_state_dict()
    assert st["is_terminal"] and st["winner"] in (-1, 0, 1)
    assert st["prefix"] == ""


@pytest.mark.parametrize("seed", [0, 4, 5])
def test_ui_payload_hides_private_cards(seed):
    gs = engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    while not gs.is_terminal:
        core = json.loads(gs.get_state_dict()["core_json"])
        assert all(c is None for c in core["deck"]) and all(c is None for c in core["final_deck"])
        for pl in core["players"]:
            assert set(pl["hand"]) <= set(pl["public_hand"]), "a hidden hand card reached the UI payload"
            assert all(c is None for c in pl["final_cards"] + pl["tucked"])
        gs.apply_action(rng.choice(gs.get_legal_actions()))


def _pouch_with_seen_and_unseen(seed):
    """Plays seed's random game to a Pouch decision whose mover shows some hand cards and hides
    others. Returns (actions so far, tuck of a shown card, tuck of a hidden card) or None."""
    gs = engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    actions = []
    while not gs.is_terminal:
        legal = gs.get_legal_actions()
        core = json.loads(gs.get_state_dict()["core_json"])
        if core["stack"] and core["stack"][-1]["kind"] == "ab_pouch":
            shown = set(core["players"][gs.current_player]["public_hand"])
            tucks = {gs.get_action_info(a)["card"]: a for a in legal
                     if gs.get_action_info(a)["name"].startswith("hand_discard:")}
            seen = [a for c, a in tucks.items() if c in shown]
            unseen = [a for c, a in tucks.items() if c not in shown]
            if seen and unseen:
                return actions, seen[0], unseen[0]
        actions.append(rng.choice(legal))
        gs.apply_action(actions[-1])
    return None


def test_face_down_tuck_hides_which_card():
    """Tucking a card the opponent saw or one it did not leaves the same public picture: every card
    it had seen in the hand is now "in the hand or tucked"."""
    found = next((seed, f) for seed in range(60) if (f := _pouch_with_seen_and_unseen(seed)))
    seed, (actions, seen, unseen) = found
    views = []
    for tuck in (seen, unseen):
        gs = engine.GameSession(GAME, seed=seed)
        for a in actions:
            gs.apply_action(a)
        tucker = gs.current_player
        gs.apply_action(tuck)
        views.append(json.loads(gs.get_state_dict()["core_json"]))
    assert views[0] == views[1]
    assert views[0]["players"][tucker]["public_hand"] == []


def test_greedy_beats_random():
    wins = 0
    for g in range(6):
        seats = ("greedy", "random") if g % 2 == 0 else ("random", "greedy")
        gs = _play(100 + g, seats)
        wins += gs.winner == seats.index("greedy")
    assert wins >= 5, f"greedy won {wins}/6 against random"


@pytest.mark.parametrize("beam", ["beam_2x3", "beam_3x8"])
def test_lookahead_heuristics_finish_a_game(beam):
    gs = _play(31, (beam, "greedy"))
    st = gs.get_state_dict()
    assert st["is_terminal"] and st["winner"] in (-1, 0, 1)


def test_belief_config_builds():
    meta = engine.game_metadata(GAME)
    cfg = json.loads((Path(__file__).parents[2] / "games/arknova/config/belief.json").read_text())
    net = create_belief_model_from_config(cfg, input_dim=meta["belief_feature_dim"],
                                          output_dim=meta["belief_logit_count"])
    net.eval()
    out = net(torch.zeros(2, meta["belief_feature_dim"]))
    assert out.shape == (2, meta["belief_logit_count"])


def _net(seed=0):
    meta = engine.game_metadata(GAME)
    torch.manual_seed(seed)
    net = create_model_from_config(
        {**NET_CFG, "attention": "matmul"}, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
        num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)
    return net.eval()


def _check_hand_tokens(features):
    """One card / project token per card in either hand (unseen opponent cards as entity 0), whatever the size."""
    tokens = decode(features)
    players = [t for t in tokens if t.type == "player"]
    cards = [t for t in tokens if t.type in ("card", "project")]
    my_hand, opp_hand = [t for t in cards if t["my_hand"]], [t for t in cards if t["opp_hand"]]
    sizes = [round(t["hand_size"] * 20) for t in players]  # me, then the opponent
    assert [len(my_hand), len(opp_hand)] == sizes
    assert all(t.entity > 0 and t.anchor == t.entity for t in my_hand), "my card c: entity = anchor = 1 + c"
    assert all(t.anchor == 0 for t in opp_hand)


def test_variable_layout_binds_every_pointer_atom_by_anchor():
    meta = engine.game_metadata(GAME)
    lay = meta["token_layout"]
    assert lay["variable_tokens"] and meta["feature_dim"] == 0
    assert set(lay["action_token"]) == {-1} and set(lay["action_head"]) == {-1}
    bound = [a for a in lay["anchor_action"] if a >= 0]
    assert len(bound) == len(set(bound))
    gs = engine.GameSession(GAME, seed=1)
    pointer = {a for a in range(meta["action_space"])
               if gs.get_action_info(a)["name"].split(":")[0] in
               ("hand", "hand_discard", "display", "display_discard", "snap", "project", "zoo", "zoo_give", "list",
                "private", "final_discard", "enclosure", "cell")}
    assert set(bound) == pointer


def test_pouch_tuck_is_observed_as_a_hidden_card():
    """A tucked card stays hidden: observers get the never-legal hidden_hand_card atom instead. A
    harbour discard (Map 4) during a Pouch decision is not a tuck: its card is shown."""
    tucks = 0
    for seed in range(6):
        gs = engine.GameSession(GAME, seed=seed)
        rng = random.Random(seed)
        while not gs.is_terminal:
            actor = gs.current_player
            legal = gs.get_legal_actions()
            action = rng.choice(legal)
            hidden = gs.get_action_info(action)["name"]
            harbor = gs.get_state_dict()["prefix"].split() == ["harbor"]
            trace = gs.apply_action_with_trace(action, 1 - actor)
            if trace["action"] != action:
                assert hidden.startswith("hand_discard:") and not harbor
                assert gs.get_action_info(trace["action"])["name"] == "hidden_hand_card"
                assert trace["action"] not in legal
                tucks += 1
    assert tucks > 0


def test_training_loop_with_variable_tokens(tmp_path):
    """Two steps of the real pipeline (compact fp16 replay, fp16 inference) on variable-length encodings."""
    from training.cli import find_game_config, find_heuristic_config, find_train_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(GAME)
    cfg["max_game_plies"] = 60
    train_cfg, _ = find_train_config(GAME)
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["network"] = {**train_cfg["network"], **NET_CFG}
    train_cfg["heuristic_guidance"] = {"name": "greedy", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2,
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5},
                             "training_filter_steps": 0}
    run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    latest = tmp_path / "models" / "model_latest.onnx"
    assert latest.exists() and (tmp_path / "models" / "fp16" / "model_latest.onnx").exists()
    ep = engine.run_selfplay_episode(game_id=GAME, seed=3, model_path=str(latest), simulations=4, max_game_plies=40)
    assert ep["samples"]


def test_token_layout_and_selfplay(tmp_path):
    meta = engine.game_metadata(GAME)
    lay = meta["token_layout"]
    assert len(lay["action_token"]) == meta["action_space"]
    path = tmp_path / "ark.onnx"
    export_onnx(_net(seed=3), path, meta["feature_dim"])
    ep = engine.run_selfplay_episode(game_id=GAME, seed=21, model_path=str(path), simulations=4, max_game_plies=150)
    assert ep["samples"]
    assert meta["has_final_score_extractor"] and len(ep["final_scores"]) == 2
    kwargs = dict(game_id=GAME, seeds=[7], latest_model_path=str(path), num_self_threads=1, simulations=8,
                  max_game_plies=60, batch_size=16, max_queue_size=64, intra_op_num_threads=1, leaf_batch_size=2,
                  fast_simulations=2, full_search_prob=0.5)
    lists = engine.run_concurrent_selfplay_step(**kwargs)["episodes"][0]
    compact = engine.run_concurrent_selfplay_step(compact_samples=True, **kwargs)["episodes"][0]
    assert {s["full_search"] for s in compact["samples"]} == {True, False}
    for a, b in zip(lists["samples"], compact["samples"], strict=True):
        assert ("features" in b) == b["full_search"]
        if b["full_search"]:
            np.testing.assert_array_equal(np.asarray(a["features"], dtype=np.float32), b["features"])
    assert compact["belief_samples"]
    for a, b in zip(lists["belief_samples"], compact["belief_samples"], strict=True):
        assert isinstance(b["features"], np.ndarray) and b["features"].dtype == np.float32
        np.testing.assert_array_equal(np.asarray(a["features"], dtype=np.float32), b["features"])
        for key in ("hand_counts", "remaining", "alive_per_opp"):
            np.testing.assert_array_equal(np.asarray(a[key]), b[key])
    lengths = set()
    for s in ep["samples"]:
        x = np.asarray(s["features"], dtype=np.float32)
        assert x.size % lay["token_dim"] == 0 and np.isfinite(x).all()
        assert np.asarray(s["legal_mask"]).any()
        _check_hand_tokens(x)
        lengths.add(x.size)
    assert len(lengths) > 1
