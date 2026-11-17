"""Ark Nova belief network: the opponent's hidden hand is written card by card, lowest card first
(belief.json type "autoregressive", engine/core/belief_sequence.h).

* The metadata's sequence layout, game.json and belief.json agree.
* The selfplay labels are possible hands of the pool: inside the unseen regular cards the features
  show, as many cards as the opponent hides, and they walk into training rows.
* Worlds dealt from a belief model search with maybe-tucked cards and play whole games
  reproducibly (the tracker refuses a hand that holds a card it cannot), and two steps of the real
  training loop train and export the belief net.

Hidden-slot independence with a belief model: test_arknova_encoding.py; the model's hands reaching
the search and not being reused for another position: test_arknova_actions.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine
from conftest import get_test_model

GAME = "arknova"
ROOT = Path(__file__).resolve().parents[2]
CLASSES = 240
# Belief feature blocks (ArknovaBeliefFeatureExtractor), CLASSES wide each, then the scalars.
UNSEEN, OPP_KNOWN, OPP_MAYBE = 0, 1, 2
SCALARS = 5 * CLASSES


def _belief_model(tmp_path, seed=7):
    """An untrained belief net with a random last layer, so that its hands depend on the features (a
    new net's last layer is zero: the uniform deal)."""
    import torch

    from training.model import create_belief_model_from_config, export_belief_onnx

    meta = engine.game_metadata(GAME)
    cfg = json.loads((ROOT / "games/arknova/config/belief.json").read_text())
    torch.manual_seed(seed)
    net = create_belief_model_from_config(cfg, input_dim=meta["belief_feature_dim"],
                                          output_dim=meta["belief_logit_count"],
                                          sequence_layout=meta["belief_sequence_layout"]).eval()
    torch.nn.init.normal_(net.head[-1].weight, std=0.3)
    path = tmp_path / f"belief_{seed}.onnx"
    export_belief_onnx(net, path, meta["belief_feature_dim"])
    return str(path)


def _block(features, k):
    return features[k * CLASSES:(k + 1) * CLASSES]


def test_metadata():
    meta = engine.game_metadata(GAME)
    variant = json.loads((ROOT / "games/arknova/config/game.json").read_text())["variants"][GAME]
    assert meta["has_belief_extractor"]
    assert meta["belief_feature_dim"] == variant["belief_feature_dim"] == SCALARS + 18
    assert meta["belief_logit_count"] == variant["belief_logit_count"] == CLASSES
    assert meta["belief_sequence_layout"] == {"classes": CLASSES, "segments": 1, "prefix_dim": 2 * CLASSES + 1}
    belief = json.loads((ROOT / "games/arknova/config/belief.json").read_text())
    assert belief["type"] == "autoregressive" and belief["training"] == {"loss": "sequence_nll"}


def test_labels_are_possible_hands_of_the_pool(tmp_path):
    """Selfplay with a belief model, every ply, both observers. The pool is the unseen regular cards
    of the features less the hand cards the mover named in its prefix; the label lies in it and has
    as many cards as the opponent hides beyond those and its maybe-tucked cards."""
    ep = engine.run_selfplay_episode(game_id=GAME, seed=21, model_path=get_test_model(GAME), simulations=6,
                                     max_game_plies=200, belief_model_path=_belief_model(tmp_path))
    assert ep["belief_samples"]
    exact = hidden_hands = 0
    for bs in ep["belief_samples"]:
        x = np.asarray(bs["features"], dtype=np.float32)
        pool = np.asarray(bs["remaining"])
        hand = np.flatnonzero(np.asarray(bs["hand_counts"])[0])
        unseen = _block(x, UNSEEN)
        assert pool.shape == (CLASSES,) and set(np.unique(pool)) <= {0, 1}
        assert np.all(unseen[pool == 1] == 1), "the pool holds a card the observer can place"
        assert np.all(pool[hand] == 1), "a label card is outside the pool"
        assert not np.any(_block(x, OPP_KNOWN)[hand]) and not np.any(_block(x, OPP_MAYBE)[hand])
        not_visible = round(float(x[SCALARS + 1]) * 20)  # opponent hand cards the observer has not seen
        named = int(unseen.sum() - pool.sum())
        maybe = int(_block(x, OPP_MAYBE).sum())
        assert not_visible - named - maybe <= len(hand) <= not_visible - named
        exact += maybe == 0
        rows = engine.belief_sequence_training_rows(CLASSES, 1, pool[None].astype(np.int32),
                                                    np.asarray(bs["hand_counts"], dtype=np.int32)[None])
        assert rows["target"].tolist() == hand.tolist()
        hidden_hands += len(hand) > 0
    assert exact > 0 and hidden_hands > 100


def test_searches_with_maybe_tucked_cards(tmp_path):
    """Worlds from the model's samples when the opponent holds maybe-tucked cards: each sample places
    them in the hand or among the tucked cards and the model writes the hand cards left."""
    model, belief = get_test_model(GAME), _belief_model(tmp_path)
    searched = 0
    for seed in range(40):
        gs = engine.GameSession(GAME, seed=seed, model_path=model, belief_model_path=belief)
        rng = random.Random(seed)
        while not gs.is_terminal and searched < 20:
            x = np.asarray(gs.belief_features(gs.current_player), dtype=np.float32)
            if _block(x, OPP_MAYBE).any() and len(gs.get_legal_actions()) > 1:
                assert gs.get_ai_action(simulations=16, temperature=0.0)["action"] in gs.get_legal_actions()
                searched += 1
            gs.apply_action(rng.choice(gs.get_legal_actions()))
        if searched >= 20:
            return
    pytest.fail(f"only {searched} searches had maybe-tucked cards")


def test_whole_games_play_out_with_a_belief_model(tmp_path):
    kw = dict(game_id=GAME, seed=33, model_path=get_test_model(GAME), simulations=8, max_game_plies=300,
              belief_model_path=_belief_model(tmp_path), dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    a, b = engine.run_selfplay_episode(**kw), engine.run_selfplay_episode(**kw)
    assert a["total_plies"] >= 200
    assert [s["action_id"] for s in a["samples"]] == [s["action_id"] for s in b["samples"]]


def test_training_loop_trains_the_belief_net(tmp_path, caplog):
    """Two steps of the real pipeline with belief.json: the belief net is trained on the selfplay
    labels, exported with its two inputs, and the next selfplay searches with it."""
    import logging

    import onnxruntime as ort

    from training.cli import find_belief_config, find_game_config, find_heuristic_config, find_train_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(GAME)
    cfg["max_game_plies"] = 60
    train_cfg, _ = find_train_config(GAME)
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["belief"], _ = find_belief_config(GAME)
    train_cfg["belief"] = {**train_cfg["belief"], "architecture": [32, 32], "head": [32]}
    train_cfg["network"] = {**train_cfg["network"], "type": "token_transformer", "d_model": 32, "num_layers": 2,
                            "num_heads": 4, "ffn_dim": 64, "relative_position_radius": 4}
    train_cfg["heuristic_guidance"] = {"name": "greedy", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "training_filter_steps": 0,
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    with caplog.at_level(logging.INFO):
        run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                          episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                          learning_rate=0.001, seed=5, save_every=0)
    assert "b_loss=" in caplog.text and "b_gain=" in caplog.text
    meta = engine.game_metadata(GAME)
    latest = tmp_path / "models" / "belief_latest.onnx"
    inputs = ort.InferenceSession(str(latest), providers=["CPUExecutionProvider"]).get_inputs()
    assert {i.name: i.shape[1] for i in inputs} == {
        "features": meta["belief_feature_dim"], "prefix": meta["belief_sequence_layout"]["prefix_dim"]}
    assert (tmp_path / "belief_checkpoint.pt").exists()
    ep = engine.run_selfplay_episode(
        game_id=GAME, seed=3, model_path=str(tmp_path / "models" / "model_latest.onnx"), simulations=4,
        max_game_plies=60, belief_model_path=str(latest))
    assert ep["samples"] and ep["belief_samples"]
