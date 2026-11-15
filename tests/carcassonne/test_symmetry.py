"""Quarter turns of the Carcassonne board as training-time augmentation.

The token encoder transforms stored samples itself
(IFeatureEncoder::apply_sample_symmetry) because the image of a rotation or
meeple action depends on the tile it is about. The reference here is the
engine: a position is written as a snapshot, turned by plain index
arithmetic, and loaded again (the loader stores the rules' own forms of
rotations and meeple positions); its encoding and legal actions must be the
transform of the original's.

* encode(turned state) == transform(encode(state)), legal masks included,
  for both perspectives, both stages and all three turns.
* Rules: an action and its image lead to positions with the same scores,
  meeple supplies and score projection.
* Turns compose (four quarter turns are the identity), padding tokens stay
  untouched, weight on something that is no action is rejected.
* Config: the augmentation is accepted for `carcassonne_tok`, rejected with
  the opponent-policy head, and the real training loop runs with it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine
from training.pipeline import pad_token_rows
from training.symmetry import SampleSymmetryAugmenter, parse_symmetry_augmentation

from . import carc_reference as ref
from .test_snapshot_equivalence import _flat, _load, _pairs

GAME = "carcassonne_tok"
ROOT = Path(__file__).resolve().parents[2]
SIZE = 20
TOKEN_DIM = 7 + 9 + 13 * 6
PLACE_ACTIONS = SIZE * SIZE * 4
MEEPLE_BASE = PLACE_ACTIONS + 1


def _turn_cell(cell, g):
    row, col = divmod(cell, SIZE)
    for _ in range(g):
        row, col = col, SIZE - 1 - row
    return row * SIZE + col


def _turn_position(pos, g):
    if 0 <= pos < 4:
        return (pos + g) % 4
    if 4 <= pos < 12:
        return 4 + (pos - 4 + 2 * g) % 8
    return pos


def _turned(payload, g):
    """The snapshot of the same position with the board turned g quarter
    turns clockwise. Rotations and meeple positions are written as plain
    turns; the loader stores them in the rules' forms."""
    out = dict(payload)
    tile = _pairs(payload["board_tile"])
    for key in ("board_tile", "board_meeple_owner", "final_meeple_scores"):
        out[key] = _flat({(_turn_cell(c, g),): v for (c,), v in _pairs(payload[key]).items()})
    out["board_rotation"] = _flat({(_turn_cell(c, g),): (r + g) % 4 if tile[(c,)] >= 0 else r
                                   for (c,), r in _pairs(payload["board_rotation"]).items()})
    out["board_meeple_pos"] = _flat({(_turn_cell(c, g),): _turn_position(p, g)
                                     for (c,), p in _pairs(payload["board_meeple_pos"]).items()})
    row, col = _pairs(payload["last_row"])[()], _pairs(payload["last_col"])[()]
    if row >= 0 and col >= 0:
        cell = _turn_cell(row * SIZE + col, g)
        out["last_row"], out["last_col"] = [[], cell // SIZE], [[], cell % SIZE]
        out["last_rotation"] = [[], (_pairs(payload["last_rotation"])[()] + g) % 4]
    h_tile = _pairs(payload["history_tile"])
    h_row, h_col, h_rot = (dict(_pairs(payload[k])) for k in ("history_row", "history_col", "history_rotation"))
    for k, t in h_tile.items():
        if t < 0:
            continue
        cell = _turn_cell(h_row[k] * SIZE + h_col[k], g)
        h_row[k], h_col[k], h_rot[k] = cell // SIZE, cell % SIZE, (h_rot[k] + g) % 4
    out["history_row"], out["history_col"], out["history_rotation"] = _flat(h_row), _flat(h_col), _flat(h_rot)
    return out


def _positions(seed, every):
    """(seat to move, snapshot) at every `every`-th ply of one game mixing
    heuristic and random moves, meeples placed often."""
    rng = random.Random(seed)
    gs = engine.GameSession(GAME, seed=seed)
    out = []
    ply = 0
    while not gs.is_terminal:
        if ply % every == 0:
            seat = gs.current_player
            out.append((seat, dict(gs.extract_initial_observation(seat)["public_snapshot"])))
        legal = gs.get_legal_actions()
        if rng.random() < 0.5:
            action = gs.get_heuristic_action("rule", 0.0)["action"]
        else:
            placing = [a for a in legal if a != ref.PASS]
            action = rng.choice(placing if placing and legal[0] >= ref.PASS else legal)
        gs.apply_action(action)
        ply += 1
    return out


def _token_set(x):
    return sorted(map(tuple, np.asarray(x, dtype=np.float32).reshape(-1, TOKEN_DIM).tolist()))


def _apply(g, features, rows):
    feats, out = engine.apply_sample_symmetries(
        GAME, np.array([g]), np.asarray([features], dtype=np.float32),
        [np.asarray([r], dtype=np.float32) for r in rows])
    return feats[0], [r[0] for r in out]


def test_encoder_reports_four_sample_symmetries_and_no_tables():
    assert engine.encoder_sample_symmetry_count(GAME) == 4
    assert engine.encoder_symmetries(GAME) == []
    assert engine.encoder_sample_symmetry_count("carcassonne") == 0
    assert engine.encoder_sample_symmetry_count("harmonies_tok") == 0, "table symmetries are not sample symmetries"


def test_encoding_of_turned_state_is_turned_encoding():
    checked = {"place": 0, "meeple": 0}
    short_period = moved_meeple_action = 0
    for seed in (41, 42, 43):
        for seat, payload in _positions(seed, every=5):
            base = _load(GAME, seat, payload)
            stage = "meeple" if base.get_state_dict()["stage"] == 1 else "place"
            for g in (1, 2, 3):
                turned = _load(GAME, seat, _turned(payload, g))
                for me in (0, 1):
                    a, b = base._encode(me), turned._encode(me)
                    feats, (mask,) = _apply(g, a["features"], [a["legal_mask"]])
                    assert _token_set(feats) == _token_set(b["features"]), (seed, stage, g, me)
                    np.testing.assert_array_equal(mask, np.asarray(b["legal_mask"], dtype=np.float32))
                checked[stage] += 1
                sd = base.get_state_dict()
                if stage == "place":
                    short_period += ref.rotation_period(sd["current_tile"]) < 4
                else:
                    legal = [a for a in base.get_legal_actions() if MEEPLE_BASE <= a < MEEPLE_BASE + 12]
                    plain = sorted(MEEPLE_BASE + _turn_position(a - MEEPLE_BASE, g) for a in legal)
                    moved_meeple_action += plain != sorted(
                        a for a in turned.get_legal_actions() if MEEPLE_BASE <= a < MEEPLE_BASE + 12)
    assert checked["place"] > 100 and checked["meeple"] > 100, checked
    assert short_period > 10, "tiles with fewer than four distinct rotations were barely placed"
    assert moved_meeple_action > 10, "no meeple action had to move to the lowest side of its feature"


def _image(g, features, action, space):
    row = np.zeros(space, dtype=np.float32)
    row[action] = 1.0
    _, (out,) = _apply(g, features, [row])
    assert out.sum() == 1.0 and out.max() == 1.0
    return int(out.argmax())


def test_an_action_and_its_image_score_the_same():
    space = engine.game_metadata(GAME)["action_space"]
    rng = random.Random(9)
    played = scored = 0
    for seat, payload in _positions(44, every=3):
        g = rng.choice((1, 2, 3))
        base, turned = _load(GAME, seat, payload), _load(GAME, seat, _turned(payload, g))
        feats = base._encode(seat)["features"]
        legal = base.get_legal_actions()
        images = sorted(_image(g, feats, a, space) for a in legal)
        assert images == sorted(turned.get_legal_actions())
        action = rng.choice(legal)
        before = base.get_state_dict()["scores"]
        base.apply_action(action)
        turned.apply_action(_image(g, feats, action, space))
        sa, sb = base.get_state_dict(), turned.get_state_dict()
        for key in ("scores", "meeples_remaining", "projected_final_scores", "stage", "is_terminal"):
            assert sa[key] == sb[key], key
        played += 1
        scored += sa["scores"] != before
    assert played > 40 and scored > 3


def test_turns_compose_and_padding_is_untouched():
    rng = np.random.default_rng(3)
    positions = _positions(45, every=11)
    rows, masks, policies = [], [], []
    for seat, payload in positions:
        enc = _load(GAME, seat, payload)._encode(seat)
        mask = np.asarray(enc["legal_mask"], dtype=np.float32)
        rows.append(np.asarray(enc["features"], dtype=np.float32))
        masks.append(mask)
        policies.append(mask * rng.random(len(mask)).astype(np.float32))
    x = pad_token_rows(rows, TOKEN_DIM)
    assert len({len(r) for r in rows}) > 1 and x.shape[1] > max(len(r) for r in rows), "the batch has no padding"
    mask, policy = np.stack(masks), np.stack(policies)
    aug = SampleSymmetryAugmenter(GAME, 4)
    sym = aug.sample(np.random.default_rng(0), len(rows))
    assert set(sym.tolist()) == {0, 1, 2, 3}
    one = np.ones(len(rows), dtype=np.int64)

    feats, (m, p) = aug.transform(x, [mask, policy], sym)
    for i, r in enumerate(rows):
        assert not feats[i, len(r):].any(), "padding tokens changed"
        if sym[i] == 0:
            np.testing.assert_array_equal(feats[i], x[i])
        np.testing.assert_array_equal(p[i] > 0, m[i] > 0)
        assert np.isclose(p[i].sum(), policy[i].sum(), rtol=1e-6)
    # g, then quarter turns until four in total, is the identity.
    for i in range(len(rows)):
        f, mm, pp = feats[i:i + 1], m[i:i + 1], p[i:i + 1]
        for _ in range((4 - sym[i]) % 4):
            f, (mm, pp) = aug.transform(f, [mm, pp], one[:1])
        assert _token_set(f[0, :len(rows[i])]) == _token_set(rows[i])
        np.testing.assert_array_equal(mm[0], mask[i])
        np.testing.assert_allclose(pp[0], policy[i], rtol=1e-6)
    # The inputs are not modified.
    np.testing.assert_array_equal(x, pad_token_rows(rows, TOKEN_DIM))


def test_weight_on_something_that_is_no_action_is_rejected():
    space = engine.game_metadata(GAME)["action_space"]
    place = meeple = None
    for seat, payload in _positions(46, every=1):
        gs = _load(GAME, seat, payload)
        sd = gs.get_state_dict()
        if sd["stage"] == 0 and place is None and ref.rotation_period(sd["current_tile"]) < 4:
            place = (gs._encode(seat)["features"], ref.rotation_period(sd["current_tile"]))
        if sd["stage"] == 1 and meeple is None:
            meeple = gs._encode(seat)["features"]
    assert place is not None and meeple is not None
    row = np.zeros(space, dtype=np.float32)
    row[5 * 4 + place[1]] = 1.0             # a rotation at the tile's period repeats rotation 0
    with pytest.raises(ValueError, match="placement that is no action"):
        _apply(1, place[0], [row])
    row = np.zeros(space, dtype=np.float32)
    row[MEEPLE_BASE] = 1.0
    with pytest.raises(ValueError, match="meeple outside the meeple stage"):
        _apply(1, place[0], [row])
    row = np.zeros(space, dtype=np.float32)
    row[0] = 1.0
    with pytest.raises(ValueError, match="placement that is no action"):
        _apply(1, meeple, [row])
    with pytest.raises(ValueError, match="symmetry out of range"):
        _apply(4, meeple, [])
    with pytest.raises(ValueError, match="no sample-dependent symmetries"):
        engine.apply_sample_symmetries("tictactoe", np.array([1]), np.zeros((1, 27), dtype=np.float32), [])


def _train_cfg():
    return json.loads((ROOT / "games/carcassonne/config/train_transformer_guided.json").read_text(encoding="utf-8"))


def test_config_enables_augmentation_and_rejects_the_opponent_policy_head():
    meta = engine.game_metadata(GAME)
    kwargs = dict(feature_dim=0, action_space=meta["action_space"], token_layout=meta["token_layout"])
    training = _train_cfg()["training"]
    assert training["symmetry_augmentation"] is True and "opponent_policy_head" not in training
    aug = parse_symmetry_augmentation(training, GAME, **kwargs)
    assert isinstance(aug, SampleSymmetryAugmenter) and aug.count == 4
    with pytest.raises(ValueError, match="cannot be combined with training.opponent_policy_head"):
        parse_symmetry_augmentation({**training, "opponent_policy_head": {"loss_weight": 0.15}}, GAME, **kwargs)
    with pytest.raises(ValueError, match="no board symmetries"):
        parse_symmetry_augmentation({"symmetry_augmentation": True}, "carcassonne", feature_dim=1,
                                    action_space=meta["action_space"], token_layout=None)


def test_training_loop_with_symmetry_augmentation(tmp_path):
    """Two steps of the real pipeline (compact fp16 replay, quarter-turn
    augmentation, variable-length tokens) on CPU."""
    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(GAME)
    cfg["max_game_plies"] = 24
    train_cfg = _train_cfg()
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {"name": "rule", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "replay_buffer_size": 2000,
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    assert (tmp_path / "models" / "model_latest.onnx").exists()
