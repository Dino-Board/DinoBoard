"""harmonies_tok / harmonies_a_tok: HarmoniesTokenEncoder (side B / side A, 2 players) + the token
transformer.

* Layout contract: every action is bound by anchor to exactly one (anchor,
  head), except ending the turn (global head).
* Every token feature of real states (side B: 30 random + 10 greedy games, side A: 12 + 4) equals an
  independent recomputation from the state dict (harmonies_reference.py).
* Each legal action is scored by exactly one (token, head) present in the
  encoding, and no two legal actions share a logit.
* docs/games/harmonies_api.md lists exactly the layout the engine reports; side A's layout is side B's
  with the water columns named for the river.
* The train config builds the network, and a tiny CPU run of the real
  training loop works with symmetry augmentation on.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine

from . import harmonies_reference as ref

GAME = "harmonies_tok"
GAMES = ["harmonies_tok", "harmonies_a_tok"]
RIVER_NAMES = {"islands": "river", "island": "river_reach", "island_split": "river_join"}
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/games/harmonies_api.md"
NET_CFG = {"d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64}


def _meta(game=GAME):
    return engine.game_metadata(game)


def _layout_json(game=GAME):
    return json.loads(engine.GameSession(game, seed=0).get_state_dict()["encoder_layout"])


def _tokens(features, game=GAME):
    return np.asarray(features, dtype=np.float32).reshape(-1, _meta(game)["token_layout"]["token_dim"])


@pytest.mark.parametrize("game", GAMES)
def test_layout_contract(game):
    meta = _meta(game)
    lay = meta["token_layout"]
    js = _layout_json(game)
    n = 25 if game == "harmonies_tok" else 23
    assert meta["action_space"] == 5 + 32 + n * 6 + n * 32 + 1
    assert meta["feature_dim"] == 0 and lay["variable_tokens"] and lay["token_positions"]
    assert lay["token_dim"] == js["token_dim"] == 86 and lay["num_pointer_heads"] == js["num_heads"] == 6
    assert lay["num_token_types"] == len(js["types"]) == 7
    assert set(lay["action_token"]) == {-1} and set(lay["action_head"]) == {-1}
    for t in js["types"]:  # contiguous feature tables inside the token
        at = 0
        for name, off, width in t["features"]:
            assert off == at, (t["name"], name)
            at += width
        assert at <= lay["token_dim"] - js["header"]
    anchors = {a[0]: (a[1], a[2]) for a in js["anchors"]}
    assert sum(c for _, c in anchors.values()) == lay["num_anchor_ids"] == n + 5 + 32 + n * 32
    aa = np.asarray(lay["anchor_action"]).reshape(lay["num_anchor_ids"], 6)
    bound = aa[aa >= 0]
    end_turn = meta["action_space"] - 1
    assert len(bound) == len(set(bound.tolist())) == meta["action_space"] - 1
    assert end_turn not in bound
    cell, cube = anchors["cell"][0], anchors["cube"][0]
    assert aa[cell - 1 + 4].tolist() == [37 + 4 * 6 + c for c in range(6)]
    assert aa[cube - 1 + 3 * 32 + 9].tolist() == [37 + n * 6 + 3 * 32 + 9] + [-1] * 5
    assert aa[anchors["slot"][0] - 1 + 2].tolist() == [2] + [-1] * 5
    assert aa[anchors["market_card"][0] - 1 + 6].tolist() == [5 + 6] + [-1] * 5


def _play(seed, policy, game=GAME):
    """(state dict, perspective, encoding, legal) along a game, every few plies."""
    gs = engine.GameSession(game, seed=seed)
    rng = random.Random(seed)
    ply = 0
    while not gs.is_terminal:
        if ply % 7 == seed % 7:
            d = gs.get_state_dict()
            me = d["current_player"] if ply % 5 else 1 - d["current_player"]
            yield d, me, gs._encode(me), gs.get_legal_actions()
        a = gs.get_heuristic_action("greedy", 0.0)["action"] if policy == "greedy" else rng.choice(gs.get_legal_actions())
        gs.apply_action(a)
        ply += 1


def _bindings(x, lay):
    """action -> list of (token row, head) scoring it, over the tokens present."""
    aa = np.asarray(lay["anchor_action"]).reshape(lay["num_anchor_ids"], lay["num_pointer_heads"])
    out = {}
    for i, t in enumerate(x):
        if t[2] > 0:
            for h, a in enumerate(aa[int(t[2]) - 1]):
                if a >= 0:
                    out.setdefault(int(a), []).append((i, h))
    return out


# Games played per side, and the least they must reach (states, cube tokens, cells with a nonzero
# card_advance). Side A shares side B's layout and code but for the water columns, so fewer games.
REFERENCE_GAMES = {
    "harmonies_tok": ((("random", range(30)), ("greedy", range(100, 110))), (300, 30, 300)),
    "harmonies_a_tok": ((("random", range(12)), ("greedy", range(100, 104))), (150, 15, 300)),
}


@pytest.mark.parametrize("game", GAMES)
def test_every_feature_matches_independent_reference(game):
    lay = _meta(game)["token_layout"]
    js = _layout_json(game)
    end_turn = _meta(game)["action_space"] - 1
    states = cubes = advances = 0
    plan, (min_states, min_cubes, min_advances) = REFERENCE_GAMES[game]
    for policy, seeds in plan:
        for seed in seeds:
            for d, me, enc, legal in _play(seed, policy, game):
                x = _tokens(enc["features"], game)
                assert np.isfinite(x).all() and (x[:, 0] == 1).all()
                exp = ref.expected_tokens(d, me)
                problems = ref.differences(x, exp, js, ref.Geometry(d["board_columns"]))
                assert not problems, f"{policy} seed {seed} perspective {me}: {problems[:3]}"
                if me == d["current_player"]:
                    assert set(np.flatnonzero(enc["legal_mask"]).tolist()) == set(legal)
                    bind = _bindings(x, lay)
                    for a in legal:
                        if a == end_turn:
                            assert a not in bind
                        else:
                            assert len(bind.get(a, [])) == 1, f"action {a} bound {bind.get(a)}"
                    used = [bind[a][0] for a in legal if a != end_turn]
                    assert len(used) == len(set(used)), "two legal actions share a logit"
                states += 1
                cubes += len(exp["cube"])
                advances += sum(any(v != 0 for v in c["card_advance"]) for c in exp["cell"].values())
    assert states > min_states and cubes > min_cubes and advances > min_advances, (states, cubes, advances)


def _doc_tables(text):
    tables, lines = {}, text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"#### `(\w+)`", line)
        if not m:
            continue
        start = next(j for j in range(i + 1, len(lines)) if lines[j].startswith("|"))
        end = next((j for j in range(start, len(lines)) if not lines[j].startswith("|")), len(lines))
        tables[m.group(1)] = [[c.strip() for c in row.strip().strip("|").split("|")] for row in lines[start:end]][2:]
    return tables


def test_docs_list_every_feature_anchor_and_entity():
    js, tables = _layout_json(), _doc_tables(DOC.read_text(encoding="utf-8"))
    for t in js["types"]:
        rows = tables[t["name"]]
        documented = [(r[2].strip("`"), int(r[0].split("-")[0]), int(r[1])) for r in rows]
        assert documented == [tuple(f) for f in [(n, o, w) for n, o, w in t["features"]]], t["name"]
        for r in rows:
            first, _, last = r[0].partition("-")
            assert int(last or first) == int(first) + int(r[1]) - 1
    assert [(r[0].strip("`"), int(r[1]), int(r[2])) for r in tables["anchors"]] == [tuple(a) for a in js["anchors"]]
    assert [(r[0].strip("`"), int(r[1]), int(r[2])) for r in tables["entities"]] == [tuple(e) for e in js["entities"]]


def test_side_a_layout_is_side_b_with_river_names():
    b, a = _layout_json("harmonies_tok"), _layout_json("harmonies_a_tok")
    assert a["token_dim"] == b["token_dim"] and a["header"] == b["header"]
    for ta, tb in zip(a["types"], b["types"]):
        assert ta["name"] == tb["name"]
        assert ta["features"] == [[RIVER_NAMES.get(n, n), o, w] for n, o, w in tb["features"]], ta["name"]
    text = DOC.read_text(encoding="utf-8")
    for name in RIVER_NAMES.values():
        assert f"`{name}`" in text, f"{name} is not documented"


def _train_cfg():
    return json.loads((ROOT / "games/harmonies/config/train_transformer.json").read_text(encoding="utf-8"))


def test_train_config_builds_model_and_enables_augmentation():
    from training.cli import validate_guidance_until
    from training.model import create_model_from_config
    path = ROOT / "games/harmonies/config/train_transformer.json"
    cfg = _train_cfg()
    validate_guidance_until(cfg["heuristic_guidance"], path)
    assert cfg["training"]["symmetry_augmentation"] is True
    meta = _meta()
    net = create_model_from_config(cfg["network"], input_dim=0, policy_dim=meta["action_space"],
                                   num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert net.token_positions and not hasattr(net, "anchor_embed")


@pytest.mark.parametrize("game", GAMES)
def test_training_loop_with_symmetry_augmentation(game, tmp_path):
    """Two steps of the real pipeline (compact fp16 replay, symmetry
    augmentation, variable-length tokens) on CPU."""
    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(game)
    cfg["max_game_plies"] = 40
    train_cfg = _train_cfg()
    train_cfg["heuristic"], _ = find_heuristic_config(game)
    train_cfg["network"] = {**train_cfg["network"], **NET_CFG}
    train_cfg["heuristic_guidance"] = {"name": "greedy", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    run_training_loop(game_id=game, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    latest = tmp_path / "models" / "model_latest.onnx"
    assert latest.exists()
    ep = engine.run_selfplay_episode(game_id=game, seed=3, model_path=str(latest), simulations=4, max_game_plies=30)
    assert ep["samples"]


def test_augmentation_is_rejected_without_symmetries():
    from training.symmetry import parse_symmetry_augmentation
    with pytest.raises(ValueError, match="no board symmetries"):
        parse_symmetry_augmentation({"symmetry_augmentation": True}, "tictactoe", feature_dim=1, action_space=9,
                                    token_layout=None)
    with pytest.raises(ValueError, match="must be a bool"):
        parse_symmetry_augmentation({"symmetry_augmentation": 1}, GAME, feature_dim=0, action_space=988,
                                    token_layout=None)
    assert parse_symmetry_augmentation({}, "tictactoe", feature_dim=1, action_space=9, token_layout=None) is None
