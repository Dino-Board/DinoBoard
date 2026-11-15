"""6nimmit: SixNimmtTokenEncoder + the token transformer, two to four players.

* Layout contract: every action is bound by anchor to exactly one (anchor,
  head); no token carries a card id as an entity; the layout is the same for
  every player count.
* Every token of whole games (random play and the heuristics, every seat at
  every ply) equals an independent recomputation (nimmt_token_reference.py)
  from what that seat could have written down: its hand, the table, the
  bullheads taken, and its own record of every card shown
  (nimmt_game_model.py), kept from the deal and the actions, not read from
  the engine. The reference never reads another player's hand or face-down
  card.
* Each legal action is scored by exactly one (token, head) of the encoding.
* The network cannot tell which card a token is except through its features:
  exchanging the anchors of two hand cards exchanges their logits.
* docs/games/6nimmit_api.md lists exactly the layout the engine reports.
* The train config builds the network, and a tiny CPU run of the real
  training loop works.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine

sys.path.insert(0, str(Path(__file__).parent))
import nimmt_game_model as gm  # noqa: E402
import nimmt_token_reference as ref  # noqa: E402

GAMES = list(gm.GAMES)
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/games/6nimmit_api.md"
TRAIN_CFG = ROOT / "games/6nimmit/config/train.json"
NUM_CARDS, NUM_ROWS = 104, 4


def _layout_json(game="6nimmit"):
    return json.loads(engine.GameSession(game, seed=0).get_state_dict()["encoder_layout"])


def _tokens(features, lay):
    return np.asarray(features, dtype=np.float32).reshape(-1, lay["token_dim"])


@pytest.mark.parametrize("game", GAMES)
def test_layout_contract(game):
    meta = engine.game_metadata(game)
    lay, js = meta["token_layout"], _layout_json(game)
    assert meta["num_players"] == gm.GAMES[game] and meta["action_space"] == NUM_CARDS + NUM_ROWS
    assert meta["feature_dim"] == 0 and lay["variable_tokens"] and lay["token_positions"]
    assert lay["num_entity_ids"] == 1, "a card id as an entity would be a one-hot card again"
    assert [t["name"] for t in js["types"]] == ["global", "player", "row", "card"]
    assert lay["num_token_types"] == 4 and lay["num_pointer_heads"] == len(js["heads"]) == 2
    widest = 0
    for t in js["types"]:  # contiguous feature tables inside the token
        at = 0
        for name, off, width in t["features"]:
            assert off == at, (t["name"], name)
            at += width
        widest = max(widest, at)
    assert lay["token_dim"] == js["token_dim"] == js["header"] + widest == 7 + 29
    assert js == _layout_json("6nimmit"), "one layout for every player count"
    assert set(lay["action_token"]) == {-1} and set(lay["action_head"]) == {-1}
    assert js["anchors"] == [["card", 1, NUM_CARDS], ["row", NUM_CARDS + 1, NUM_ROWS]]
    assert lay["num_anchor_ids"] == NUM_CARDS + NUM_ROWS
    aa = np.asarray(lay["anchor_action"]).reshape(lay["num_anchor_ids"], 2)
    play, take = js["heads"].index("play"), js["heads"].index("take")
    for card in range(1, NUM_CARDS + 1):  # anchor id = card number, action = play that card
        assert aa[card - 1, play] == card - 1 and aa[card - 1, take] == -1
    for row in range(NUM_ROWS):
        assert aa[NUM_CARDS + row, take] == NUM_CARDS + row and aa[NUM_CARDS + row, play] == -1
    assert meta["belief_feature_dim"] == 0, "hands are sampled uniformly, without a belief network"


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


POLICIES = [("random", gm.random_policy, range(30)), ("rule_v1", gm.heuristic_policy("rule_v1"), range(100, 125)),
            ("rule_v1_noisy", gm.heuristic_policy("rule_v1", epsilon=0.15), range(200, 225)),
            ("lkf_v1", gm.heuristic_policy("lkf_v1"), range(300, 304))]


@pytest.mark.parametrize("game", GAMES)
def test_every_feature_matches_independent_reference(game):
    n = gm.GAMES[game]
    lay, js = engine.game_metadata(game)["token_layout"], _layout_json(game)
    seen = dict(states=0, choosing=0, own_face_down=0, sixth=0, safe=0, small=0, preview_me=0, preview_others=0,
                ahead=0, behind=0)
    max_tokens = 0
    for name, policy, seeds in POLICIES:
        for seed in seeds:
            for gs, model, d in gm.drive(engine, game, seed, policy):
                if model.terminal:
                    continue
                legal = gs.get_legal_actions()
                for me in range(n):
                    view = model.view(me)
                    enc = gs._encode(me)
                    x = _tokens(enc["features"], lay)
                    assert np.isfinite(x).all() and (x[:, 0] == 1).all() and (x[:, 1] == 0).all()
                    max_tokens = max(max_tokens, len(x))
                    tokens = ref.expected_tokens(view)
                    problems = ref.differences(x, tokens, js)
                    assert not problems, f"{name} seed {seed} ply {d['ply']} seat {me}: {problems[:3]}"
                    anchors = x[x[:, 2] > 0, 2].tolist()
                    assert len(anchors) == len(set(anchors)), "two tokens share an anchor"
                    if me == d["current_player"]:
                        assert view["to_move"]
                        assert set(np.flatnonzero(enc["legal_mask"]).tolist()) == set(legal)
                        bind = _bindings(x, lay)
                        assert set(bind) == {c - 1 for c in view["hand"]} | {NUM_CARDS + r for r in range(NUM_ROWS)}
                        assert all(len(bind[a]) == 1 for a in legal)
                    cards = [t["feats"] for t in tokens if t["type"] == "card"]
                    rows = [t["feats"] for t in tokens if t["type"] == "row"]
                    players = [t["feats"] for t in tokens if t["type"] == "player"]
                    seen["states"] += 1
                    seen["choosing"] += view["choosing_row"]
                    seen["own_face_down"] += bool(view["shown"]) and not view["choosing_row"]
                    seen["sixth"] += any(c["sixth"] for c in cards)
                    seen["safe"] += any(c["gap_safe"] for c in cards)
                    seen["small"] += any(c["small"] for c in cards)
                    seen["preview_me"] += any(r["if_taken_mine"] > 0 for r in rows)
                    seen["preview_others"] += any(r["if_taken_others"] > 0 for r in rows)
                    seen["ahead"] += players[0]["margin"] > 0
                    seen["behind"] += players[0]["margin"] < 0
    assert seen["states"] > 2500, seen
    assert all(v > 20 for v in seen.values()), seen
    # Before the cards are turned over: my hand and my own played card. After: one card fewer
    # in the hand, and every player's card face up.
    assert max_tokens == 1 + n + NUM_ROWS + max(10 + 1, 9 + n)


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


def test_docs_list_every_feature_and_anchor():
    js, tables = _layout_json(), _doc_tables(DOC.read_text(encoding="utf-8"))
    for t in js["types"]:
        rows = tables[t["name"]]
        documented = [(r[2].strip("`"), int(r[0].split("-")[0]), int(r[1])) for r in rows]
        assert documented == [tuple(f) for f in t["features"]], t["name"]
        for r in rows:
            first, _, last = r[0].partition("-")
            assert int(last or first) == int(first) + int(r[1]) - 1
    assert [[r[0].strip("`"), int(r[1]), int(r[2])] for r in tables["anchors"]] == js["anchors"]


def _train_cfg():
    return json.loads(TRAIN_CFG.read_text(encoding="utf-8"))


def _network(game="6nimmit", seed=0, **overrides):
    import torch
    from training.model import create_model_from_config
    meta = engine.game_metadata(game)
    torch.manual_seed(seed)
    net = create_model_from_config({**_train_cfg()["network"], **overrides}, input_dim=0,
                                   policy_dim=meta["action_space"], num_players=meta["num_players"],
                                   token_layout=meta["token_layout"])
    return net.eval()


@pytest.mark.parametrize("game", GAMES)
def test_train_config_builds_a_network_without_card_ids(game):
    from training.cli import validate_guidance_until
    cfg = _train_cfg()
    validate_guidance_until(cfg["heuristic_guidance"], TRAIN_CFG)
    gs = engine.GameSession(game, seed=0)  # the warm-start heuristic exists and plays
    assert gs.get_heuristic_action(cfg["heuristic_guidance"]["name"], 0.0)["action"] in gs.get_legal_actions()
    assert cfg["network"]["anchor_embedding"] is False, "an anchor embedding is a learned vector per card number"
    net = _network(game)
    assert net.token_positions and not hasattr(net, "anchor_embed")
    assert net.entity_embed.num_embeddings == 1


@pytest.mark.parametrize("game", ["6nimmit", "6nimmit_4p"])
def test_a_card_is_scored_by_what_it_is_not_by_its_number(game):
    """The anchor only says which action a token's logit belongs to: give two hand
    cards each other's anchors and their logits change places, nothing else moves."""
    import torch
    lay = engine.game_metadata(game)["token_layout"]
    net = _network(game, seed=3, attention="matmul", d_model=32, num_layers=2, ffn_dim=64)
    gs = engine.GameSession(game, seed=5)
    hand = gs.get_state_dict()["players"][0]["hand"]
    x = _tokens(gs._encode(0)["features"], lay)
    a, b = hand[2], hand[7]
    ia, ib = (int(np.flatnonzero(x[:, 2] == c)[0]) for c in (a, b))
    swapped = x.copy()
    swapped[ia, 2], swapped[ib, 2] = b, a
    with torch.no_grad():
        p0, v0 = net(torch.from_numpy(x.reshape(1, -1)))
        p1, v1 = net(torch.from_numpy(swapped.reshape(1, -1)))
    p0, p1 = p0[0].numpy(), p1[0].numpy()
    assert abs(p0[a - 1] - p0[b - 1]) > 1e-4, "pick two cards the network tells apart"
    assert np.allclose(p1[a - 1], p0[b - 1], atol=1e-5) and np.allclose(p1[b - 1], p0[a - 1], atol=1e-5)
    rest = [c - 1 for c in hand if c not in (a, b)] + [NUM_CARDS + r for r in range(NUM_ROWS)]
    assert np.allclose(p1[rest], p0[rest], atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)


def test_the_same_hand_reads_differently_once_cards_are_known_to_be_gone():
    """Card counting reaches the network: with more cards marked as shown in this hand, the same
    hand and table encode differently, and only through the counts of unseen cards."""
    game = "6nimmit"
    lay, js = engine.game_metadata(game)["token_layout"], _layout_json(game)
    gs = engine.GameSession(game, seed=9)
    d = gs.get_state_dict()
    hand, table = d["players"][0]["hand"], [[c for c in row if c] for row in d["table"]]
    view = {"me": 0, "players": 2, "round": 0, "choosing_row": False, "chooser": -1, "to_move": True, "hand": hand,
            "table": table, "gone": set(d["used"]), "shown": {}, "heads": [0, 0]}
    fresh = _tokens(gs._encode(0)["features"], lay)
    assert not ref.differences(fresh, ref.expected_tokens(view), js)
    # 16 more cards, spread over the deck, are known to be gone (as many as eight rounds show).
    in_play = set(hand) | set(d["players"][1]["hand"]) | set(d["used"])
    earlier = sorted(set(range(1, 105)) - in_play)[::5][:16]
    for c in earlier:
        gs._force_write_field("used", [c - 1], 1)
    late = _tokens(gs._encode(0)["features"], lay)
    view.update(gone=set(d["used"]) | set(earlier))
    assert not ref.differences(late, ref.expected_tokens(view), js)
    assert len(late) == len(fresh)
    touched = set()  # names of the header fields / features that differ
    for i, j in zip(*np.nonzero(~np.isclose(fresh, late))):
        if j < js["header"]:
            touched.add(js["header_fields"][j])
        else:
            features = js["types"][int(fresh[i, 3])]["features"]
            touched |= {name for name, off, width in features if off <= j - js["header"] < off + width}
    allowed = {"x", "unseen", "density", "low_unseen", "low_expected", "reach_unseen", "reach_expected",
               "reach_heads", "percentile", "below_expected", "gap_safe", "gap_unseen", "gap_expected"}
    assert touched <= allowed, sorted(touched - allowed)
    assert {"x", "unseen", "density", "gap_unseen"} <= touched


@pytest.mark.parametrize("game", ["6nimmit", "6nimmit_3p"])
def test_training_loop(game, tmp_path):
    """Two steps of the real pipeline (whole games, variable-length tokens, positions) on CPU."""
    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(game)
    train_cfg = _train_cfg()
    train_cfg["heuristic"], _ = find_heuristic_config(game)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 2, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {"name": "rule_v1", "hold_steps": 1, "steps": 2, "initial_ratio": 1.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "simulations_start": 4}
    train_cfg["mcts"]["selfplay"]["simulations"] = 4
    train_cfg["mcts"]["eval"]["simulations"] = 4
    run_training_loop(game_id=game, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    latest = tmp_path / "models" / "model_latest.onnx"
    assert latest.exists()
    ep = engine.run_selfplay_episode(game_id=game, seed=3, model_path=str(latest), simulations=4,
                                     max_game_plies=cfg["max_game_plies"])
    assert ep["samples"] and ep["custom_stats"]["most_heads"] >= ep["custom_stats"]["fewest_heads"] >= 0
