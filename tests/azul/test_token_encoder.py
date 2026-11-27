"""azul_tok / azul_tok_3p / azul_tok_4p: AzulTokenEncoder + the token transformer.

* Layout contract: every action is pointer head `target` on the pile
  (source, colour) it takes.
* Every token of real positions equals an independent recomputation from the
  state dict, and the pile tokens show exactly the legal moves.
* The order of the factories does not matter: reordering them gives the same
  set of tokens, and the network returns the same value and the same policy
  for the corresponding moves.
* The old MLP (pv_512x4, flat encoder) keeps working inside the token
  variants: it plays the same moves as in the flat variants and can guide the
  training of the transformer.
* The train config builds the network and runs through the real pipeline.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from conftest import get_test_model
from training.model import TokenTransformerPVNet, create_model_from_config, export_onnx

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CFG = ROOT / "games/azul/config/train_transformer.json"
VARIANTS = [("azul_tok", "azul", 2), ("azul_tok_3p", "azul_3p", 3), ("azul_tok_4p", "azul_4p", 4)]
GAME = "azul_tok"
NET_CFG = {"type": "token_transformer", "d_model": 64, "num_layers": 2, "num_heads": 4, "ffn_dim": 128}

COLORS = ROWS = 5
TARGETS = 6            # five pattern lines + the floor
FLOOR_TARGET = 5
HEADER = 2             # [valid, entity id]
GLOBAL, PLAYER, ROW, PILE = range(4)
FLOOR_COST = [1, 1, 2, 2, 2, 3, 3]


def _meta(game=GAME):
    return engine.game_metadata(game)


def _bases(players):
    row_base = 1 + players
    pile_base = row_base + players * ROWS
    return row_base, pile_base, pile_base + (2 * players + 2) * COLORS


def _tokens(features, game=GAME):
    lay = _meta(game)["token_layout"]
    return np.asarray(features, dtype=np.float32).reshape(lay["num_tokens"], lay["token_dim"])


def _net(game=GAME, seed=0):
    meta = _meta(game)
    torch.manual_seed(seed)
    net = create_model_from_config(
        NET_CFG, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
        num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)
    return net.eval()


def _random_positions(game, seeds, every=1):
    """Sessions along uniformly random games (they reach floors that overflow,
    full lines and late rounds much more often than search does)."""
    for seed in seeds:
        rng = random.Random(seed)
        gs = engine.GameSession(game, seed=seed)
        ply = 0
        while not gs.is_terminal and ply < 300:
            if ply % every == 0:
                yield gs
            gs.apply_action(rng.choice(gs.get_legal_actions()))
            ply += 1


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_layout_contract(game, flat, players):
    meta, flat_meta = _meta(game), _meta(flat)
    lay = meta["token_layout"]
    row_base, pile_base, num_tokens = _bases(players)
    assert flat_meta["token_layout"] is None
    assert meta["num_players"] == players and meta["action_space"] == flat_meta["action_space"]
    assert lay["num_tokens"] == num_tokens and lay["num_tokens"] * lay["token_dim"] == meta["feature_dim"]
    assert not lay["variable_tokens"] and not lay["token_positions"]
    assert lay["num_entity_ids"] == 1 and lay["num_pointer_heads"] == TARGETS and lay["num_token_types"] == 4
    assert lay["token_types"] == ([GLOBAL] + [PLAYER] * players + [ROW] * (players * ROWS)
                                  + [PILE] * (num_tokens - pile_base))
    # action = (source * 5 + colour) * 6 + target; no action uses the global head
    assert lay["action_token"] == [pile_base + a // TARGETS for a in range(meta["action_space"])]
    assert lay["action_head"] == [a % TARGETS for a in range(meta["action_space"])]
    assert meta["action_space"] == (num_tokens - pile_base) * TARGETS


# ---------------------------------------------------------------------------
# Token contents against an independent reference
# ---------------------------------------------------------------------------

def _wall_col(row, color):
    return (color + row) % COLORS


def _floor_cost(first_slot, tiles):
    return sum(FLOOR_COST[first_slot:first_slot + tiles])


def _placement_score(wall, row, col):
    def run(dr, dc):
        n, r, c = 0, row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLORS and wall[r][c]:
            n, r, c = n + 1, r + dr, c + dc
        return n
    across, down = 1 + run(0, -1) + run(0, 1), 1 + run(-1, 0) + run(1, 0)
    return across + down if across > 1 and down > 1 else max(across, down)


def _final_bonus(wall):
    rows = sum(all(r) for r in wall)
    cols = sum(all(wall[r][c] for r in range(ROWS)) for c in range(COLORS))
    sets = sum(all(wall[r][_wall_col(r, k)] for r in range(ROWS)) for k in range(COLORS))
    return 2 * rows + 7 * cols + 10 * sets


def _settle(player):
    """Walls each row is scored against, points per row, and whether the
    settlement completes a wall row (rows settle top to bottom)."""
    wall = [list(r) for r in player["wall"]]
    walls, gains = [], []
    for row, line in enumerate(player["pattern_lines"]):
        walls.append([list(r) for r in wall])
        gain = 0
        if line["length"] == row + 1:
            col = _wall_col(row, line["color"])
            if not wall[row][col]:
                wall[row][col] = 1
                gain = _placement_score(wall, row, col)
        gains.append(gain)
    return walls, gains, any(all(r) for r in wall)


def _fits(player, row, color):
    line = player["pattern_lines"][row]
    if line["length"] >= row + 1 or (line["length"] > 0 and line["color"] != color):
        return False
    return not player["wall"][row][_wall_col(row, color)]


def _reference_tokens(sd, starter, me, token_dim):
    players = sd["num_players"]
    factories = 2 * players + 1
    row_base, pile_base, num_tokens = _bases(players)
    tok = np.zeros((num_tokens, token_dim), dtype=np.float64)
    marker = bool(sd["first_player_token_in_center"])
    settled = [_settle(p) for p in sd["players"]]

    g = tok[0]
    g[0] = 1
    g = g[HEADER:]
    g[0] = min(sd["round_index"], 10) / 10
    for c in range(COLORS):
        g[1 + c] = sd["bag_counts"][c] / 20
        g[6 + c] = sd["box_counts"][c] / 20
    g[11] = sum(sd["bag_counts"]) / 100
    g[12] = marker
    g[13] = sd["current_player"] == me
    piles = [(f, c, n) for f, fac in enumerate(sd["factories"]) for c, n in enumerate(fac) if n]
    piles += [(factories, c, n) for c, n in enumerate(sd["center"]) if n]
    g[14] = sum(n for _, _, n in piles) / (4 * factories)
    g[15] = len(piles) / 20

    for pid, p in enumerate(sd["players"]):
        seat = (pid - me) % players
        walls, gains, completes = settled[pid]
        t = tok[1 + seat]
        t[0] = 1
        f = t[HEADER:]
        f[seat] = 1
        f[4] = sd["current_player"] == pid
        f[5] = (not marker) and starter == pid
        f[6] = p["score"] / 100
        f[7] = p["floor_count"] / 7
        f[8] = _floor_cost(0, p["floor_count"]) / 14
        f[9] = _floor_cost(p["floor_count"], 1) / 3
        assert len(p["floor"]) == p["floor_count"]
        for tile in p["floor"]:
            if tile >= 0:
                f[10 + tile] += 1 / 7
        f[15] = sum(gains) / 20
        f[16] = completes
        f[17] = _final_bonus(p["wall"]) / 50
        for c in range(COLORS):
            f[18 + c] = sum(p["wall"][r][c] for r in range(ROWS)) / 5
            f[23 + c] = sum(p["wall"][r][_wall_col(r, c)] for r in range(ROWS)) / 5
        for row, line in enumerate(p["pattern_lines"]):
            t = tok[row_base + seat * ROWS + row]
            t[0] = 1
            f = t[HEADER:]
            f[seat] = 1
            f[4 + row] = 1
            if line["length"] > 0:
                f[9 + line["color"]] = 1
            f[14] = line["length"] / (row + 1)
            f[15] = (row + 1 - line["length"]) / 5
            f[16] = line["length"] == row + 1
            for c in range(COLORS):
                col = _wall_col(row, c)
                f[22 + c] = p["wall"][row][c]
                if p["wall"][row][col]:
                    f[17 + c] = 1
                else:
                    f[27 + c] = _placement_score(walls[row], row, col) / 10
            f[32] = gains[row] / 10

    mine, nxt = sd["players"][me], sd["players"][(me + 1) % players]
    my_walls = settled[me][0]
    for source, color, count in piles:
        center = source == factories
        takes_marker = int(center and marker)
        t = tok[pile_base + source * COLORS + color]
        t[0] = 1
        f = t[HEADER:]
        f[0] = center
        f[1] = takes_marker
        f[2 + color] = 1
        f[7] = min(count, 16) / 8
        if not center:
            for c in range(COLORS):
                if c != color:
                    f[8 + c] = sd["factories"][source][c] / 4
        for row in range(ROWS):
            if not _fits(mine, row, color):
                continue
            free = row + 1 - mine["pattern_lines"][row]["length"]
            to_line = min(free, count)
            b = f[13 + 5 * row:]
            b[0] = 1
            b[1] = to_line / 5
            if to_line == free:
                b[2] = 1
                b[3] = _placement_score(my_walls[row], row, _wall_col(row, color)) / 10
            b[4] = _floor_cost(mine["floor_count"], count - to_line + takes_marker) / 14
        f[38] = _floor_cost(mine["floor_count"], count + takes_marker) / 14
        for row in range(ROWS):
            if _fits(nxt, row, color):
                f[39] = 1
                if count >= row + 1 - nxt["pattern_lines"][row]["length"]:
                    f[40] = 1
    return tok


def _starter(gs):
    """first_player_next_round is not in the state dict; the public snapshot has it."""
    return gs.extract_initial_observation(0)["public_snapshot"]["first_player_next_round"][1]


@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_tokens_match_reference(game, flat, players):
    lay = _meta(game)["token_layout"]
    row_base, pile_base, _ = _bases(players)
    checked = scoring_lines = marker_taken = overflow = 0
    for gs in _random_positions(game, range(4)):
        sd = gs.get_state_dict()
        starter = _starter(gs)
        for me in range(players):
            enc = gs._encode(me)
            tok = _tokens(enc["features"], game)
            np.testing.assert_allclose(tok, _reference_tokens(sd, starter, me, lay["token_dim"]),
                                       atol=1e-6, rtol=0)
            assert not tok[tok[:, 0] == 0].any(), "tokens that are not there must be all-zero"
            assert not tok[:, 1].any(), "there are no entity ids"
            assert np.abs(tok[:, HEADER:]).max() <= 2.0
            if me != sd["current_player"]:
                continue
            # The pile tokens show the legal moves: target t of a pile is legal
            # iff the pile is there and, for a pattern line, its "accepts" flag is set.
            piles = tok[pile_base:]
            shown = np.zeros((len(piles), TARGETS), dtype=bool)
            shown[:, :ROWS] = piles[:, HEADER + 13:HEADER + 38:5] == 1.0
            shown[:, FLOOR_TARGET] = piles[:, 0] == 1.0
            assert shown.reshape(-1).tolist() == [bool(v) for v in enc["legal_mask"]]
            checked += 1
            scoring_lines += int(tok[row_base:pile_base, HEADER + 32].any())
            marker_taken += int(tok[1:row_base, HEADER + 5].any())
            overflow += int(piles[:, HEADER + 17::5][:, :ROWS].any())
    # the random games reached the interesting cases
    assert checked > 150 and scoring_lines > 20 and marker_taken > 20 and overflow > 20


@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_final_scores_feed_the_score_head(game, flat, players):
    """final_score_extractor (training.score_head) reports each seat's points."""
    for g in (game, flat):
        assert _meta(g)["has_final_score_extractor"]
    finished = 0
    for seed in range(3):
        ep = engine.run_selfplay_episode(game_id=game, seed=seed, model_path=get_test_model(game), simulations=4,
                                         max_game_plies=400)
        scores = ep["final_scores"]
        assert len(scores) == players and all(v >= 0 and v == int(v) for v in scores)
        if ep["winner"] >= 0:
            finished += 1
            assert max(scores) > 0 and scores[ep["winner"]] == max(scores)
    assert finished > 0


# ---------------------------------------------------------------------------
# The order of the factories does not matter
# ---------------------------------------------------------------------------

def _with_factories_reordered(gs, game, seat, perm):
    """The same position with factory j holding what factory perm[j] held,
    built through the public snapshot (so the encoder sees a real state)."""
    snap = copy.deepcopy(gs.extract_initial_observation(seat)["public_snapshot"])
    flat = snap["factories"]
    held = {tuple(flat[i]): flat[i + 1] for i in range(0, len(flat), 2)}
    snap["factories"] = [x for j in range(len(perm)) for c in range(COLORS)
                         for x in ([j, c], held[(perm[j], c)])]
    other = engine.GameSession(game, 12345, "", False, "")
    other.apply_initial_observation(seat, {"public_snapshot": snap, "tracker_init": {}})
    return other


@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_factory_order_does_not_change_tokens_policy_or_value(game, flat, players):
    meta = _meta(game)
    factories = 2 * players + 1
    net = _net(game)
    rng = np.random.default_rng(0)
    checked = 0
    for gs in _random_positions(game, range(3), every=5):
        seat = gs.current_player
        sd = gs.get_state_dict()
        if sum(any(f) for f in sd["factories"]) < 2:
            continue
        perm = [int(v) for v in rng.permutation(factories)]
        while [sd["factories"][p] for p in perm] == sd["factories"]:
            perm = [int(v) for v in rng.permutation(factories)]
        other = _with_factories_reordered(gs, game, seat, perm)
        assert other.get_state_dict()["factories"] == [sd["factories"][p] for p in perm]
        a, b = gs._encode(seat), other._encode(seat)
        ta, tb = _tokens(a["features"], game), _tokens(b["features"], game)
        assert not np.array_equal(ta, tb)
        # Same tokens in other slots: nothing in a token says which factory it is.
        assert sorted(map(tuple, ta.tolist())) == sorted(map(tuple, tb.tolist()))
        # new action (source j, colour, target) is the old action on source perm[j]
        src = perm + [factories]
        act = np.asarray([(src[x // (COLORS * TARGETS)] * COLORS * TARGETS) + x % (COLORS * TARGETS)
                          for x in range(meta["action_space"])])
        assert [bool(v) for v in b["legal_mask"]] == [bool(a["legal_mask"][i]) for i in act]
        x = torch.from_numpy(np.stack([ta.reshape(-1), tb.reshape(-1)]))
        with torch.no_grad():
            pol, val = net(x)
        torch.testing.assert_close(val[1], val[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pol[1], pol[0][torch.from_numpy(act)], atol=1e-5, rtol=1e-5)
        legal = torch.tensor([bool(v) for v in a["legal_mask"]])
        assert pol[0][legal].std() > 1e-3, "the check needs a policy that tells moves apart"
        checked += 1
    assert checked >= 15


# ---------------------------------------------------------------------------
# Network, ONNX, old MLP inside the token variants, training
# ---------------------------------------------------------------------------

def _selfplay_features(game=GAME):
    ep = engine.run_selfplay_episode(game_id=game, seed=21, model_path=get_test_model(game), simulations=8,
                                     max_game_plies=40)
    feats = np.asarray([s["features"] for s in ep["samples"]], dtype=np.float32)
    assert len(feats) > 10
    return feats


def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    feats = _selfplay_features()
    net = _net(seed=1)
    path = tmp_path / "tok.onnx"
    export_onnx(net, path, _meta()["feature_dim"])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pol_o, val_o = sess.run(["policy", "value"], {"features": feats})
    with torch.no_grad():
        pol_t, val_t = net(torch.from_numpy(feats))
    np.testing.assert_allclose(pol_o, pol_t.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(val_o, val_t.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_old_mlp_plays_the_same_moves_inside_the_token_variant(game, flat, players):
    """pv_512x4 is the shipped MLP over the flat encoder. In the token variant
    it must still be fed flat features: same search, same moves, while the
    recorded samples carry tokens for the transformer to learn from."""
    kwargs = dict(seed=5, temperature=0.0, max_game_plies=4, heuristic_selector="pv_512x4")
    old = engine.run_heuristic_episode(game_id=flat, **kwargs)
    new = engine.run_heuristic_episode(game_id=game, **kwargs)
    assert [s["action_id"] for s in new["samples"]] == [s["action_id"] for s in old["samples"]]
    assert len(new["samples"]) == 4
    for s_old, s_new in zip(old["samples"], new["samples"]):
        assert s_new["policy_action_visits"] == s_old["policy_action_visits"]
        assert len(s_old["features"]) == _meta(flat)["feature_dim"]
        assert len(s_new["features"]) == _meta(game)["feature_dim"]


def test_concurrent_selfplay_with_transformer_and_mlp_guidance(tmp_path):
    path = tmp_path / "tok.onnx"
    export_onnx(_net(seed=2), path, _meta()["feature_dim"])
    out = engine.run_concurrent_selfplay_step(
        game_id=GAME, seeds=[1, 2], latest_model_path=str(path), num_self_threads=2,
        simulations=32, max_game_plies=6, batch_size=32, max_queue_size=64,
        intra_op_num_threads=1, leaf_batch_size=4)
    assert sum(len(ep["samples"]) for ep in out["episodes"]) > 0
    out = engine.run_concurrent_selfplay_step(
        game_id=GAME, seeds=[3], latest_model_path=str(path), num_self_threads=1,
        simulations=16, max_game_plies=3, batch_size=32, max_queue_size=64,
        intra_op_num_threads=1, heuristic_guidance_ratio=1.0, heuristic_selector="pv_512x4")
    assert len(out["episodes"]) == 1


def _train_cfg():
    return json.loads(TRAIN_CFG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("game,flat,players", VARIANTS)
def test_train_config_builds_model(game, flat, players):
    from training.cli import find_train_config, validate_guidance_until
    cfg = _train_cfg()
    validate_guidance_until(cfg["heuristic_guidance"], TRAIN_CFG)
    # MCTS profiles are read from the base train.json; the recipe must not carry its own.
    assert "mcts" not in cfg and "belief" not in cfg
    assert find_train_config(game)[1] == find_train_config("azul")[1]
    # The old network is the guide.
    assert cfg["heuristic_guidance"]["name"] == "pv_512x4"
    # The KataGo-style options of the recipe: three in this file, FPU in the MCTS profiles.
    assert {"playout_cap", "score_head", "ema"} <= set(cfg["training"])
    assert "opponent_policy_head" not in cfg["training"]
    from training import mcts_profile
    for name in ("selfplay", "eval"):
        assert mcts_profile.resolve_profile(game, name).fpu_reduction == 0.2
    meta = _meta(game)
    net = create_model_from_config(cfg["network"], input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
                                   num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)


def test_training_loop_guided_by_the_old_network(tmp_path, caplog):
    """Two steps of the real pipeline with the shipped recipe (shrunk): step 1
    is played by the old MLP (guidance ratio 1), step 2 by the transformer."""
    import logging
    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    caplog.set_level(logging.INFO, logger="training.pipeline")
    cfg = find_game_config(GAME)
    train_cfg = _train_cfg()
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 1, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {**train_cfg["heuristic_guidance"], "hold_steps": 1, "steps": 2}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=1, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Step ")]
    assert len(steps) == 2 and "h_ratio=1.00" in steps[0] and "h_ratio" not in steps[1]
    assert all("s_loss=" in s and "o_loss=" not in s for s in steps)
    latest = tmp_path / "models" / "model_latest.onnx"
    assert latest.exists()
    ep = engine.run_selfplay_episode(game_id=GAME, seed=3, model_path=str(latest), simulations=4, max_game_plies=30)
    assert ep["samples"]
