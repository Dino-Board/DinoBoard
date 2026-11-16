"""6 nimmt! belief network: the other players' hands are written card by card, lowest first
(belief.json type "autoregressive", engine/core/belief_sequence.h).

* The tracker remembers every finished round (the rows it started from and each seat's card);
  every seat remembers the same, and it is what really happened.
* The belief features equal an independent recomputation from what the seat could have written
  down (its hand, the cards shown, the rounds), kept here from the deal and the actions; a
  session fed only observations produces the same features as the real game's view of that seat.
* The training labels are the other seats' real hidden cards, and they are a possible deal of
  the pool the features show.
* With a belief model the search still cannot tell what lies in slots its seat cannot see, the
  model's hands reach the search, and whole games play out.
* Two steps of the real training loop train and export the belief net.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "framework"))

from conftest import get_test_model, load_game_config  # noqa: E402

GAMES = {"6nimmit": 2, "6nimmit_3p": 3, "6nimmit_4p": 4}
NUM_CARDS, NUM_ROWS, ROW_CAPACITY, HAND = 104, 4, 5, 10
games = pytest.mark.parametrize("game", list(GAMES))


def _dim(n):
    others = n - 1
    return 2 * NUM_CARDS + 2 * others * NUM_CARDS + HAND * (2 * NUM_ROWS + n) + (1 + 2 * others + n + 2 * NUM_ROWS)


def _belief_model(game, tmp_path, *, seed=5, silent=False):
    """An untrained belief net for `game`; unless `silent`, with a random last layer so that its
    hands depend on the features (a new net's last layer is zero: the uniform deal)."""
    import torch

    from training.model import create_belief_model_from_config, export_belief_onnx

    meta = engine.game_metadata(game)
    cfg = json.loads((ROOT / "games/6nimmit/config/belief.json").read_text())
    torch.manual_seed(seed)
    net = create_belief_model_from_config(cfg, input_dim=meta["belief_feature_dim"],
                                          output_dim=meta["belief_logit_count"],
                                          sequence_layout=meta["belief_sequence_layout"]).eval()
    if not silent:
        torch.nn.init.normal_(net.head[-1].weight, std=0.3)
    path = tmp_path / f"belief_{game}_{seed}_{int(silent)}.onnx"
    export_belief_onnx(net, path, meta["belief_feature_dim"])
    return str(path)


# ---- an independent record of a game ---------------------------------------------------------

def _rows(table):
    """(end, length) of each row, in table order."""
    out = []
    for row in table:
        cards = [c for c in row if c]
        out.append((cards[-1] if cards else 0, len(cards)))
    return out


class Record:
    """A game as a spectator who also sees every hand would write it down."""

    def __init__(self, gs, n):
        sd = gs.get_state_dict()
        self.n = n
        self.shown = {c for row in sd["table"] for c in row if c}  # cards that have lain face up
        self.rounds = []       # finished rounds: (rows at the start, card of each seat)
        self.start_rows = None  # rows the current round started from
        self.committed = {}    # seat -> card played this round, not placed yet
        self.update(sd)

    def update(self, sd):
        self.hands = [set(p["hand"]) for p in sd["players"]]
        self.heads = [p["hand_heads"] for p in sd["players"]]
        self.round = sd["current_round"]
        self.sealed = bool(sd["small_phase_active"]) and not sd["is_terminal"]
        self.table = _rows(sd["table"])
        if self.sealed:
            self.shown |= set(self.committed.values())

    def play(self, gs, action, viewer):
        """Apply `action` on the real game; returns the trace `viewer` gets."""
        actor = gs.current_player
        if action < NUM_CARDS:
            if not self.committed:
                self.start_rows = self.table
            self.committed[actor] = action + 1
        trace = gs.apply_action_with_trace(action, viewer)
        placed = {ev["payload"]["player"]: ev["payload"]["card"] for ev in trace["events"]}
        if placed:
            assert placed == self.committed, "a round's cards are placed together"
            self.rounds.append((self.start_rows, [self.committed[p] for p in range(self.n)]))
            self.shown |= set(placed.values())
            self.committed = {}
        self.update(gs.get_state_dict())
        return trace

    def others(self, me):
        return [(me + 1 + i) % self.n for i in range(self.n - 1)]

    def pool(self, me):
        """Cards `me` cannot place."""
        mine = self.hands[me] | ({self.committed[me]} if me in self.committed else set())
        return [0 if (c in self.shown or c in mine) else 1 for c in range(1, NUM_CARDS + 1)]

    def hidden(self, me):
        """The cards each other seat holds that `me` has not seen: its hand and its face-down card."""
        out = []
        for p in self.others(me):
            cards = set(self.hands[p])
            if p in self.committed and not self.sealed:
                cards.add(self.committed[p])
            out.append(sorted(cards))
        return out

    def memory(self):
        """The tracker's snapshot of the finished rounds."""
        flat = []
        for rows, cards in self.rounds:
            flat += [e for e, _ in rows] + [k for _, k in rows] + cards
        return flat

    def features(self, me):
        f32 = np.float32
        others = self.others(me)
        x = [np.asarray(self.pool(me), dtype=f32),
             np.asarray([1 if c in self.hands[me] else 0 for c in range(1, NUM_CARDS + 1)], dtype=f32)]
        for p in others:
            played, skipped = np.zeros(NUM_CARDS, dtype=f32), np.zeros(NUM_CARDS, dtype=f32)
            for rows, cards in self.rounds:
                card = cards[p]
                played[card - 1] = 1
                below = [e for e, _ in rows if e < card]
                if below:  # the cards that would have fitted the same row more closely
                    skipped[max(below):card - 1] += f32(0.25)
            x += [played, skipped]
        for t in range(HAND):
            block = np.zeros(2 * NUM_ROWS + self.n, dtype=f32)
            if t < len(self.rounds):
                rows, cards = self.rounds[t]
                rows = sorted(rows)
                block[:NUM_ROWS] = [f32(e) / f32(NUM_CARDS) for e, _ in rows]
                block[NUM_ROWS:2 * NUM_ROWS] = [f32(k) / f32(ROW_CAPACITY) for _, k in rows]
                block[2 * NUM_ROWS:] = [f32(cards[p]) / f32(NUM_CARDS) for p in [me] + others]
            x.append(block)
        tail = [f32(self.round) / f32(HAND)]
        for p in others:
            tail += [f32(HAND - self.round - (1 if self.sealed else 0)) / f32(HAND), f32(self.sealed)]
        tail += [f32(self.heads[p]) / f32(30) for p in [me] + others]
        rows = sorted(self.table)
        tail += [f32(e) / f32(NUM_CARDS) for e, _ in rows] + [f32(k) / f32(ROW_CAPACITY) for _, k in rows]
        x.append(np.asarray(tail, dtype=f32))
        return np.concatenate(x)


# ---- layout ----------------------------------------------------------------------------------

@games
def test_metadata(game):
    n = GAMES[game]
    meta, cfg = engine.game_metadata(game), load_game_config(game)["variants"][game]
    assert meta["has_belief_extractor"]
    assert meta["belief_feature_dim"] == cfg["belief_feature_dim"] == _dim(n)
    assert meta["belief_logit_count"] == cfg["belief_logit_count"] == NUM_CARDS
    layout = meta["belief_sequence_layout"]
    assert layout == {"classes": NUM_CARDS, "segments": n - 1,
                      "prefix_dim": (n - 1) * NUM_CARDS + (n - 1) + NUM_CARDS}
    belief = json.loads((ROOT / "games/6nimmit/config/belief.json").read_text())
    assert belief["type"] == "autoregressive"


def test_belief_config_loads_with_the_train_config():
    from training.cli import find_belief_config
    cfg, _ = find_belief_config("6nimmit_3p")
    assert cfg["type"] == "autoregressive" and cfg["training"] == {"loss": "sequence_nll"}


# ---- memory and features ---------------------------------------------------------------------

@games
def test_memory_and_features_are_what_each_seat_could_write_down(game):
    """Random whole games. After every move: every seat's observation-fed session remembers the
    same finished rounds, which are the ones really played; its belief features equal the
    recomputation, and equal the features of the real game's own view of that seat."""
    n = GAMES[game]
    pauses = positions = 0
    for seed in range(4):
        rng = random.Random(seed)
        # One real game per seat, in lockstep (same seed: same deal), each tracing for its seat.
        truths = [engine.GameSession(game, seed=seed, model_path="", use_filter=False) for _ in range(n)]
        records = [Record(t, n) for t in truths]
        seats = []
        for p in range(n):
            s = engine.GameSession(game, seed=100 + seed + p, model_path="", use_filter=False)
            s.apply_initial_observation(p, truths[p].extract_initial_observation(p))
            seats.append(s)
        while not truths[0].is_terminal:
            action = rng.choice(truths[0].get_legal_actions())
            for p in range(n):
                trace = records[p].play(truths[p], action, p)
                seats[p].apply_observation(trace["action"], events=trace["events"],
                                           public_snapshot=trace["public_snapshot"])
            record = records[0]
            for p in range(n):
                assert seats[p].get_belief_snapshot() == {"rounds": record.memory()}
            if truths[0].is_terminal:
                break
            pauses += record.sealed
            for p in range(n):
                want = record.features(p)
                got = np.asarray(seats[p].belief_features(p), dtype=np.float32)
                assert got.shape == want.shape == (_dim(n),)
                assert np.array_equal(got, want), np.flatnonzero(got != want)[:10]
                assert truths[p].belief_features(p) == seats[p].belief_features(p)
                positions += 1
        assert len(records[0].rounds) == HAND
    assert pauses >= 2, "no seat had to take a row: the face-up pause was not covered"
    assert positions > 100


# ---- labels ----------------------------------------------------------------------------------

@games
def test_labels_are_the_real_hidden_cards(game, tmp_path):
    """Selfplay with a belief model: at every ply, for every observer, the label is the cards the
    other seats really hold unseen (replayed here from the episode's actions), the features are
    the recomputed ones, and the label is a possible deal of the pool (training rows exist)."""
    n = GAMES[game]
    cfg = load_game_config(game)
    ep = engine.run_selfplay_episode(
        game_id=game, seed=21, model_path=get_test_model(game), simulations=6,
        max_game_plies=cfg["max_game_plies"], belief_model_path=_belief_model(game, tmp_path))
    actions = [s["action_id"] for s in ep["samples"]]
    by_ply = {}
    for bs in ep["belief_samples"]:
        by_ply.setdefault(bs["ply"], {})[bs["observer"]] = bs
    gt = engine.GameSession(game, seed=21, model_path="", use_filter=False)
    record = Record(gt, n)
    checked = 0
    for ply, action in enumerate(actions):
        assert sorted(by_ply[ply]) == list(range(n))
        for me, bs in by_ply[ply].items():
            hidden = record.hidden(me)
            assert [list(np.flatnonzero(row) + 1) for row in np.asarray(bs["hand_counts"])] == hidden
            assert list(bs["remaining"]) == record.pool(me)
            assert all(record.pool(me)[c - 1] for hand in hidden for c in hand)
            assert np.array_equal(np.asarray(bs["features"], dtype=np.float32), record.features(me))
            rows = engine.belief_sequence_training_rows(
                NUM_CARDS, n - 1, np.asarray([bs["remaining"]], dtype=np.int32),
                np.asarray([bs["hand_counts"]], dtype=np.int32))
            assert rows["target"].tolist() == [c - 1 for hand in hidden for c in hand]
            checked += 1
        record.play(gt, action, 0)
    assert gt.is_terminal and checked == len(actions) * n
    assert len(by_ply) == len(actions)


# ---- search ----------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["truth", "garbage"])
@pytest.mark.parametrize("game", ["6nimmit", "6nimmit_3p"])
def test_search_with_a_belief_model_ignores_hidden_slots(game, mode, tmp_path):
    """tests/framework/test_search_ignores_hidden_bytes.py, with a belief model whose hands depend
    on its features: two sessions of one seat, same seed, one with the slots hidden from the seat
    overwritten before every decision. Their searches must be identical."""
    from test_search_ignores_hidden_bytes import _search, _tamper

    model, belief = get_test_model(game), _belief_model(game, tmp_path)
    n = GAMES[game]
    roots = tampered = 0
    for seed in range(4):
        rng = random.Random(seed * 31 + n)
        gt = engine.GameSession(game, seed=seed, model_path="", use_filter=False)
        seat = seed % n
        twins = []
        for _ in range(2):
            ai = engine.GameSession(game, seed=seed + 4242, model_path=model, use_filter=False,
                                    belief_model_path=belief)
            ai.apply_initial_observation(seat, gt.extract_initial_observation(seat))
            twins.append(ai)
        clean, dirty = twins
        while not gt.is_terminal:
            action = rng.choice(gt.get_legal_actions())
            if gt.current_player == seat:
                tampered += _tamper(dirty, gt, seat, game, n, rng, mode)
                expected, got = _search(clean), _search(dirty)
                assert got == expected, f"[{game}/{mode}] seed={seed} round={gt.get_state_dict()['current_round']}"
                assert dirty.belief_features(seat) == clean.belief_features(seat)
                roots += 1
                if rng.random() < 0.5:
                    action = expected[0]
            trace = gt.apply_action_with_trace(action, seat)
            for ai in twins:
                ai.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
    assert roots >= 30 and tampered > 0


def test_the_models_hands_reach_the_search(tmp_path):
    """Same seat, same seed, same position: a silent belief model (the uniform deal) and one with
    opinions search different worlds, so their searches differ somewhere; each is reproducible."""
    game, model = "6nimmit", get_test_model("6nimmit")
    silent = _belief_model(game, tmp_path, silent=True)
    loud = _belief_model(game, tmp_path, seed=9)
    different = 0
    for seed in range(3):
        sessions = [engine.GameSession(game, seed=seed, model_path=model, use_filter=False, belief_model_path=b)
                    for b in (silent, silent, loud)]
        rng = random.Random(seed)
        while not sessions[0].is_terminal:
            if len(sessions[0].get_legal_actions()) > 1:
                stats = [s.get_ai_action(simulations=48, temperature=0.0)["stats"] for s in sessions]
                found = [(s["root_actions"], s["root_action_visits"], s["root_values"]) for s in stats]
                assert found[0] == found[1]
                different += found[0] != found[2]
            action = rng.choice(sessions[0].get_legal_actions())
            for s in sessions:
                s.apply_action(action)
    assert different >= 5


@games
def test_whole_games_play_out_with_a_belief_model(game, tmp_path):
    """Every world of every search is dealt by the model (the tracker refuses a hand that holds a
    card it cannot); the episode is reproducible."""
    cfg = load_game_config(game)
    kw = dict(game_id=game, seed=33, model_path=get_test_model(game), simulations=24,
              max_game_plies=cfg["max_game_plies"], belief_model_path=_belief_model(game, tmp_path),
              dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    a, b = engine.run_selfplay_episode(**kw), engine.run_selfplay_episode(**kw)
    assert a["total_plies"] >= HAND * GAMES[game]
    assert [s["action_id"] for s in a["samples"]] == [s["action_id"] for s in b["samples"]]


# ---- training --------------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["6nimmit", "6nimmit_3p"])
def test_training_loop_trains_the_belief_net(game, tmp_path, caplog):
    """Two steps of the real pipeline with belief.json: the belief net is trained on the selfplay
    labels, exported with its two inputs, and the next selfplay searches with it."""
    import logging

    import onnxruntime as ort

    from training.cli import find_belief_config, find_game_config, find_heuristic_config, find_train_config
    from training.pipeline import run_training_loop
    cfg = find_game_config(game)
    train_cfg, _ = find_train_config(game)
    train_cfg["heuristic"], _ = find_heuristic_config(game)
    train_cfg["belief"], _ = find_belief_config(game)
    train_cfg["belief"] = {**train_cfg["belief"], "architecture": [32, 32], "head": [32]}
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 2, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {"name": "rule_v1", "hold_steps": 1, "steps": 2, "initial_ratio": 1.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "simulations_start": 4}
    train_cfg["mcts"]["selfplay"]["simulations"] = 4
    train_cfg["mcts"]["eval"]["simulations"] = 4
    with caplog.at_level(logging.INFO):
        run_training_loop(game_id=game, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                          episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                          learning_rate=0.001, seed=5, save_every=0)
    assert "b_loss=" in caplog.text and "b_gain=" in caplog.text
    meta = engine.game_metadata(game)
    latest = tmp_path / "models" / "belief_latest.onnx"
    inputs = ort.InferenceSession(str(latest), providers=["CPUExecutionProvider"]).get_inputs()
    assert {i.name: i.shape[1] for i in inputs} == {
        "features": meta["belief_feature_dim"], "prefix": meta["belief_sequence_layout"]["prefix_dim"]}
    assert (tmp_path / "belief_checkpoint.pt").exists()
    ep = engine.run_selfplay_episode(
        game_id=game, seed=3, model_path=str(tmp_path / "models" / "model_latest.onnx"), simulations=4,
        max_game_plies=cfg["max_game_plies"], belief_model_path=str(latest))
    assert ep["samples"] and ep["belief_samples"]
