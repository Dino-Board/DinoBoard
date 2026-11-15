"""Selfplay exploration options against policy lock-in, all off unless configured:

- `training.opening_exploration`: a share of the episodes opens with a prefix
  of plies sampled from the raw policy (no search, never a training sample).
- MCTS selfplay profile `root_policy_temperature`: root priors p^(1/T).
- MCTS selfplay profile `dirichlet_total_alpha`: root noise alpha = total / N.
- Search-diversity read-out on every searched ply (prior / visit entropy,
  largest prior, KL from visits to prior), averaged into the step log.
"""
import dataclasses
import math

import dinoboard_engine as engine
import pytest

from conftest import get_test_model
from training import mcts_profile, pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config

OPENING = {"episode_ratio": 1.0, "min_plies": 3, "max_plies": 3, "temperature": 1.0, "uniform_mix": 0.0}


def _episode(game_id, seed=7, **kw):
    args = dict(game_id=game_id, seed=seed, model_path=get_test_model(game_id),
                simulations=48, max_game_plies=30, temperature=0.0,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    args.update(kw)
    return engine.run_selfplay_episode(**args)


def _visits(ep):
    return [tuple(s["policy_action_visits"]) for s in ep["samples"]]


def _actions(ep):
    return [s["action_id"] for s in ep["samples"]]


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["quoridor", "loveletter"])
def test_absent_and_explicit_off_are_the_same_selfplay(game_id):
    base = _episode(game_id, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    off = _episode(game_id, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, exploration={
        "dirichlet_total_alpha": 0.0, "root_policy_temperature_initial": 1.0,
        "root_policy_temperature_final": 1.0, "root_policy_temperature_decay_plies": 0,
        "opening_exploration": dict(OPENING, episode_ratio=0.0)})
    assert _visits(base) == _visits(off) and _actions(base) == _actions(off)
    assert base["opening_exploration_plies"] == 0
    assert not any(s["opening_exploration"] for s in base["samples"])


# ---------------------------------------------------------------------------
# Opening exploration
# ---------------------------------------------------------------------------

def _check_prefix(ep, prefix):
    assert len(ep["samples"]) == ep["total_plies"]
    assert ep["opening_exploration_plies"] == min(prefix, ep["total_plies"])
    for i, s in enumerate(ep["samples"]):
        assert s["opening_exploration"] == (i < prefix)
        if i < prefix:  # sampled from the raw policy: no search, no policy target
            assert not s["full_search"] and s["simulations"] == 0
            assert list(s["policy_action_ids"]) == [] and list(s["policy_action_visits"]) == []
            assert not s["has_policy_stats"]
        else:
            assert s["full_search"] and sum(s["policy_action_visits"]) > 0
        assert len(s["z_values"]) == 2


@pytest.mark.parametrize("game_id", ["quoridor", "azul", "loveletter"])
def test_prefix_plies_are_flagged_and_unsearched(game_id):
    _check_prefix(_episode(game_id, exploration={"opening_exploration": OPENING}), 3)


def test_prefix_reaches_every_selfplay_binding():
    model = get_test_model("quoridor")
    common = dict(game_id="quoridor", simulations=16, max_game_plies=12,
                  exploration={"opening_exploration": OPENING})
    batch = engine.run_selfplay_episodes_batch(seeds=[1, 2], model_paths=[model] * 2, **common)
    for ep in batch["episodes"]:
        _check_prefix(ep, 3)
    for compact in (False, True):
        step = engine.run_concurrent_selfplay_step(
            seeds=[1, 2, 3], latest_model_path=model, num_self_threads=2, batch_size=8,
            max_queue_size=16, intra_op_num_threads=1, compact_samples=compact, **common)
        for ep in step["episodes"]:
            _check_prefix(ep, 3)
            # Like playout-cap fast plies, prefix plies carry no features in compact mode.
            assert all(("features" in s) == (not compact or s["full_search"]) for s in ep["samples"])


def test_episode_ratio_and_prefix_length_are_drawn_per_episode():
    block = dict(OPENING, episode_ratio=0.5, min_plies=2, max_plies=6)
    lengths = [_episode("quoridor", seed=s, simulations=8, max_game_plies=10,
                        exploration={"opening_exploration": block})["opening_exploration_plies"]
               for s in range(1, 25)]
    explored = [n for n in lengths if n > 0]
    assert 0 < len(explored) < len(lengths)
    assert all(2 <= n <= 6 for n in explored) and len(set(explored)) > 1
    again = _episode("quoridor", seed=3, simulations=8, max_game_plies=10,
                     exploration={"opening_exploration": block})
    assert again["opening_exploration_plies"] == lengths[2]  # a function of the seed


def test_prefix_moves_follow_the_policy_temperature_and_uniform_mix():
    """Tic-tac-toe opens from one fixed position, so the first move isolates the sampler."""
    def first_moves(**opening):
        block = dict(OPENING, min_plies=1, max_plies=1, **opening)
        return {_actions(_episode("tictactoe", seed=s, simulations=8,
                                  exploration={"opening_exploration": block}))[0]
                for s in range(1, 21)}

    greedy = first_moves(temperature=1e-4)   # argmax of the prior, whatever the seed
    assert len(greedy) == 1
    assert len(first_moves(uniform_mix=1.0)) >= 5        # uniform over the 9 cells
    assert len(first_moves(temperature=1e4)) >= 5        # flattened prior ~ uniform
    # A mix of 1 ignores the policy altogether.
    assert first_moves(temperature=1e-4, uniform_mix=1.0) == first_moves(uniform_mix=1.0)


def test_prefix_keeps_the_observation_trace_whole():
    ep = _episode("loveletter", trace_perspective=0, exploration={"opening_exploration": OPENING})
    assert len(ep["observation_trace"]) == ep["total_plies"]
    assert [t["ply"] for t in ep["observation_trace"]] == list(range(ep["total_plies"]))


def test_heuristic_guidance_does_not_act_inside_the_prefix():
    ep = engine.run_selfplay_episode(
        game_id="quoridor", seed=5, model_path=get_test_model("quoridor"), simulations=8,
        max_game_plies=8, heuristic_guidance_ratio=1.0, heuristic_selector="simple_max",
        exploration={"opening_exploration": OPENING})
    flags = [s["opening_exploration"] for s in ep["samples"]]
    assert flags[:3] == [True] * 3 and not any(flags[3:])
    # Guided plies carry the heuristic's policy target (and run no search of
    # ours); prefix plies carry no target at all.
    assert all(s["simulations"] == 0 and sum(s["policy_action_visits"]) > 0
               for s in ep["samples"][3:])


@pytest.mark.parametrize("bad, match", [
    ({"opening_exploration": dict(OPENING, episode_ratio=1.5)}, "episode_ratio"),
    ({"opening_exploration": dict(OPENING, min_plies=0)}, "min_plies"),
    ({"opening_exploration": dict(OPENING, min_plies=4)}, "min_plies"),
    ({"opening_exploration": dict(OPENING, temperature=0.0)}, "temperature"),
    ({"opening_exploration": dict(OPENING, uniform_mix=-0.1)}, "uniform_mix"),
    ({"opening_exploration": {"episode_ratio": 0.5}}, "missing"),
    ({"opening_exploration": dict(OPENING, plies=3)}, "unknown"),
    ({"opening_explore": OPENING}, "unknown"),
    ({"root_policy_temperature_final": 0.0}, "root_policy_temperature"),
    ({"root_policy_temperature_initial": -1.0}, "root_policy_temperature"),
    ({"dirichlet_total_alpha": -1.0}, "dirichlet_total_alpha"),
])
def test_bindings_reject_invalid_exploration(bad, match):
    with pytest.raises(Exception, match=match):
        _episode("quoridor", exploration=bad)


# ---------------------------------------------------------------------------
# Root policy temperature
# ---------------------------------------------------------------------------

def _root_temperature(t):
    return {"root_policy_temperature_initial": t, "root_policy_temperature_final": t}


def test_root_policy_temperature_reshapes_the_root_search_only():
    """Ply 0 of tic-tac-toe: same root for every run, so the temperature is the only difference."""
    def ply0(t):
        return _episode("tictactoe", simulations=24, exploration=_root_temperature(t))["samples"][0]

    neutral, sharp, flat = ply0(1.0), ply0(0.05), ply0(50.0)
    assert tuple(neutral["policy_action_visits"]) == _visits(_episode("tictactoe", simulations=24))[0]
    # T -> 0 piles the prior onto one move, T -> inf spreads it evenly.
    assert sharp["visit_entropy"] < flat["visit_entropy"]
    assert max(sharp["policy_action_visits"]) > max(flat["policy_action_visits"])
    # The read-out reports the network's own prior, not the reshaped one.
    assert sharp["prior_entropy"] == neutral["prior_entropy"] == flat["prior_entropy"]


def test_root_policy_temperature_schedule_is_linear_in_the_ply():
    sharp_then_off = {"root_policy_temperature_initial": 0.05,
                      "root_policy_temperature_final": 1.0,
                      "root_policy_temperature_decay_plies": 1}
    scheduled = _episode("tictactoe", simulations=24, exploration=sharp_then_off)
    always_sharp = _episode("tictactoe", simulations=24, exploration=_root_temperature(0.05))
    plain = _episode("tictactoe", simulations=24)
    assert _visits(scheduled)[0] == _visits(always_sharp)[0] != _visits(plain)[0]
    # From ply 1 on the schedule has reached 1 = plain search of whatever position it is in.
    if _actions(scheduled)[0] == _actions(plain)[0]:
        assert _visits(scheduled)[1] == _visits(plain)[1]


def test_fast_searches_ignore_the_root_policy_temperature():
    model = get_test_model("tictactoe")

    def fast_visits(**kw):
        out = engine.run_concurrent_selfplay_step(
            game_id="tictactoe", seeds=[3], latest_model_path=model, num_self_threads=1,
            simulations=24, fast_simulations=12, full_search_prob=1e-9, dirichlet_alpha=0.0,
            temperature=0.0, batch_size=8, max_queue_size=16, intra_op_num_threads=1, **kw)
        samples = out["episodes"][0]["samples"]
        assert not any(s["full_search"] for s in samples)
        return [tuple(s["policy_action_visits"]) for s in samples]

    assert fast_visits() == fast_visits(exploration=_root_temperature(50.0))


# ---------------------------------------------------------------------------
# Dirichlet noise sized by the number of legal actions
# ---------------------------------------------------------------------------

def test_total_alpha_is_alpha_times_legal_actions():
    """Tic-tac-toe ply 0 has 9 legal moves: total 4.5 is exactly alpha 0.5 there."""
    def ply0(**kw):
        return _visits(_episode("tictactoe", simulations=64, dirichlet_epsilon=0.5, **kw))[0]

    per_action = ply0(dirichlet_alpha=0.5)
    assert ply0(exploration={"dirichlet_total_alpha": 4.5}) == per_action
    assert ply0(exploration={"dirichlet_total_alpha": 0.9}) != per_action   # alpha 0.1 at ply 0
    assert ply0(dirichlet_alpha=0.0) != per_action                          # noise does act


def test_total_alpha_and_alpha_are_exclusive():
    with pytest.raises(Exception, match="only one"):
        _episode("tictactoe", dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 exploration={"dirichlet_total_alpha": 4.5})


# ---------------------------------------------------------------------------
# Search-diversity read-out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["quoridor", "splendor"])
def test_policy_stats_on_searched_plies(game_id):
    ep = _episode(game_id, simulations=64)
    seen = 0
    for s in ep["samples"]:
        n = len(s["policy_action_ids"])
        searched = n >= 2 and not s["tail_solved"]
        assert s["has_policy_stats"] == searched
        if not searched:
            assert s["prior_entropy"] == s["visit_entropy"] == s["policy_kl"] == 0.0
            continue
        seen += 1
        visits = s["policy_action_visits"]
        pi = [v / sum(visits) for v in visits]
        assert s["visit_entropy"] == pytest.approx(-sum(p * math.log(p) for p in pi if p > 0), abs=1e-4)
        assert 0.0 <= s["prior_entropy"] <= math.log(n) + 1e-4
        assert 1.0 / n - 1e-6 <= s["prior_top1"] <= 1.0
        assert s["policy_kl"] >= 0.0
    assert seen > 0


def test_guided_plies_have_no_policy_stats():
    ep = engine.run_selfplay_episode(
        game_id="quoridor", seed=5, model_path=get_test_model("quoridor"), simulations=8,
        max_game_plies=6, heuristic_guidance_ratio=1.0, heuristic_selector="simple_max")
    assert all(s["simulations"] == 0 for s in ep["samples"])  # every ply was heuristic-guided
    assert not any(s["has_policy_stats"] for s in ep["samples"])
    assert pipeline.policy_diversity_stats(ep["samples"]) is None


def test_policy_diversity_stats_averages_the_flagged_samples():
    row = dict(has_policy_stats=True, prior_entropy=1.0, prior_top1=0.5, visit_entropy=0.6, policy_kl=0.2)
    rows = [row, dict(row, prior_entropy=2.0, prior_top1=0.3, visit_entropy=1.0, policy_kl=0.4),
            dict(row, has_policy_stats=False, prior_entropy=99.0)]
    assert pipeline.policy_diversity_stats(rows) == pytest.approx(
        {"prior_ent": 1.5, "prior_top1": 0.4, "search_ent": 0.8, "policy_kl": 0.3})


# ---------------------------------------------------------------------------
# Config: MCTS profile fields
# ---------------------------------------------------------------------------

def _root_exploration_fields(profile):
    return (profile.dirichlet_total_alpha, profile.root_policy_temperature_initial,
            profile.root_policy_temperature_final, profile.root_policy_temperature_decay_plies)


def test_profile_defaults_are_off():
    for game_id in ("tictactoe", "quoridor", "splendor"):
        for name in ("selfplay", "eval", "web_expert"):
            assert _root_exploration_fields(mcts_profile.resolve_profile(game_id, name)) == (0.0, 1.0, 1.0, 0)


@pytest.mark.parametrize("game_id", ["harmonies", "harmonies_tok"])
def test_configured_game_explores_in_selfplay_only(game_id):
    """Harmonies sets both fields in train.json::mcts.selfplay (variants read
    the base game's file); every other profile of the game stays off."""
    selfplay = mcts_profile.resolve_profile(game_id, "selfplay")
    assert selfplay.dirichlet_total_alpha > 0.0 and selfplay.dirichlet_alpha == 0.0
    assert selfplay.root_policy_temperature_initial > selfplay.root_policy_temperature_final > 1.0
    assert selfplay.root_policy_temperature_decay_plies > 0
    for name in ("eval", "web_expert", "web_casual", "analysis"):
        assert _root_exploration_fields(mcts_profile.resolve_profile(game_id, name)) == (0.0, 1.0, 1.0, 0)


def test_profile_root_policy_temperature_forms():
    flat = mcts_profile._flatten_layer({"root_policy_temperature": 1.1, "simulations": 5})
    assert flat == {"simulations": 5, "root_policy_temperature_initial": 1.1,
                    "root_policy_temperature_final": 1.1, "root_policy_temperature_decay_plies": 0}
    sched = mcts_profile._flatten_layer(
        {"root_policy_temperature": {"initial": 1.25, "final": 1.1, "decay_plies": 30}})
    assert sched == {"root_policy_temperature_initial": 1.25, "root_policy_temperature_final": 1.1,
                     "root_policy_temperature_decay_plies": 30}
    for bad in ("hot", True, {"initial": 1.25, "final": 1.1}, {"initial": 1.25, "final": 1.1, "decay_plies": 1.5},
                {"initial": "a", "final": 1.1, "decay_plies": 3}):
        with pytest.raises(ValueError, match="root_policy_temperature"):
            mcts_profile._flatten_layer({"root_policy_temperature": bad})


@pytest.mark.parametrize("fields, match", [
    (dict(dirichlet_total_alpha=-1.0), "dirichlet_total_alpha"),
    (dict(dirichlet_total_alpha=10.0, dirichlet_alpha=0.3), "only one"),
    (dict(root_policy_temperature_initial=0.0), "root_policy_temperature"),
    (dict(root_policy_temperature_initial=1.25, root_policy_temperature_final=1.1), "decay_plies"),
    (dict(root_policy_temperature_decay_plies=-1), "decay_plies"),
])
def test_profile_rejects_invalid_values(fields, match):
    with pytest.raises(ValueError, match=match):
        mcts_profile._validate_resolved("selfplay", "tictactoe", dict(mcts_profile._BASE_DICT, **fields))


def test_profile_fields_belong_to_the_selfplay_profile_only():
    for field in ("dirichlet_total_alpha", "root_policy_temperature"):
        mcts_profile._validate_profile_node("selfplay", "tictactoe", {field: 1.1}, "x")
        with pytest.raises(ValueError, match="mcts.selfplay"):
            mcts_profile._reject_selfplay_profile_only_fields({field: 1.1}, "mcts.json")


# ---------------------------------------------------------------------------
# Config: training.opening_exploration and the training loop
# ---------------------------------------------------------------------------

def test_parse_opening_exploration():
    assert pipeline.parse_opening_exploration({}, 100) is None
    block = {"episode_ratio": 0.25, "min_plies": 2, "max_plies": 12, "temperature": 1.5, "uniform_mix": 0.1}
    assert pipeline.parse_opening_exploration({"opening_exploration": block}, 100) == block
    for bad, match in [
        (dict(block, episode_ratio=1.01), "episode_ratio"),
        (dict(block, min_plies=0), "min_plies"),
        (dict(block, min_plies=13), "min_plies"),
        (dict(block, max_plies=100), "max_game_plies"),
        (dict(block, max_plies=12.0), "integer"),
        (dict(block, temperature=0), "temperature"),
        (dict(block, uniform_mix=2), "uniform_mix"),
        ({k: v for k, v in block.items() if k != "uniform_mix"}, "exactly"),
        (dict(block, extra=1), "exactly"),
    ]:
        with pytest.raises(ValueError, match=match):
            pipeline.parse_opening_exploration({"opening_exploration": bad}, 100)


def _tictactoe_train_cfg(concurrent, **training):
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    sc = {"enabled": False}
    if concurrent:
        sc = {"enabled": True, "leaf_batch_size": 1,
              "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                            "intra_op_num_threads": 1, "num_inference_threads": 1}}
    train_cfg["training"] = {**train_cfg["training"], "simulations_start": 12,
                             "train_batches_per_step": 1, "selfplay_concurrency": sc, **training}
    return train_cfg


def _train(train_cfg, out, episodes=6):
    pipeline.run_training_loop(
        game_id="tictactoe", game_config=find_game_config("tictactoe"), train_config=train_cfg,
        output_dir=out, steps=1, episodes_per_step=episodes, eval_every=0, eval_games=2,
        max_workers=1, batch_size=16, learning_rate=0.001, seed=4)


def _patch_selfplay_profile(monkeypatch, **fields):
    """Small selfplay search for the loop tests, plus the given profile fields."""
    real_resolve = pipeline.resolve_profile
    monkeypatch.setattr(pipeline, "resolve_profile", lambda g, n: dataclasses.replace(
        real_resolve(g, n), simulations=12, **fields) if n == "selfplay" else real_resolve(g, n))


@pytest.mark.parametrize("concurrent", [False, True])
def test_training_loop_forwards_exploration_and_logs_it(tmp_path, monkeypatch, caplog, concurrent):
    caplog.set_level("INFO", logger="training.pipeline")
    _patch_selfplay_profile(
        monkeypatch, dirichlet_alpha=0.0, dirichlet_total_alpha=4.5,
        root_policy_temperature_initial=1.3, root_policy_temperature_final=1.1,
        root_policy_temperature_decay_plies=4)
    block = {"episode_ratio": 1.0, "min_plies": 1, "max_plies": 2, "temperature": 1.0, "uniform_mix": 0.5}
    seen = []
    if concurrent:  # the legacy path runs the binding in worker processes, out of the spy's reach
        real = engine.run_concurrent_selfplay_step
        monkeypatch.setattr(engine, "run_concurrent_selfplay_step",
                            lambda **kw: seen.append(kw["exploration"]) or real(**kw))

    _train(_tictactoe_train_cfg(concurrent, opening_exploration=block), tmp_path)

    if concurrent:
        assert seen == [{"dirichlet_total_alpha": 4.5, "root_policy_temperature_initial": 1.3,
                         "root_policy_temperature_final": 1.1,
                         "root_policy_temperature_decay_plies": 4, "opening_exploration": block}]
    assert "opening_exploration: {'episode_ratio': 1.0" in caplog.text
    assert "dirichlet_total_alpha=4.5" in caplog.text
    step_line = next(line for line in caplog.text.splitlines() if "Step 1/1:" in line)
    assert "expl_eps=6, expl_plies=" in step_line
    for key in ("prior_ent=", "prior_top1=", "search_ent=", "policy_kl="):
        assert key in step_line


def test_training_loop_without_the_block_logs_diversity_but_no_exploration(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    _patch_selfplay_profile(monkeypatch)
    _train(_tictactoe_train_cfg(False), tmp_path, episodes=2)
    assert "opening_exploration: disabled" in caplog.text
    step_line = next(line for line in caplog.text.splitlines() if "Step 1/1:" in line)
    assert "prior_ent=" in step_line and "expl_eps=" not in step_line


def test_training_loop_rejects_a_bad_block(tmp_path):
    bad = {"episode_ratio": 0.5, "min_plies": 1, "max_plies": 9, "temperature": 1.0, "uniform_mix": 0.0}
    with pytest.raises(ValueError, match="max_game_plies"):
        _train(_tictactoe_train_cfg(False, opening_exploration=bad), tmp_path)
