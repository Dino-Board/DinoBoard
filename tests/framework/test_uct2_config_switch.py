"""Train-config switch for the UCT2 parent count (`uct2_arrival_visits`).

A profile field of train.json's `mcts.selfplay` / `mcts.eval` (absent = true =
arrivals at the child; false = the incoming edge's total visits, the search
before the switch existed). It must reach every search the training process
starts, and `GameSession.get_ai_action` takes it as an optional argument.
mcts_pv heuristics do not follow it: they are pinned to false.
"""
import dataclasses

import dinoboard_engine as engine
import pytest

from conftest import get_test_model
from training import mcts_profile, pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config

MULTI, SINGLE = "loveletter", "quoridor"  # multi-outcome edges / none


def _visits(ep):
    return [tuple(s["policy_action_visits"]) for s in ep["samples"]]


def test_profile_default_and_validation():
    for name in ("selfplay", "eval", "web_expert"):
        assert mcts_profile.resolve_profile("splendor", name).uct2_arrival_visits is True
    for bad in (0, 1, "false", None):
        merged = dict(mcts_profile._BASE_DICT, uct2_arrival_visits=bad)
        with pytest.raises(ValueError, match="uct2_arrival_visits"):
            mcts_profile._validate_resolved("eval", "tictactoe", merged)
    with pytest.raises(ValueError, match="unknown"):
        mcts_profile._validate_profile_node("eval", "tictactoe", {"uct2_arrival_visit": False}, "x")
    # Baseline mcts.json / web.json may not carry it (the web platform would ignore it).
    with pytest.raises(ValueError, match="mcts.selfplay"):
        mcts_profile._reject_train_profile_only_fields({"uct2_arrival_visits": False}, "mcts.json")
    mcts_profile._validate_profile_node("eval", "tictactoe", {"uct2_arrival_visits": False}, "x")


@pytest.mark.parametrize("game_id, changes", [(MULTI, True), (SINGLE, False)])
def test_selfplay_bindings_honour_the_switch(game_id, changes):
    model = get_test_model(game_id)

    def episode(**kw):
        return _visits(engine.run_selfplay_episode(
            game_id=game_id, seed=1, model_path=model, simulations=200, max_game_plies=12, **kw))

    def batch(**kw):
        return _visits(engine.run_selfplay_episodes_batch(
            game_id=game_id, seeds=[1], model_paths=[model] * 2, simulations=200,
            max_game_plies=12, **kw)["episodes"][0])

    def concurrent(**kw):
        return _visits(engine.run_concurrent_selfplay_step(
            game_id=game_id, seeds=[1], latest_model_path=model, num_self_threads=1,
            simulations=200, max_game_plies=12, batch_size=8, max_queue_size=16,
            intra_op_num_threads=1, **kw)["episodes"][0])

    for run in (episode, batch, concurrent):
        assert run() == run(uct2_arrival_visits=True)  # default = new behaviour
        assert (run(uct2_arrival_visits=False) != run()) == changes
        with pytest.raises(TypeError):
            run(uct2_arrival_visits=0)


def test_eval_vs_heuristic_bindings_honour_the_switch():
    model = get_test_model("carcassonne")

    def legacy(seed, **kw):
        return list(engine.run_constrained_eval_vs_heuristic(
            game_id="carcassonne", seed=seed, model_path=model, simulations=64,
            heuristic_selector="rule", constrained=False, max_game_plies=60, **kw)["action_history"])

    def batched(seed, **kw):
        return list(engine.run_concurrent_eval_vs_heuristic_batch(
            game_id="carcassonne", model_path=model, heuristic_selector="rule",
            tasks=[dict(seed=seed, model_is_player=0)], num_threads=1, simulations=64,
            max_game_plies=60, batch_size=8, max_queue_size=16, intra_op_num_threads=1,
            **kw)["results"][0]["action_history"])

    for run in (legacy, batched):
        assert run(1) == run(1, uct2_arrival_visits=True)
        assert any(run(s, uct2_arrival_visits=False) != run(s) for s in (1, 2, 3))


def test_get_ai_action_argument_selects_the_recorded_count():
    """The debug trace (whose counts test_uct2_arrival_visits recounts) runs
    the same search as get_ai_action with default arguments."""
    gs = engine.GameSession(MULTI, seed=42, model_path=get_test_model(MULTI))
    visits = {}
    for flag in (True, False):
        got = gs.get_ai_action(simulations=200, temperature=0.0, uct2_arrival_visits=flag)
        trace = gs._debug_run_mcts_with_edge_traces(simulations=200, uct2_arrival_visits=flag)
        visits[flag] = list(got["stats"]["root_action_visits"])
        assert visits[flag] == list(trace["root_action_visits"])
    assert visits[True] != visits[False]
    assert list(gs.get_ai_action(simulations=200, temperature=0.0)["stats"]["root_action_visits"]) == visits[True]
    with pytest.raises(TypeError):
        gs.get_ai_action(simulations=8, temperature=0.0, uct2_arrival_visits="off")
    with pytest.raises(ValueError, match="must be a bool"):
        engine.run_arena_match(game_id=MULTI, seed=1, seat_configs=[
            dict(model_path=get_test_model(MULTI), uct2_arrival_visits=0)] * 2)


def test_heuristics_are_pinned_and_report_it():
    assert engine.heuristic_settings("splendor", "pv_512x4")["uct2_arrival_visits"] is False


@pytest.mark.parametrize("switch", [True, False])
def test_training_loop_passes_the_profile_switch_to_every_search(tmp_path, monkeypatch, switch):
    real_resolve = pipeline.resolve_profile
    monkeypatch.setattr(pipeline, "resolve_profile", lambda g, n: dataclasses.replace(
        real_resolve(g, n), uct2_arrival_visits=switch))
    seen = {}

    def spy(name, pick):
        real = getattr(engine, name)

        def wrapper(*args, **kwargs):
            seen.setdefault(name, []).extend(pick(kwargs))
            return real(*args, **kwargs)
        monkeypatch.setattr(engine, name, wrapper)

    spy("run_concurrent_selfplay_step", lambda kw: [kw["uct2_arrival_visits"]])
    spy("run_concurrent_eval_vs_heuristic_batch", lambda kw: [kw["uct2_arrival_visits"]])
    spy("run_concurrent_arena_batch", lambda kw: [
        seat["uct2_arrival_visits"] for task in kw["tasks"] for seat in task["seat_configs"]])

    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    train_cfg["training"] = {
        **train_cfg["training"], "train_batches_per_step": 1,
        "selfplay_concurrency": {
            "enabled": True, "leaf_batch_size": 1,
            "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                          "intra_op_num_threads": 1, "num_inference_threads": 1}},
        "playout_cap": {"full_simulations": 16, "fast_simulations": 4, "full_search_prob": 0.5},
    }
    pipeline.run_training_loop(
        game_id="tictactoe", game_config=find_game_config("tictactoe"),
        train_config=train_cfg, output_dir=tmp_path, steps=1, episodes_per_step=4,
        eval_every=1, eval_games=2, eval_benchmarks=["heuristic_free"], max_workers=1,
        batch_size=16, learning_rate=0.001, seed=4)
    assert set(seen) == {"run_concurrent_selfplay_step",
                         "run_concurrent_eval_vs_heuristic_batch",
                         "run_concurrent_arena_batch"}  # selfplay, eval, gating
    for name, values in seen.items():
        assert values and all(v is switch for v in values), name
