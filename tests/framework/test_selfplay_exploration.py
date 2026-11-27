"""Selfplay exploration options against policy lock-in, all off unless configured:

- `training.opening_exploration`: a share of the episodes opens with a prefix
  of plies sampled from the raw policy (no search, never a training sample).
- `training.midgame_start`: a mutually exclusive share fast-forwards along a
  sharp raw-policy trajectory into a midgame position and optionally takes a
  KataGo tournament fork move on the last prefix ply (never a training sample).
- MCTS selfplay profile `root_policy_temperature`: root priors p^(1/T) with
  KataGo exponential halflife decay `final + (initial - final) * 0.5^(ply / H)`.
- MCTS selfplay profile `dirichlet_total_alpha` + `dirichlet_shaped`: root noise
  concentration sized by action count, either uniform (`total / N`) or KataGo
  shaped (50% uniform + 50% excess log-prior above mean).
- Search-diversity read-out on every searched ply (prior / visit entropy,
  largest prior, KL from visits to prior), averaged into the step log.
"""
import dataclasses
import math

import dinoboard_engine as engine
import pytest
import torch
import torch.nn as nn

from conftest import get_test_model
from training import mcts_profile, pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config
from training.model import export_onnx

OPENING = {"episode_ratio": 1.0, "min_plies": 3, "max_plies": 3, "temperature": 1.0, "uniform_mix": 0.0}
MIDGAME = {
    "episode_ratio": 1.0, "min_plies": 3, "max_plies": 3,
    "temperature": 0.7, "fork_min_choices": 2, "fork_max_choices": 4,
}


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


class _ConstLogitPVNet(nn.Module):
    """Constant-logit PV model for deterministic root-prior / fork / shaped-noise tests."""

    def __init__(self, logits: list[float], num_players: int = 2):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32), requires_grad=False)
        self.num_players = num_players

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        pol = self.logits.unsqueeze(0).expand(b, -1) + (x[:, :1] * 0.0)
        val = torch.zeros((b, self.num_players), dtype=torch.float32, device=x.device) + (x[:, :1] * 0.0)
        return pol, val


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["quoridor", "loveletter"])
def test_absent_and_explicit_off_are_the_same_selfplay(game_id):
    base = _episode(game_id, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    off = _episode(game_id, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, exploration={
        "dirichlet_total_alpha": 0.0, "dirichlet_shaped": False,
        "root_policy_temperature_initial": 1.0,
        "root_policy_temperature_final": 1.0,
        "root_policy_temperature_halflife_plies": 0.0,
        "opening_exploration": dict(OPENING, episode_ratio=0.0),
        "midgame_start": dict(MIDGAME, episode_ratio=0.0)})
    assert _visits(base) == _visits(off) and _actions(base) == _actions(off)
    assert base["opening_exploration_plies"] == 0
    assert base["midgame_start_plies"] == 0
    assert not any(s["opening_exploration"] or s["midgame_start"] for s in base["samples"])


# ---------------------------------------------------------------------------
# Opening exploration
# ---------------------------------------------------------------------------

def _check_prefix(ep, prefix, *, flag="opening_exploration", count_key="opening_exploration_plies"):
    other_flag = "midgame_start" if flag == "opening_exploration" else "opening_exploration"
    other_count = "midgame_start_plies" if flag == "opening_exploration" else "opening_exploration_plies"
    assert len(ep["samples"]) == ep["total_plies"]
    assert ep[count_key] == min(prefix, ep["total_plies"])
    assert ep[other_count] == 0
    for i, s in enumerate(ep["samples"]):
        assert s[flag] == (i < prefix)
        assert not s[other_flag]
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
    assert all(s["simulations"] == 0 and sum(s["policy_action_visits"]) > 0
               for s in ep["samples"][3:])


# ---------------------------------------------------------------------------
# Midgame start + KataGo tournament fork
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["quoridor", "azul", "loveletter"])
def test_midgame_start_plies_are_flagged_and_unsearched(game_id):
    ep = _episode(game_id, exploration={"midgame_start": MIDGAME})
    _check_prefix(ep, 3, flag="midgame_start", count_key="midgame_start_plies")


def test_midgame_start_reaches_every_selfplay_binding():
    model = get_test_model("quoridor")
    common = dict(game_id="quoridor", simulations=16, max_game_plies=12,
                  exploration={"midgame_start": MIDGAME})
    batch = engine.run_selfplay_episodes_batch(seeds=[1, 2], model_paths=[model] * 2, **common)
    for ep in batch["episodes"]:
        _check_prefix(ep, 3, flag="midgame_start", count_key="midgame_start_plies")
    for compact in (False, True):
        step = engine.run_concurrent_selfplay_step(
            seeds=[1, 2, 3], latest_model_path=model, num_self_threads=2, batch_size=8,
            max_queue_size=16, intra_op_num_threads=1, compact_samples=compact, **common)
        for ep in step["episodes"]:
            _check_prefix(ep, 3, flag="midgame_start", count_key="midgame_start_plies")
            assert all(("features" in s) == (not compact or s["full_search"]) for s in ep["samples"])


def test_opening_and_midgame_starts_are_mutually_exclusive_per_episode():
    open_block = dict(OPENING, episode_ratio=0.45, min_plies=2, max_plies=3)
    mid_block = dict(MIDGAME, episode_ratio=0.45, min_plies=4, max_plies=5)
    saw_open = saw_mid = saw_normal = False
    for s in range(1, 60):
        ep = _episode("quoridor", seed=s, simulations=8, max_game_plies=10,
                      exploration={"opening_exploration": open_block, "midgame_start": mid_block})
        o_plies = ep["opening_exploration_plies"]
        m_plies = ep["midgame_start_plies"]
        assert not (o_plies > 0 and m_plies > 0), "episode cannot run both prefixes"
        if o_plies > 0:
            saw_open = True
            assert 2 <= o_plies <= 3
        elif m_plies > 0:
            saw_mid = True
            assert 4 <= m_plies <= 5
        else:
            saw_normal = True
    assert saw_open and saw_mid and saw_normal


def test_midgame_tournament_fork_branches_away_from_top1_and_never_picks_worst(tmp_path):
    """On 9-move Tic-Tac-Toe ply 0 with strict logit order 0 < 1 < ... < 8:
    - k=2 tournament fork draws 2 distinct moves and plays the higher-prior one:
      * action 0 (unique worst move) can NEVER beat its opponent -> 0 hits;
      * action 8 (unique top-1 move) only enters the 2-subset 2/9 of the time ->
        deviates onto strong alternatives (actions 4..7) most of the time.
    - fork_max_choices=0 disables the fork and falls back to temperature sampling
      (at T=1e-4, always plays action 8).
    """
    meta = engine.game_metadata("tictactoe")
    model_path = tmp_path / "strict_order.onnx"
    export_onnx(_ConstLogitPVNet([float(i) for i in range(9)], meta["num_players"]),
                model_path, meta["feature_dim"])

    def fork_moves(fork_min, fork_max, temp=1e-4):
        ms = {
            "episode_ratio": 1.0, "min_plies": 1, "max_plies": 1,
            "temperature": temp,
            "fork_min_choices": fork_min, "fork_max_choices": fork_max,
        }
        return [
            _actions(engine.run_selfplay_episode(
                game_id="tictactoe", seed=s, model_path=str(model_path),
                simulations=8, max_game_plies=9, temperature=0.0,
                dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
                exploration={"midgame_start": ms}))[0]
            for s in range(1, 61)
        ]

    no_fork = fork_moves(0, 0, temp=1e-4)
    assert set(no_fork) == {8}

    k2 = fork_moves(2, 2, temp=1e-4)
    assert 0 not in k2, "unique worst move cannot win a k=2 tournament"
    assert len(set(k2)) >= 4, "k=2 tournament should branch across multiple moves"
    assert sum(1 for a in k2 if a != 8) > sum(1 for a in k2 if a == 8)

    # k=9 gets clamped to N-1 = 8: action 8 enters 8/9 of subsets, action 7 wins the other 1/9!
    k_clamped = fork_moves(9, 12, temp=1e-4)
    assert set(k_clamped) <= {7, 8}
    assert 7 in k_clamped and 8 in k_clamped


@pytest.mark.parametrize("bad, match", [
    ({"opening_exploration": dict(OPENING, episode_ratio=1.5)}, "episode_ratio"),
    ({"opening_exploration": dict(OPENING, min_plies=0)}, "min_plies"),
    ({"opening_exploration": dict(OPENING, min_plies=4)}, "min_plies"),
    ({"opening_exploration": dict(OPENING, temperature=0.0)}, "temperature"),
    ({"opening_exploration": dict(OPENING, uniform_mix=-0.1)}, "uniform_mix"),
    ({"opening_exploration": {"episode_ratio": 0.5}}, "missing"),
    ({"opening_exploration": dict(OPENING, plies=3)}, "unknown"),
    ({"opening_explore": OPENING}, "unknown"),
    ({"midgame_start": dict(MIDGAME, episode_ratio=-0.1)}, "episode_ratio"),
    ({"midgame_start": dict(MIDGAME, min_plies=0)}, "min_plies"),
    ({"midgame_start": dict(MIDGAME, min_plies=5)}, "min_plies"),
    ({"midgame_start": dict(MIDGAME, temperature=0.0)}, "temperature"),
    ({"midgame_start": dict(MIDGAME, fork_min_choices=0, fork_max_choices=3)}, "fork"),
    ({"midgame_start": dict(MIDGAME, fork_min_choices=5, fork_max_choices=3)}, "fork"),
    ({"midgame_start": {"episode_ratio": 0.2}}, "missing"),
    ({"midgame_start": dict(MIDGAME, extra=1)}, "unknown"),
    ({"opening_exploration": dict(OPENING, episode_ratio=0.7),
      "midgame_start": dict(MIDGAME, episode_ratio=0.4)}, "episode_ratio"),
    ({"root_policy_temperature_final": 0.0}, "root_policy_temperature"),
    ({"root_policy_temperature_initial": -1.0}, "root_policy_temperature"),
    ({"root_policy_temperature_initial": 1.25, "root_policy_temperature_final": 1.1,
      "root_policy_temperature_halflife_plies": 0.0}, "halflife_plies"),
    ({"root_policy_temperature_halflife_plies": -1.0}, "halflife_plies"),
    ({"root_policy_temperature_decay_plies": 10}, "unknown"),
    ({"dirichlet_total_alpha": -1.0}, "dirichlet_total_alpha"),
    ({"dirichlet_shaped": True}, "dirichlet_shaped"),
    ({"dirichlet_total_alpha": 10.83, "dirichlet_shaped": 1}, "bool"),
])
def test_bindings_reject_invalid_exploration(bad, match):
    with pytest.raises(Exception, match=match):
        _episode("quoridor", exploration=bad)


# ---------------------------------------------------------------------------
# Root policy temperature (exponential halflife decay)
# ---------------------------------------------------------------------------

def _root_temperature(t):
    return {"root_policy_temperature_initial": t, "root_policy_temperature_final": t}


def test_root_policy_temperature_reshapes_the_root_search_only():
    """Ply 0 of tic-tac-toe: same root for every run, so the temperature is the only difference."""
    def ply0(t):
        return _episode("tictactoe", simulations=24, exploration=_root_temperature(t))["samples"][0]

    neutral, sharp, flat = ply0(1.0), ply0(0.05), ply0(50.0)
    assert tuple(neutral["policy_action_visits"]) == _visits(_episode("tictactoe", simulations=24))[0]
    assert sharp["visit_entropy"] < flat["visit_entropy"]
    assert max(sharp["policy_action_visits"]) > max(flat["policy_action_visits"])
    assert sharp["prior_entropy"] == neutral["prior_entropy"] == flat["prior_entropy"]


def test_root_policy_temperature_schedule_is_exponential_halflife(tmp_path):
    """At ply 0, T(0) = initial; at ply = halflife_plies = 2, T(2) = (initial + final) / 2."""
    meta = engine.game_metadata("tictactoe")
    model_path = tmp_path / "skewed_ttt.onnx"
    export_onnx(
        _ConstLogitPVNet([0.5 * i for i in range(9)], meta["num_players"]),
        model_path, meta["feature_dim"],
    )

    def run_ttt(exploration=None):
        kw = dict(
            game_id="tictactoe", seed=7, model_path=str(model_path),
            simulations=64, max_game_plies=9, temperature=0.0,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        if exploration is not None:
            kw["exploration"] = exploration
        return engine.run_selfplay_episode(**kw)

    sched = {
        "root_policy_temperature_initial": 3.0,
        "root_policy_temperature_final": 1.0,
        "root_policy_temperature_halflife_plies": 2.0,
    }
    scheduled = run_ttt(sched)
    at_t3 = run_ttt(_root_temperature(3.0))
    at_t2 = run_ttt(_root_temperature(2.0))
    plain = run_ttt()

    # Ply 0 uses T(0) = 3.0 exactly.
    assert _visits(scheduled)[0] == _visits(at_t3)[0] != _visits(plain)[0]

    # Strict logit order ensures both trajectories play [8, 7] on plies 0..1,
    # reaching the identical board at ply 2 where T(2) = 1.0 + 2.0 * 0.5^(2/2) = 2.0!
    assert _actions(scheduled)[:2] == _actions(at_t2)[:2] == [8, 7]
    assert _visits(scheduled)[2] == _visits(at_t2)[2]
    assert _visits(scheduled)[2] != _visits(plain)[2]
    assert _visits(scheduled)[2] != _visits(at_t3)[2]


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
# Dirichlet noise: total_alpha + KataGo shaped alpha
# ---------------------------------------------------------------------------

def test_total_alpha_is_alpha_times_legal_actions():
    """Tic-tac-toe ply 0 has 9 legal moves: total 4.5 is exactly alpha 0.5 there."""
    def ply0(**kw):
        return _visits(_episode("tictactoe", simulations=64, dirichlet_epsilon=0.5, **kw))[0]

    per_action = ply0(dirichlet_alpha=0.5)
    assert ply0(exploration={"dirichlet_total_alpha": 4.5}) == per_action
    assert ply0(exploration={"dirichlet_total_alpha": 0.9}) != per_action
    assert ply0(dirichlet_alpha=0.0) != per_action


def test_shaped_dirichlet_matches_uniform_on_flat_priors_and_focuses_on_plausible_moves(tmp_path):
    """KataGo shaped Dirichlet verification on Quoridor ply 0 (N > 100 legal moves,
    with training_filter_ratio=0.0 so all wall placements remain legal at root):
    1. With flat priors, excess log-prior is 0 everywhere -> shaped falls back to
       uniform total_alpha / N (6 moves out of ~131 win only ~4/80 draws).
    2. With 1 dominant move (p~0.88), 5 medium 'blind-spot' moves (p~0.024 each, above
       cap=0.01), and ~125 garbage wall moves (p~1e-6), setting epsilon=1.0 and
       simulations=1 samples argmax(Dir(alpha)):
       - Unshaped splits alpha evenly over all ~131 moves -> garbage wins >85% of draws.
       - Shaped puts 50% of 10.83 equally onto the 6 capped plausible moves ->
         the 6 plausible moves win the majority of draws, AND the 5 blind-spot
         moves collectively win far more draws than the single dominant move!
    """
    meta = engine.game_metadata("quoridor")
    gs = engine.GameSession("quoridor", 1, "", False, "")
    legal = list(gs.get_all_legal_actions())
    assert len(legal) >= 100
    fav = legal[0]
    blind_spots = set(legal[1:6])
    plausible = {fav} | blind_spots

    def top_noise_picks(model_path, shaped: bool) -> list[int]:
        picks = []
        for s in range(1, 81):
            ep = engine.run_selfplay_episode(
                game_id="quoridor", seed=s, model_path=str(model_path),
                simulations=1, max_game_plies=1, temperature=0.0,
                training_filter_ratio=0.0,
                dirichlet_alpha=0.0, dirichlet_epsilon=1.0,
                exploration={"dirichlet_total_alpha": 10.83, "dirichlet_shaped": shaped})
            s0 = ep["samples"][0]
            assert len(s0["policy_action_ids"]) == len(legal)
            best_i = max(range(len(s0["policy_action_visits"])),
                         key=lambda i: s0["policy_action_visits"][i])
            picks.append(s0["policy_action_ids"][best_i])
        return picks

    # 1. Flat priors -> shaped has zero excess log-prior and falls back to uniform.
    flat_path = tmp_path / "quoridor_flat.onnx"
    export_onnx(_ConstLogitPVNet([0.0] * meta["action_space"], meta["num_players"]),
                flat_path, meta["feature_dim"])
    flat_shaped_picks = top_noise_picks(flat_path, True)
    assert sum(1 for a in flat_shaped_picks if a in plausible) <= 15

    # 2. Skewed priors: 1 dominant + 5 blind-spot plausible + ~125 garbage walls.
    logits = [-14.0] * meta["action_space"]
    logits[fav] = 2.0
    for a in blind_spots:
        logits[a] = -1.6  # softmax ~ 0.024 > cap=0.01
    skew_path = tmp_path / "quoridor_skew.onnx"
    export_onnx(_ConstLogitPVNet(logits, meta["num_players"]),
                skew_path, meta["feature_dim"])

    unshaped_picks = top_noise_picks(skew_path, False)
    shaped_picks = top_noise_picks(skew_path, True)

    unshaped_plausible = sum(1 for a in unshaped_picks if a in plausible)
    shaped_plausible = sum(1 for a in shaped_picks if a in plausible)
    shaped_blind = sum(1 for a in shaped_picks if a in blind_spots)
    shaped_fav = sum(1 for a in shaped_picks if a == fav)

    assert unshaped_plausible <= 15
    assert shaped_plausible >= 40
    assert shaped_blind >= shaped_fav * 2


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
    assert all(s["simulations"] == 0 for s in ep["samples"])
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
    return (profile.dirichlet_total_alpha, profile.dirichlet_shaped,
            profile.root_policy_temperature_initial,
            profile.root_policy_temperature_final,
            profile.root_policy_temperature_halflife_plies)


def test_profile_defaults_are_off():
    for game_id in ("tictactoe", "quoridor"):
        for name in ("selfplay", "eval", "web_expert"):
            assert _root_exploration_fields(mcts_profile.resolve_profile(game_id, name)) == (
                0.0, False, 1.0, 1.0, 0.0)


@pytest.mark.parametrize("game_id", [
    "harmonies", "harmonies_tok", "splendor", "splendor_tok",
    "azul", "azul_tok", "carcassonne", "jaipur",
])
def test_configured_game_explores_in_selfplay_only(game_id):
    """Configured games set shaped adaptive Dirichlet + exponential root-policy
    temperature in train.json::mcts.selfplay; every other profile stays off."""
    selfplay = mcts_profile.resolve_profile(game_id, "selfplay")
    assert selfplay.dirichlet_total_alpha > 0.0 and selfplay.dirichlet_alpha == 0.0
    assert selfplay.dirichlet_shaped is True
    assert selfplay.root_policy_temperature_initial > selfplay.root_policy_temperature_final > 1.0
    assert selfplay.root_policy_temperature_halflife_plies > 0.0
    for name in ("eval", "web_expert", "web_casual", "analysis"):
        assert _root_exploration_fields(mcts_profile.resolve_profile(game_id, name)) == (
            0.0, False, 1.0, 1.0, 0.0)


def test_profile_root_policy_temperature_forms():
    flat = mcts_profile._flatten_layer({"root_policy_temperature": 1.1, "simulations": 5})
    assert flat == {"simulations": 5, "root_policy_temperature_initial": 1.1,
                    "root_policy_temperature_final": 1.1,
                    "root_policy_temperature_halflife_plies": 0.0}
    sched = mcts_profile._flatten_layer(
        {"root_policy_temperature": {"initial": 1.25, "final": 1.1, "halflife_plies": 4.5}})
    assert sched == {"root_policy_temperature_initial": 1.25,
                     "root_policy_temperature_final": 1.1,
                     "root_policy_temperature_halflife_plies": 4.5}
    for bad in (
        "hot", True,
        {"initial": 1.25, "final": 1.1},
        {"initial": 1.25, "final": 1.1, "decay_plies": 10},
        {"initial": 1.25, "final": 1.1, "halflife_plies": True},
        {"initial": "a", "final": 1.1, "halflife_plies": 3},
    ):
        with pytest.raises(ValueError, match="root_policy_temperature"):
            mcts_profile._flatten_layer({"root_policy_temperature": bad})


@pytest.mark.parametrize("fields, match", [
    (dict(dirichlet_total_alpha=-1.0), "dirichlet_total_alpha"),
    (dict(dirichlet_total_alpha=10.0, dirichlet_alpha=0.3), "only one"),
    (dict(dirichlet_shaped=True, dirichlet_total_alpha=0.0), "dirichlet_shaped"),
    (dict(dirichlet_shaped="yes", dirichlet_total_alpha=10.0), "dirichlet_shaped"),
    (dict(root_policy_temperature_initial=0.0), "root_policy_temperature"),
    (dict(root_policy_temperature_initial=1.25, root_policy_temperature_final=1.1), "halflife_plies"),
    (dict(root_policy_temperature_halflife_plies=-1.0), "halflife_plies"),
    (dict(temperature_schedule_enabled=True, temperature_initial=0.8,
          temperature_final=0.1, temperature_halflife_plies=0.0), "halflife_plies"),
])
def test_profile_rejects_invalid_values(fields, match):
    with pytest.raises(ValueError, match=match):
        mcts_profile._validate_resolved("selfplay", "tictactoe", dict(mcts_profile._BASE_DICT, **fields))


def test_profile_fields_belong_to_the_selfplay_profile_only():
    for field, val in (("dirichlet_total_alpha", 1.1), ("dirichlet_shaped", True),
                       ("root_policy_temperature", 1.1)):
        mcts_profile._validate_profile_node("selfplay", "tictactoe", {field: val}, "x")
        with pytest.raises(ValueError, match="mcts.selfplay"):
            mcts_profile._reject_selfplay_profile_only_fields({field: val}, "mcts.json")


# ---------------------------------------------------------------------------
# Config: training.opening_exploration, training.midgame_start, and loop
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


def test_parse_midgame_start():
    assert pipeline.parse_midgame_start({}, 100) is None
    block = {
        "episode_ratio": 0.15, "min_plies": 10, "max_plies": 24,
        "temperature": 0.7, "fork_min_choices": 3, "fork_max_choices": 8,
    }
    assert pipeline.parse_midgame_start({"midgame_start": block}, 100) == block
    no_fork = dict(block, fork_min_choices=0, fork_max_choices=0)
    assert pipeline.parse_midgame_start({"midgame_start": no_fork}, 100) == no_fork
    for bad, match in [
        (dict(block, episode_ratio=-0.01), "episode_ratio"),
        (dict(block, min_plies=0), "min_plies"),
        (dict(block, min_plies=25), "min_plies"),
        (dict(block, max_plies=100), "max_game_plies"),
        (dict(block, fork_min_choices=3.0), "integer"),
        (dict(block, fork_min_choices=True), "integer"),
        (dict(block, temperature=0), "temperature"),
        (dict(block, fork_min_choices=0, fork_max_choices=4), "fork"),
        (dict(block, fork_min_choices=6, fork_max_choices=4), "fork"),
        ({k: v for k, v in block.items() if k != "fork_max_choices"}, "exactly"),
        (dict(block, extra=1), "exactly"),
    ]:
        with pytest.raises(ValueError, match=match):
            pipeline.parse_midgame_start({"midgame_start": bad}, 100)


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
        monkeypatch, dirichlet_alpha=0.0, dirichlet_total_alpha=4.5, dirichlet_shaped=True,
        root_policy_temperature_initial=1.3, root_policy_temperature_final=1.1,
        root_policy_temperature_halflife_plies=2.0)
    open_block = {"episode_ratio": 0.5, "min_plies": 1, "max_plies": 2, "temperature": 1.0, "uniform_mix": 0.5}
    mid_block = {"episode_ratio": 0.5, "min_plies": 2, "max_plies": 3, "temperature": 0.7,
                 "fork_min_choices": 2, "fork_max_choices": 4}
    seen = []
    if concurrent:
        real = engine.run_concurrent_selfplay_step
        monkeypatch.setattr(engine, "run_concurrent_selfplay_step",
                            lambda **kw: seen.append(kw["exploration"]) or real(**kw))

    _train(_tictactoe_train_cfg(concurrent, opening_exploration=open_block, midgame_start=mid_block),
           tmp_path, episodes=10)

    if concurrent:
        assert seen == [{
            "dirichlet_total_alpha": 4.5, "dirichlet_shaped": True,
            "root_policy_temperature_initial": 1.3,
            "root_policy_temperature_final": 1.1,
            "root_policy_temperature_halflife_plies": 2.0,
            "opening_exploration": open_block,
            "midgame_start": mid_block,
        }]
    assert "opening_exploration: {'episode_ratio': 0.5" in caplog.text
    assert "midgame_start: {'episode_ratio': 0.5" in caplog.text
    assert "dirichlet_total_alpha=4.5, shaped=True" in caplog.text
    step_line = next(line for line in caplog.text.splitlines() if "Step 1/1:" in line)
    assert "expl_eps=" in step_line and "mid_eps=" in step_line
    for key in ("prior_ent=", "prior_top1=", "search_ent=", "policy_kl="):
        assert key in step_line


def test_training_loop_without_the_block_logs_diversity_but_no_exploration(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    _patch_selfplay_profile(monkeypatch)
    _train(_tictactoe_train_cfg(False), tmp_path, episodes=2)
    assert "opening_exploration: disabled" in caplog.text
    assert "midgame_start: disabled" in caplog.text
    step_line = next(line for line in caplog.text.splitlines() if "Step 1/1:" in line)
    assert "prior_ent=" in step_line and "expl_eps=" not in step_line and "mid_eps=" not in step_line


def test_training_loop_rejects_bad_blocks(tmp_path):
    bad = {"episode_ratio": 0.5, "min_plies": 1, "max_plies": 9, "temperature": 1.0, "uniform_mix": 0.0}
    with pytest.raises(ValueError, match="max_game_plies"):
        _train(_tictactoe_train_cfg(False, opening_exploration=bad), tmp_path)

    open_ok = {"episode_ratio": 0.6, "min_plies": 1, "max_plies": 2, "temperature": 1.0, "uniform_mix": 0.0}
    mid_excess = {"episode_ratio": 0.5, "min_plies": 2, "max_plies": 3, "temperature": 0.7,
                  "fork_min_choices": 2, "fork_max_choices": 4}
    with pytest.raises(ValueError, match="<= 1.0"):
        _train(_tictactoe_train_cfg(False, opening_exploration=open_ok, midgame_start=mid_excess), tmp_path)
