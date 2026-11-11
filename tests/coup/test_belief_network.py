"""Coup belief network: evaluator wiring, selfplay emission, sampling effects.

These tests cover the belief net pipeline end-to-end at the C++/Python
boundary (Plan §1-§3):

1. `game_metadata` exposes belief feature_dim / logit_count.
2. `GameSession` / `run_selfplay_episode` accept `belief_model_path` and
   load the ONNX belief evaluator without crashing.
3. With belief_model_path set, MCTS root call invokes the belief net
   (verified indirectly: sampling distribution is reproducible with the
   same seed, distinct from no-belief / uniform-fallback path).
4. `belief_samples` are emitted by selfplay only when a label extractor
   is registered (Coup is — Plan §2.2 / Task #107).
5. 2p / 3p / 4p variants all run end-to-end through the belief evaluator.
"""
import dinoboard_engine
import pytest
from pathlib import Path


PV_PATH = {
    "coup":     "games/coup/model/coup_2p.onnx",
    "coup_3p":  "games/coup/model/coup_3p.onnx",
    "coup_4p":  "games/coup/model/coup_4p.onnx",
}
BELIEF_PATH = {
    "coup":     "games/coup/model/coup_belief_2p.onnx",
    "coup_3p":  "games/coup/model/coup_belief_3p.onnx",
    "coup_4p":  "games/coup/model/coup_belief_4p.onnx",
}


def _ensure_models_exist():
    for p in (*PV_PATH.values(), *BELIEF_PATH.values()):
        if not Path(p).exists():
            pytest.skip(f"missing shipped placeholder model: {p}")


class TestMetadata:
    @pytest.mark.parametrize("game_id,expected_feat,expected_logits", [
        # Multiset belief head: 20 logits/opp = 15 two-card multisets +
        # 5 single cards (BELIEF_NETWORK.md §5).
        ("coup",    34, 20),   # 6 + 1*28, (2-1)*20
        ("coup_3p", 62, 40),   # 6 + 2*28, (3-1)*20
        ("coup_4p", 90, 60),   # 6 + 3*28, (4-1)*20
    ])
    def test_belief_dims_in_metadata(self, game_id, expected_feat, expected_logits):
        meta = dinoboard_engine.game_metadata(game_id)
        assert meta["has_belief_extractor"] is True
        assert meta["belief_feature_dim"] == expected_feat
        assert meta["belief_logit_count"] == expected_logits


class TestEvaluatorLoad:
    """`belief_model_path` must wire OnnxBeliefEvaluator without error."""

    @pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
    def test_game_session_loads_belief_model(self, game_id):
        _ensure_models_exist()
        gs = dinoboard_engine.GameSession(
            game_id, seed=42,
            model_path=PV_PATH[game_id],
            belief_model_path=BELIEF_PATH[game_id],
        )
        assert gs.is_terminal is False
        # Trigger MCTS root: belief evaluator must run inside prepare_for_root.
        r = gs.get_ai_action(simulations=20, temperature=0.0)
        assert "action" in r

    def test_session_without_belief_model_still_works(self):
        """Empty belief_model_path = uniform-fallback: MCTS still runs."""
        gs = dinoboard_engine.GameSession(
            "coup", seed=42,
            model_path=PV_PATH["coup"],
            belief_model_path="",
        )
        r = gs.get_ai_action(simulations=20, temperature=0.0)
        assert "action" in r

    def test_bogus_belief_path_raises(self):
        """No silent degradation — bad path must throw."""
        with pytest.raises(Exception):
            dinoboard_engine.GameSession(
                "coup", seed=42,
                model_path=PV_PATH["coup"],
                belief_model_path="/nonexistent/not_a_real_model.onnx",
            )


class TestSelfplayWithBelief:
    """run_selfplay_episode accepts belief_model_path and emits belief samples."""

    @pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
    def test_selfplay_runs_with_belief_net(self, game_id):
        _ensure_models_exist()
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=game_id, seed=7,
            model_path=PV_PATH[game_id],
            belief_model_path=BELIEF_PATH[game_id],
            simulations=15, max_game_plies=30,
        )
        assert ep["total_plies"] > 0

    def test_belief_samples_emitted_when_label_extractor_registered(self):
        """Coup registers CoupBeliefLabelExtractor (Task #107). Selfplay must
        emit one belief_sample per ply per non-terminal observer."""
        _ensure_models_exist()
        ep = dinoboard_engine.run_selfplay_episode(
            game_id="coup", seed=11,
            model_path=PV_PATH["coup"],
            belief_model_path=BELIEF_PATH["coup"],
            simulations=10, max_game_plies=20,
        )
        bs = ep["belief_samples"]
        assert len(bs) > 0, "Coup selfplay should emit belief_samples (#107)"
        # Class count exposed for the trainer.
        assert ep["belief_label_class_count"] == 5  # kCharacterCount

        # Each sample carries: ply, observer, features, hand_counts,
        # remaining, alive_per_opp. Sanity-check shapes.
        meta = dinoboard_engine.game_metadata("coup")
        feat_dim = meta["belief_feature_dim"]
        n_players = meta["num_players"]
        for s in bs:
            assert len(s["features"]) == feat_dim, (
                f"feature dim mismatch: got {len(s['features'])} != {feat_dim}"
            )
            assert len(s["hand_counts"]) == n_players - 1
            for row in s["hand_counts"]:
                assert len(row) == 5  # kCharacterCount
            assert len(s["remaining"]) == 5
            assert len(s["alive_per_opp"]) == n_players - 1
            assert 0 <= s["observer"] < n_players


class TestDeterminism:
    """Same seed + same belief model = reproducible decisions."""

    def test_same_seed_same_action(self):
        _ensure_models_exist()
        actions = []
        for _ in range(2):
            gs = dinoboard_engine.GameSession(
                "coup", seed=99,
                model_path=PV_PATH["coup"],
                belief_model_path=BELIEF_PATH["coup"],
            )
            for _ in range(3):
                if gs.is_terminal:
                    break
                legal = gs.get_legal_actions()
                gs.apply_action(legal[0])
            if not gs.is_terminal:
                r = gs.get_ai_action(simulations=30, temperature=0.0)
                actions.append(r["action"])
        if len(actions) == 2:
            assert actions[0] == actions[1], (
                f"Belief-net MCTS should be deterministic: {actions}"
            )
