"""Coup complete acceptance checklist.

Coup is asymmetric-hidden-info with bluffing. It exercises:
  - belief_tracker with per-player private hand AND a heuristic-weighted
    sampler (claim/challenge history biases opp role priors)
  - public_event_extractor / applier
  - initial observation via framework walker (viz=1 slots only,
    no per-game extractor/applier hooks)
  - elimination + multiplayer (2p / 3p / 4p)
  - encoder must zero out opponent known_hand block
  - uniform-random heuristic (RandomHeuristic) (web 'heuristic' fallback)
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer

Build status: Coup is gated by `games/manifest.json` (`enabled` flag).
The whole file skips gracefully when the engine wasn't built with Coup.
"""
import dinoboard_engine
import pytest

from conftest import (
    assert_api_belief_matches_selfplay,
    get_test_model,
    load_game_config,
    run_random_episode_states,
)


def _coup_available() -> bool:
    try:
        dinoboard_engine.game_metadata("coup")
        return True
    except Exception:
        return False


# Skip the whole file if Coup is not built — applied per-class below for
# clear reasons in test output.
coup_skip = pytest.mark.skipif(
    not _coup_available(), reason="coup not built (manifest disabled)")


GAME = "coup"
VARIANTS = ["coup", "coup_3p", "coup_4p"]


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

@coup_skip
class TestRegistration:

    def test_2p_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_all_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games, f"{v} not registered"

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        cfg = load_game_config(GAME)
        assert meta["action_space"] == cfg["action_space"]
        assert meta["feature_dim"] == cfg["feature_dim"]
        assert meta["num_players"] == 2

    def test_player_counts(self):
        assert dinoboard_engine.GameSession("coup_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("coup_4p", seed=42).num_players == 4


# ---------------------------------------------------------------------------
# 2. GameSession basics
# ---------------------------------------------------------------------------

@coup_skip
class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert not gs.is_terminal

    def test_legal_actions_in_range(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        action_space = load_game_config(GAME)["action_space"]
        assert len(legal) > 0
        assert all(0 <= a < action_space for a in legal)

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:20]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0


# ---------------------------------------------------------------------------
# 3. Feature encoding (must zero opponent's known_hand)
# ---------------------------------------------------------------------------

@coup_skip
class TestEncoder:
    """Encoder dim sanity. Information-barrier invariants (opponent private
    block all-zero, self hand reflects truth) live in
    tests/framework/test_is_mcts_correctness.py::TestCoupEncoderInfoBarrier
    against the current schema-driven layout."""

    def test_encode_state_correct_dim(self):
        cfg = load_game_config(GAME)
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == cfg["feature_dim"]
        assert len(info["legal_mask"]) == cfg["action_space"]


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

@coup_skip
class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=60,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        cfg = load_game_config(GAME)
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == cfg["feature_dim"]
            assert len(s["legal_mask"]) == cfg["action_space"]
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=12345, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=30,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]

    def test_seed_diversity(self):
        diff = 0
        for i in range(20):
            ep1 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=30,
            )
            ep2 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i + 10000, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=30,
            )
            if [s["action_id"] for s in ep1["samples"]] != [s["action_id"] for s in ep2["samples"]]:
                diff += 1
        assert diff > 10, f"only {diff}/20 coup games differed across distant seeds"


# ---------------------------------------------------------------------------
# 5. MCTS / arena
# ---------------------------------------------------------------------------

@coup_skip
class TestMcts:

    def test_apply_ai_action_advances_state(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        before = gs.get_state_dict()
        gs.apply_ai_action(simulations=10, temperature=0.0)
        after = gs.get_state_dict()
        assert before != after

    def test_arena_completes(self):
        m = get_test_model(GAME)
        result = dinoboard_engine.run_arena_match(
            game_id=GAME, seed=42, model_paths=[m, m],
            simulations_list=[10, 10], temperature=0.0,
        )
        assert "winner" in result and result["total_plies"] > 0


# ---------------------------------------------------------------------------
# 6. Heuristic (uniform-random fallback)
# ---------------------------------------------------------------------------

@coup_skip
class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            if gs.is_terminal:
                continue
            result = gs.get_heuristic_action("rule")
            assert "action" in result
            assert result["action"] in gs.get_all_legal_actions()


# ---------------------------------------------------------------------------
# 7. Temperature variety
# ---------------------------------------------------------------------------

@coup_skip
class TestTemperature:

    def test_nonzero_temperature_adds_variety(self):
        actions = set()
        model = get_test_model(GAME)
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed, model_path=model)
            if gs.is_terminal:
                continue
            result = gs.get_ai_action(simulations=10, temperature=1.0)
            if "action" in result:
                actions.add(result["action"])
        assert len(actions) > 1, "temperature=1.0 always picked the same action across seeds"


# ---------------------------------------------------------------------------
# 8. Components NOT registered
# ---------------------------------------------------------------------------

@coup_skip
class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )


# ---------------------------------------------------------------------------
# 9. AI API belief / public state / legal actions equivalence
# (independent-seed API session must match self-play tracker step-by-step)
# ---------------------------------------------------------------------------

@coup_skip
class TestApiBeliefEquivalence:

    PUBLIC_KEYS = [
        "current_player", "is_terminal", "winner", "num_players",
        "ply", "stage", "active_player", "declared_action",
        "action_target", "blocker", "challenger", "deck_size",
    ]

    def test_belief_matches_selfplay_under_independent_seed(self):
        assert_api_belief_matches_selfplay(GAME, self.PUBLIC_KEYS)


# ---------------------------------------------------------------------------
# 10. Rule invariants (game-specific conservation laws)
# ---------------------------------------------------------------------------

# Coup deck: 5 characters × 3 copies = 15 total cards. Cards flow:
#   court_deck → influence slot (drawn at game start, drawn during exchange)
#   influence slot → court_deck (returned during exchange, by Ambassador)
# Revealed influence cards stay in their slot but are publicly known and
# do NOT return to the deck. So:
#   court_deck.size() + sum(influence slots that are not revealed AND not -1)
#   + sum(exchange_drawn that aren't -1) == 15
_COUP_TOTAL_CARDS = 15
_COUP_CHARACTERS = 5


def _count_coup_cards(state: dict) -> int:
    """Sum every card slot regardless of where it sits.

    Counts a card iff its character slot is in [0, 5). Influence character
    is -1 when a slot is empty (e.g. eliminated player after both reveals,
    or assassinate-pending state that briefly cleared a slot). Revealed
    cards are still in the slot — they count.
    """
    n = state["deck_size"]
    for p in state["players"]:
        for inf in p["influences"]:
            if 0 <= inf["character"] < _COUP_CHARACTERS:
                n += 1
    # exchange_drawn: only populated during ExchangeReturn1/Return2.
    # Outside those stages it's empty list. -1 means "slot already
    # returned during Return1".
    for c in state.get("exchange_drawn", []):
        if 0 <= c < _COUP_CHARACTERS:
            n += 1
    return n


def _assert_coup_invariants(state: dict) -> None:
    n = state["num_players"]

    # Card conservation: total visible cards (deck + influence slots +
    # exchange_drawn) == 15.
    total = _count_coup_cards(state)
    assert total == _COUP_TOTAL_CARDS, (
        f"coup card count broken: total={total} expected={_COUP_TOTAL_CARDS}, "
        f"deck_size={state['deck_size']}, exchange_drawn={state.get('exchange_drawn', [])}")

    # Per-player constraints.
    alive_count = 0
    for i, p in enumerate(state["players"]):
        # Coins always non-negative; in standard Coup the cap is 12 (force
        # coup at >=10, but a player can hold more if no coup target — be
        # permissive but bounded).
        assert 0 <= p["coins"] <= 12, (
            f"player {i} coins out of range: {p['coins']}")
        # Influences: exactly 2 slots.
        assert len(p["influences"]) == 2, (
            f"player {i} has {len(p['influences'])} influence slots (must be 2)")

        revealed = sum(1 for inf in p["influences"] if inf["revealed"])

        if p["alive"]:
            alive_count += 1
            # Alive → not both influences revealed. (A slot may transiently
            # hold character=-1 during mid-action stages, e.g. while
            # awaiting a reveal/return resolution.)
            assert revealed < 2, (
                f"player {i} alive but both influences revealed: {p['influences']}")
        else:
            # Dead → both influences revealed.
            assert revealed == 2, (
                f"player {i} dead but only {revealed} revealed influences")

    # exchange_drawn only present mid-exchange.
    ed = state.get("exchange_drawn", [])
    if ed:
        # During exchange-return stages, list has 2 entries. Each is either
        # a valid character (0..4) or -1 (already returned in Return1).
        assert len(ed) == 2
        for c in ed:
            assert c == -1 or 0 <= c < _COUP_CHARACTERS, (
                f"exchange_drawn invalid value: {c}")

    # current_player in range when not terminal; alive count consistent.
    if state["is_terminal"]:
        assert alive_count <= 1, (
            f"terminal state with {alive_count} alive players (must be ≤1)")
    else:
        assert alive_count >= 2, (
            f"non-terminal state with {alive_count} alive (must be ≥2)")
        assert 0 <= state["current_player"] < n


@coup_skip
class TestRuleInvariants:
    """Per-ply assertions on Coup's conservation laws.

    Catches: card duplication / loss across deck ↔ influences ↔ exchange,
    coin overflow / underflow, alive/influence inconsistency, terminal
    reached with multiple players still alive.
    """

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_invariants_hold_along_random_episode(self, variant, seed):
        for state in run_random_episode_states(variant, seed=seed, max_plies=200):
            _assert_coup_invariants(state)
