"""6nimmit simultaneous-action three-slot pattern guards.

Locks the cursor scan / acts_mask / pending_commits invariants from
docs/plans/simultaneous_actions.md so future framework or game changes
that break the pattern fail loudly.
"""
import pytest
import dinoboard_engine

GAME = "6nimmit"


def _state_dict(gs):
    return gs.get_state_dict()


def test_root_current_player_is_zero_at_episode_start():
    """Stage-1 cursor invariant — at episode start every seat has acts_mask=1
    and pending_commits=0, so the cursor anchors at p0."""
    gs = dinoboard_engine.GameSession(GAME, seed=42)
    sd = _state_dict(gs)
    assert sd["current_player"] == 0
    assert sd["small_phase_active"] is False
    for p in range(sd["num_players"]):
        assert sd["players"][p]["pending_card"] == 0


def test_cursor_advances_after_p0_commits():
    """Stage-2 cursor invariant — after p0 commits, the public cursor moves
    to p1 (next acts_mask=1 ∧ pending=0 seat). Carries the 'GT advanced past
    p0' signal across the snapshot wire even though opp's pending stays viz=0
    on a session where opp hasn't committed yet."""
    gs = dinoboard_engine.GameSession(GAME, seed=42)
    legal = gs.get_legal_actions()
    gs.apply_action(legal[0])  # p0 commits its first legal card
    sd = _state_dict(gs)
    # Round didn't end: p1 still uncommitted.
    assert sd["small_phase_active"] is False
    assert sd["players"][1]["pending_card"] == 0
    # Cursor moved to p1.
    assert sd["current_player"] == 1


def test_full_round_resets_acts_and_advances_round():
    """After both seats commit and the apply loop drains, current_round
    bumps and pending_commits / cursor reset to round-0 state."""
    gs = dinoboard_engine.GameSession(GAME, seed=42)
    sd0 = _state_dict(gs)
    gs.apply_action(gs.get_legal_actions()[0])  # p0
    gs.apply_action(gs.get_legal_actions()[0])  # p1
    sd = _state_dict(gs)
    if sd["small_phase_active"]:
        gs.apply_action(gs.get_legal_actions()[0])
        sd = _state_dict(gs)
    # Round advanced.
    assert sd["current_round"] == sd0["current_round"] + 1
    # Cursor reset to a fresh seat with viz=1 acts_mask + viz=0 pending.
    assert sd["current_player"] in (0, 1)
    assert sd["players"][sd["current_player"]]["pending_card"] == 0


def test_legal_actions_belong_to_cursor_player():
    """Whoever current_player() returns is the seat whose legal_actions are
    served. Walks several plies to cover commit + SMALL phases."""
    gs = dinoboard_engine.GameSession(GAME, seed=42)
    plies = 0
    while not gs.is_terminal and plies < 30:
        sd = _state_dict(gs)
        cur = sd["current_player"]
        legal = gs.get_legal_actions()
        if sd["small_phase_active"]:
            assert sd["small_player"] == cur
            assert all(a >= 104 for a in legal)  # SMALL_CHOOSE_ROW range
        else:
            # Commit phase: cursor must NOT have already committed
            # (pending_commits[cur] must be 0).
            assert sd["players"][cur]["pending_card"] == 0
            assert all(a < 104 for a in legal)  # COMMIT range
        gs.apply_action(legal[0])
        plies += 1


def test_mcts_runs_without_apply_loop_unknown_state():
    """Regression — sim descent on a session whose opp pending is viz=0
    must not trip rules' apply-loop invariant. The randomize_unseen
    placeholder-→-0 normalize is the structural fix; this test wedges a
    full game through MCTS to catch any future regression."""
    import os
    m = "games/6nimmit/model/6nimmit_2p.onnx"
    if not os.path.exists(m):
        pytest.skip("deployed 6nimmit_2p.onnx not on disk")
    gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
    plies = 0
    while not gs.is_terminal and plies < 50:
        result = gs.get_ai_action(simulations=20, temperature=0.0)
        gs.apply_action(result["action"])
        plies += 1
    assert plies > 0
