"""Azul regressions: round end without the first-player marker being taken,
and exact undo for the tail solver.

A1 (tail solver reported false wins): `will_round_end_after_action` ignored
that a pick from the centre also takes the first-player marker. When that
pick removed the last tiles the action really triggered the full round
settlement, but the undo record was the cheap partial one — other players'
boards, scores, factories, bag and lid stayed in their post-settlement state
and later sibling moves were evaluated on a corrupted position.

A3 (dead end): the round only ended once the marker had left the centre. If
every pick of a round emptied a single-colour factory nobody ever took from
the centre, the last tiles left, and the state was non-terminal with no legal
action (MCTS raised). Rule now: the round ends when no tiles remain; an
untaken marker stays in the centre, nobody is penalised, and the player who
started the round starts the next one.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

BLUE, YELLOW, RED, BLACK, WHITE = range(5)
CENTER_2P = 5          # source index of the centre in the 2-player game


def aid(source, color, target):
    return (source * 5 + color) * 6 + target


def _clear_table(gs, n_factories=5):
    for f in range(n_factories):
        for c in range(5):
            gs._force_write_field("factories", [f, c], 0)
    for c in range(5):
        gs._force_write_field("center", [c], 0)


# ---------------------------------------------------------------------------
# A1 — tail solver on a round-ending centre pick with the marker still there
# ---------------------------------------------------------------------------

def _false_win_position(model=""):
    """R = player 0 to move, P = player 1. Only factory 0 = {yellow, red} is
    left, the centre is empty and still holds the first-player marker.

    R completes a wall row at this round's settlement whatever happens (game
    ends; R finishes on 10 + 5 + 2 = 17). P has 17 points and a pattern line
    with 2 red (capacity 3) next to an occupied wall cell.
      R takes yellow -> P takes red + marker from the centre:
            red -> line 2 (index 1): 17 - 1     = 16  R wins
            red -> line 3 (index 2): 17 + 2 - 1 = 18  P wins   <- refutation
      R takes red    -> P takes yellow + marker: at best 16    R wins
    So "take red" is the only forced win for R.
    """
    gs = engine.GameSession("azul", seed=7, model_path=model)
    w = gs._force_write_field
    _clear_table(gs)
    w("factories", [0, YELLOW], 1)
    w("factories", [0, RED], 1)
    w("factories", [1, BLUE], 1)            # P's set-up move below
    w("first_player_token_in_center", [], 1)
    w("current_player", [], 1)
    # R
    w("player_wall_mask", [0, 0], 0b01111)
    w("player_line_color", [0, 0], WHITE)
    w("player_line_len", [0, 0], 1)
    w("player_line_color", [0, 4], BLUE)
    w("player_line_len", [0, 4], 4)
    w("player_score", [0], 10)
    w("scores", [0], 10)
    # P
    w("player_wall_mask", [1, 0], 0b00110)
    w("player_wall_mask", [1, 2], 0b01000)
    w("player_line_color", [1, 2], RED)
    w("player_line_len", [1, 2], 2)
    w("player_line_color", [1, 4], BLUE)
    w("player_line_len", [1, 4], 1)
    w("player_score", [1], 17)
    w("scores", [1], 17)
    # One real move so every AI view is synced from the truth snapshot.
    gs.apply_action(aid(1, BLUE, 3))
    assert gs.current_player == 0
    return gs


def _winner_after(seq):
    gs = _false_win_position()
    for a in seq:
        assert a in gs.get_legal_actions()
        gs.apply_action(a)
    assert gs.is_terminal
    return gs.winner


def _is_forced_win_for_r(action):
    probe = _false_win_position()
    probe.apply_action(action)
    return all(_winner_after([action, reply]) == 0 for reply in probe.get_legal_actions())


def test_position_has_one_refuted_and_one_winning_colour():
    # Ground truth by actually playing every line with do_action_fast.
    assert _winner_after([aid(0, YELLOW, 1), aid(CENTER_2P, RED, 2)]) == 1
    assert not _is_forced_win_for_r(aid(0, YELLOW, 1))
    assert _is_forced_win_for_r(aid(0, RED, 1))


def test_tail_solver_only_reports_a_real_forced_win():
    """The solver used to return `tail_solved` with "take yellow", which
    loses to red -> line 3."""
    gs = _false_win_position(get_test_model("azul"))
    gs.configure_tail_solve(True, 5, 200000, 0.0)
    res = gs.get_ai_action(simulations=8, temperature=0.0)
    stats = res["stats"]
    assert stats["tail_solve_attempted"]
    assert stats["tail_solved"], "a forced win exists (take red) and must be found"
    assert stats["tail_solve_value"] >= 1.0
    assert _is_forced_win_for_r(res["action"]), (
        f"tail solver claimed a proven win with action {res['action']} "
        f"({res['action_info']['color_name']} -> {res['action_info']['target_name']}), "
        f"but the opponent has a winning reply")


def _assert_roundtrips_exact(gs, ctx):
    r = gs._debug_do_undo_roundtrip(nested=3)
    base = r["baseline"]
    assert base["state"] == gs.get_state_dict()
    for rt in r["roundtrips"]:
        for key in ("state", "state_hash", "perspective_hashes", "step_count"):
            assert rt[key] == base[key], (
                f"{ctx}: do_action_deterministic + undo_action of action "
                f"{rt['action']} did not restore '{key}'")
    return len(r["roundtrips"])


def test_undo_is_exact_for_the_round_ending_centre_pick():
    gs = _false_win_position()
    gs.apply_action(aid(0, YELLOW, 1))      # centre = {red} + marker, P to move
    sd = gs.get_state_dict()
    assert sd["first_player_token_in_center"] and sum(sd["center"]) == 1
    assert _assert_roundtrips_exact(gs, "marker-in-centre round end") == 3


@pytest.mark.parametrize("variant,n_factories", [("azul", 5), ("azul_3p", 7), ("azul_4p", 9)])
def test_undo_is_exact_along_random_games(variant, n_factories):
    """Every legal action at every ply of random games, including the picks
    that end a round and the ones that end the game."""
    rng = random.Random(11)
    round_ends = game_ends = checked = 0
    for seed in range(6):
        gs = engine.GameSession(variant, seed=seed)
        while not gs.is_terminal:
            checked += _assert_roundtrips_exact(gs, f"{variant} seed={seed}")
            before = gs.get_state_dict()
            gs.apply_action(rng.choice(gs.get_legal_actions()))
            round_ends += gs.get_state_dict()["round_index"] != before["round_index"]
        game_ends += 1
    assert checked > 1000 and round_ends >= 12 and game_ends == 6


# ---------------------------------------------------------------------------
# A3 — round end when the marker was never taken
# ---------------------------------------------------------------------------

def _marker_never_taken_position(model=""):
    """Factory 0 holds 4 blue, nothing else is on the table, the marker is
    still in the centre, player 0 is to move (after a set-up move by 1)."""
    gs = engine.GameSession("azul", seed=7, model_path=model)
    w = gs._force_write_field
    _clear_table(gs)
    w("factories", [0, BLUE], 4)
    w("factories", [1, YELLOW], 1)
    w("first_player_token_in_center", [], 1)
    w("current_player", [], 1)
    for p in (0, 1):
        w("player_score", [p], 10)
        w("scores", [p], 10)
    gs.apply_action(aid(1, YELLOW, 5))       # player 1: yellow -> floor
    return gs


def test_round_ends_when_no_tiles_remain_even_if_marker_untaken():
    gs = _marker_never_taken_position()
    before = gs.get_state_dict()
    assert before["current_player"] == 0 and before["first_player_token_in_center"]
    gs.apply_action(aid(0, BLUE, 4))         # last tiles: 4 blue -> line 5
    sd = gs.get_state_dict()
    assert not gs.is_terminal
    assert gs.get_legal_actions(), "non-terminal state without a legal action"
    assert sd["round_index"] == before["round_index"] + 1
    assert sum(sum(f) for f in sd["factories"]) == 20, "factories were not refilled"
    # The marker stays in the centre and nobody pays for it: player 1 only
    # loses the point for its one floor tile, player 0 loses nothing.
    assert sd["first_player_token_in_center"] is True
    assert sd["scores"] == [10, 9]
    assert all(-2 not in p["floor"] for p in sd["players"])
    # Nobody took the marker: the player who started the previous round
    # (seat 0, the initial first player) starts again.
    assert sd["current_player"] == 0


def test_game_ends_when_no_tile_can_be_dealt():
    """Bag and lid empty and the last tiles go onto the walls: the next round
    cannot be dealt, so the game ends with the normal final scoring instead
    of leaving a round with no legal action."""
    gs = engine.GameSession("azul", seed=7)
    w = gs._force_write_field
    _clear_table(gs)
    for c in range(5):
        w("bag_counts", [c], 0)
        w("box_lid_counts", [c], 0)
    w("factories", [0, BLUE], 1)
    w("factories", [1, YELLOW], 1)
    w("first_player_token_in_center", [], 1)
    w("current_player", [], 1)
    w("player_score", [0], 5)
    w("scores", [0], 5)
    w("player_score", [1], 3)
    w("scores", [1], 3)
    gs.apply_action(aid(1, YELLOW, 0))       # player 1: yellow -> line 1 (full)
    assert not gs.is_terminal
    gs.apply_action(aid(0, BLUE, 0))         # player 0: blue -> line 1 (full)
    sd = gs.get_state_dict()
    assert sum(sd["bag_counts"]) + sum(sd["box_counts"]) == 0
    assert gs.is_terminal, "no tile left to deal, yet the game goes on"
    assert sd["scores"] == [6, 4]            # one isolated wall tile each
    assert gs.winner == 0 and gs.winners == [0] and gs.losers == [1]


def test_search_survives_the_marker_never_taken_position():
    """MCTS used to raise 'legal_actions returned empty for non-terminal
    state' one ply before the dead end."""
    gs = _marker_never_taken_position(get_test_model("azul"))
    res = gs.get_ai_action(simulations=60, temperature=0.0)
    assert res["action"] in gs.get_legal_actions()


def test_random_four_player_games_never_dead_end():
    """Same random stream that reached the dead end in the audit (azul_4p,
    game 3095: a single 4-red factory was all that could be refilled)."""
    rng = random.Random(3000)
    unfinished = 0
    for gi in range(150):
        gs = engine.GameSession("azul_4p", seed=3000 + gi)
        plies = 0
        # 500 plies is far beyond any real game (random 4p games take ~110).
        # A random game may legitimately not finish: e.g. game 3095 is left
        # with 4 red tiles that the round's starter can only put on the
        # floor, so they cycle floor -> lid -> factory forever (runners cut
        # that off with max_game_plies); what must never happen is a state
        # with nothing to do.
        while not gs.is_terminal and plies < 500:
            legal = gs.get_legal_actions()
            assert legal, f"seed {3000 + gi} ply {plies}: non-terminal state without legal actions"
            gs.apply_action(rng.choice(legal))
            plies += 1
        unfinished += not gs.is_terminal
    assert unfinished <= 5, f"{unfinished}/150 random games did not finish"
