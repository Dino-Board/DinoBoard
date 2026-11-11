"""Multi-player games: a tie for first place between SOME of the players.

Convention (IGameState::winners / losers + DefaultStateValueModel):
  - one winner            -> winners = [w], losers = everyone else
  - partial tie for first -> winners = every player tied for first,
                             losers  = everyone else
                             values: +1/len(winners) each, -1/len(losers) each
  - everyone tied (and every 2-player tie)
                          -> winners = losers = [], all values 0 (a draw)
`winner` stays -1 whenever there is no single winner.

Before this was implemented, Azul / Harmonies / Splendor reported any tie as
winner = -1 with empty winners/losers, so in a 3-4 player game a player who
clearly finished behind the tied leaders received the same value (0) as the
leaders — "lose clearly" and "share first place" were indistinguishable to
the value head.

The value vector itself is checked end-to-end on real episodes (Harmonies
3p/4p, Azul 3p/4p) against a ranking recomputed here from the final state;
rare configurations are built directly with `_force_write_field`.
"""
from __future__ import annotations

import pytest

import dinoboard_engine as engine


def expected_values(keys):
    """keys[p] = ranking key (bigger is better). Returns (winners, losers, z)."""
    n = len(keys)
    best = max(keys)
    leaders = [p for p in range(n) if keys[p] == best]
    if len(leaders) == n:
        return [], [], [0.0] * n
    others = [p for p in range(n) if p not in leaders]
    z = [0.0] * n
    for p in leaders:
        z[p] = 1.0 / len(leaders)
    for p in others:
        z[p] = -1.0 / len(others)
    return leaders, others, z


# ---------------------------------------------------------------------------
# End to end: runner value targets vs. an independent ranking
# ---------------------------------------------------------------------------

def _harmonies_keys(sd):
    return [(p["score"], sum(p["has_animal"])) for p in sd["players"]]


def _azul_keys(sd):
    return [(p["score"], sum(1 for row in p["wall"] if all(row))) for p in sd["players"]]


# Seed windows chosen to contain real partial ties for first place that the
# cube tie-break does not resolve (3p: seeds 37, 127, 145; 4p: seed 79).
@pytest.mark.parametrize("game_id,heuristic,keys_of,seeds", [
    ("harmonies_a_3p", "rule", _harmonies_keys, list(range(30, 45)) + [127, 145]),
    ("harmonies_a_4p", "rule", _harmonies_keys, range(72, 84)),
    ("azul_3p", "rule_v1", _azul_keys, range(1, 31)),
    ("azul_4p", "rule_v1", _azul_keys, range(1, 31)),
])
def test_episode_values_follow_the_ranking(game_id, heuristic, keys_of, seeds):
    kinds = {"single": 0, "partial_tie": 0, "full_tie": 0}
    for seed in seeds:
        ep = engine.run_heuristic_episode(
            game_id=game_id, seed=seed, temperature=1.0, max_game_plies=600,
            heuristic_selector=heuristic)
        gs = engine.GameSession(game_id, seed=seed)
        for s in ep["samples"]:
            gs.apply_action(s["action_id"])
        assert gs.is_terminal, f"{game_id} seed={seed}: episode did not finish"
        winners, losers, z = expected_values(keys_of(gs.get_state_dict()))
        kinds["full_tie" if not winners else ("single" if len(winners) == 1 else "partial_tie")] += 1
        ctx = f"{game_id} seed={seed} keys={keys_of(gs.get_state_dict())}"
        assert gs.winners == winners and gs.losers == losers, ctx
        assert gs.winner == (winners[0] if len(winners) == 1 else -1), ctx
        assert ep["samples"][-1]["z_values"] == pytest.approx(z, abs=1e-6), ctx
        assert sum(ep["samples"][-1]["z_values"]) == pytest.approx(0.0, abs=1e-6), ctx
        assert ep["draw"] == (not winners), ctx
    assert kinds["single"] > 0
    if game_id.startswith("harmonies"):
        # Random Harmonies games tie often enough to exercise the convention
        # on real episodes; Azul ties are covered by the built positions below.
        assert kinds["partial_tie"] > 0, kinds


# ---------------------------------------------------------------------------
# Built positions
# ---------------------------------------------------------------------------

def _harmonies_final(n, totals, cubes):
    """Play one full round on a fresh n-player game with the end already
    triggered; just before the last END_TURN set every player's animal score
    so the final totals are `totals`, and give them `cubes` animal cubes."""
    # Side A board: the action ids (175, 911) and 23 hexes below are its layout.
    gs = engine.GameSession(f"harmonies_a_{n}p", seed=5)
    gs._force_write_field("end_triggered", [], 1)
    for p in range(n):
        gs.apply_action(next(a for a in gs.get_legal_actions() if a < 5))
        for _ in range(3):
            gs.apply_action(next(a for a in gs.get_legal_actions() if 37 <= a < 175))
        if p == n - 1:
            sd = gs.get_state_dict()
            for q in range(n):
                land = sd["players"][q]["landscape_score"]
                assert totals[q] >= land
                gs._force_write_field("animal_scores", [q], totals[q] - land)
                empty = [h for h in range(23) if sd["players"][q]["tower_type"][h] == 0]
                for h in empty[:cubes[q]]:
                    gs._force_write_field("has_animal", [q, h], 1)
        gs.apply_action(911)
    assert gs.is_terminal
    sd = gs.get_state_dict()
    assert [pl["score"] for pl in sd["players"]] == totals
    return gs


@pytest.mark.parametrize("totals,cubes,winner,winners,losers", [
    ([40, 30, 40], [2, 2, 2], -1, [0, 2], [1]),         # partial tie, cubes equal
    ([40, 30, 40], [2, 5, 3], 2, [2], [0, 1]),          # tie-break: most cubes
    ([40, 40, 40], [1, 1, 1], -1, [], []),              # everyone tied: draw
    ([40, 40, 40], [1, 2, 2], -1, [1, 2], [0]),         # tie-break leaves two
    ([30, 40, 35], [9, 0, 9], 1, [1], [0, 2]),          # points first, cubes only break ties
])
def test_harmonies_three_player_final_ranking(totals, cubes, winner, winners, losers):
    gs = _harmonies_final(3, totals, cubes)
    assert (gs.winner, gs.winners, gs.losers) == (winner, winners, losers)


@pytest.mark.parametrize("totals,cubes,winner,winners,losers", [
    ([40, 40], [3, 3], -1, [], []),                     # 2-player tie stays a draw
    ([40, 40], [3, 4], 1, [1], [0]),                    # rulebook tie-break
    ([41, 40], [0, 9], 0, [0], [1]),
])
def test_harmonies_two_player_final_ranking(totals, cubes, winner, winners, losers):
    gs = _harmonies_final(2, totals, cubes)
    assert (gs.winner, gs.winners, gs.losers) == (winner, winners, losers)


def _azul_3p_final(scores_before, row_completers):
    """3-player Azul, last two picks of the game. Players in `row_completers`
    hold a wall row with 4 tiles plus the matching full pattern line, so they
    complete a row at the settlement (+5 for the tile, +2 row bonus).
    Player 1 then player 2 each put one tile on the floor (-1)."""
    gs = engine.GameSession("azul_3p", seed=3)
    w = gs._force_write_field
    for f in range(7):
        for c in range(5):
            w("factories", [f, c], 0)
    for c in range(5):
        w("center", [c], 0)
    w("factories", [0, 0], 1)
    w("factories", [1, 1], 1)
    w("first_player_token_in_center", [], 0)
    w("current_player", [], 1)
    for p in range(3):
        w("player_score", [p], scores_before[p])
        w("scores", [p], scores_before[p])
        if p in row_completers:
            w("player_wall_mask", [p, 0], 0b01111)
            w("player_line_color", [p, 0], 4)
            w("player_line_len", [p, 0], 1)
    gs.apply_action((1 * 5 + 1) * 6 + 5)      # player 1: yellow from factory 1 -> floor
    gs.apply_action((0 * 5 + 0) * 6 + 5)      # player 2: blue from factory 0 -> floor
    assert gs.is_terminal
    return gs


def test_azul_three_player_partial_shared_victory():
    # 0: 10 + 7 = 17 (1 row); 1: 11 + 7 - 1 = 17 (1 row); 2: 5 - 1 = 4
    gs = _azul_3p_final([10, 11, 5], row_completers=(0, 1))
    assert gs.get_state_dict()["scores"] == [17, 17, 4]
    assert (gs.winner, gs.winners, gs.losers) == (-1, [0, 1], [2])


def test_azul_three_player_tie_broken_by_completed_rows():
    # 0: 17 with a completed row; 1: 18 - 1 = 17 without one -> 0 wins alone
    gs = _azul_3p_final([10, 18, 5], row_completers=(0,))
    assert gs.get_state_dict()["scores"] == [17, 17, 4]
    assert (gs.winner, gs.winners, gs.losers) == (0, [0], [1, 2])


def test_azul_three_player_full_tie_is_a_draw():
    # 0: 17; 1: 11 + 7 - 1 = 17; 2: 11 + 7 - 1 = 17, one row each
    gs = _azul_3p_final([10, 11, 11], row_completers=(0, 1, 2))
    assert gs.get_state_dict()["scores"] == [17, 17, 17]
    assert (gs.winner, gs.winners, gs.losers) == (-1, [], [])


def _splendor_final(n, points, cards):
    """Force the last turn of the final round: the last player of the round
    takes gems, which ends the game with the given points / card counts."""
    gs = engine.GameSession(f"splendor_{n}p", seed=11)
    w = gs._force_write_field
    w("first_player", [], 0)
    w("current_player", [], n - 1)
    w("final_round_remaining", [], 1)
    for p in range(n):
        w("player_points", [p], points[p])
        w("player_cards_count", [p], cards[p])
    take = next(a for a in gs.get_legal_actions() if gs.get_action_info(a)["type"].startswith("take"))
    gs.apply_action(take)
    assert gs.is_terminal
    return gs


@pytest.mark.parametrize("n,points,cards,winner,winners,losers", [
    (3, [16, 12, 16], [5, 4, 5], -1, [0, 2], [1]),      # partial shared victory
    (3, [16, 12, 16], [5, 4, 4], 2, [2], [0, 1]),       # fewest cards breaks the tie
    (3, [16, 16, 16], [5, 5, 5], -1, [], []),           # everyone tied: draw
    (4, [15, 17, 17, 17], [9, 6, 6, 6], -1, [1, 2, 3], [0]),
    (2, [16, 16], [5, 5], -1, [], []),                  # 2-player tie stays a draw
])
def test_splendor_final_ranking(n, points, cards, winner, winners, losers):
    gs = _splendor_final(n, points, cards)
    assert (gs.winner, gs.winners, gs.losers) == (winner, winners, losers)
