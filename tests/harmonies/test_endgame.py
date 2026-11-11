"""Harmonies end-of-game regressions.

H1  Rulebook p.6: "In case of a tie, the player who has placed the most
    Animal cubes wins. In the event of another tie, players share the
    victory." The engine used to declare every tie on points a draw.
H3  max_game_plies was 200; about a third of 4-player games need more and
    were cut off (no adjudicator -> every seat got z = 0). The cap is now at
    least the provable maximum game length, so no game can be truncated.
H4  Completed animal cards left no public trace, so randomize_unseen could
    put them back into the sampled deck. `completed_cards` is now a public
    state field and the sampled deck equals the real deck membership.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import dinoboard_engine as engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _layout(game_id):
    return engine.GameSession(game_id, seed=0).get_state_dict()["action_layout"]


def _greedy_game(game_id, seed):
    gs = engine.GameSession(game_id, seed=seed)
    plies = 0
    while not gs.is_terminal:
        gs.apply_action(gs.get_heuristic_action("greedy", 0.0)["action"])
        plies += 1
        assert plies < 2000
    return gs, plies


def _rulebook_winner(sd):
    """Most points; tie -> most animal cubes; still tied -> shared (-1)."""
    keys = [(p["score"], sum(p["has_animal"])) for p in sd["players"]]
    best = max(keys)
    leaders = [i for i, k in enumerate(keys) if k == best]
    return (leaders[0] if len(leaders) == 1 else -1), keys


# ---------------------------------------------------------------------------
# H1 — tie-break by animal cubes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["harmonies_a", "harmonies"])
def test_winner_follows_rulebook_tie_break_in_real_games(game_id):
    tied_on_points = decided_by_cubes = 0
    for seed in range(1, 61):
        gs, _ = _greedy_game(game_id, seed)
        sd = gs.get_state_dict()
        expected, keys = _rulebook_winner(sd)
        assert gs.winner == expected and sd["winner"] == expected, (
            f"seed={seed}: (points, cubes) = {keys}, rulebook winner {expected}, "
            f"engine winner {gs.winner}")
        if keys[0][0] == keys[1][0]:
            tied_on_points += 1
            decided_by_cubes += expected >= 0
    # Side A: seeds 14 / 30 / 50 end level on points with different cube counts.
    if game_id == "harmonies_a":
        assert tied_on_points >= 2 and decided_by_cubes >= 2


# ---------------------------------------------------------------------------
# H3 — the ply cap can no longer truncate a game
# ---------------------------------------------------------------------------

def _max_possible_plies(n_players, n_hexes):
    """Upper bound on actions in one game.

    Turns: the pouch holds 120 tokens, 15 start on the central board, every
    turn refills 3 -> the 36th refill finds the pouch empty and triggers the
    end; 36 is a multiple of 2, 3 and 4, so the round is complete right there.
    Per turn: choose slot + 3 token placements + end turn = 5 actions.
    Optional actions: each of the 32 animal cards can be taken once; every
    cube lands on a different space of its owner's board (23 or 25 spaces) and there
    are only as many cubes as the cards carry."""
    cards = json.loads((PROJECT_ROOT / "games/harmonies/animal_cards.json").read_text(encoding="utf-8"))["cards"]
    assert len(cards) == 32
    total_cubes = sum(len(c["scores"]) for c in cards)
    return 36 * 5 + 32 + min(n_hexes * n_players, total_cubes)


def test_ply_cap_is_not_below_the_longest_possible_game():
    cfg = json.loads((PROJECT_ROOT / "games/harmonies/config/game.json").read_text(encoding="utf-8"))
    cap = cfg["max_game_plies"]
    for variant, meta in cfg["variants"].items():
        hexes = (meta["action_space"] - 38) // 38
        bound = _max_possible_plies(meta["num_players"], hexes)
        assert cap >= bound, (
            f"{variant}: max_game_plies={cap} is below the longest possible "
            f"game ({bound} actions) — episodes can be cut off before the end")


def test_four_player_games_finish_under_the_cap():
    """Real 4-player games: many run past the old cap of 200, none reaches the
    new one. Runs through the selfplay-style runner with the configured cap."""
    cfg = json.loads((PROJECT_ROOT / "games/harmonies/config/game.json").read_text(encoding="utf-8"))
    cap = cfg["max_game_plies"]
    lengths = []
    for seed in range(1, 25):
        ep = engine.run_heuristic_episode(
            game_id="harmonies_4p", seed=seed, temperature=0.0,
            max_game_plies=cap, heuristic_selector="greedy")
        gs = engine.GameSession("harmonies_4p", seed=seed)
        for s in ep["samples"]:
            gs.apply_action(s["action_id"])
        assert gs.is_terminal, f"seed={seed}: cut off after {ep['total_plies']} plies"
        assert ep["total_plies"] <= _max_possible_plies(4, 25)
        lengths.append(ep["total_plies"])
    assert max(lengths) > 200, "no game needed more than the old cap; test lost its point"


def test_turn_count_never_exceeds_36():
    rng = random.Random(4)
    for game_id in ("harmonies", "harmonies_3p", "harmonies_4p", "harmonies_a", "harmonies_a_4p"):
        end_turn = _layout(game_id)["end_turn"]
        for seed in range(15):
            gs = engine.GameSession(game_id, seed=seed)
            turns = 0
            while not gs.is_terminal:
                a = rng.choice(gs.get_legal_actions())
                turns += a == end_turn
                gs.apply_action(a)
            assert turns <= 36


# ---------------------------------------------------------------------------
# H4 — completed cards are public and never return to the deck
# ---------------------------------------------------------------------------

def test_completed_cards_are_tracked_and_every_card_is_accounted_for():
    completed_total = 0
    lay = _layout("harmonies")
    for seed in range(1, 9):
        gs = engine.GameSession("harmonies", seed=seed)
        tally = [0] * 32
        while not gs.is_terminal:
            before = gs.get_state_dict()
            p = before["current_player"]
            a = gs.get_heuristic_action("greedy", 0.0)["action"]
            if lay["place_cube"] <= a < lay["end_turn"]:
                # The greedy player rarely finishes a card on its own: make
                # every cube it places the card's last one.
                cid = (a - lay["place_cube"]) % 32 + 1
                slot = list(before["players"][p]["held_cards"]).index(cid)
                gs._force_write_field("cube_remaining", [p, slot], 1)
                tally[cid - 1] = 1
            gs.apply_action(a)
            sd = gs.get_state_dict()
            if lay["place_cube"] <= a < lay["end_turn"]:
                assert cid not in sd["players"][p]["held_cards"]
            assert sd["completed_cards"] == tally
            held = {c for pl in sd["players"] for c in pl["held_cards"] if c > 0}
            market = {c for c in sd["animal_market"] if c > 0}
            deck = {i + 1 for i, v in enumerate(sd["animal_deck"]) if v}
            done = {i + 1 for i, v in enumerate(sd["completed_cards"]) if v}
            parts = [held, market, deck, done]
            assert sum(len(s) for s in parts) == 32 and set().union(*parts) == set(range(1, 33))
        completed_total += sum(tally)
    assert completed_total >= 20


def test_sampled_deck_never_contains_a_completed_card():
    """randomize_unseen (via the test_belief_tracker binding: random plies,
    then sampled clones): the sampled deck must be exactly the real deck —
    in Harmonies every card outside the deck is publicly located."""
    states_with_completed = 0
    for seed in range(1, 301):
        for plies in (60, 90):
            r = engine.test_belief_tracker("harmonies", seed, plies, 4)
            o = r["original_state"]
            deck_true = [i + 1 for i, v in enumerate(o["animal_deck"]) if v]
            held = {c for pl in o["players"] for c in pl["held_cards"] if c > 0}
            market = {c for c in o["animal_market"] if c > 0}
            completed = set(range(1, 33)) - set(deck_true) - held - market
            states_with_completed += bool(completed)
            for t in r["trial_states"]:
                deck_sim = [i + 1 for i, v in enumerate(t["animal_deck"]) if v]
                assert not (set(deck_sim) & completed), (
                    f"seed={seed} plies={plies}: sampled deck holds completed "
                    f"card(s) {sorted(set(deck_sim) & completed)}")
                assert deck_sim == deck_true
    assert states_with_completed >= 5
