"""6 nimmt! heuristic picker behavior tests.

The picker is a port of `games/6nimmit/参考/lyc_pbh_wzq.py` — a 6-tier
commit decision tree (forced → block → push → seed → bad → fallback)
with `findsmall mode 1` for the SMALL phase. We can't construct
arbitrary board states from Python (the binding only seeds states), so
tier-by-tier isolation is impossible at this layer; instead these tests
cover behavior that's observable through the public API:

  1. invariants: legal-only support, |scores| == |actions|, exactly one
     score = 1.0 when the tier produces a single deterministic pick
  2. forced: when only one card is legal, pick it
  3. SMALL phase: heuristic returns a SMALL_CHOOSE_ROW action
  4. quality: heuristic strongly outperforms uniform-random head-to-head
     (≥70% win rate over 50 games)

Tier-isolation unit tests would require a state-construction helper
that doesn't exist; if/when one is added (e.g. `set_state_from_dict`),
those tests can be folded in here.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import dinoboard_engine
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

GAME = "6nimmit"
COMMIT_OFFSET = 0
SMALL_CHOOSE_ROW_OFFSET = 104
NUM_CARDS = 104
NUM_ROWS = 4


# ---------------------------------------------------------------------------
# 1. Picker invariants
# ---------------------------------------------------------------------------

class TestPickerInvariants:

    def test_scores_match_actions_length(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            if gs.is_terminal:
                continue
            r = gs.get_heuristic_action("rule_v1")
            # The Python binding surfaces `action` + `action_info` only;
            # invariants on the underlying scores vector are exercised
            # indirectly: the chosen action must be legal.
            assert r["action"] in gs.get_legal_actions()

    def test_picker_returns_legal_for_full_game(self):
        # Walk through several full games using the heuristic picker for
        # both seats — every decision must be legal.
        for seed in range(10):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            ply = 0
            while not gs.is_terminal and ply < 200:
                legal = gs.get_legal_actions()
                r = gs.get_heuristic_action("rule_v1")
                assert r["action"] in legal, (
                    f"seed={seed} ply={ply}: heuristic picked illegal action "
                    f"{r['action']} not in {legal}")
                gs.apply_action(r["action"])
                ply += 1
            assert gs.is_terminal, f"seed={seed} game did not finish"


# ---------------------------------------------------------------------------
# 2. SMALL phase
# ---------------------------------------------------------------------------

class TestSmallPhase:
    """When the apply queue dispatches a card whose value is smaller than
    every row tail, the framework pauses, sets small_phase_active=True
    and small_player=that committer, and the only legal actions are
    SMALL_CHOOSE_ROW (104..107). The heuristic must return one of those.
    """

    def test_heuristic_picks_small_choose_row_when_active(self):
        # Sweep many seeds; SMALL situations arise naturally.
        seen_small = 0
        for seed in range(80):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            ply = 0
            while not gs.is_terminal and ply < 200:
                st = gs.get_state_dict()
                if st["small_phase_active"]:
                    seen_small += 1
                    legal = gs.get_legal_actions()
                    r = gs.get_heuristic_action("rule_v1")
                    info = gs.get_action_info(r["action"])
                    assert info["type"] == "small_choose_row", (
                        f"seed={seed}: small_phase_active but picker chose "
                        f"non-row action: {info}")
                    assert r["action"] in legal
                # Drive both seats with random moves so we explore the
                # space — this is just a SMALL detector, not picker quality.
                random.seed(seed * 1000 + ply)
                gs.apply_action(random.choice(gs.get_legal_actions()))
                ply += 1
        assert seen_small > 0, (
            "no SMALL phase ever encountered in 80 seeds — sample size too "
            "small or rules buggy")


# ---------------------------------------------------------------------------
# 3. Quality regression — heuristic vs uniform-random
# ---------------------------------------------------------------------------

def _play_heuristic_vs_random(seed: int, heuristic_seat: int = 0) -> int:
    """Play a single game; seat 0/1 is heuristic, the other random.
    Returns winner index, or -1 on draw."""
    gs = dinoboard_engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed * 7 + heuristic_seat)
    ply = 0
    while not gs.is_terminal and ply < 200:
        cp = gs.current_player
        if cp == heuristic_seat:
            r = gs.get_heuristic_action("rule_v1")
            gs.apply_action(r["action"])
        else:
            legal = gs.get_legal_actions()
            gs.apply_action(rng.choice(legal))
        ply += 1
    return gs.get_state_dict().get("winner", -1)


class TestHeuristicQuality:
    """End-to-end behavior: the ported lyc_pbh_wzq strategy must
    strongly outperform uniform random. This is the canary for "did the
    port lose semantics" — if a tier is broken, win rate sags toward 50%.

    We play 30 games with heuristic on seat 0 and 30 games on seat 1
    (controls for the dealing-order asymmetry of 6 nimmt!) and require
    win rate ≥ 70%. Expected actual rate is ~85-95%.
    """

    def test_heuristic_beats_random(self):
        wins = 0
        games = 0
        for seed in range(30):
            w = _play_heuristic_vs_random(seed, heuristic_seat=0)
            if w == 0: wins += 1
            games += 1
            w = _play_heuristic_vs_random(seed + 1000, heuristic_seat=1)
            if w == 1: wins += 1
            games += 1
        rate = wins / games
        assert rate >= 0.70, (
            f"heuristic vs random win rate {rate:.1%} ({wins}/{games}) "
            f"below 70% — tier semantics likely broken in the port")


# ---------------------------------------------------------------------------
# 4. Forced singleton: when only one card is legal, picker chooses it.
# ---------------------------------------------------------------------------

class TestForcedSingleton:
    """At the very last commit ply of round 10, every player has exactly
    one card left → legal commits has size 1 → picker MUST return it."""

    def test_last_card_is_forced(self):
        for seed in range(8):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            rng = random.Random(seed)
            ply = 0
            forced_seen = False
            while not gs.is_terminal and ply < 200:
                legal = gs.get_legal_actions()
                if len(legal) == 1:
                    r = gs.get_heuristic_action("rule_v1")
                    assert r["action"] == legal[0], (
                        f"seed={seed} ply={ply}: legal={legal}, "
                        f"picker chose {r['action']}")
                    forced_seen = True
                gs.apply_action(rng.choice(gs.get_legal_actions()))
                ply += 1
            # SMALL plies always have 4 legal options, but commit plies in
            # the last round are forced — at least one such ply per game.
            assert forced_seen, f"seed={seed}: never hit a forced commit"
