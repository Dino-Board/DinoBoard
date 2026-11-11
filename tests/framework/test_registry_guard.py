"""Framework guard: a hidden-info game (belief_tracker != null) MUST register
the public-event protocol.

Why this lint exists: a game that adds a `belief_tracker` but forgets to call
`install_event_protocol(...)` (or directly assign `public_state_applier` /
`public_event_extractor`) silently desyncs. With no `public_state_applier`,
`advance_per_seat_states` / `apply_observation` fall back to replaying
`do_action_fast` on the session's masked seat — which reads viz=0 hidden slots
(unread bytes) and draws from a divergent RNG. The session's public projection
drifts from truth, and the symptom is `legal_actions(ai_view)` computed on a
wrong board → illegal moves / empty legal sets, surfacing mid-training, not at
startup. This is the harmonies commit 632ebe2 failure mode.

The runtime guard lives in `GameRegistry::create_game` (engine/core/
game_registry.h): `belief_tracker && !public_state_applier` (and the matching
extractor check) throw at session creation. This file pins that guard with:

  1. Runtime positive: every tracker game constructs a GameSession (which
     calls create_game) without raising — the guard must not false-fire on
     correctly-registered games.
  2. Static positive: every tracker game's register.cpp actually wires the
     protocol — catches the "forgot to register" mistake at lint time, before
     the runtime guard is even reachable.
  3. Guard presence: the throw text exists in game_registry.h, so a future
     refactor can't silently delete the guard.
"""
from __future__ import annotations

import re
from pathlib import Path

import dinoboard_engine
import pytest

from conftest import games_with_tracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"
REGISTRY_HEADER = PROJECT_ROOT / "engine" / "core" / "game_registry.h"


def _register_cpp(game_id: str) -> Path:
    """Locate <game>_register.cpp for a base game id. Tracker games all
    register through a single register.cpp shared across player-count
    variants (e.g. harmonies_3p uses harmonies_register.cpp)."""
    base = re.sub(r"_\d+p$", "", game_id)
    # games_with_tracker() returns base ids (manifest capability), but be
    # defensive: strip any variant suffix.
    candidates = sorted(GAMES_DIR.glob(f"{base}/*register*.cpp"))
    assert candidates, f"no *register*.cpp under games/{base}/"
    return candidates[0]


def test_guard_throw_text_present_in_registry_header():
    """The runtime guard's error strings must remain in game_registry.h.
    If someone refactors the guard away, this fails loudly instead of
    silently re-opening the desync hole."""
    text = REGISTRY_HEADER.read_text(encoding="utf-8")
    assert "registers a belief_tracker but no" in text, (
        "create_game guard for belief_tracker-without-protocol is missing "
        "from game_registry.h — was it deleted? See harmonies 632ebe2.")
    assert "public_state_applier" in text
    assert "public_event_extractor" in text


@pytest.mark.parametrize("game_id", games_with_tracker())
def test_tracker_game_creates_session_without_guard_fire(game_id):
    """Runtime positive: every tracker game's create_game (reached via
    GameSession construction) must succeed — the guard must not false-fire
    on a correctly-registered game. model_path omitted: create_game runs
    before any model is loaded, which is exactly where the guard fires."""
    gs = dinoboard_engine.GameSession(game_id, seed=42)
    assert gs.game_id == game_id


@pytest.mark.parametrize("game_id", games_with_tracker())
def test_tracker_game_registers_event_protocol(game_id):
    """Static positive: every tracker game's register.cpp must wire the
    public-event protocol. `install_event_protocol(...)` sets all three
    fields (events_only_extractor / public_event_extractor /
    public_state_applier) in one call — the normal path. Games may instead
    assign public_state_applier / public_event_extractor directly (Azul-style
    for a tracker-less snapshot game, though no tracker game does this today).
    Either wiring satisfies the create_game guard.

    A tracker game that does NEITHER is the harmonies-632ebe2 bug: it would
    pass static analysis but desync at runtime — except the create_game guard
    now catches it. This lint catches it even earlier (no build needed)."""
    src = _register_cpp(game_id).read_text(encoding="utf-8")
    has_helper = bool(re.search(r"\binstall_event_protocol\s*\(", src))
    has_applier = bool(re.search(r"\bpublic_state_applier\s*=", src))
    has_extractor = bool(re.search(r"\bpublic_event_extractor\s*=", src))
    assert has_helper or (has_applier and has_extractor), (
        f"{game_id} registers a belief_tracker but its register.cpp wires "
        f"neither install_event_protocol(...) nor direct "
        f"public_state_applier/public_event_extractor assignment. The AI "
        f"session would replay do_action_fast on a masked seat and desync "
        f"(harmonies 632ebe2 failure mode). Call install_event_protocol(...).")
