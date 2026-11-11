"""§A.c.6 structural test: in selfplay, the AI path's MCTS / encoder
output depends ONLY on the observation history — not on truth's
internal hidden state.

Background (CLAUDE.md "AI Pipeline Independence" + ALGORITHM_OVERVIEW
§9.8 + DEC-003):

  selfplay_runner advances per-seat session state via the public-event
  protocol after each truth do_action_fast. MCTS / encoder /
  legal_actions on the AI path read from per_seat_states[player], not
  from the truth state. Per DEC-003, the session itself NEVER calls
  randomize_unseen — viz=0 slots on the session are unread bytes
  (kHiddenHashSentinel in the hash, kPlaceholder in the encoder); only
  MCTS sims sample fresh worlds at sim entry on a cloned sim_tracker.

  Operationally this means: for two selfplay episodes that differ
  only in something invisible to the perspective player (e.g. truth's
  step-rng path through hidden draws, the draw order in face-down
  decks), the AI must produce identical decisions ply-for-ply,
  because:
    - per-seat session public state is rebuilt from public_snapshot,
      which is derived from public_event_extractor — itself invariant
      under truth's hidden internals
    - the session's own viz=0 bytes do not feed the hash or the
      encoder, so even if they happen to differ between seeds (they
      do — initial reset_with_seed leaves whatever the rules wrote)
      decision-side reads cannot distinguish them; sim-entry
      determinization on a clone is the only path that reads viz=0,
      and it draws from the tracker, not from session bytes

  We can't easily perturb only "truth-internal hidden" through the
  Python binding without also changing the action stream (selfplay's
  actions depend on MCTS, which now ignores truth-hidden — but the
  episode_seed seeds both GT step rng and AI session rng, so a single
  seed change perturbs both at once). What we CAN test cheaply:

    Two selfplay runs with the SAME episode_seed must be byte-equal
    on every observable AI output (action sequence, sample policy
    distributions, sample features). This is the deterministic-repro
    half: any non-determinism that crept into the AI path under
    session refactor would manifest as a divergence here.

    Two API sessions seeded DIFFERENTLY but driven through the same
    observation trace must produce IDENTICAL public state and (for
    games whose tracker is deterministic given history) identical
    belief — i.e. truth's seed cannot leak through into the AI's
    public-state view. (test_api_belief_matches_selfplay covers this
    for hidden-info games.)

Scope (IN_SCOPE_GAMES below): TicTacToe / Quoridor / Azul / Splendor /
LoveLetter — the five games whose per-seat session state is currently
driven by the public-event protocol. Coup is excluded pending its
schema-driven tracker rewrite (perspective-baked private knowledge in
the tracker hasn't migrated to viz=0 schema slots yet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


# Per-seat in-scope: every game whose perspective-private knowledge is
# carried by `state.viz_` (rules-driven via reveal_slot / reveal_slot_to
# / reset_to_base). Coup is still carved out — its tracker carries
# perspective-baked private knowledge that hasn't been migrated.
IN_SCOPE_GAMES = ["tictactoe", "quoridor", "azul", "splendor", "loveletter"]


def _samples_to_signature(samples):
    """A deterministic, comparable summary of a sample list."""
    sig = []
    for s in samples:
        sig.append((
            s["ply"], s["player"], s["action_id"],
            tuple(s["policy_action_ids"]),
            tuple(s["policy_action_visits"]),
        ))
    return sig


@pytest.mark.parametrize("game_id", IN_SCOPE_GAMES)
def test_selfplay_deterministic_under_same_seed(game_id):
    """Two selfplay runs with identical seed must produce identical AI
    decisions and policy targets. Catches any non-determinism that
    crept into the AI path under the per-seat session refactor — e.g.
    a stray std::random_device in randomize_unseen, an unhashed pointer
    in DAG node lookup, etc."""
    model_path = get_test_model(game_id)

    def _run():
        return engine.run_selfplay_episode(
            game_id=game_id, seed=12345, model_path=model_path,
            simulations=12, max_game_plies=20,
        )

    ep_a = _run()
    ep_b = _run()

    assert _samples_to_signature(ep_a["samples"]) == \
        _samples_to_signature(ep_b["samples"]), (
            f"[{game_id}] selfplay with the same seed produced different "
            f"AI decisions across runs. Non-determinism in MCTS or in "
            f"per-seat session advance — every input the AI path reads "
            f"must be reproducible from (episode_seed, ply, seat).")

    # Feature vectors are part of the AI output (the encoder reads from
    # the per-seat session state under §A.a). They must match too.
    assert len(ep_a["samples"]) == len(ep_b["samples"])
    for i, (sa, sb) in enumerate(zip(ep_a["samples"], ep_b["samples"])):
        fa = np.asarray(sa["features"], dtype=np.float32)
        fb = np.asarray(sb["features"], dtype=np.float32)
        assert np.array_equal(fa, fb), (
            f"[{game_id}] sample {i}: features differ between two runs of "
            f"the same seed — encoder is reading non-reproducible state.")


@pytest.mark.parametrize("game_id", IN_SCOPE_GAMES)
def test_api_session_independent_of_truth_seed(game_id):
    """Two API sessions started with DIFFERENT seeds, then driven
    through the same observation trace, must converge to the same
    public-state view. If they don't, the AI's public state still
    secretly depends on truth's hidden-internal RNG path, which would
    mean public_state_applier or randomize_unseen is leaking.

    For TicTacToe / Quoridor (no public_event_extractor, no hidden
    randomness) this is public state equivalence under apply_action
    replay. For Azul / Splendor / Love Letter (snapshot or tracker
    registered — Azul has fully-public counts but resolves bag draws
    via session_rng inside do_action_fast, so the session must be
    driven by snapshot replay rather than apply_action replay) this
    test verifies their public state under the per-seat session
    ignores the API session's own seed.
    """
    perspective = 0
    seed_truth = 42

    model_path = get_test_model(game_id)

    # Generate the canonical observation trace from a perspective.
    if game_id in ("tictactoe", "quoridor"):
        # Fully deterministic public games (no public_event_extractor):
        # drive both API sessions via apply_action against the truth's
        # action history. Both must reach byte-equal public state since
        # do_action_fast on these games is deterministic.
        ep = engine.run_selfplay_episode(
            game_id=game_id, seed=seed_truth, model_path=model_path,
            simulations=10, max_game_plies=30,
        )
        actions = [s["action_id"] for s in ep["samples"]]

        def _drive(sess_seed):
            sess = engine.GameSession(
                game_id, seed=sess_seed, model_path="", use_filter=False)
            for a in actions:
                if sess.is_terminal:
                    break
                sess.apply_action(a)
            return sess.get_state_dict()

        d_a = _drive(11111)
        d_b = _drive(99999)
        assert d_a == d_b, (
            f"[{game_id}] two fully-public sessions with different seeds "
            f"diverged after replaying the same action history: "
            f"public state depends on session seed.")
        return

    # Hidden-info-with-tracker games: use the observation_trace path.
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=10, max_game_plies=30,
        trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    if not trace:
        pytest.skip(f"{game_id}: no trace produced (game ended too early)")

    # Public fields these games expose via get_state_dict.
    PUBLIC_KEYS = {
        # Azul: bag/box_lid are per-color counts (all_public). Two
        # snapshot-driven sessions with different session_rng seeds must
        # arrive at byte-equal public state, including bag_counts and
        # box_counts — those flow through the snapshot, not through
        # session_rng.
        "azul": [
            "current_player", "is_terminal", "winner", "num_players",
            "round_index", "first_player_token_in_center", "scores",
            "factories", "center", "players",
            "bag_counts", "bag_total", "box_counts",
        ],
        "splendor": [
            "current_player", "is_terminal", "winner", "num_players",
            "bank", "tableau", "nobles", "players",
        ],
        # LL: `players[p]` bundles each seat's hand (owner-only). Two
        # sessions driven from perspective 0 will agree on hand[0] but
        # disagree on hand[1..] (those are belief samples, re-drawn by
        # session_rng on every observe — that's exactly the contract).
        # We compare only fields that are public to perspective 0:
        # public scalars, deck_size, 2p face_up_removed, and a slimmed
        # `players` view that strips the hand fields.
        "loveletter": [
            "current_player", "is_terminal", "winner", "num_players",
            "ply", "deck_size", "face_up_removed",
        ],
    }[game_id]

    def _drive(sess_seed):
        sess = engine.GameSession(
            game_id, seed=sess_seed, model_path="", use_filter=False)
        sess.apply_initial_observation(perspective, ep["initial_observation"])
        for step in trace:
            sess.apply_observation(
                step["action"],
                events=step["events"],
                public_snapshot=step.get("public_snapshot", {}),
            )
        return sess.get_state_dict()

    d_a = _drive(11111)
    d_b = _drive(99999)

    for key in PUBLIC_KEYS:
        assert d_a[key] == d_b[key], (
            f"[{game_id}] public field '{key}' diverged between two "
            f"differently-seeded API sessions driven through the same "
            f"observation trace:\n  seed=11111: {d_a[key]}\n  seed=99999: "
            f"{d_b[key]}\nThis means the AI's public state secretly "
            f"depends on session-internal RNG (BUG-028 surface).")

    # LL: compare per-player public-only fields explicitly. `hand` /
    # `hand_name` are owner-only and only legitimately public for
    # perspective itself; in a hidden-info game, two sessions can
    # carry different belief samples for opp seats — that is the
    # contract, not a leak.
    if game_id == "loveletter":
        public_pp_keys = ["alive", "protected", "discards"]
        assert len(d_a["players"]) == len(d_b["players"])
        for p, (pa, pb) in enumerate(zip(d_a["players"], d_b["players"])):
            for k in public_pp_keys:
                assert pa[k] == pb[k], (
                    f"[loveletter] players[{p}].{k} diverged: "
                    f"{pa[k]} vs {pb[k]}")
            # perspective's own hand is also legitimately visible.
            if p == perspective:
                assert pa["hand"] == pb["hand"], (
                    f"[loveletter] perspective's own hand diverged: "
                    f"{pa['hand']} vs {pb['hand']}")
