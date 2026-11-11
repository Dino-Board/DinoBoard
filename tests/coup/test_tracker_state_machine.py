"""Unit tests for CoupBeliefTracker's stage-boundary state machine.

These tests drive specific (action, expected events, expected raw counts)
sequences through GameSession and assert on `get_belief_snapshot()`.
They cover bugs that the smoke-only `test_heuristic_sampling.py` cannot
catch:

1. Single-counting: in N>2 games every opp Allow after a claim must
   collapse into ONE `claim_unchallenged` event total (not N-1) so
   pending state clears exactly once.
2. Pending-claim tracking: pending_claimer / pending_claim_role /
   pending_challenged update on claim, challenge, and resolve.
3. v0.2 raw-event accumulators (pre/post claim & challenge counts,
   last_reshuffle_kind, last_revealed_role) are the belief-network
   feature extractor's input — schema, shapes, and reshuffle promotion
   must stay stable.

Handcraft signal-based scoring was retired with the multiset belief net
rewrite (2026-05). The tracker no longer maintains the additive
`signals_` array; sampling fallback is uniform-from-pool shuffle.

The tracker is perspective-invariant, so we only test the session's
shared `bt_` via `gs.get_belief_snapshot()`.
"""
import dinoboard_engine
import pytest


# Coup action IDs (mirrors games/coup/coup_state.h).
INCOME = 0
FOREIGN_AID = 1
COUP_OFFSET = 2
TAX = 6
ASSASSIN_OFFSET = 7
STEAL_OFFSET = 11
EXCHANGE = 15
CHALLENGE = 16
ALLOW = 17
BLOCK_DUKE = 18
BLOCK_CONTESSA = 19
BLOCK_AMBASSADOR = 20
BLOCK_CAPTAIN = 21
ALLOW_NO_BLOCK = 22

# Character ids.
DUKE = 0
ASSASSIN = 1
CAPTAIN = 2
AMBASSADOR = 3
CONTESSA = 4
NUM_CHARS = 5


def _zero_counts(n_players):
    return [[0] * NUM_CHARS for _ in range(n_players)]


def _claim_counts(snap, kind, n_players):
    flat = snap[f"{kind}_claim_counts"]
    return [
        list(flat[p * NUM_CHARS:(p + 1) * NUM_CHARS])
        for p in range(n_players)
    ]


def _challenge_counts(snap, kind, n_players):
    flat = snap[f"{kind}_challenge_initiated"]
    return [
        list(flat[p * NUM_CHARS:(p + 1) * NUM_CHARS])
        for p in range(n_players)
    ]


class TestInitialState:
    """Fresh session: no pending claim, no accrued counts."""

    @pytest.mark.parametrize("game_id,n", [
        ("coup_2p", 2), ("coup_3p", 3), ("coup_4p", 4)
    ])
    def test_fresh_session(self, game_id, n):
        gs = dinoboard_engine.GameSession(game_id, seed=1, model_path="")
        snap = gs.get_belief_snapshot()
        assert _claim_counts(snap, "pre", n) == _zero_counts(n)
        assert _claim_counts(snap, "post", n) == _zero_counts(n)
        assert _challenge_counts(snap, "pre", n) == _zero_counts(n)
        assert _challenge_counts(snap, "post", n) == _zero_counts(n)
        assert snap["pending_claimer"] == -1
        assert snap["pending_claim_role"] == -1
        assert snap["pending_challenged"] is False


class TestPendingClaimLifecycle:
    """A claim sets pending_*; a Challenge flips pending_challenged.
    `claim_unchallenged` collapses N-1 Allows into one pending-clear."""

    def test_2p_tax_allow_clears_pending(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        snap = gs.get_belief_snapshot()
        assert snap["pending_claimer"] == 0
        assert snap["pending_claim_role"] == DUKE
        assert snap["pending_challenged"] is False
        # post_claim_counts must have bumped on the claimer.
        assert _claim_counts(snap, "post", 2)[0][DUKE] == 1

        gs.apply_action(ALLOW)          # P1 Allows → claim_unchallenged
        snap = gs.get_belief_snapshot()
        assert snap["pending_claimer"] == -1
        assert snap["pending_claim_role"] == -1
        assert snap["pending_challenged"] is False

    def test_3p_two_allows_no_double_clear(self):
        """N=3: P1 Allows then P2 Allows. Pending must remain set after
        the first Allow (no event yet) and clear after the second."""
        gs = dinoboard_engine.GameSession("coup_3p", seed=1, model_path="")
        gs.apply_action(TAX)
        gs.apply_action(ALLOW)
        mid = gs.get_belief_snapshot()
        assert mid["pending_claimer"] == 0, (
            "intermediate Allow shouldn't fire claim_unchallenged"
        )
        gs.apply_action(ALLOW)
        snap = gs.get_belief_snapshot()
        assert snap["pending_claimer"] == -1

    def test_4p_three_allows_no_triple_clear(self):
        gs = dinoboard_engine.GameSession("coup_4p", seed=1, model_path="")
        gs.apply_action(TAX)
        gs.apply_action(ALLOW)
        gs.apply_action(ALLOW)
        mid = gs.get_belief_snapshot()
        assert mid["pending_claimer"] == 0
        gs.apply_action(ALLOW)
        snap = gs.get_belief_snapshot()
        assert snap["pending_claimer"] == -1

    def test_challenge_sets_pending_challenged(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)
        gs.apply_action(CHALLENGE)
        snap = gs.get_belief_snapshot()
        # Action-level: challenger accrues post_challenge_initiated.
        assert _challenge_counts(snap, "post", 2)[1][DUKE] == 1
        # Pending stays set (cleared on resolve event, not on challenge).
        assert snap["pending_challenged"] is True


class TestRawCountsV02:
    """v0.2 raw-event accumulators consumed by the belief-net feature
    extractor: pre/post claim counts, pre/post challenge-initiated,
    last_reshuffle_kind, last_revealed_role."""

    KIND_NONE = 0
    KIND_EXCHANGE = 1
    KIND_REVEAL_TRUTHFUL = 2

    def test_initial_zero(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        snap = gs.get_belief_snapshot()
        assert _claim_counts(snap, "pre", 2) == _zero_counts(2)
        assert _claim_counts(snap, "post", 2) == _zero_counts(2)
        assert _challenge_counts(snap, "pre", 2) == _zero_counts(2)
        assert _challenge_counts(snap, "post", 2) == _zero_counts(2)
        assert list(snap["last_reshuffle_kind"]) == [self.KIND_NONE] * 2
        assert list(snap["last_revealed_role"]) == [-1] * 2

    def test_claim_lands_on_post_not_pre(self):
        """First claim before any reshuffle: post += 1, pre stays 0."""
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)
        snap = gs.get_belief_snapshot()
        assert _claim_counts(snap, "post", 2)[0][DUKE] == 1
        assert _claim_counts(snap, "pre", 2) == _zero_counts(2)
        assert list(snap["last_reshuffle_kind"]) == [self.KIND_NONE] * 2

    def test_challenge_lands_on_post_initiated(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)
        gs.apply_action(CHALLENGE)
        snap = gs.get_belief_snapshot()
        assert _challenge_counts(snap, "post", 2)[1][DUKE] == 1
        assert _challenge_counts(snap, "pre", 2) == _zero_counts(2)


class TestStageBoundaryEventsFromSelfplay:
    """End-to-end: run a real selfplay episode and verify the tracker's
    terminal serialize() yields a sane schema (right shapes, all entries
    non-negative and bounded)."""

    @pytest.mark.parametrize("game_id,n", [
        ("coup_2p", 2), ("coup_3p", 3), ("coup_4p", 4)
    ])
    def test_counts_within_bounds_after_selfplay_drive(self, game_id, n):
        gs = dinoboard_engine.GameSession(game_id, seed=42, model_path="")
        for _ in range(20):
            if gs.is_terminal:
                break
            legal = gs.get_legal_actions()
            gs.apply_action(legal[0])
        snap = gs.get_belief_snapshot()
        for kind in ("pre", "post"):
            mat = _claim_counts(snap, kind, n)
            for p in range(n):
                for c in range(NUM_CHARS):
                    assert mat[p][c] >= 0
                    assert mat[p][c] <= 50
            mat = _challenge_counts(snap, kind, n)
            for p in range(n):
                for c in range(NUM_CHARS):
                    assert mat[p][c] >= 0
                    assert mat[p][c] <= 50

    @pytest.mark.parametrize("game_id", ["coup_2p", "coup_3p", "coup_4p"])
    def test_serialize_keys_stable(self, game_id):
        gs = dinoboard_engine.GameSession(game_id, seed=7, model_path="")
        snap = gs.get_belief_snapshot()
        # Serialize keys must be stable (used for tracker-equality assertion
        # in test_sim_tracker_descent_maintained / perspective invariance).
        assert set(snap.keys()) == {
            "pre_claim_counts",
            "post_claim_counts",
            "pre_challenge_initiated",
            "post_challenge_initiated",
            "last_reshuffle_kind",
            "last_revealed_role",
            "exchange_complete_counts",
            "pending_claimer",
            "pending_claim_role",
            "pending_challenged",
        }
