"""What the human seat may be sent — the single choke point of the web boundary.

`GameSession.get_state_dict()` is the truth dump (every hidden card), for
tests and tools. Nothing in `platform/` may put it into a response while a
game is running; responses are built from the helpers below instead, so
that everything the browser receives is a function of what the human seat
can see under the game's visibility schema (engine/core/viewer_state.h).
Once the game is over the full state is shown.

tests/framework/test_web_no_hidden_info.py plays every hidden-information
game through the platform and fails if any response changes when values
hidden from the human are changed.
"""
from __future__ import annotations

# Games whose state serializer emits EVERY private slot of a seat (as seen
# by that seat). For these, "does the player to move know more than the
# human" is decided from the two seats' viewer states, which tells an empty
# private slot from an occupied one; all other games use the schema alone
# (any slot the mover sees and the human does not => opaque), which is
# always safe but treats e.g. an empty reserve slot as a secret.
# Opt-in only: a serializer that omits private state must NOT be listed.
_PRIVATE_STATE_FULLY_SERIALIZED = frozenset({"splendor"})


def viewer_state(gs, viewer: int) -> dict:
    """State dict for seat `viewer`: leaves that depend on slots hidden from
    the viewer are None (finished game: the full state)."""
    if gs.is_terminal:
        return gs.get_state_dict()
    return gs.get_public_state_dict(viewer)


def _covers(mine, theirs) -> bool:
    """True iff `theirs` holds no value that `mine` lacks (both viewer states
    of the same game)."""
    if isinstance(theirs, dict):
        if not isinstance(mine, dict):
            return False
        return all(k in mine and _covers(mine[k], v) for k, v in theirs.items())
    if isinstance(theirs, list):
        if not isinstance(mine, list) or len(mine) != len(theirs):
            return False
        return all(_covers(a, b) for a, b in zip(mine, theirs))
    return theirs is None or (type(mine) is type(theirs) and mine == theirs)


def actor_is_opaque(gs, viewer: int) -> bool:
    """True iff the player to move knows something `viewer` does not.

    Its legal actions (the cards it holds), its search statistics and a
    hint computed for it then depend on information hidden from the
    viewer and must not be sent. False for the viewer's own turn and for
    opponents that hold nothing private (perfect-information games, games
    whose only hidden state is a common deck)."""
    if gs.is_terminal:
        return False
    actor = gs.current_player
    if actor < 0 or actor == viewer or not gs.seat_sees_more_than(actor, viewer):
        return False
    if gs.game_id.split("_")[0] in _PRIVATE_STATE_FULLY_SERIALIZED:
        return not _covers(gs.get_public_state_dict(viewer),
                           gs.get_public_state_dict(actor))
    return True


def viewer_legal_actions(gs, viewer: int) -> tuple[list, bool]:
    """(legal actions, hidden?) for the player to move as `viewer` may know
    them. A player's options are a function of what that player sees, so
    they are withheld exactly when that player is opaque to the viewer."""
    if actor_is_opaque(gs, viewer):
        return [], True
    return gs.get_legal_actions(), False
