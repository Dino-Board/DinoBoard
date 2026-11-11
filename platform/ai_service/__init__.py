"""AI inference API service.

Session-based API: a third-party game server outsources AI decisions to us by
sending ONLY observations (action IDs and, for snapshot-path games, the
public-event trace + public_snapshot) and receives ONLY action IDs back.

The API contract never accepts or returns a game state object. The separation
is structural, not procedural — see CLAUDE.md "AI Pipeline Independence from
Game State" for why the AI physically cannot read truth: the belief tracker
takes no state pointer, public state is rebuilt wholesale from the
`public_snapshot` message each ply, and the session's viz=0 slots are never
touched outside MCTS sim entry — sim_tracker.randomize_unseen on a clone is
the sole place hidden slots get filled, and the session's own viz=0 bytes are
unread by hash / encoder / decision logic.

Three categories of game:

  - **Snapshot-path with tracker** (Love Letter, Splendor, Coup): caller
    must pass `events` + `public_snapshot` on every observe; events feed
    the belief tracker, snapshot wholesale-rebuilds public state. The
    session is created with an `initial_observation` carrying perspective-
    private facts known at game start (own starting hand).
  - **Snapshot-path without tracker** (Azul): caller still passes
    `public_snapshot` (and `events`, even though no consumer reads them
    today) — Azul is fully public, so there's no asymmetric hidden info,
    but the snapshot remains the only operative channel for syncing
    public state on the AI side.
  - **Fully-public no-snapshot** (TicTacToe, Quoridor): only `action_id`
    is needed; `events` / `public_snapshot` are ignored. The session
    advances by replaying the action through `do_action_fast` on the
    AI-side seat state — there are no hidden slots to maintain.
"""
