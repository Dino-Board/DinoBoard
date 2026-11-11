"""Build a 6nimmit initial observation from judge-supplied facts.

Used by `games/6nimmit/judge_adapter/adapter.py`. The 2017 judge
(`games/6nimmit/参考/main.cpp`) gives each player on round 0:
  - their 10-card hand
  - the four row heads
We materialize a GT-side `engine.GameSession("6nimmit_<N>p")`, then
force-overwrite seat 0's hand and the four row-head slots so they match
what the judge dealt. We never need to reconstruct the opponents' hands —
those stay viz=0 to seat 0 (the perspective the AI plays as), and MCTS
sim-entry `randomize_unseen` will sample them on a clone per the
standard ISMCTS protocol (§1.6 of the plan).

The simplification (per 2026-05-17 user decision, plan §1.8.0): we always
configure the AI as **seat 0** on the server. The judge-side player_id is
remapped to 0 here; opponents are server seats 1..N-1. Since the AI is
called BEFORE any commit happens this round (judge fork() pattern means
adapter sees the post-resolution state of the previous round), seat 0 is
always next to commit — no special "I have to play first" handling needed
in the rules engine, that's just framework default `current_player=0`.
"""
from __future__ import annotations

import dinoboard_engine as engine


def build_initial_observation(num_players: int, hand: list[int],
                               row_heads: list[int],
                               seed: int = 0) -> tuple[engine.GameSession, dict]:
    """Construct a 6nimmit GT session matching the judge's deal.

    Args:
        num_players: 2/3/4. Selects the variant id `6nimmit_<N>p`.
        hand: 10 distinct ints in [1, 104] — seat 0's starting hand.
        row_heads: 4 distinct ints in [1, 104] — the four initial row tails,
            in the same order the judge prints them on the third line+.
        seed: GT session RNG seed; affects only the *unused* random fill
            for opponent hands, which the AI doesn't see anyway.

    Returns:
        (gt_session, initial_observation_dict). The dict is what you
        pass as `initial_observation` to POST /ai/sessions.

    Raises:
        ValueError on input shape / range mismatches.
    """
    if num_players not in (2, 3, 4):
        raise ValueError(f"num_players must be 2/3/4 (got {num_players})")
    if len(hand) != 10:
        raise ValueError(f"hand must be 10 cards (got {len(hand)})")
    if len(row_heads) != 4:
        raise ValueError(f"row_heads must be 4 cards (got {len(row_heads)})")

    used = set()
    for c in list(hand) + list(row_heads):
        if not (1 <= c <= 104):
            raise ValueError(f"card {c} out of range [1, 104]")
        if c in used:
            raise ValueError(f"duplicate card {c} in hand+row_heads")
        used.add(c)

    game_id = f"6nimmit_{num_players}p"
    gs = engine.GameSession(game_id, seed=seed)

    # Clear seat 0's hand (the C++ reset_with_seed dealt random cards)
    # and write the judge-supplied 10. Opponent hands stay as whatever
    # the C++ deal happened to give — irrelevant since they're viz=0 to
    # seat 0 and sim-entry resamples them.
    for c in range(104):
        gs._force_write_field("hand", [0, c], 0)
    for c in hand:
        gs._force_write_field("hand", [0, c - 1], 1)

    # Clear the table (also dealt by reset_with_seed) and plant the
    # judge-supplied row heads. Each row holds card_number at slot 0,
    # zeros elsewhere — initial state.
    for r in range(4):
        for s in range(5):
            gs._force_write_field("table", [r, s], 0)
        gs._force_write_field("table", [r, 0], int(row_heads[r]))

    # Important caveat: opponent hands now CAN overlap with the judge's
    # row_heads / our hand if the C++ deal accidentally drew the same
    # card for them. That breaks the "104 distinct cards in play"
    # invariant on truth, but it's only relevant inside this GT session
    # for two operations: (a) `extract_initial_observation(0)` — which
    # walks the schema viz=1 slots, all of which we just wrote correctly;
    # (b) the GT session is held by the adapter only to ship the snapshot,
    # then discarded. We never use this GT to advance anything. The
    # invariant ON THE AI SIDE — that what it sees matches a legal initial
    # configuration — is preserved.
    obs = gs.extract_initial_observation(0)
    return gs, obs
