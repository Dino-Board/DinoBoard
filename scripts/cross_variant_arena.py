"""Cross-variant arena: pit a 2p coup model against a 4p coup model in 1v1.

Strategy:
- Run a 2p ground-truth GameSession (no model) to manage the real game.
- For the 2p-model seat: use a GameSession("coup", model_path=2p) for get_ai_action.
- For the 4p-model seat: use a GameSession("coup_4p", model_path=4p) driven via
  apply_observation with padded snapshots (2 ghost dead players, unrevealed).
- Each action from the 2p GT is traced (apply_action_with_trace) to get
  public_snapshot + events, then padded and fed to the 4p session.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dinoboard_engine as engine


def pad_snapshot_2p_to_4p(snap_2p: dict) -> dict:
    """Pad a 2p public_snapshot to look like a 4p snapshot with players 2/3 dead (unrevealed)."""
    snap = {}
    for key, val in snap_2p.items():
        snap[key] = val

    # alive: add dead players 2/3
    if 'alive' in snap:
        snap['alive'] = list(snap['alive']) + [[2], False, [3], False]

    # coins: players 2/3 have 0
    if 'coins' in snap:
        snap['coins'] = list(snap['coins']) + [[2], 0, [3], 0]

    # revealed: players 2/3 both slots revealed (dead)
    if 'revealed' in snap:
        snap['revealed'] = list(snap['revealed']) + [
            [2, 0], True, [2, 1], True,
            [3, 0], True, [3, 1], True,
        ]

    # influence: players 2/3 slots — placeholder values, viz=0 so randomize_unseen fills them
    if 'influence' in snap:
        snap['influence'] = list(snap['influence']) + [
            [2, 0], 0, [2, 1], 0,
            [3, 0], 0, [3, 1], 0,
        ]

    # deck_size: must shrink by 4 to account for the 4 ghost hidden influence slots.
    # Pool balance: remaining = 15 - visible_cards, hidden_slots = opp(2) + ghost(4) + deck.
    # For these to match: deck = original_deck - 4.
    if 'deck_size' in snap:
        orig = snap['deck_size']
        if isinstance(orig, list) and len(orig) == 2:
            snap['deck_size'] = [[], orig[1] - 4]

    # __viz__: need to expand viewer dimension from 2 to 4 for all fields
    if '__viz__' in snap:
        snap['__viz__'] = pad_viz_2p_to_4p(snap['__viz__'])

    # Optional all-viewers viz tensors are shaped for 2 viewers; without them
    # the 4p session syncs its own column only (the pre-existing behaviour).
    snap.pop('__viz_all_viewers__', None)

    return snap


def pad_viz_2p_to_4p(viz_2p: dict) -> dict:
    """Expand viz from 2p to 4p format.

    Viz is a per-field mask array. Structure:
    - Scalar fields (current_player, stage, ...): [1] — unchanged
    - Per-player fields (alive, coins): [1]*N → need to extend from N=2 to N=4
    - influence: [N*2] — owner_only pattern. 2p=[1,1,0,0], 4p=[1,1,0,0,0,0,0,0]
    - revealed: [N*2] — all public. 2p=[1,1,1,1], 4p=[1,1,1,1,1,1,1,1]
    - deck_count: [0]*5 — unchanged (N-independent)
    - exchange_drawn: [0,0] — unchanged (N-independent)
    """
    viz_4p = dict(viz_2p)

    # Per-player public: extend from 2 to 4 entries
    for field in ('alive', 'coins'):
        if field in viz_4p and isinstance(viz_4p[field], list):
            viz_4p[field] = viz_4p[field] + [1, 1]  # ghost players visible

    # revealed: [N*2] all-1s. 2p=4 entries, 4p=8 entries
    if 'revealed' in viz_4p and isinstance(viz_4p['revealed'], list):
        viz_4p['revealed'] = viz_4p['revealed'] + [1, 1, 1, 1]

    # influence: [N*2] owner-only. 2p=[1,1,0,0], 4p needs [1,1,0,0, 0,0,0,0]
    # Ghost players' influence has viz=0 so randomize_unseen samples into them
    if 'influence' in viz_4p and isinstance(viz_4p['influence'], list):
        viz_4p['influence'] = viz_4p['influence'] + [0, 0, 0, 0]

    # deck_known: [N*5] owner-only rows -> ghost rows hidden.
    if 'deck_known' in viz_4p and isinstance(viz_4p['deck_known'], list):
        viz_4p['deck_known'] = viz_4p['deck_known'] + [0] * 10

    # deck_known_taint: [N*N*2] owner-only, row p = flags for slots (q, s).
    # Re-lay 2x2x2 into 4x4x2: a real seat's row keeps its visibility
    # (now over all 4 q's), ghost rows stay hidden.
    if 'deck_known_taint' in viz_4p and isinstance(viz_4p['deck_known_taint'], list):
        old = viz_4p['deck_known_taint']
        new = [0] * (4 * 4 * 2)
        for p in range(2):
            for q in range(4):
                for s in range(2):
                    new[(p * 4 + q) * 2 + s] = old[(p * 2) * 2]
        viz_4p['deck_known_taint'] = new

    return viz_4p


def pad_initial_obs_2p_to_4p(obs_2p: dict, seat_4p: int) -> dict:
    """Pad a 2p initial_observation to 4p format."""
    result = {}
    if 'public_snapshot' in obs_2p:
        result['public_snapshot'] = pad_snapshot_2p_to_4p(obs_2p['public_snapshot'])
    if 'tracker_init' in obs_2p:
        result['tracker_init'] = obs_2p['tracker_init']
    return result


def run_match(model_2p: str, belief_2p: str, model_4p: str, belief_4p: str,
              seed: int, sims: int, seat_4p: int = 1) -> dict:
    """Run one 1v1 match. seat_4p is which seat (0 or 1) the 4p model plays."""
    seat_2p = 1 - seat_4p

    # Ground truth — no model, just rules engine
    gt = engine.GameSession('coup', seed, '', False)

    # 2p model session
    gs2 = engine.GameSession('coup', seed, model_2p, False, belief_model_path=belief_2p)

    # 4p model session — driven via apply_observation
    gs4 = engine.GameSession('coup_4p', seed, model_4p, False, belief_model_path=belief_4p)
    # Apply padded initial observation for the 4p seat
    init_obs_2p = gt.extract_initial_observation(seat_4p)
    init_obs_4p = pad_initial_obs_2p_to_4p(init_obs_2p, seat_4p)
    gs4.apply_initial_observation(seat_4p, init_obs_4p)

    plies = 0
    while not gt.is_terminal and plies < 200:
        cp = gt.current_player
        if cp == seat_2p:
            result = gs2.get_ai_action(simulations=sims, temperature=1.0, top_k=5,
                                       opponent_selection='prior', c_puct=2.0)
            action = result['action']
        else:
            result = gs4.get_ai_action(simulations=sims, temperature=1.0, top_k=5,
                                       opponent_selection='prior', c_puct=2.0)
            action = result['action']

        # Apply to GT and get trace
        trace = gt.apply_action_with_trace(action, seat_4p)
        snap = pad_snapshot_2p_to_4p(trace['public_snapshot'])
        events = trace.get('events', [])

        # Feed to both AI sessions
        gs2.apply_action(action)
        gs4.apply_observation(trace['action'], events, snap)

        plies += 1

    winner = gt.winner
    return {
        'winner': winner,
        'plies': plies,
        'seat_2p_won': winner == seat_2p,
        'seat_4p_won': winner == seat_4p,
        'draw': winner < 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-variant arena: 2p model vs 4p model')
    parser.add_argument('--model2p', required=True, help='Path to 2p policy model (.onnx)')
    parser.add_argument('--model4p', required=True, help='Path to 4p policy model (.onnx)')
    parser.add_argument('--belief2p', default='', help='Path to 2p belief model')
    parser.add_argument('--belief4p', default='', help='Path to 4p belief model')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--sims', type=int, default=500)
    parser.add_argument('--seed-base', type=int, default=10000)
    args = parser.parse_args()

    wins_2p = 0
    wins_4p = 0
    draws = 0

    for i in range(args.games):
        seed = args.seed_base + i
        # Alternate seats
        seat_4p = i % 2
        try:
            result = run_match(args.model2p, args.belief2p, args.model4p, args.belief4p,
                               seed, args.sims, seat_4p)
        except Exception as e:
            print(f'Game {i} (seed={seed}, 4p_seat={seat_4p}) ERROR: {e}')
            continue

        if result['seat_2p_won']:
            wins_2p += 1
        elif result['seat_4p_won']:
            wins_4p += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            total = wins_2p + wins_4p + draws
            print(f'[{i+1}/{args.games}] 2p={wins_2p} 4p={wins_4p} draw={draws} '
                  f'(2p wr={wins_2p/total:.1%}, 4p wr={wins_4p/total:.1%})')

    total = wins_2p + wins_4p + draws
    print(f'\nFinal ({total} games): 2p_model={wins_2p} ({wins_2p/max(1,total):.1%}) '
          f'4p_model={wins_4p} ({wins_4p/max(1,total):.1%}) draws={draws}')


if __name__ == '__main__':
    main()
