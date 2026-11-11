"""UCT2 parent count when one edge has several outcomes
(NetMctsConfig::uct2_arrival_visits).

The search has no chance nodes: an edge leads to a different child whenever
the action's result differs between simulations — a visible chance outcome
(splendor card reveal, carcassonne tile draw) or the next player's hand in the
sampled world (loveletter). The exploration term of a node must then use the
number of traversals of the incoming edge that ARRIVED AT THAT NODE, not the
edge's total over all its outcomes (which overstates it by 1/p for an outcome
of probability p).

The engine reports, per descent step, (parent node, edge, child node, parent
count used). The expected counts are recomputed here from the recorded paths
alone: a count used by simulation s is the number of earlier simulations
(completed or still in flight under leaf batching) that made the same step.
"""
import collections

import dinoboard_engine as engine
import pytest

from conftest import get_test_model

SIMS = 300
MULTI_OUTCOME_GAMES = ["splendor", "loveletter", "carcassonne"]
SINGLE_OUTCOME_GAMES = ["tictactoe", "quoridor"]


def _trace(game_id, *, arrival_visits, leaf_batch_size=1):
    gs = engine.GameSession(game_id, seed=42, model_path=get_test_model(game_id))
    out = gs._debug_run_mcts_with_edge_traces(
        simulations=SIMS, uct2_arrival_visits=arrival_visits,
        leaf_batch_size=leaf_batch_size)
    assert out["simulations"] == SIMS
    return out["edges"]


def _recount(edges):
    """Per descent step: (count used by the engine, earlier traversals of the
    edge, earlier traversals of the edge that reached the same child, earlier
    visits of the child, child). Also the set of incoming (parent, edge) pairs
    of every node."""
    by_sim = collections.defaultdict(list)
    for sim, step, parent, edge, child, used in edges:
        by_sim[sim].append((step, parent, edge, child, used))
    edge_seen = collections.Counter()
    arrival_seen = collections.Counter()
    node_seen = collections.Counter()
    rows = []
    for sim in sorted(by_sim):
        at = 0  # every path starts at the root
        for i, (step, parent, edge, child, used) in enumerate(by_sim[sim]):
            assert step == i and parent == at, "descent steps must chain from the root"
            at = child
            rows.append((used, edge_seen[(parent, edge)],
                         arrival_seen[(parent, edge, child)], node_seen[child], child))
        for _step, parent, edge, child, _used in by_sim[sim]:
            edge_seen[(parent, edge)] += 1
            arrival_seen[(parent, edge, child)] += 1
            node_seen[child] += 1
    incoming = collections.defaultdict(set)
    for parent, edge, child in arrival_seen:
        incoming[child].add((parent, edge))
    return rows, incoming


@pytest.mark.parametrize("leaf_batch_size", [1, 8])
@pytest.mark.parametrize("game_id", MULTI_OUTCOME_GAMES)
def test_parent_count_is_arrivals_at_the_child(game_id, leaf_batch_size):
    rows, incoming = _recount(
        _trace(game_id, arrival_visits=True, leaf_batch_size=leaf_batch_size))
    assert all(used == arrivals for used, _edge, arrivals, _node, _child in rows)
    # The position must exercise the difference, or the check above is vacuous.
    inflated = [r for r in rows if r[1] != r[2]]
    assert len(inflated) > 20, f"[{game_id}] no multi-outcome edges searched"
    # No transposition into the child => plain PUCT: the child's own visit count.
    single_parent = [r for r in rows if len(incoming[r[4]]) == 1]
    assert single_parent and all(used == node for used, _e, _a, node, _c in single_parent)


@pytest.mark.parametrize("leaf_batch_size", [1, 8])
@pytest.mark.parametrize("game_id", MULTI_OUTCOME_GAMES)
def test_legacy_switch_counts_every_outcome_of_the_edge(game_id, leaf_batch_size):
    rows, _ = _recount(
        _trace(game_id, arrival_visits=False, leaf_batch_size=leaf_batch_size))
    assert all(used == edge_total for used, edge_total, _a, _n, _c in rows)
    assert any(used != arrivals for used, _e, arrivals, _n, _c in rows)


@pytest.mark.parametrize("game_id", SINGLE_OUTCOME_GAMES)
def test_single_outcome_edges_are_unaffected(game_id):
    """Deterministic, fully public games: every edge has one outcome, so both
    counts coincide and the two settings run the identical search."""
    new = _trace(game_id, arrival_visits=True)
    old = _trace(game_id, arrival_visits=False)
    assert new == old
    rows, _ = _recount(new)
    assert rows and all(used == edge_total == arrivals for used, edge_total, arrivals, _n, _c in rows)


def _arena(game_id, seed, arrival_visits, batched):
    seat = dict(model_path=get_test_model(game_id), simulations=200, temperature=0.0,
                uct2_arrival_visits=arrival_visits)
    if batched:
        out = engine.run_concurrent_arena_batch(
            game_id=game_id, tasks=[dict(seed=seed, seat_configs=[seat, seat])],
            num_threads=1, max_game_plies=60, batch_size=8, max_queue_size=16,
            intra_op_num_threads=1)["results"][0]
    else:
        out = engine.run_arena_match(
            game_id=game_id, seed=seed, max_game_plies=60, seat_configs=[seat, seat])
    return list(out["action_history"])


@pytest.mark.parametrize("batched", [False, True])
def test_arena_seat_config_selects_the_parent_count(batched):
    """Per-player switch (used for A/B): no effect without multi-outcome
    edges, a different search with them."""
    for seed in (1, 2, 3):
        assert _arena("quoridor", seed, True, batched) == _arena("quoridor", seed, False, batched)
    assert any(_arena("splendor", seed, True, batched) != _arena("splendor", seed, False, batched)
               for seed in (1, 2, 3))
