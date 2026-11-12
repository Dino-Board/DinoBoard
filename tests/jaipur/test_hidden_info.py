"""Jaipur hidden-information regression tests.

Background (audit J1-J5): the draw pile used to be a pre-shuffled vector that
was not part of the visibility schema. Masking, snapshots and
`randomize_unseen` therefore never replaced it, and from 27 cards left every
MCTS world drew the TRUE next cards. The fix removes the stored order
altogether: the pile is a hidden per-type count (`deck_count`), a draw is a
weighted sample with the caller's rng, and `randomize_unseen` rebuilds every
hidden slot from public counts. These tests pin the behaviour, not the
implementation:

  - simulated refills do not follow the truth (GameSession and external-API
    sessions),
  - every sampled world conserves cards / bonus tokens and leaves all public
    fields untouched,
  - hidden bytes of the session state cannot influence the search,
  - the rule heuristic prices bonus tokens from public knowledge only,
  - a mid-game bootstrap restores the tracker memory.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

GAME = "jaipur"
TOTAL = [6, 6, 6, 8, 8, 10]            # goods cards per type (diamond..leather)
CAMELS = 11
# Goods tokens, top (first taken) first.
TOKENS = [[7, 7, 5, 5, 5], [6, 6, 5, 5, 5], [5, 5, 5, 5, 5],
          [5, 3, 3, 2, 2, 1, 1], [5, 3, 3, 2, 2, 1, 1], [4, 3, 2, 1, 1, 1, 1, 1, 1]]
BONUS_VALUE = [[1, 2, 3], [4, 5, 6], [8, 9, 10]]
BONUS_INIT = [[2, 3, 2], [2, 2, 2], [2, 1, 2]]
A_TAKE, A_CAMELS, A_SELL, A_TRADE = 0, 6, 7, 49


def _sell(g, k):
    return A_SELL + g * 7 + (k - 1)


def _is_sell(a):
    return A_SELL <= a < A_TRADE


def _idle(sd):
    return sum(sd["trade_take"]) == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _play_to_take_position(seed, rng, model, deck_pred):
    """Random play until an idle-phase position with a legal single take and
    `deck_pred(deck_remaining)`. Returns (session, actions) or None."""
    gt = engine.GameSession(GAME, seed=seed, model_path=model)
    actions = []
    for _ in range(400):
        if gt.is_terminal:
            return None
        sd = gt.get_state_dict()
        legal = gt.get_legal_actions()
        takes = [a for a in legal if A_TAKE <= a < A_TAKE + 6]
        if _idle(sd) and takes and deck_pred(sd["deck_remaining"]):
            return gt, actions
        a = rng.choice(takes) if (takes and rng.random() < 0.6) else rng.choice(legal)
        gt.apply_action(a)
        actions.append(a)
    return None


def _true_next_card(seed, actions):
    """The card the real game draws next: replay on a twin session and take
    one good there. Returns a market index (0..5 goods, 6 camel)."""
    twin = engine.GameSession(GAME, seed=seed)
    for a in actions:
        twin.apply_action(a)
    sd = twin.get_state_dict()
    take = next(a for a in twin.get_legal_actions() if A_TAKE <= a < A_TAKE + 6)
    before = list(sd["market_count"])
    twin.apply_action(take)
    after = list(twin.get_state_dict()["market_count"])
    before[take - A_TAKE] -= 1
    drawn = [i for i in range(7) if after[i] > before[i]]
    assert len(drawn) == 1
    return drawn[0]


def _revealed_refills(searcher, me, market, sims):
    """Cards the simulations drew to refill the market after the root player
    took one good. Inferred from the per-step sim-tracker traces: when the
    opponent's reply in a sim takes a good type that was NOT in the market
    after the root take, that type can only be the refill card."""
    opp = 1 - me
    tr = searcher._debug_run_mcts_with_sim_traces(sims)
    root_net = list(tr["root_tracker"]["known"])
    by_sim = {}
    for e in tr["sim_traces"]:
        by_sim.setdefault(e["sim"], {})[e["step"]] = e["tracker"]
    revealed = Counter()
    for steps in by_sim.values():
        if 0 not in steps or 1 not in steps:
            continue
        n0 = list(steps[0]["known"])
        n1 = list(steps[1]["known"])
        d0 = [n0[i] - root_net[i] for i in range(12)]
        if sum(abs(x) for x in d0) != 1 or 1 not in d0:
            continue
        i0 = d0.index(1)
        if i0 // 6 != me:
            continue
        d1 = [n1[i] - n0[i] for i in range(12)]
        if sum(abs(x) for x in d1) != 1 or 1 not in d1:
            continue
        i1 = d1.index(1)
        if i1 // 6 != opp:
            continue
        g0, g1 = i0 % 6, i1 % 6
        if market[g1] - (1 if g1 == g0 else 0) == 0:
            revealed[g1] += 1
    return revealed


def _leak_stats(api: bool, positions=10, sims=2500):
    """(#positions, #positions with >1 distinct refill card, reveal events,
    events equal to the true next card) over late positions."""
    model = get_test_model(GAME)
    rng = random.Random(2024)
    n_pos = multi = events = matches = 0
    for seed in range(1, 400):
        if n_pos >= positions:
            break
        r = _play_to_take_position(seed, rng, model, lambda d: 3 <= d <= 27)
        if r is None:
            continue
        gt, actions = r
        me = gt.current_player
        market = list(gt.get_state_dict()["market_count"])
        searcher = gt
        if api:
            # External-API path: a session with a DIFFERENT seed that only
            # ever saw the current position from `me`'s side.
            searcher = engine.GameSession(GAME, seed=seed + 777_777, model_path=model)
            searcher.apply_initial_observation(me, gt.extract_initial_observation(me))
        rev = _revealed_refills(searcher, me, market, sims)
        if sum(rev.values()) < 30:
            continue
        truth = _true_next_card(seed, actions)
        n_pos += 1
        multi += len(rev) > 1
        events += sum(rev.values())
        matches += rev[truth]
    return n_pos, multi, events, matches


# ---------------------------------------------------------------------------
# J1 — simulated draws must not follow the true deck
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("api", [False, True], ids=["game_session", "api_session"])
def test_simulated_refills_do_not_follow_the_true_deck(api):
    """With <= 27 cards left the old engine drew the true next card in every
    simulated world (100% of refills, a single card per position). A correct
    determinization draws from the unseen pool: several different cards per
    position, and the true card only at its chance rate (~15-25%)."""
    n_pos, multi, events, matches = _leak_stats(api)
    assert n_pos >= 8, "not enough late positions with observable refills"
    assert multi >= n_pos // 2, (
        f"only {multi}/{n_pos} positions saw more than one distinct refill "
        f"card across simulations — sims are replaying a fixed draw order")
    assert matches / events < 0.6, (
        f"{matches}/{events} simulated refills equal the TRUE next card — "
        f"MCTS worlds are drawing from the real deck")


def test_truth_draws_are_reproducible_from_the_seed():
    """Removing the stored order must not make the real game irreproducible:
    same seed + same actions => same public history."""
    rng = random.Random(5)
    a = engine.GameSession(GAME, seed=31)
    b = engine.GameSession(GAME, seed=31)
    c = engine.GameSession(GAME, seed=32)
    differs = a.get_state_dict() != c.get_state_dict()
    for _ in range(300):
        if a.is_terminal:
            break
        act = rng.choice(a.get_legal_actions())
        a.apply_action(act)
        b.apply_action(act)
        assert a.get_state_dict() == b.get_state_dict()
    assert differs, "different seeds produced the same deal"


def test_initial_deal_is_a_legal_jaipur_setup():
    for seed in range(200):
        sd = engine.GameSession(GAME, seed=seed).get_state_dict()
        assert sd["market_count"][6] >= 3 and sum(sd["market_count"]) == 5
        for p in (0, 1):
            assert sum(sd[f"hand_count_{p}"]) + sd[f"camel_count_{p}"] == 5
        assert sd["deck_remaining"] == 40 and sum(sd["deck_count"]) == 40
        for g in range(6):
            assert (sd["market_count"][g] + sd["hand_count_0"][g] + sd["hand_count_1"][g]
                    + sd["deck_count"][g]) == TOTAL[g]
        assert (sd["market_count"][6] + sd["camel_count_0"] + sd["camel_count_1"]
                + sd["deck_count"][6]) == CAMELS


# ---------------------------------------------------------------------------
# J1 / J2 / J3 — every sampled world is consistent with what the observer knows
# ---------------------------------------------------------------------------

PUBLIC_KEYS = (
    ["market_count", "deck_remaining", "piles_exhausted",
     "bonus_remaining_3", "bonus_remaining_4", "bonus_remaining_5",
     "trade_take", "trade_give"]
    + [f"{k}_{p}" for p in (0, 1)
       for k in ("camel_count", "rupees", "bonus_count", "goods_tokens_taken")]
    + [f"price_remaining_{g}" for g in range(6)]
    + [f"current_price_{g}" for g in range(6)]
)
OBSERVER_PRIVATE_KEYS = ["hand_count_0", "bonus_total_0"]


def _tracker_states():
    """`test_belief_tracker` plays random plies (tracker = seat 0), then
    calls randomize_unseen on clones of the truth."""
    for seed in range(1, 121):
        for plies in (10, 35, 70, 110, 150):
            yield f"seed={seed} plies={plies}", engine.test_belief_tracker(GAME, seed, plies, 6)


def test_randomize_unseen_keeps_public_fields_and_hand_size():
    """Every sampled world must leave all public fields and the observer's
    own private fields as they are (J2: the public bonus-pile sizes used to
    grow by the opponent's token count; J3: the public deck size used to
    grow by a "ghost" card), and give the opponent exactly its real hand
    SIZE, which is public (J3)."""
    worlds = oversold = 0
    for ctx, r in _tracker_states():
        o = r["original_state"]
        taken = sum(len(TOKENS[g]) - o[f"price_remaining_{g}"] for g in range(6))
        in_play = (sum(o["market_count"]) + sum(o["hand_count_0"]) + sum(o["hand_count_1"])
                   + o["camel_count_0"] + o["camel_count_1"] + o["deck_remaining"])
        oversold += (55 - in_play) > taken
        for t in r["trial_states"]:
            worlds += 1
            for k in PUBLIC_KEYS + OBSERVER_PRIVATE_KEYS:
                assert t[k] == o[k], f"{ctx}: randomize_unseen changed '{k}': {o[k]} -> {t[k]}"
            assert sum(t["hand_count_1"]) == sum(o["hand_count_1"]), (
                f"{ctx}: sampled opponent hand has {sum(t['hand_count_1'])} cards, "
                f"the real (public) hand size is {sum(o['hand_count_1'])}")
    assert worlds > 2000
    assert oversold > 20, "playouts never sold more cards than value tokens were left (J3 not exercised)"


def test_randomize_unseen_conserves_cards_and_bonus_tokens():
    """Every sampled world accounts for every card and every bonus token
    exactly once, and respects what the tracker knows about the opponent."""
    worlds = 0
    for ctx, r in _tracker_states():
        o = r["original_state"]
        known = list(r["belief_snapshot"]["known"])
        for t in r["trial_states"]:
            worlds += 1
            assert t["discard_count"] == o["discard_count"], ctx
            assert sum(t["deck_count"]) == o["deck_remaining"], ctx
            assert min(t["deck_count"]) >= 0 and min(t["hand_count_1"]) >= 0, ctx
            for g in range(6):
                assert (t["market_count"][g] + t["hand_count_0"][g] + t["hand_count_1"][g]
                        + t["deck_count"][g] + t["discard_count"][g]) == TOTAL[g], (
                    f"{ctx}: goods type {g} not conserved in a sampled world")
                # known information about the opponent is respected
                assert t["hand_count_1"][g] >= known[6 + g], ctx
            assert (t["market_count"][6] + t["camel_count_0"] + t["camel_count_1"]
                    + t["deck_count"][6]) == CAMELS, ctx
            opp_total = 0
            for tier in range(3):
                for v in range(3):
                    i = tier * 3 + v
                    assert t["bonus_taken_0"][i] == o["bonus_taken_0"][i], ctx
                    assert min(t["bonus_taken_1"][i], t["bonus_pile"][i]) >= 0, ctx
                    assert (t["bonus_taken_0"][i] + t["bonus_taken_1"][i]
                            + t["bonus_pile"][i]) == BONUS_INIT[tier][v], (
                        f"{ctx}: bonus tokens of tier {tier} not conserved")
                    opp_total += t["bonus_taken_1"][i] * BONUS_VALUE[tier][v]
                assert sum(t["bonus_taken_1"][tier * 3:tier * 3 + 3]) == o["bonus_count_1"][tier], ctx
                assert sum(t["bonus_pile"][tier * 3:tier * 3 + 3]) == o[f"bonus_remaining_{tier + 3}"], ctx
            assert t["bonus_total_1"] == opp_total, ctx
    assert worlds > 2000


def test_sampled_worlds_vary_and_are_not_the_truth():
    """The hidden draw-pile composition is resampled per world; it must not
    be a copy of the truth."""
    eligible = varied = equal_truth = worlds = 0
    for seed, plies in [(s, p) for s in range(1, 41) for p in (6, 14, 30)]:
        r = engine.test_belief_tracker(GAME, seed, plies, 12)
        o = r["original_state"]
        # Only positions where the observer does NOT know the opponent's
        # whole hand (otherwise the pile composition is fully determined).
        unknown = sum(o["hand_count_1"]) - sum(r["belief_snapshot"]["known"][6:12])
        if o["is_terminal"] or unknown < 2:
            continue
        eligible += 1
        decks = {tuple(t["deck_count"]) for t in r["trial_states"]}
        varied += len(decks) > 1
        worlds += len(r["trial_states"])
        equal_truth += sum(tuple(t["deck_count"]) == tuple(o["deck_count"]) for t in r["trial_states"])
    assert eligible >= 40
    assert varied >= 0.8 * eligible, f"{varied}/{eligible} positions resampled the draw pile"
    assert equal_truth < 0.5 * worlds, f"{equal_truth}/{worlds} sampled piles equal the true pile"


# ---------------------------------------------------------------------------
# Hidden bytes of the session state must not reach the search
# ---------------------------------------------------------------------------

def test_hidden_slots_of_the_session_do_not_influence_search():
    """Two external sessions with the same seed, bootstrapped from the same
    position; in one of them every slot hidden from the observer is
    overwritten with garbage. Search results must be identical — hash and
    encoder never read those slots and randomize_unseen replaces all of them."""
    model = get_test_model(GAME)
    rng = random.Random(77)
    checked = 0
    for seed in range(1, 60):
        if checked >= 4:
            break
        r = _play_to_take_position(seed, rng, "", lambda d: 5 <= d <= 30)
        if r is None:
            continue
        gt, _ = r
        me = gt.current_player
        opp = 1 - me
        obs = gt.extract_initial_observation(me)
        clean = engine.GameSession(GAME, seed=4242, model_path=model)
        dirty = engine.GameSession(GAME, seed=4242, model_path=model)
        clean.apply_initial_observation(me, obs)
        dirty.apply_initial_observation(me, obs)
        for g in range(6):
            dirty._force_write_field("hand_count", [opp, g], (g * 3 + 1) % 5)
        dirty._force_write_field("bonus_total", [opp], 37)
        for i in range(7):
            dirty._force_write_field("deck_count", [i], (i * 5 + 2) % 9)
        for t in range(3):
            for v in range(3):
                dirty._force_write_field("bonus_pile", [t, v], (t + 2 * v) % 3)
                dirty._force_write_field("bonus_taken", [opp, t, v], (2 * t + v) % 3)
        assert (dirty.state_hash_for_perspective(me) == clean.state_hash_for_perspective(me))
        ra = clean.get_ai_action(simulations=300, temperature=0.0)
        rb = dirty.get_ai_action(simulations=300, temperature=0.0)
        assert ra["stats"]["root_actions"] == rb["stats"]["root_actions"]
        assert ra["stats"]["root_action_visits"] == rb["stats"]["root_action_visits"], (
            f"seed={seed}: hidden bytes of the session state changed the search result")
        checked += 1
    assert checked == 4


def test_draw_from_an_unsampled_observer_state_is_refused():
    """Rules must never draw from hidden slots that were not produced by the
    truth or by randomize_unseen (an observer session's hidden slots are
    stale): they raise instead of inventing a card."""
    rng = random.Random(3)
    r = None
    for seed in range(1, 40):
        r = _play_to_take_position(seed, rng, "", lambda d: 5 <= d <= 30)
        if r is not None:
            break
    gt, _ = r
    me = gt.current_player
    observer = engine.GameSession(GAME, seed=999)
    observer.apply_initial_observation(me, gt.extract_initial_observation(me))
    take = next(a for a in observer.get_legal_actions() if A_TAKE <= a < A_TAKE + 6)
    with pytest.raises(RuntimeError, match="randomize_unseen was not applied"):
        observer.apply_action(take)


def test_every_runner_entry_point_plays_full_games():
    """The rules now refuse to draw from a state whose hidden slots were not
    sampled. Every search entry point must therefore hand a sampled world to
    the rules: selfplay (per-seat states), heuristic-guided selfplay, arena,
    eval-vs-heuristic (search on the truth state with per-seat trackers),
    the batched concurrent paths and the MCTS-backed heuristic."""
    m = get_test_model(GAME)
    ep = engine.run_selfplay_episode(
        game_id=GAME, seed=3, model_path=m, simulations=24, max_game_plies=200)
    assert ep["total_plies"] > 30
    ep = engine.run_selfplay_episode(
        game_id=GAME, seed=4, model_path=m, simulations=24, max_game_plies=200,
        heuristic_guidance_ratio=0.5, heuristic_selector="rule")
    assert ep["total_plies"] > 30
    r = engine.run_arena_match(
        game_id=GAME, seed=5, model_paths=[m, m], simulations_list=[24, 24], temperature=0.0)
    assert r["total_plies"] > 30
    for seat in (0, 1):
        r = engine.run_constrained_eval_vs_heuristic(
            game_id=GAME, seed=6 + seat, model_path=m, simulations=24, model_is_player=seat,
            constrained=False, heuristic_temperature=0.0, heuristic_selector="rule",
            max_game_plies=200)
        assert r["total_plies"] > 30
    out = engine.run_concurrent_eval_vs_heuristic_batch(
        game_id=GAME, model_path=m, heuristic_selector="rule",
        tasks=[dict(seed=8, model_is_player=0), dict(seed=9, model_is_player=1)],
        num_threads=2, simulations=24, max_game_plies=200, batch_size=16,
        max_queue_size=64, intra_op_num_threads=1)
    assert all(r["total_plies"] > 30 for r in out["results"])
    out = engine.run_concurrent_selfplay_step(
        game_id=GAME, seeds=[11, 12], latest_model_path=m, pool_paths=[],
        num_self_threads=2, num_pool_threads=0, simulations=24, max_game_plies=200,
        tail_solve_enabled=False, batch_size=32, max_queue_size=64,
        intra_op_num_threads=1, leaf_batch_size=1)
    assert all(ep["total_plies"] > 30 for ep in out["episodes"])
    ep = engine.run_heuristic_episode(
        game_id=GAME, seed=13, temperature=0.0, max_game_plies=60,
        heuristic_selector="rule_mcts")
    assert ep["total_plies"] > 30


# ---------------------------------------------------------------------------
# J3 — public discard counts
# ---------------------------------------------------------------------------

def test_discard_count_tracks_sold_cards():
    rng = random.Random(8)
    oversold_sales = 0
    for seed in range(25):
        s = engine.GameSession(GAME, seed=seed)
        tally = [0] * 6
        while not s.is_terminal:
            before = s.get_state_dict()
            p = s.current_player
            a = rng.choice(s.get_legal_actions())
            s.apply_action(a)
            st = s.get_state_dict()
            if _is_sell(a):
                g, n = (a - A_SELL) // 7, (a - A_SELL) % 7 + 1
                tally[g] += n
                oversold_sales += n > before[f"price_remaining_{g}"]
            assert st["discard_count"] == tally
            for g in range(6):
                assert (st["market_count"][g] + st["hand_count_0"][g] + st["hand_count_1"][g]
                        + st["deck_count"][g] + st["discard_count"][g]) == TOTAL[g]
    assert oversold_sales > 0


# ---------------------------------------------------------------------------
# J4 — heuristics price bonus tokens from public knowledge only
# ---------------------------------------------------------------------------

def _tokens_gained(g, n, remaining):
    k0 = len(TOKENS[g]) - remaining
    return sum(TOKENS[g][k0:k0 + min(n, remaining)])


def test_rule_heuristic_bonus_estimate_uses_public_pool_only():
    """The rule heuristic scores `sell g` as (value tokens + expected bonus).
    `run_heuristic_episode` publishes softmax(scores / T) as visit counts, so
    the bonus term the heuristic used can be recovered from the ratio between
    a sell and a take action (whose score has no bonus term).

    It must equal the mean of the residual value pool the SELLER knows (all
    tokens of the tier minus its own draws). The old code used the mean of
    the real face-down pile, i.e. it knew which values the opponent drew."""
    T = 6.0
    tol = 0.09
    checked = informative = 0
    for seed in range(1, 61):
        ep = engine.run_heuristic_episode(
            game_id=GAME, seed=seed, temperature=T, max_game_plies=400,
            heuristic_selector="rule")
        gt = engine.GameSession(GAME, seed=seed)
        drawn = [[[], [], []], [[], [], []]]      # drawn[p][tier] = values p drew
        for s in ep["samples"]:
            st = gt.get_state_dict()
            p = s["player"]
            assert p == gt.current_player
            if _idle(st):
                visits = dict(zip(s["policy_action_ids"], s["policy_action_visits"]))
                hand = st[f"hand_count_{p}"]
                takes = [(visits[a], a) for a in visits if A_TAKE <= a < A_TAKE + 6]
                for g in range(6):
                    n = hand[g]
                    sell_all = _sell(g, n)
                    if n < 3 or sell_all not in visits or not takes:
                        continue
                    v_ref, a_ref = max(takes)
                    if visits[sell_all] < 300 or v_ref < 300:
                        continue        # rounding of the visit counts too coarse
                    gr = a_ref - A_TAKE
                    rem = st[f"price_remaining_{gr}"]
                    score_ref = (6.0 - gr) + hand[gr] + (0.15 * TOKENS[gr][len(TOKENS[gr]) - rem] if rem else 0.0)
                    score_sell = score_ref + T * math.log(visits[sell_all] / v_ref)
                    tier = 2 if n >= 5 else (1 if n == 4 else 0)
                    used = score_sell / (0.8 if g >= 3 else 1.0) - _tokens_gained(g, n, st[f"price_remaining_{g}"])
                    full_sum = sum(c * v for c, v in zip(BONUS_INIT[tier], BONUS_VALUE[tier]))
                    full_n = sum(BONUS_INIT[tier])
                    pile = st[f"bonus_remaining_{tier + 3}"]
                    own = drawn[p][tier]
                    if pile == 0 or full_n == len(own):
                        continue
                    public_mean = (full_sum - sum(own)) / (full_n - len(own))
                    truth_mean = (full_sum - sum(own) - sum(drawn[1 - p][tier])) / pile
                    checked += 1
                    informative += abs(truth_mean - public_mean) > 2 * tol
                    assert abs(used - public_mean) < tol, (
                        f"seed={seed} ply={s['ply']}: heuristic priced a tier-{tier} bonus at "
                        f"{used:.3f}; public expectation is {public_mean:.3f}, the hidden "
                        f"pile's true mean is {truth_mean:.3f}")
            a = s["action_id"]
            before = st
            gt.apply_action(a)
            after = gt.get_state_dict()
            if _is_sell(a):
                for tier in range(3):
                    if after[f"bonus_count_{p}"][tier] > before[f"bonus_count_{p}"][tier]:
                        drawn[p][tier].append(after[f"bonus_total_{p}"] - before[f"bonus_total_{p}"])
    assert checked >= 30, f"only {checked} usable (sell >= 3 cards) decisions"
    assert informative >= 8, (
        f"only {informative} decisions where the hidden pile mean differs from the "
        f"public expectation — the test would not notice a leak")


# ---------------------------------------------------------------------------
# J5 — mid-game bootstrap restores the tracker memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [3, 17, 42])
def test_midgame_bootstrap_matches_streamed_session(seed):
    """belief_tracker.h contract: a session bootstrapped from
    `extract_initial_observation` at ply N holds the same belief as one that
    streamed every action from ply 0."""
    seat = 0
    gt = engine.GameSession(GAME, seed=seed)
    streamed = engine.GameSession(GAME, seed=seed + 1000)
    streamed.apply_initial_observation(seat, gt.extract_initial_observation(seat))
    rng = random.Random(seed)
    nontrivial = 0
    for ply in range(120):
        if gt.is_terminal:
            break
        tr = gt.apply_action_with_trace(rng.choice(gt.get_legal_actions()), seat)
        streamed.apply_observation(tr["action"], tr["events"], tr["public_snapshot"])
        if gt.is_terminal or ply % 5:
            continue
        decide = gt.current_player
        attached = engine.GameSession(GAME, seed=seed + 2000 + ply)
        attached.apply_initial_observation(decide, gt.extract_initial_observation(decide))
        snap = attached.get_belief_snapshot()
        assert snap == streamed.get_belief_snapshot(), (
            f"seed={seed} ply={ply}: tracker memory lost by a mid-game bootstrap")
        assert snap == gt.get_belief_snapshot()
        nontrivial += any(snap["known"])
        assert attached.get_legal_actions() == gt.get_legal_actions()
    assert nontrivial >= 5


@pytest.mark.parametrize("seed", [7, 23])
def test_midgame_bootstrap_can_search(seed):
    model = get_test_model(GAME)
    gt = engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    for _ in range(45):
        if gt.is_terminal:
            break
        gt.apply_action(rng.choice(gt.get_legal_actions()))
    assert not gt.is_terminal
    decide = gt.current_player
    attached = engine.GameSession(GAME, seed=seed + 2000, model_path=model)
    attached.apply_initial_observation(decide, gt.extract_initial_observation(decide))
    result = attached.get_ai_action(simulations=200, temperature=0.0)
    assert result["action"] in gt.get_legal_actions()
