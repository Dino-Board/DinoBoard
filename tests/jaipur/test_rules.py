"""Jaipur rule + tracker correctness tests.

One turn is one BGA request: take one good, take all camels, sell k cards of
a type (partial sales allowed), or one trade. A trade is a factored action:
take atoms in ascending type, then give atoms in ascending type (camels
last); it resolves on the give atom that pays the last card.

Covered here: the legal trade atoms against a brute-force enumeration of all
trades, deck / card conservation, sale semantics, termination, and the
belief tracker's lower-bound invariant (known <= truth).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import dinoboard_engine as engine  # noqa: E402

GAME = "jaipur"
INIT_TOKENS = {0: 5, 1: 5, 2: 5, 3: 7, 4: 7, 5: 9}
A_TAKE, A_CAMELS, A_SELL, A_TRADE_TAKE, A_TRADE_GIVE, N_ACTIONS = 0, 6, 7, 49, 79, 114


def sell(g, k):
    return A_SELL + g * 7 + (k - 1)


def trade_take(g, k):
    return A_TRADE_TAKE + g * 5 + (k - 1)


def trade_give(x, k):
    return A_TRADE_GIVE + x * 5 + (k - 1)


def is_sell(a):
    return A_SELL <= a < A_TRADE_TAKE


def idle(st):
    return sum(st["trade_take"]) == 0


def _heur_action(s, selector="rule"):
    res = s.get_heuristic_action(selector, 0.0)
    return res.get("action") if isinstance(res, dict) else res


def _multisets(counts, size, lo=0):
    """Count vectors c <= counts with sum(c) == size, non-zero only at types >= lo."""
    if size == 0:
        yield [0] * len(counts)
        return
    for t in range(lo, len(counts)):
        for k in range(1, min(counts[t], size) + 1):
            for rest in _multisets(counts, size - k, t + 1):
                rest[t] = k
                yield rest


def _all_trades(st, p):
    """Every legal trade from the rulebook: take 2..5 market goods, give as
    many hand goods of other types and / or camels, hand <= 7 afterwards."""
    hand, camels, mkt = st[f"hand_count_{p}"], st[f"camel_count_{p}"], st["market_count"]
    camel_cap = max(0, min(camels, 7 - sum(hand)))
    for n in range(2, 6):
        for take in _multisets(mkt[:6], n):
            avail = [0 if take[g] else hand[g] for g in range(6)] + [camel_cap]
            for give in _multisets(avail, n):
                yield tuple(take), tuple(give)


def _reference_trade_atoms(st, p):
    """Next atoms of every trade that extends the state's public prefix."""
    pt, pg = st["trade_take"], st["trade_give"]
    last_take = max([g for g in range(6) if pt[g]], default=-1)
    last_give = max([x for x in range(7) if pg[x]], default=-1)
    atoms = set()
    for take, give in _all_trades(st, p):
        if sum(pg):
            if list(take) != pt or any(give[x] != pg[x] for x in range(last_give + 1)):
                continue
            x = next(x for x in range(last_give + 1, 7) if give[x])
            atoms.add(trade_give(x, give[x]))
            continue
        if any(take[g] != pt[g] for g in range(last_take + 1)):
            continue
        nxt = [g for g in range(last_take + 1, 6) if take[g]]
        if nxt:
            atoms.add(trade_take(nxt[0], take[nxt[0]]))
        elif sum(pt) >= 2:
            x = next(x for x in range(7) if give[x])
            atoms.add(trade_give(x, give[x]))
    return atoms


def _random_games(seeds, trade_bias=0.5, rng_seed=0):
    """Yield (session, state_before, player, action) over random games biased
    towards trade atoms, so trades are exercised often."""
    rng = random.Random(rng_seed)
    for seed in seeds:
        s = engine.GameSession(GAME, model_path="", seed=seed)
        plies = 0
        while not s.is_terminal and plies < 600:
            st = s.get_state_dict()
            legal = s.get_all_legal_actions()
            assert legal, f"seed {seed} ply {plies}: empty legal set"
            trades = [a for a in legal if a >= A_TRADE_TAKE]
            a = rng.choice(trades) if trades and rng.random() < trade_bias else rng.choice(legal)
            p = s.current_player
            s.apply_action(a)
            yield s, st, p, a
            plies += 1
        assert s.is_terminal, f"seed {seed}: did not terminate"


def test_metadata():
    md = engine.game_metadata(GAME)
    assert md["action_space"] == N_ACTIONS
    assert md["feature_dim"] == 242
    layout = md["token_layout"]
    assert layout["num_tokens"] == 11 and not layout["variable_tokens"]
    # every action is scored by a pointer head on the token of its type
    assert all(t >= 0 for t in layout["action_token"])


def test_config_matches_engine():
    import json
    cfg = json.loads((PROJECT_ROOT / "games/jaipur/config/game.json").read_text())
    v = cfg["variants"]["jaipur"]
    md = engine.game_metadata(GAME)
    assert (v["feature_dim"], v["action_space"]) == (md["feature_dim"], md["action_space"])


def test_trade_atoms_match_brute_force_enumeration():
    """The legal trade atoms are exactly the next atoms of the rulebook's
    trades that extend the chosen prefix — no trade missing, none extra, and
    hence no prefix that cannot be completed."""
    checked = in_trade = 0
    rng = random.Random(1)
    for seed in range(60):
        s = engine.GameSession(GAME, model_path="", seed=seed)
        while not s.is_terminal:
            st = s.get_state_dict()
            p = s.current_player
            legal = s.get_all_legal_actions()
            got = {a for a in legal if a >= A_TRADE_TAKE}
            want = _reference_trade_atoms(st, p)
            assert got == want, (
                f"seed {seed} ply {st['plies']}: engine-only {sorted(got - want)}, "
                f"reference-only {sorted(want - got)}")
            if not idle(st):
                assert set(legal) == got, "only trade atoms are legal mid-trade"
                in_trade += 1
            checked += 1
            trades = sorted(got)
            s.apply_action(rng.choice(trades) if trades and rng.random() < 0.5 else rng.choice(legal))
    assert checked > 3000 and in_trade > 500


def test_turn_actions_follow_the_rulebook():
    """Single takes need a free hand slot, camels need camels in the market,
    sells need min 2 for diamond / gold / silver and allow partial sales."""
    for seed in range(40):
        s = engine.GameSession(GAME, model_path="", seed=seed)
        rng = random.Random(seed)
        while not s.is_terminal:
            st = s.get_state_dict()
            p = s.current_player
            legal = set(s.get_all_legal_actions())
            if idle(st):
                hand, mkt = st[f"hand_count_{p}"], st["market_count"]
                for g in range(6):
                    assert ((A_TAKE + g) in legal) == (mkt[g] > 0 and sum(hand) < 7)
                    for k in range(1, 8):
                        ok = k <= hand[g] and k >= (2 if g <= 2 else 1)
                        assert (sell(g, k) in legal) == ok, (seed, g, k, hand)
                assert (A_CAMELS in legal) == (mkt[6] > 0)
            s.apply_action(rng.choice(sorted(legal)))


def test_partial_sell():
    """Selling k < hand cards keeps the rest, pays the top k tokens, and only
    3+ cards earn a bonus token."""
    seen = 0
    for seed in range(300):
        s = engine.GameSession(GAME, model_path="", seed=seed)
        st = s.get_state_dict()
        hand = st["hand_count_0"]
        g = next((g for g in range(3, 6) if hand[g] >= 2), None)
        if g is None:
            continue
        s.apply_action(sell(g, 1))
        after = s.get_state_dict()
        assert after["hand_count_0"][g] == hand[g] - 1
        assert after["discard_count"][g] == 1
        assert after["price_remaining_" + str(g)] == INIT_TOKENS[g] - 1
        assert after["bonus_count_0"] == [0, 0, 0]
        assert after["current_player"] == 1
        seen += 1
    assert seen >= 20


def test_card_conservation_and_trade_effects():
    """Every card is accounted for (55 total), and a resolved trade moves
    exactly the prefix's counts between market, hand and herd."""
    trades = 0
    for s, st, p, a in _random_games(range(40)):
        after = s.get_state_dict()
        on_table = (sum(after["market_count"]) + sum(after["hand_count_0"])
                    + sum(after["hand_count_1"]) + after["camel_count_0"] + after["camel_count_1"])
        assert on_table + after["deck_remaining"] + sum(after["discard_count"]) == 55
        if a >= A_TRADE_GIVE and idle(after):
            x, k = (a - A_TRADE_GIVE) // 5, (a - A_TRADE_GIVE) % 5 + 1
            give = list(st["trade_give"])
            give[x] += k
            take = st["trade_take"]
            assert sum(give) == sum(take) >= 2
            for g in range(6):
                assert after["market_count"][g] == st["market_count"][g] - take[g] + give[g]
                assert after[f"hand_count_{p}"][g] == st[f"hand_count_{p}"][g] + take[g] - give[g]
                assert not (take[g] and give[g]), "took and gave the same type"
            assert after[f"camel_count_{p}"] == st[f"camel_count_{p}"] - give[6]
            assert sum(after[f"hand_count_{p}"]) <= 7
            assert after["deck_remaining"] == st["deck_remaining"], "a trade draws no card"
            trades += 1
        elif a >= A_TRADE_TAKE:
            assert after["current_player"] == p, "trade atoms before the last keep the turn"
            assert after["market_count"] == st["market_count"]
    assert trades > 200


def test_rule_heuristics_play_full_games():
    for selector, seeds in (("rule", range(20)), ("rule_smart", (0, 3))):
        for seed in seeds:
            s = engine.GameSession(GAME, model_path="", seed=seed)
            plies = 0
            while not s.is_terminal and plies < 400:
                a = _heur_action(s, selector)
                assert a in s.get_all_legal_actions(), f"{selector} picked an illegal action"
                s.apply_action(a)
                plies += 1
            assert s.is_terminal, f"{selector} seed {seed}: stalled ({plies} plies)"


def test_round_ends_only_when_market_refill_fails():
    """Rulebook: the round ends when the market cannot be refilled. An empty
    deck with a full market is not terminal; a deck-driven end leaves the
    market short."""
    full_market_empty_deck_seen = 0
    for seed in range(60):
        s = engine.GameSession(GAME, model_path="", seed=seed)
        while not s.is_terminal:
            s.apply_action(_heur_action(s, "rule"))
            st = s.get_state_dict()
            if st["deck_remaining"] == 0 and sum(st["market_count"]) == 5 \
                    and st["piles_exhausted"] < 3:
                assert not s.is_terminal, f"seed {seed}: ended with a full market"
                full_market_empty_deck_seen += 1
        st = s.get_state_dict()
        if st["piles_exhausted"] < 3:
            assert st["deck_remaining"] == 0 and sum(st["market_count"]) < 5, (
                f"seed {seed}: terminal without 3 empty piles or a failed refill")
    assert full_market_empty_deck_seen > 0


def test_goods_tokens_taken_per_player():
    """Per-player goods-token counts (third tie-break) add up to the tokens
    removed from the piles, and only the seller's count grows on a sale."""
    for s, st, p, a in _random_games(range(20), rng_seed=3):
        after = s.get_state_dict()
        taken = [after[f"goods_tokens_taken_{q}"] for q in range(2)]
        removed = sum(INIT_TOKENS[g] - after[f"price_remaining_{g}"] for g in range(6))
        assert sum(taken) == removed
        if is_sell(a):
            g = (a - A_SELL) // 7
            assert taken[1 - p] == st[f"goods_tokens_taken_{1 - p}"]
            assert taken[p] - st[f"goods_tokens_taken_{p}"] == (
                st[f"price_remaining_{g}"] - after[f"price_remaining_{g}"])


def test_bonus_value_split_public_vs_private():
    """rupees (public) must NOT include bonus token value; bonus_total
    (owner-only) carries it. Selling 3+ of a type yields a bonus."""
    for seed in range(500):
        s = engine.GameSession(GAME, model_path="", seed=seed)
        st = s.get_state_dict()
        hc = st["hand_count_0"]
        g = next((g for g in range(6) if hc[g] >= 3), None)
        if g is None:
            continue
        s.apply_action(sell(g, hc[g]))
        st2 = s.get_state_dict()
        assert sum(st2["bonus_count_0"]) == 1
        assert st2["bonus_total_0"] > 0
        tokens = sum(sorted([7, 7, 5, 5, 5] if g == 0 else [6, 6, 5, 5, 5] if g == 1 else
                            [5] * 5 if g == 2 else [5, 3, 3, 2, 2, 1, 1] if g < 5 else
                            [4, 3, 2, 1, 1, 1, 1, 1, 1], reverse=True)[:hc[g]])
        assert st2["rupees_0"] - st["rupees_0"] == tokens, "bonus value leaked into rupees"
        return
    pytest.skip("no seed with a 3+ sellable hand found")


def test_tracker_known_is_a_lower_bound():
    """known[p][g] never exceeds what p holds; it is exact after p takes
    cards it did not have and sells / gives them all."""
    checked = exact = 0
    for s, st, p, a in _random_games(range(30), rng_seed=7):
        snap = s.get_belief_snapshot()
        known = snap["known"]
        after = s.get_state_dict()
        for q in range(2):
            hc = after[f"hand_count_{q}"]
            for g in range(6):
                assert known[q * 6 + g] <= hc[g], (q, g, known, hc)
            exact += sum(known[q * 6:q * 6 + 6]) == sum(hc)
        checked += 1
    assert checked > 1000 and exact > 50
