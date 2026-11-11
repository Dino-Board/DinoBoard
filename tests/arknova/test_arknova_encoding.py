"""Ark Nova token encoder: every token decoded and compared with independent sources, the layout the
network is built from, hidden information, and the documentation of the layout."""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

import dinoboard_engine as engine
from conftest import get_test_model

sys.path.insert(0, str(Path(__file__).parent))
import arknova_checks as checks  # noqa: E402
from arknova_reference import (GAME, ROOT, atom_names, card_index, cards, decode, known_to, layout,  # noqa: E402
                               own_slots)

# Features that need a rare situation (Hypnosis, Snapping 2, Dominance, the Engineer, an extra
# action offered as a decision, a donation followed by another decision, ...). The games below do
# not reach them with a non-zero value; they are compared whenever they occur, and the larger runs
# of the same checker (see docs/games/arknova_api.md, Verification) cover them; Dominance has
# test_dominance_offers_only_the_primates_project.
RARE = {("decision", "turn_borrowed"), ("decision", "extra_from_opponent"), ("decision", "can_replenish"),
        ("decision", "replenish_between"), ("decision", "only_one"), ("decision", "turn_xtoken"),
        ("decision", "engineer_used"), ("decision", "chosen_map_token"), ("decision", "optional"),
        ("decision", "extra_actions"), ("decision", "donated"), ("global", "final_turns"),
        ("project", "public_list"), ("project", "opp_maybe_tucked"), ("enclosure", "target")}


def test_every_token_decodes_to_the_state():
    """Every feature of every token, in random and greedy games, against the UI serialization of the
    same state, the truth's private slots, the card table and map parsed from the sources, and (for
    the "can I do this now" flags) the legal actions. Also: the selfplay sample equals the encoding of
    the ground-truth state bit for bit, and the other player's view of the same state passes the
    same checks and mirrors this one."""
    ck = checks.Checker()
    for seed in (3, 8, 15, 32):
        checks.check_episode(ck, seed, "random", stride=5, perspective_every=6)
    for seed in (5, 6):
        checks.check_episode(ck, seed, "greedy", stride=3, perspective_every=8)
    assert ck.decisions > 600
    assert not ck.unchecked(), f"features never compared: {ck.unchecked()}"
    missing = set(ck.never_nonzero()) - RARE
    assert not missing, f"features never compared with a non-zero value: {sorted(missing)}"


def test_dominance_offers_only_the_primates_project():
    """decision.only_one. A Dominance animal played while base project 108 (primates) is still in the
    base project pile lets its owner take only that project (abilities: kDOMINANCE). Greedy game 47
    reaches it: there the flag is set, the legal atoms are that project's list atom and skip, and the
    whole encoding passes the full check; in every other decision of the game the flag is 0."""
    table, idx = cards(), card_index()
    dominance = next(c["id_str"] for c in table if "DOMINANCE" in c["abilities"])
    ck = checks.Checker()
    episode = checks.Episode(47, "greedy")
    found = 0
    for n, sample, me, state, core, legal in episode.steps():
        toks = decode(sample["features"])
        top = core["stack"][-1]
        if top["kind"] != "ab_project" or "only" not in top["data"]:
            assert toks[1]["only_one"] == 0, f"decision {n}"
            continue
        assert toks[1]["only_one"] == 1.0
        assert top["data"]["only"] == ["108"] and dominance in core["players"][me]["zoo_cards"]
        pile = core["base_project_deck"]
        assert len(pile) > 1, "the pile holds other projects that Dominance must not offer"
        assert set(legal) == {f"list:{pile.index('108')}", "skip"}
        public = [t for t in toks if t.type == "project" and t["public_list"] == 1.0]
        assert sorted(t.card() for t in public) == sorted(idx[c] for c in pile)
        checks.check_state(ck, toks, me, core, episode.own, state, episode.tracker, legal)
        found += 1
    assert found >= 1


def test_discard_and_unseen_pools_account_for_every_card():
    """global.discard_* / unseen_*. Over a greedy game: when cards are discarded the discard columns
    change, and the regular cards counted by the discard and unseen type columns plus those in the
    zones visible to the perspective make up every regular card of the game exactly once. The
    unseen cards are exactly as many as the deck plus the opponent's hidden regular cards, and
    every one of those is unseen."""
    table, idx = cards(), card_index()
    types = ("animal", "sponsor", "project")
    totals = np.array([sum(c["type"] == t for c in table) for t in types])
    regular = {i for i, c in enumerate(table) if c["type"] != "endgame"}
    episode = checks.Episode(6, "greedy")
    previous, grew = None, 0
    for n, sample, me, state, core, legal in episode.steps():
        own, opp, tracker = episode.own, 1 - me, episode.tracker
        g = decode(sample["features"])[0]
        discard = [idx[c] for c in core["discard"]]
        if previous is not None and len(discard) > len(previous[0]):
            assert not np.array_equal(g["discard_type"], previous[1]) or not np.array_equal(
                g["discard_icons"], previous[2]), f"decision {n}: cards discarded, discard columns unchanged"
            grew += 1
        previous = (discard, g["discard_type"], g["discard_icons"])
        # Every regular card, once: the discard pile, the zones I see, and the unseen pool.
        visible = [idx[c] for c in [c for c in core["display"] if c is not None] + core["base_projects"] +
                   core["base_project_deck"] + core["projects"] + core["players"][0]["zoo_cards"] +
                   core["players"][1]["zoo_cards"]]
        visible += [idx[c] for f in core["stack"] if f["kind"] == "ab_hunter" for c in f["data"]["cards"]]
        visible += list(own[me]["hand"]) + own[me]["tucked"] + [c for fr in own[me]["private"] for c in fr]
        visible += sorted(tracker.shown(opp) | tracker.maybe[opp])
        visible = [c for c in visible if c in regular]
        unseen = sorted(regular - set(discard) - set(visible))
        counts = collections.Counter(discard + visible + unseen)
        assert set(counts) == regular and set(counts.values()) == {1}, f"decision {n}: a card in two places"
        by_type = lambda cs: np.array([sum(table[c]["type"] == t for c in cs) for t in types])  # noqa: E731
        from_encoder = np.rint((g["discard_type"] + g["unseen_type"]) * totals).astype(int)
        assert (from_encoder + by_type(visible) == totals).all(), f"decision {n}"
        assert (np.rint(g["unseen_type"] * totals) == by_type(unseen)).all()
        # The truth: the deck and the opponent's hidden regular cards are the unseen pool.
        hidden = ((own[opp]["hand"] - tracker.shown(opp)) | set(own[opp]["tucked"]) |
                  {c for fr in own[opp]["private"] for c in fr}) - tracker.maybe[opp]
        hidden &= regular
        assert hidden <= set(unseen) and len(unseen) == len(core["deck"]) + len(hidden), f"decision {n}"
    assert grew >= 5


def test_pools_do_not_read_hidden_cards():
    """The global token (pools included) and the whole encoding are the same for two worlds that
    differ only in the opponent's hidden hand, tucked, final and private slots: the replayed truth,
    and a copy where each of those slots holds another card I have not seen."""
    for seed, p in ((2, 0), (9, 1)):
        truth = engine.GameSession(GAME, seed=seed)
        rng, noise = random.Random(seed), random.Random(seed + 7)
        actions, decisions, compared = [], 0, 0
        while not truth.is_terminal and compared < 12:
            decisions += truth.current_player == p
            if truth.current_player == p and decisions % 6 == 0:
                world = engine.GameSession(GAME, seed=seed)
                for a in actions:
                    world.apply_action(a)
                g = decode(truth._encode(p)["features"])[0]
                core = json.loads(truth.get_state_dict()["core_json"])
                own = {q: own_slots(truth, q) for q in (0, 1)}
                shown = {card_index()[c] for c in core["players"][1 - p]["public_hand"]}
                known = known_to(p, core, own, shown)
                pool = [c for c in range(len(cards())) if c not in known]
                hand = own[1 - p]["hand"] - shown
                for c in hand:
                    world._force_write_field("hand", [1 - p, c], 0)
                for c in noise.sample(sorted(set(pool) - hand), len(hand)):
                    world._force_write_field("hand", [1 - p, c], 1)
                for field, n in (("final_cards", 6), ("tucked", 24), ("final_known", 4)):
                    for i in range(n):
                        world._force_write_field(field, [1 - p, i], noise.choice(pool))
                for k in range(4):
                    for j in range(6):
                        world._force_write_field("private_cards", [1 - p, k, j], noise.choice(pool))
                assert own_slots(world, 1 - p) != own[1 - p] or not hand
                assert np.array_equal(truth._encode(p)["features"], world._encode(p)["features"]), (
                    f"seed {seed} after {len(actions)} moves")
                assert g["unseen_type"].any()
                compared += 1
            a = rng.choice(truth.get_legal_actions())
            truth.apply_action(a)
            actions.append(a)
        assert compared >= 6


def test_layout_is_what_the_network_is_built_from():
    meta = engine.game_metadata(GAME)
    tl, lay = meta["token_layout"], layout()
    assert tl["variable_tokens"] and tl["token_positions"] and meta["feature_dim"] == 0
    assert tl["token_dim"] == lay.token_dim and lay.header == 7
    assert tl["num_token_types"] == len(lay.type_names) == 8
    assert tl["num_pointer_heads"] == lay.raw["num_heads"] == 6
    widest = max(o + w for feats in lay.features.values() for o, w in feats.values())
    assert lay.token_dim == lay.header + widest
    for name, feats in lay.features.items():  # contiguous, in order, no overlap
        at = 0
        for feature, (offset, width) in feats.items():
            assert offset == at and width > 0, f"{name}.{feature}"
            at += width
    assert tl["num_anchor_ids"] == sum(n for _, n in lay.anchors.values())
    assert tl["num_entity_ids"] == 1 + sum(n for _, n in lay.entities.values())
    assert set(tl["action_token"]) == {-1} and set(tl["action_head"]) == {-1}
    assert len(tl["anchor_action"]) == tl["num_anchor_ids"] * tl["num_pointer_heads"]
    # Each pointer atom class sits on one anchor range and one head.
    names, heads = atom_names(), tl["num_pointer_heads"]
    where = {}
    for i, a in enumerate(tl["anchor_action"]):
        if a >= 0:
            where.setdefault(names[a].split(":")[0], set()).add((lay.anchor_name(i // heads + 1)[0], i % heads))
    cell = {("cell", h) for h in range(6)}
    assert where == {
        "hand": {("hand", 0)}, "hand_discard": {("hand", 1)}, "zoo": {("zoo", 0)}, "zoo_give": {("zoo", 1)},
        "display": {("display", 0)}, "display_discard": {("display", 1)}, "snap": {("display", 2)},
        "project": {("project", 0)}, "list": {("public_list", 0)}, "private": {("private_list", 0)},
        "final_discard": {("final", 1)}, "enclosure": {("enclosure", 0)}, "cell": cell}
    assert meta["action_space"] == json.loads((ROOT / "games/arknova/config/game.json").read_text())[
        "variants"]["arknova"]["action_space"]


def test_search_does_not_depend_on_hidden_slots():
    """Three sessions with the same seed (so the search and its world sampling use the same random
    numbers) whose states differ only in the slots hidden from the searching player: the truth's own
    view (placeholders), an observation-fed session still holding the initial deal's true values
    (like selfplay's per-seat states), and one whose hidden slots are overwritten with garbage
    before every search. The root visit counts and the searched values must be identical."""
    model = get_test_model(GAME)
    table = cards()
    for seed, p in ((2, 0), (4, 1)):
        truth = engine.GameSession(GAME, seed=seed, model_path=model)
        stale = engine.GameSession(GAME, seed=seed, model_path=model)
        garbage = engine.GameSession(GAME, seed=seed, model_path=model)
        first = truth.extract_initial_observation(p)
        stale.apply_initial_observation(p, first)
        garbage.apply_initial_observation(p, first)
        rng, noise = random.Random(seed), random.Random(seed + 100)
        searched = decisions = 0
        while not truth.is_terminal and searched < 25:
            legal = truth.get_legal_actions()
            if truth.current_player == p:
                decisions += 1
                assert sorted(stale.get_legal_actions()) == sorted(legal)
                if decisions % 9 == 0 and len(legal) > 1:
                    shown = {card_index()[c] for c in
                             json.loads(garbage.get_state_dict()["core_json"])["players"][1 - p]["public_hand"]}
                    for c in range(len(table)):
                        if c not in shown:
                            garbage._force_write_field("hand", [1 - p, c], noise.randint(0, 1))
                    for field, n in (("final_cards", 6), ("tucked", 24), ("final_known", 4)):
                        for i in range(n):
                            garbage._force_write_field(field, [1 - p, i], noise.randint(-1, len(table) - 1))
                    for k in range(4):
                        for j in range(6):
                            garbage._force_write_field("private_cards", [1 - p, k, j], noise.randint(-1, len(table) - 1))
                    kw = dict(simulations=24, temperature=0.0, opponent_selection="prior", world_cap=8)
                    stats = [g.get_ai_action(**kw)["stats"] for g in (truth, stale, garbage)]
                    # Visit counts, and the network's values averaged over the searched nodes as exact
                    # floats: an encoding that read a hidden slot would move them.
                    found = [(s["root_actions"], s["root_action_visits"], s["root_values"], s["action_values"])
                             for s in stats]
                    assert found[0] == found[1] == found[2], f"seed {seed} decision {decisions}"
                    searched += 1
            trace = truth.apply_action_with_trace(rng.choice(legal), p)
            for g in (stale, garbage):
                g.apply_observation(trace["action"], events=trace["events"], public_snapshot=trace["public_snapshot"])
        assert searched >= 15


DOC = ROOT / "docs" / "games" / "arknova_api.md"


def _doc_tables(text):
    """`### name ...` heading -> the rows of the first table under it, as lists of cells."""
    tables = {}
    for section in re.split(r"^### ", text, flags=re.M)[1:]:
        title = section.split("\n", 1)[0]
        lines = section.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
        if start is None:
            continue
        end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
        rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines[start:end]]
        tables[title.split()[0].strip("`")] = rows[2:]  # without the header and the |---| row
    return tables


def test_docs_list_every_feature_and_atom():
    """docs/games/arknova_api.md documents the layout the engine reports: per token type every
    feature with its offset and width, the anchor and entity ranges, and every atom range."""
    lay, tables = layout(), _doc_tables(DOC.read_text())
    for name, feats in lay.features.items():
        documented = [(row[2].strip("`"), int(row[0].split("-")[0]), int(row[1])) for row in tables[name]]
        assert documented == [(f, o, w) for f, (o, w) in feats.items()], f"token type {name}"
        for row in tables[name]:  # "a-b" ranges are written out for wide features
            first, _, last = row[0].partition("-")
            assert int(last or first) == int(first) + int(row[1]) - 1
    assert [(r[0].strip("`"), int(r[1]), int(r[2])) for r in tables["anchors"]] == [
        (n, base, count) for n, (base, count) in lay.anchors.items()]
    assert [(r[0].strip("`"), int(r[1]), int(r[2])) for r in tables["entities"]] == [
        (n, base, count) for n, (base, count) in lay.entities.items()]
    names = atom_names()
    at = 0
    for row in tables["atoms"]:
        first, count, pattern = int(row[0]), int(row[1]), row[2].strip("`")
        assert first == at, f"atom table: {pattern} should start at {at}"
        for a in range(first, first + count):
            assert re.fullmatch(pattern, names[a]), f"atom {a} is {names[a]}, documented as {pattern}"
        at += count
    assert at == len(names) == engine.game_metadata(GAME)["action_space"]
