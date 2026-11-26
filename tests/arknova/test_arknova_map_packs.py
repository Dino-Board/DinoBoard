"""The Zoo Map Packs (maps 9-14) and Tournament 1, against their rulebooks and the Marine Worlds glossary:
random games on each map, every whole option compared before and after with what the rule says."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import dinoboard_engine as engine

sys.path.insert(0, str(Path(__file__).parent))
from arknova_reference import GAME, atom_names, card_index, cards, own_slots, slot_strength, zoo_map  # noqa: E402

PACK_MAPS = ("plan9", "plan10", "plan11", "plan12", "plan13", "plan14", "plant1")
GEO_MONEY, GEO_APPEAL = 4, 2
DRAWING_BOARD = (("money", 3), ("appeal", 2), ("reputation", 1), None)  # the 4th area: Hunter 4


def _core(gs):
    return json.loads(gs.get_state_dict()["core_json"])


def _seeds(key, n):
    out, seed = [], 0
    while len(out) < n:
        if _core(engine.GameSession(GAME, seed=seed))["map"] == key:
            out.append(seed)
        seed += 1
    return out


def _covered(pl):
    return {tuple(c) for b in pl["buildings"] for c in b["cells"]}


def _play(seed, check, max_options=4000):
    """Random game `seed`, one whole option at a time: check(before, key, after, gs_before_own, gs)."""
    gs = engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    names = atom_names()
    n = 0
    while not gs.is_terminal and n < max_options:
        before, own = _core(gs), {p: own_slots(gs, p) for p in (0, 1)}
        key, atoms = rng.choice(gs.external_options())
        chosen = gs.get_state_dict()["prefix"].split()
        for i, a in enumerate(atoms):
            if i < len(chosen) and chosen[i] == names[a]:
                continue  # already in the prefix (applied without asking)
            if gs.is_terminal or a not in gs.get_legal_actions():
                break  # the rest of the option was applied without asking
            gs.apply_action(a)
        check(before, json.loads(key), _core(gs), own, gs)
        n += 1
    assert gs.is_terminal, f"seed {seed}: not over after {max_options} options"
    return n


BGA_MAPS = {*(f"plan{i}a" for i in range(1, 9)), "plan11", "plan12", "plan13", "plan14", "plant1"}


def test_bga_variant_deals_only_board_game_arenas_maps():
    """The maps are in two groups: the 13 Board Game Arena deals (its maps 1-8 are the "a" sides) and
    the others. arknova deals all 25, arknova_bga only BGA's; both share the base game's config."""
    maps = json.loads((Path(__file__).resolve().parents[2] / "games/arknova/arknova_maps.json").read_text())["maps"]
    assert {m["key"] for m in maps if m["bga"]} == BGA_MAPS and len(maps) == 25
    dealt = {g: set() for g in (GAME, "arknova_bga")}
    for g in dealt:
        for seed in range(300):
            dealt[g].add(_core(engine.GameSession(g, seed=seed))["map"])
    assert dealt[GAME] == {m["key"] for m in maps}
    assert dealt["arknova_bga"] == BGA_MAPS
    assert engine.game_metadata("arknova_bga")["action_space"] == engine.game_metadata(GAME)["action_space"]
    from training.cli import find_game_config
    assert find_game_config("arknova_bga")["action_space"] == engine.game_metadata(GAME)["action_space"]


def test_left_edge_abilities_are_bgas():
    """A left-edge token resolves an animal ability only where BGA's map prints it: Snapping on the
    first token of maps 9-14, a card in reputation range or from the deck on Tournament 1's."""
    root = Path(__file__).resolve().parents[2] / "games/arknova"
    pack = json.loads((root / "bga_pack_maps.json").read_text())["maps"]
    maps = {m["key"]: m for m in json.loads((root / "arknova_maps.json").read_text())["maps"]}
    for k, b in pack.items():
        snapping = "Snapping" in b["bonusSpaces"][0]["bonus"]
        assert ("0" in maps[f"plan{k.lower()}"]["token_abilities"]) == snapping, k


def test_scavenging_draws_the_known_discard_cards_first():
    """A board outside the engine (BGA) has already revealed what Scavenging draws from the discard
    pile: discard_next names those cards, and the engine's draw then holds them (Tournament 1's
    Scavenging 3 space here)."""
    idx = card_index()
    checked = 0
    for seed in range(40):
        host = engine.ExternalHost(GAME, seed, "plant1")
        rng = random.Random(seed)
        while not host.is_terminal and checked < 3:
            before = json.loads(host.state_json())
            key = rng.choice(host.options())[0]
            host.step(key)
            after = json.loads(host.state_json())
            top = after["stack"][-1] if after["stack"] else None
            if not top or top["kind"] != "ab_pick" or not set(top["data"]["cards"]) <= set(before["discard"]):
                continue
            others = [c for c in before["discard"] if c not in top["data"]["cards"]]
            if not others:
                continue
            before["discard_next"] = [others[0]]
            gs = engine.GameSession(GAME, 0, "")
            gs.load_external_state(json.dumps(before))
            atoms = dict(gs.external_options())[key]
            for a in atoms:
                if a in gs.get_legal_actions():
                    gs.apply_action(a)
            p = top["player"]
            drawn = {c for frame in own_slots(gs, p)["private"] for c in frame}
            assert idx[others[0]] in drawn, "the known card was not drawn"
            checked += 1
        if checked >= 3:
            break
    assert checked


def test_strength_of_the_action_cards():
    """Map 12 conceals slots 1, 2, 4, 5 in turn (a card there has the next visible number); Tournament 1
    adds 1 after a hand card discard. The turn's base strength is the card's (X-tokens aside)."""
    seen = {"ai": 0, "boost": 0}

    def check(before, key, after, own, gs):
        if key[0] == "hand_boost":
            assert after["players"][before["current"]]["map_pack"]["hand_boost"] == "pending"
            seen["boost"] += 1
        if key[0] != "act" or before["stack"][-1]["kind"] != "turn" or not after["turn"]:
            return
        pl = before["players"][before["current"]]
        expected = slot_strength(pl, key[1], before["map"]) - 2 * ("constriction" in pl["action_tokens"].get(key[1], []))
        if pl["map_pack"]["hand_boost"] == "pending":
            expected += 1
            assert after["players"][before["current"]]["map_pack"]["hand_boost"] == "used"
        turn = after["turn"]
        while turn.get("outer"):
            turn = turn["outer"]
        assert turn["base_strength"] == expected, (before["map"], key, pl["slots"], pl["map_pack"])
        seen["ai"] += pl["map_pack"]["concealed"] > 0

    for key, n in (("plan12", 4), ("plant1", 4)):
        for seed in _seeds(key, n):
            _play(seed, check)
    assert seen["ai"] and seen["boost"], seen


def test_geographical_zoo_markers():
    """Map 9: a housed animal reaching the area of its continent may remove that marker for one of 5
    bonuses; the token removes any; the 5th marker removed also gives 1 conservation point."""
    table, idx, m = cards(), card_index(), zoo_map("plan9")
    continents = ["africa", "americas", "asia", "australia", "europe"]
    removed = 0

    def check(before, key, after, own, gs):
        nonlocal removed
        top = before["stack"][-1]
        if top["kind"] != "continent_marker":
            return
        p = top["player"]
        pb, pa = before["players"][p], after["players"][p]
        if not top["data"].get("any"):
            for k, cont in enumerate(continents):
                if top["data"]["areas"] >> k & 1:
                    assert cont in pb["map_pack"]["continent_markers"]
                    area = {tuple(c) for c in m["areas"][k]}
                    assert any(cont in table[idx[c]]["continents"] and
                               area & {tuple(x) for x in pb["buildings"][pb["housing"][c][0]]["cells"]}
                               for c in pb["housing"]), "no housed animal of that continent in its area"
        if key[0] != "continent":
            return
        removed += 1
        assert key[1] not in pa["map_pack"]["continent_markers"] and key[1] in pb["map_pack"]["continent_markers"]
        if key[2] == "money":
            assert pa["money"] >= pb["money"] + GEO_MONEY
        if key[2] == "appeal":
            assert pa["appeal"] >= pb["appeal"] + GEO_APPEAL
        if not pa["map_pack"]["continent_markers"]:
            assert pa["conservation"] >= pb["conservation"] + 1

    for seed in _seeds("plan9", 6):
        _play(seed, check)
    assert removed


def test_mark_shark_attack_store_cut_down():
    """Mark puts the player's token on a display animal; Shark Attack discards animals in reputation
    range for half their appeal; the Caves store a hand card face down; Cut Down removes an empty
    standard enclosure for 2 money per space."""
    table, idx = cards(), card_index()
    done = {"mark": 0, "shark": 0, "store": 0, "cut_down": 0}

    def check(before, key, after, own, gs):
        top = before["stack"][-1]
        p = top["player"]
        pb, pa = before["players"][p], after["players"][p]
        if key[0] == "mark":
            c = before["display"][key[1]]
            assert table[idx[c]]["type"] == "animal" and before["display_marks"][key[1]] is None
            if after["display"][key[1]] == c:
                assert after["display_marks"][key[1]] == p
            done["mark"] += 1
        elif top["kind"] == "ab_shark" and not any(f["kind"] == "ab_shark" for f in after["stack"]) and after["turn"]:
            appeal = top["data"]["appeal"]
            if key[0] == "shark":
                appeal += table[idx[before["display"][key[1]]]]["appeal"]
            assert pa["appeal"] - pb["appeal"] >= appeal // 2
            done["shark"] += appeal > 0
        elif key[0] == "store":
            c = idx[key[1]]
            assert c in own_slots(gs, p)["stored"] and c not in own_slots(gs, p)["hand"]
            assert key[1] not in pa["public_hand"]
            assert len(pa["map_pack"]["stored"]) == len(pb["map_pack"]["stored"]) + 1 <= 3
            done["store"] += 1
        elif key[0] == "cut_down":
            b = pb["buildings"][key[1]]
            assert b["kind"] == "enclosure" and not b["occupied"]
            assert len(pa["buildings"]) == len(pb["buildings"]) - 1
            assert pa["money"] >= pb["money"] + 2 * b["size"]
            done["cut_down"] += 1

    for key, n in (("plan11", 3), ("plan12", 3), ("plan13", 4), ("plan14", 4), ("plant1", 3)):
        for seed in _seeds(key, n):
            _play(seed, check)
    assert all(done.values()), done


def test_drawing_board_areas_and_rescue_station():
    """Map 13: the areas covered now are those fully covered (and no full-cover appeal); map 10: the
    animals under the map are zoo cards (never released), at most 3, none a petting zoo animal."""
    table, idx = cards(), card_index()
    board = zoo_map("plan13")
    rescued = 0

    def check(before, key, after, own, gs):
        nonlocal rescued
        for pl in after["players"]:
            if after["map"] == "plan13":
                covered = _covered(pl)
                full = [a for a, area in enumerate(board["areas"]) if {tuple(c) for c in area} <= covered]
                assert sorted(pl["map_pack"]["quarters"]) == full
            for c in pl["map_pack"]["rescued"]:
                assert c in pl["zoo_cards"] and "pet" not in table[idx[c]]["tags"]
                assert c not in pl["housing"]
            assert len(pl["map_pack"]["rescued"]) <= 3
            rescued = max(rescued, len(pl["map_pack"]["rescued"]))

    for key in ("plan13", "plan10"):
        for seed in _seeds(key, 5):
            _play(seed, check)
    assert rescued


def test_lagoon_person_sponsors():
    """Map 14: each player finds a person sponsor at the setup (face up, so the setup keeps 5 of 9),
    and every new association worker finds one."""
    table, idx = cards(), card_index()
    found = 0
    for seed in _seeds("plan14", 4):
        gs = engine.GameSession(GAME, seed=seed)
        start = _core(gs)
        for p in (0, 1):
            hand = own_slots(gs, p)["hand"]
            assert len(hand) == 9
            people = [c for c in start["players"][p]["public_hand"] if table[idx[c]]["human"]]
            assert len(people) == 1 and idx[people[0]] in hand

        def check(before, key, after, own, gs):
            nonlocal found
            for p in (0, 1):
                gained = after["players"][p]["workers"] - before["players"][p]["workers"]
                if gained > 0 and before["stack"][-1]["kind"] != "setup_keep":
                    new = set(after["players"][p]["public_hand"]) - set(before["players"][p]["public_hand"])
                    waiting = any(f["kind"] == "drawn_effect" for f in after["stack"])  # below another decision
                    assert any(table[idx[c]]["human"] for c in new) or waiting or not after["deck"], "no person sponsor found"
                    found += not waiting

        _play(seed, check)
    assert found
