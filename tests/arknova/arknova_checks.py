"""Decodes an Ark Nova encoding token by token and compares every feature with an independent source.

Sources: the UI serialization of the same state (core_json), the players' own private slots (from the
ground truth session), the card table / map parsed from the sources (arknova_reference) and, for the
"can I do this now" features, the rules themselves (legal actions of the same state).
"""
from __future__ import annotations

import collections
import json

import numpy as np

import dinoboard_engine as engine

from arknova_reference import (ACTIONS, BUILD_ACTION, BUILDING_TYPES, DONATION_COSTS, GAME, SPONSOR_TRIGGERS,
                               TASKS, adjacent_terrain, animal_cost, appeal_income, atom_index, atom_names,
                               PLACEMENT_BONUS_EFFECT, building_cells, building_type, card_index, cards, decode,
                               effect_vector, enums, final_card_points, final_scoring_cards, kiosk_income_of, known_to,
                               layout, map_rule_state, neighbors, own_slots, pool_composition, project_count,
                               regular_cards, reputation_folders, sponsor_endgame, sponsor_immediate, sponsor_income,
                               sponsor_level, strength_short, target_appeal, unmet_requirements, zoo_icons, zoo_map)

AUTO_FRAMES = {"break", "extra_end", "final_scoring", "next_turn", "turn_end"}
HOUSING = ("enclosure", "petting_zoo", "reptile_house", "aviary")
LOCATIONS = ["my_hand", "opp_hand", "display", "my_zoo", "opp_zoo", "public_list", "private_list", "my_final",
             "in_play", "opp_maybe_tucked"]
# The locations each token type has a feature for (a project is never in a zoo or a final card, only
# projects are in play).
TYPE_LOCATIONS = {"card": [loc for loc in LOCATIONS if loc != "in_play"],
                  "project": [loc for loc in LOCATIONS if loc not in ("my_zoo", "opp_zoo", "my_final")]}
# Effect flags (immediate, icon trigger, income, endgame) of a few sponsors, from their card texts.
EFFECT_SPOT = {201: (1, 0, 1, 1), 220: (0, 0, 1, 0), 243: (0, 1, 0, 1), 227: (1, 0, 0, 0), 253: (1, 1, 0, 0),
               203: (0, 0, 0, 0), 257: (0, 0, 1, 1), 236: (0, 1, 0, 0)}


def onehot(n, i):
    v = [0.0] * n
    v[i] = 1.0
    return v


def binding() -> dict:
    """atom -> (anchor id, head) for the pointer atoms."""
    lay = engine.game_metadata(GAME)["token_layout"]
    heads = lay["num_pointer_heads"]
    return {a: (i // heads + 1, i % heads) for i, a in enumerate(lay["anchor_action"]) if a >= 0}


class Checker:
    """Collects which features were compared, and which were compared with a non-zero value."""

    def __init__(self):
        self.checked = collections.Counter()
        self.nonzero = collections.Counter()
        self.frames = collections.Counter()
        self.decisions = 0
        self.tokens = []
        self.context = ""

    def eq(self, tok, name, expected):
        got = tok[name]
        if isinstance(got, float):  # a feature of width 1
            expected = float(expected)
            if abs(got - expected) > 1e-5:
                raise AssertionError(f"{tok.type}.{name}: encoder {got} != reference {expected} [{self.context}]")
            self.mark(tok.type, name, expected != 0.0)
            return
        exp = np.asarray(expected, dtype=np.float64)
        assert got.shape == exp.shape, f"{tok.type}.{name}: width {got.shape} vs {exp.shape}"
        if not np.allclose(got, exp, atol=1e-5):
            raise AssertionError(f"{tok.type}.{name}: encoder {got.tolist()} != reference {exp.tolist()} [{self.context}]")
        self.mark(tok.type, name, bool(exp.any()))

    def mark(self, type_name, name, nonzero):
        self.checked[(type_name, name)] += 1
        self.nonzero[(type_name, name)] += bool(nonzero)

    def unchecked(self):
        return [(t, n) for t, feats in layout().features.items() for n in feats if not self.checked[(t, n)]]

    def never_nonzero(self):
        return [(t, n) for t, feats in layout().features.items() for n in feats if not self.nonzero[(t, n)]]


class HandTracker:
    """What each player publicly knows about the other's hand, rebuilt from the moves: a card is
    known when it entered the hand from a public place; a face-down Pouch tuck turns the known cards
    into "in the hand or tucked" (maybe)."""

    def __init__(self, own):
        self.known = {0: set(), 1: set()}
        self.maybe = {0: set(), 1: set()}
        self.hand = {q: own[q]["hand"] for q in (0, 1)}

    def update(self, own_after, core_before, actor, name, revealed=False, harbor=False):
        """revealed: the move turned deck cards face up until one went to the hand (Waza, Hollywood);
        harbor: the move completes a harbour trade (the card goes face up to the discard pile)."""
        idx = card_index()
        top = core_before["stack"][-1]["kind"]
        public = set(core_before["display"]) | set(core_before["base_project_deck"])
        for f in core_before["stack"]:
            if f["kind"] == "ab_hunter":
                public |= set(f["data"]["cards"])
        public = {idx[c] for c in public if c is not None}
        after = {q: own_after[q]["hand"] for q in (0, 1)}
        for q in (0, 1):
            left, entered = self.hand[q] - after[q], after[q] - self.hand[q]
            if actor == q and top == "ab_pouch" and name.startswith("hand_discard:") and not harbor:
                seen = left & (self.known[q] | self.maybe[q])
                self.maybe[q] |= self.known[q] & after[q]
                self.maybe[q] = (self.maybe[q] - left) | seen
            else:
                self.maybe[q] -= left
            self.known[q] -= left
            for c in entered:
                if c in public or c in self.hand[1 - q] or revealed:
                    self.known[q].add(c)
        self.hand = after

    def shown(self, q):
        return self.known[q] - self.maybe[q]


def hollywood_reveals(before, after, p) -> bool:
    """Hollywood Hills (Map 8): every H space a building covers reveals deck cards until a sponsor,
    which goes to the hand."""
    m = zoo_map(before["map"])
    if m["rule"] != "hollywood":
        return False

    def covered(core):
        return {tuple(c) for b in core["players"][p]["buildings"] for c in b["cells"]} & m["rule_cells"]
    return bool(covered(after) - covered(before))


def turn_chain(core):
    t = core["turn"]
    while t:
        yield t
        t = t.get("outer")


def check_state(ck: Checker, toks, me, core, own, state_dict, tracker, legal):
    """Every token of `toks` (the encoding for player `me`) against the references. `own` are both
    players' private slots (truth), `legal` the names of the legal atoms (the mover's)."""
    lay, e, table, idx = layout(), enums(), cards(), card_index()
    opp = 1 - me
    pls = core["players"]
    mine, theirs = pls[me], pls[opp]
    stack = core["stack"]
    top = stack[-1]
    prefix = state_dict["prefix"].split()
    by_type = collections.defaultdict(list)
    for t in toks:
        by_type[t.type].append(t)
    assert [t.type for t in toks[:14]] == ["global", "decision", "player", "player"] + ["action_card"] * 10
    assert len(by_type["global"]) == 1 and len(by_type["decision"]) == 1 and len(by_type["cell"]) == 58
    ck.decisions += 1
    ck.frames[top["kind"]] += 1
    ck.tokens.append(len(toks))

    # ---- global ----
    g = by_type["global"][0]
    assert g.entity == 0 and g.anchor == 0 and g.pos is None
    ck.eq(g, "break_pos", core["break_pos"] / 9)
    ck.eq(g, "break_by", [core["break_by"] == me, core["break_by"] == opp])
    ck.eq(g, "deck", len(core["deck"]) / 200)
    ck.eq(g, "discard", len(core["discard"]) / 200)
    ck.eq(g, "final_deck", len(core["final_deck"]) / 12)
    ck.eq(g, "donations", core["donations"] / 5)
    ck.eq(g, "donation_cost", (DONATION_COSTS[core["donations"]] if core["donations"] < 4 else 12) / 12)
    ck.eq(g, "end_by", [core["end_by"] == me, core["end_by"] == opp])
    ck.eq(g, "final_turns", (core["final_turns"] or 0) / 2)
    ck.eq(g, "final_card_discarded", core["final_card_discarded"])
    ck.eq(g, "my_turn", core["current"] == me)
    ck.eq(g, "partner_zoo_available", [c in core["partner_zoos_available"] for c in e["continents"]])
    ck.eq(g, "university_available", [u in core["universities_available"] for u in e["universities"]])
    tiles = [0.0] * 18
    for sp, space in enumerate(("5", "8")):
        for name in core["bonus_tiles"][space]:
            tiles[sp * 9 + e["bonus_tiles"].index(name)] = 1.0
    ck.eq(g, "bonus_tiles", tiles)
    ck.eq(g, "base_pile", len(core["base_project_deck"]) / 10)
    ck.eq(g, "stack_depth", min(len(stack), 10) / 10)
    below = collections.Counter(f["kind"] for f in stack[:-1])
    ck.eq(g, "pending", [min(below[k], 4) / 4 for k in e["frames"]])
    ck.eq(g, "pending_opponent", any(f["player"] == opp and f["kind"] not in AUTO_FRAMES for f in stack[:-1]))
    after = collections.Counter(a[0] for t in turn_chain(core) for a in t["after"])
    assert set(after) <= {"clever", "ab_boost", "extra_action"}
    ck.eq(g, "after_clever", min(after["clever"], 4) / 4)
    ck.eq(g, "after_boost", min(after["ab_boost"], 4) / 4)
    ck.eq(g, "after_extra_action", min(after["extra_action"], 4) / 4)
    # The pools: the discard pile, and the cards whose place I do not know (neither public nor in my
    # slots nor seen in the opponent's hand), less the opponent's maybe-tucked cards.
    for name, v in pool_composition(idx[c] for c in core["discard"]).items():
        ck.eq(g, "discard_" + name, v)
    known = known_to(me, core, own, tracker.shown(opp)) | tracker.maybe[opp]
    for name, v in pool_composition(c for c in regular_cards() if c not in known).items():
        ck.eq(g, "unseen_" + name, v)
    ck.eq(g, "unseen_final", [c not in known for c in final_scoring_cards()])
    ck.eq(g, "trade_used", core["turn_flags"].get("harbor", False))

    # ---- decision ----
    d = by_type["decision"][0]
    kind, data, mover = top["kind"], top["data"], top["player"]
    ck.eq(d, "frame", onehot(42, e["frames"].index(kind)))
    ck.eq(d, "to_move", state_dict["current_player"] == me)
    ck.eq(d, "prefix_len", len(prefix) / 6)
    turn = core["turn"]
    ck.eq(d, "turn_action", onehot(5, ACTIONS.index(turn["action"])) if turn else [0] * 5)
    ck.eq(d, "turn_strength", turn.get("strength", 0) / 10 if turn else 0)
    ck.eq(d, "turn_xtoken", bool(turn) and turn["xtoken"])
    ck.eq(d, "turn_depth", len(list(turn_chain(core))) / 4)
    ck.eq(d, "turn_borrowed", core["borrowed"] is not None)

    def side2(action):  # the side of the action card in use, which may be the opponent's (Hypnosis)
        if core["borrowed"] and core["borrowed"][0] == action:
            return core["borrowed"][1]
        return action in pls[mover]["upgraded"]

    exp = dict(budget=0, remaining=0, done_count=0, side2=0, discard_after=0, may_snap=0, normal_done=0,
               engineer_used=0, animals_phase=[0] * 3, all_small=0, tasks_used=[0] * 4, donated=0, optional=0,
               extra_actions=[0] * 5, extra_from_opponent=0, boost_action=[0] * 5, tile_space=[0, 0],
               can_replenish=0, replenish_between=0, only_one=0, built=[0] * 11, offered=[0] * 11, list_len=0)
    if kind == "act_cards":
        exp.update(remaining=(data["draw"] - data["drawn"]) / 5, done_count=data["drawn"] / 5, side2=data["upgraded"],
                   discard_after=data["discard"], may_snap=data["may_snap"] and data["drawn"] == 0)
    elif kind == "act_build":
        exp.update(budget=data["budget"] / 10, done_count=len(data["built"]) / 5, side2=data["upgraded"],
                   normal_done=data["normal_done"], engineer_used=data["engineer_used"])
        for k, size in data["built"]:
            exp["built"][building_type(k, size)] = 1
    elif kind == "act_animals":
        exp.update(remaining=data["remaining"] / 5, done_count=data["played"] / 5, side2=side2("animals"),
                   animals_phase=onehot(3, ["play", "bonus", "take"].index(data["phase"])), all_small=data["all_small"])
    elif kind == "act_association":
        exp.update(budget=data["remaining"] / 10, side2=data["side2"], donated=data["donated"],
                   tasks_used=[t in data["used"] for t in TASKS])
    elif kind == "act_sponsors":
        exp.update(budget=data["budget"] / 10, done_count=data["played"] / 5, side2=side2("sponsors"))
    elif kind in ("ab_sun", "ab_dig", "ab_pouch"):
        exp.update(remaining=data["left"] / 5)
    elif kind == "ab_snap":
        exp.update(remaining=data["left"] / 5, can_replenish=data["can_replenish"],
                   replenish_between=data["replenish_between"])
    elif kind == "ab_pick":
        exp.update(remaining=data["count"] / 5, list_len=len(data["cards"]) / 10)
    elif kind in ("ab_hunter", "ab_resist"):
        exp.update(list_len=len(data["cards"]) / 10)
    elif kind == "ab_project":
        exp.update(list_len=len(core["base_project_deck"]) / 10, only_one="only" in data)
    elif kind == "discard_hand":
        hand = len(own[mover]["hand"])
        left = max(0, hand - data["limit"]) if "limit" in data else data.get("count", 1)
        exp.update(remaining=min(left, hand) / 5)
    elif kind == "extra_action":
        exp.update(optional=data["optional"], extra_actions=[a in data["actions"] for a in ACTIONS],
                   extra_from_opponent=data["owner"] != mover)
    elif kind == "ab_boost":
        exp.update(boost_action=onehot(5, ACTIONS.index(data["action"])))
    elif kind == "conservation_tile":
        exp.update(tile_space=[data["space"] == 5, data["space"] == 8])
    elif kind == "free_build":
        for k, size in data["kinds"]:
            exp["offered"][building_type(k, size)] = 1
    for name, value in exp.items():
        ck.eq(d, name, value)
    chosen = dict(chosen_verb=[0] * 3, chosen_type=[0] * 11, chosen_position=[0] * 3, chosen_map_token=[0] * 7,
                  chosen_harbor=0)
    for atom in prefix:
        head, _, arg = atom.partition(":")
        if head == "harbor":
            chosen["chosen_harbor"] = 1
        elif head in ("build", "engineer", "place"):
            chosen["chosen_verb"][("build", "engineer", "place").index(head)] = 1
            chosen["chosen_type"][BUILDING_TYPES.index(arg)] = 1
        elif head == "position":
            chosen["chosen_position"][int(arg) - 1] = 1
        elif head == "map_token":
            chosen["chosen_map_token"][int(arg)] = 1
    for name, value in chosen.items():
        ck.eq(d, name, value)

    # ---- players ----
    for tok, p in zip(by_type["player"], (me, opp)):
        pl = pls[p]
        ck.eq(tok, "me", p == me)
        ck.eq(tok, "opponent", p == opp)
        ck.eq(tok, "to_move", state_dict["current_player"] == p)
        ck.eq(tok, "appeal", pl["appeal"] / 60)
        ck.eq(tok, "money", pl["money"] / 50)
        ck.eq(tok, "conservation", pl["conservation"] / 30)
        ck.eq(tok, "reputation", pl["reputation"] / 15)
        ck.eq(tok, "xtokens", pl["xtokens"] / 5)
        ck.eq(tok, "workers", pl["workers"] / 4)
        out = [pl["workers_out"].get(t, 0) for t in TASKS]
        ck.eq(tok, "active_workers", (pl["workers"] - sum(out)) / 4)
        ck.eq(tok, "workers_out", [n / 4 for n in out])
        ck.eq(tok, "hand_size", len(own[p]["hand"]) / 20)
        ck.eq(tok, "hand_limit", (5 if "hand" in pl["universities"] else 3) / 5)
        ck.eq(tok, "zoo_cards", len(pl["zoo_cards"]) / 30)
        ck.eq(tok, "tucked", len(own[p]["tucked"]) / 10)
        ck.eq(tok, "final_cards", len(own[p]["final"]) / 6)
        ck.eq(tok, "partner_zoos", [c in pl["partner_zoos"] for c in e["continents"]])
        ck.eq(tok, "universities", [u in pl["universities"] for u in e["universities"]])
        ck.eq(tok, "map_tokens", pl["tokens"])
        icons = zoo_icons(pl)
        ck.eq(tok, "icons", [icons[i] / 10 for i in e["icons"]])
        ck.eq(tok, "victory_points", (pl["appeal"] - target_appeal(pl["conservation"])) / 30)
        ck.eq(tok, "poisoned", any("venom" in v for v in pl["action_tokens"].values()))
        ck.eq(tok, "covered_cells", sum(len(b["cells"]) for b in pl["buildings"]) / 50)
        kinds = collections.Counter(b["kind"] for b in pl["buildings"])
        ck.eq(tok, "buildings", [kinds[k] / 10 for k in e["building_kinds"]])
        kiosks = [k for k, b in enumerate(pl["buildings"]) if b["kind"] == "kiosk"]
        ck.eq(tok, "kiosk_income", sum(kiosk_income_of(pl["buildings"], k) for k in kiosks) / 10)
        ck.eq(tok, "appeal_income", appeal_income(pl["appeal"]) / 40)
        ck.eq(tok, "display_range", reputation_folders(pl["reputation"]) / 6)
        # placeable: checked against the Build action's legal atoms in check_playable; here only
        # what follows from the card alone (side II buildings, built special enclosures, no unique).
        placeable = tok["placeable"]
        built_kinds = {b["kind"] for b in pl["buildings"]}
        for i, (name, _, side2_only) in enumerate(BUILD_ACTION):
            if (side2_only and "build" not in pl["upgraded"]) or (i >= 7 and name in built_kinds):
                assert placeable[i] == 0.0, f"player {p}: {name} placeable"
        assert placeable[BUILDING_TYPES.index("unique")] == 0.0
        ck.mark("player", "placeable", placeable.any())
        zm = zoo_map(core["map"])
        n_pz, n_uni = len(pl["partner_zoos"]), len(pl["universities"])
        ck.eq(tok, "next_partner_zoo", effect_vector(zm["partner_zoo_slots"][n_pz] if n_pz < 4 else None))
        ck.eq(tok, "next_university", effect_vector(zm["university_slots"][n_uni] if n_uni < 3 else None))
        ck.eq(tok, "last_worker", effect_vector(zm["last_worker"] if pl["workers"] < 4 else None))
        rs = map_rule_state(pl, core["map"])
        ck.eq(tok, "map_income", rs["map_income"] / 5)
        ck.eq(tok, "can_trade", rs["can_trade"])
        ck.eq(tok, "ignore_conditions", rs["ignore_conditions"])
        ck.eq(tok, "sponsor_discount", rs["sponsor_discount"])
        ck.eq(tok, "group_left", rs["group_left"] / 3)

    # ---- action cards ----
    for i, tok in enumerate(by_type["action_card"]):
        p, a = (me, opp)[i // 5], i % 5
        pl, name = pls[p], ACTIONS[a]
        assert tok.entity == lay.entities["action_card"][0] + a and tok.anchor == 0 and tok.pos is None
        strength = pl["slots"].index(name) + 1
        on_card = pl["action_tokens"].get(name, [])
        poisoned = any("venom" in v for v in pl["action_tokens"].values())
        borrowed = core["borrowed"] and core["borrowed"][0] == name
        ck.eq(tok, "mine", p == me)
        ck.eq(tok, "opponent", p == opp)
        ck.eq(tok, "action", onehot(5, a))
        ck.eq(tok, "slot", onehot(5, strength - 1))
        ck.eq(tok, "strength", strength / 5)
        ck.eq(tok, "effective_strength", (strength - 2 * ("constriction" in on_card)) / 5)  # Constriction: -2
        ck.eq(tok, "upgraded", name in pl["upgraded"])
        ck.eq(tok, "venom", "venom" in on_card)
        ck.eq(tok, "constriction", "constriction" in on_card)
        ck.eq(tok, "multiplier", on_card.count("multiplier") / 4)
        ck.eq(tok, "venom_cost", poisoned and "venom" not in on_card)
        ck.eq(tok, "in_use", bool(turn) and turn["action"] == name and p == (1 - core["current"] if borrowed else core["current"]))

    # ---- cards and projects, by location ----
    card_base = lay.entities["card"][0]
    cardish = by_type["card"] + by_type["project"]
    for t in cardish:
        assert sum(t[loc] for loc in TYPE_LOCATIONS[t.type]) == 1.0, "a card token has exactly one location"
        known = t.card() is not None
        assert known != (t.type == "card" and t["unknown"] == 1.0)
        if known:
            assert (t.type == "project") == (table[t.card()]["type"] == "project")
        assert t.pos is None

    def at(loc):
        return [t for t in toks if t.type in ("card", "project") and loc in TYPE_LOCATIONS[t.type] and t[loc] == 1.0]

    def ids(tokens):
        return [t.card() for t in tokens]

    display = [(i, c) for i, c in enumerate(core["display"]) if c is not None]
    folders = reputation_folders(mine["reputation"])
    ctx = dict(me=me, core=core, prefix=prefix, top=top)

    hand = at("my_hand")
    assert set(ids(hand)) == own[me]["hand"] and len(hand) == len(own[me]["hand"])
    for t in hand:
        assert t.anchor == lay.anchors["hand"][0] + t.card()
        check_card(ck, t, "my_hand", -1, f"hand:{t.card()}", f"hand_discard:{t.card()}", **ctx)

    opp_hand = at("opp_hand")
    shown = tracker.shown(opp)
    seen = [t for t in opp_hand if t.card() is not None]
    assert set(ids(seen)) == shown == {idx[c] for c in theirs["public_hand"]}
    assert len(opp_hand) == len(own[opp]["hand"]), "one token per card in the opponent's hand"
    for t in opp_hand:
        assert t.anchor == 0
        if t.card() is None:
            assert t.entity == 0 and t["opp_hand"] == 1.0 and t["unknown"] == 1.0
            assert np.count_nonzero(t.f) == 2, "an unseen card carries nothing but its location"
            ck.mark("card", "unknown", True)
            ck.mark("card", "opp_hand", True)
        else:
            check_card(ck, t, "opp_hand", -1, None, None, **ctx)

    maybe = at("opp_maybe_tucked")
    assert set(ids(maybe)) == tracker.maybe[opp] and len(maybe) == len(tracker.maybe[opp])
    for t in maybe:
        assert t.anchor == 0
        check_card(ck, t, "opp_maybe_tucked", -1, None, None, **ctx)

    shown_tokens = at("display")
    assert [(round(t["display_pos"] * 6) - 1, t.card()) for t in shown_tokens] == [(i, idx[c]) for i, c in display]
    for t in shown_tokens:
        i = round(t["display_pos"] * 6) - 1
        assert t.anchor == lay.anchors["display"][0] + i
        ck.eq(t, "in_range", i < folders)
        check_card(ck, t, "display", i, f"display:{i}", None, **ctx)

    for loc, p in (("my_zoo", me), ("opp_zoo", opp)):
        zoo = at(loc)
        assert ids(zoo) == [idx[c] for c in pls[p]["zoo_cards"]]
        for t in zoo:
            assert t.anchor == (lay.anchors["zoo"][0] + t.card() if p == me else 0)
            check_card(ck, t, loc, -1, None, None, **ctx)

    public = at("public_list")
    if kind == "ab_hunter":
        assert ids(public) == [idx[c] for c in data["cards"]]
    elif kind == "ab_project":
        assert sorted(ids(public)) == sorted(idx[c] for c in core["base_project_deck"])
    else:
        assert not public
    for j, t in enumerate(public):
        assert t.anchor == lay.anchors["public_list"][0] + j
        check_card(ck, t, "public_list", -1, None, None, **ctx)

    private = at("private_list")
    if kind in ("ab_pick", "ab_resist") and mover == me:
        k = sum(f["kind"] in ("ab_pick", "ab_resist") and f["player"] == me for f in stack[:-1])
        assert ids(private) == own[me]["private"][k]
    else:
        assert not private
    for j, t in enumerate(private):
        assert t.anchor == lay.anchors["private_list"][0] + j
        check_card(ck, t, "private_list", -1, None, None, **ctx)

    final = at("my_final")
    assert ids(final) == own[me]["final"]
    for i, t in enumerate(final):
        assert t.anchor == lay.anchors["final"][0] + i
        check_card(ck, t, "my_final", -1, None, None, **ctx)

    in_play = at("in_play")
    assert ids(in_play) == [idx[c] for c in core["base_projects"] + core["projects"]]
    for slot, t in enumerate(in_play):
        assert t.type == "project" and t.anchor == lay.anchors["project"][0] + slot
        check_card(ck, t, "in_play", -1, f"project:{slot}", None, **ctx)
    assert len(cardish) == sum(len(at(loc)) for loc in LOCATIONS)
    # Why the token types lack those location features: in the state itself no zoo card or final
    # card is a project, every project in play is one.
    assert all(table[idx[c]]["type"] != "project" for p in (me, opp) for c in pls[p]["zoo_cards"])
    assert all(table[c]["type"] == "endgame" for c in own[me]["final"])
    assert all(table[idx[c]]["type"] == "project" for c in core["base_projects"] + core["projects"])
    # No token names a card hidden from me.
    hidden = (own[opp]["hand"] - shown) | set(own[opp]["final"]) | set(own[opp]["tucked"])
    hidden |= {c for frame in own[opp]["private"] for c in frame}
    named = {t.card() for t in cardish} - {None}
    assert not (named & hidden) - tracker.maybe[opp], "a token names a card hidden from the perspective"

    # ---- the map: what it prints, my buildings, the opponent's ----
    m = zoo_map(core["map"])
    covered = {tuple(c): b for b in mine["buildings"] for c in b["cells"]}
    opp_covered = {tuple(c): b for b in theirs["buildings"] for c in b["cells"]}
    near, opp_near = neighbors(covered), neighbors(opp_covered)
    kiosk_cells = {qr for qr, b in covered.items() if b["kind"] == "kiosk"}
    special, rule = m["rule_cells"], m["rule"]
    near_special = {n for r in special for n in m["cells"][r]["nbr"]}
    for i, tok in enumerate(by_type["cell"]):
        qr = m["order"][i]
        cell = m["cells"][qr]
        assert tok.pos == qr and tok.entity == lay.entities["cell"][0] + i and tok.anchor == lay.anchors["cell"][0] + i
        ck.eq(tok, "terrain", onehot(3, ["plain", "water", "rock"].index(cell["terrain"])))
        ck.eq(tok, "build2_only", cell["build2"])
        ck.eq(tok, "border", cell["border"])
        assert cell["bonus"] in e["placement_bonus"]
        ck.eq(tok, "bonus", effect_vector(PLACEMENT_BONUS_EFFECT.get(cell["bonus"])))
        ck.eq(tok, "landmark", qr in special)
        ck.eq(tok, "fill_appeal", rule == "observationtower" and qr in near_special)  # +2 appeal, / 2
        ck.eq(tok, "extra_capacity", rule == "outdoorarea" and qr in near_special)  # +2 spaces, / 2
        ck.eq(tok, "break_money", sum(r in special for r in cell["nbr"]) if rule == "parkrestaurant" else 0)
        ck.eq(tok, "unlocks_trade", rule == "harbor" and qr in special)
        ck.eq(tok, "unlocks_condition", rule == "researchinstitute" and qr in special)
        ck.eq(tok, "cover_draw_sponsor", rule == "hollywood" and qr in special)
        ck.eq(tok, "group", rule in ("icecream", "hollywood") and qr in special)
        ck.eq(tok, "group_kiosk_income", rule == "icecream" and qr in special)
        ck.eq(tok, "group_sponsor_discount", rule == "hollywood" and qr in special)
        ob = opp_covered.get(qr)
        ck.eq(tok, "opp_kind", onehot(7, e["building_kinds"].index(ob["kind"])) if ob else [0] * 7)
        ck.eq(tok, "opp_adjacent", qr in opp_near)
        ck.eq(tok, "covered", qr in covered)
        ck.eq(tok, "adjacent", qr in near)
        free_terrain = cell["terrain"] == "plain" or "219" in mine["zoo_cards"]  # Diversity Researcher
        ck.eq(tok, "buildable", qr not in covered and free_terrain and (not cell["build2"] or "build" in mine["upgraded"]))
        b = covered.get(qr)
        low = min(building_cells(b)) if b else qr
        ck.eq(tok, "kind", onehot(7, e["building_kinds"].index(b["kind"])) if b else [0] * 7)
        ck.eq(tok, "anchor", b is not None and low == qr)
        ck.eq(tok, "anchor_dq", (low[0] - qr[0]) / 4)
        ck.eq(tok, "anchor_dr", (low[1] - qr[1]) / 4)
        ck.eq(tok, "rock_neighbors", cell["rock"] / 6)
        ck.eq(tok, "water_neighbors", cell["water"] / 6)
        ck.eq(tok, "kiosk_income", kiosk_income_of(mine["buildings"], b["id"]) / 6 if b and b["kind"] == "kiosk" else 0)
        ck.eq(tok, "kiosks_adjacent", sum(n in kiosk_cells for n in cell["nbr"]) / 3)

    # ---- enclosures of both zoos ----
    enclosures = by_type["enclosure"]
    want = [(p, b) for p in (me, opp) for b in pls[p]["buildings"] if b["kind"] in HOUSING]
    assert len(enclosures) == len(want)
    for tok, (p, b) in zip(enclosures, want):
        pl, cells = pls[p], building_cells(b)
        assert tok.entity == 0
        if p == me:
            assert tok.pos == min(cells) and tok.anchor == lay.anchors["enclosure"][0] + b["id"]
        else:
            assert tok.pos is None and tok.anchor == 0
        standard = b["kind"] == "enclosure"
        capacity = b["size"] + b["extra_size"] if standard else b["size"]
        ck.eq(tok, "mine", p == me)
        ck.eq(tok, "opponent", p == opp)
        ck.eq(tok, "kind", onehot(7, e["building_kinds"].index(b["kind"])))
        ck.eq(tok, "size", b["size"] / 5)
        ck.eq(tok, "capacity", capacity / 5)
        ck.eq(tok, "occupied", standard and b["occupied"])
        ck.eq(tok, "used", 0 if standard else b["used"] / 5)
        ck.eq(tok, "free", (0 if b["occupied"] else capacity) / 5 if standard else (b["size"] - b["used"]) / 5)
        ck.eq(tok, "rock", adjacent_terrain(cells, "rock", core["map"]) / 6)
        ck.eq(tok, "water", adjacent_terrain(cells, "water", core["map"]) / 6)
        near_b = neighbors(cells)
        ck.eq(tok, "kiosks_adjacent", sum(o["kind"] == "kiosk" and bool(near_b & set(building_cells(o)))
                                          for o in pl["buildings"]) / 3)
        ck.eq(tok, "animals", sum(h[0] == b["id"] for h in pl["housing"].values()) / 4)
        ck.eq(tok, "target", kind == "move_animals" and mover == p and data["building"] == b["id"])

    # ---- the map's left-edge tokens ----
    rewards = by_type["reward"]
    assert len(rewards) == 7
    for i, tok in enumerate(rewards):
        effect, repeats = m["tokens"][i]
        assert tok.entity == 0 and tok.anchor == 0 and tok.pos is None
        ck.eq(tok, "index", onehot(7, i))
        ck.eq(tok, "effect", effect_vector(effect))
        ck.eq(tok, "repeats", repeats)
        ck.eq(tok, "taken_me", not mine["tokens"][i])
        ck.eq(tok, "taken_opp", not theirs["tokens"][i])
        ck.eq(tok, "chosen", f"map_token:{i}" in prefix)

    # ---- header: anchors are unique, and every legal atom has its logit ----
    anchors = [t.anchor for t in toks if t.anchor]
    assert len(anchors) == len(set(anchors)), "an anchor id appears on two tokens"
    if state_dict["current_player"] == me:
        bind, index = binding(), atom_index()
        sources = set()
        for name in legal:
            source = bind.get(index[name])
            if source is not None:
                assert source[0] in anchors, f"legal atom {name} has no anchor token"
            sources.add(source if source is not None else name)
        assert len(sources) == len(legal), "two legal atoms share a logit"
        check_playable(ck, toks, legal, kind, data, prefix, mine, folders, side2, core["map"])


def check_card(ck, t, loc, display_i, take, discard, *, me, core, prefix, top):
    """One card or project token against the card table and the state."""
    lay, e, table = layout(), enums(), cards()
    pls = core["players"]
    mine, theirs = pls[me], pls[1 - me]
    holder = theirs if loc in ("opp_hand", "opp_maybe_tucked", "opp_zoo") else mine
    c = table[t.card()]
    ck.context = f"{c['id_str']} {c['name']} at {loc}"
    for other in TYPE_LOCATIONS[t.type]:
        ck.eq(t, other, other == loc)
    ck.eq(t, "chosen", take in prefix)
    ck.eq(t, "chosen_discard", discard in prefix)
    ck.eq(t, "display_pos", (display_i + 1) / 6 if display_i >= 0 else 0)
    if display_i < 0:
        ck.eq(t, "in_range", 0)
    if t.type == "card":
        ck.eq(t, "unknown", 0)
        ck.eq(t, "pending", loc in ("my_zoo", "opp_zoo") and top["kind"] == "sponsor_building" and
              top["data"]["card"] == c["id_str"])
    folder = display_i + 1 if display_i >= 0 else 0
    if c["type"] == "project":
        ck.eq(t, "category", onehot(4, e["proj_categories"].index(c["proj_category"])))
        ck.eq(t, "icon", onehot(19, e["proj_icons"].index(c["proj_icon"])))
        slots = c["slots"]
        ck.eq(t, "level_required", [max(s["required"], 0) / 8 for s in slots])
        ck.eq(t, "level_size_min", [max(s["size_min"], 0) / 5 for s in slots])
        ck.eq(t, "level_size_max", [max(s["size_max"], 0) / 5 for s in slots])
        ck.eq(t, "level_conservation", [s["rewards"].get("conservation", 0) / 5 for s in slots])
        ck.eq(t, "level_reputation", [s["rewards"].get("reputation", 0) / 5 for s in slots])
        bonus = c["play_bonus"]
        ck.eq(t, "play_bonus", [bonus.get("money", 0) / 10, bonus.get("appeal", 0) / 10,
                                bonus.get("conservation", 0) / 5, bonus.get("reputation", 0) / 5])
        held = core["project_tokens"].get(c["id_str"])
        assert (held is not None) == (loc == "in_play")
        levels = [None if held is None else held.get(str(k), "free") for k in (1, 2, 3)]
        ck.eq(t, "occupied_free", [v == "free" for v in levels])
        ck.eq(t, "occupied_me", [v == me and v != "free" and v != "blocked" for v in levels])
        ck.eq(t, "occupied_opp", [v == 1 - me and v != "free" and v != "blocked" for v in levels])
        ck.eq(t, "occupied_blocked", [v == "blocked" for v in levels])
        ck.eq(t, "count_me", project_count(mine, c["proj_icon"]) / 8)
        ck.eq(t, "count_opp", project_count(theirs, c["proj_icon"]) / 8)
        ck.eq(t, "affordable", mine["money"] >= folder)
        ck.eq(t, "strength_short", strength_short(holder, c, core["map"]))
        return
    animal, sponsor = c["type"] == "animal", c["type"] == "sponsor"
    ck.eq(t, "type", [animal, sponsor, c["type"] == "endgame"])
    ck.eq(t, "cost", c["cost"] / 36)
    ck.eq(t, "size", c["size"] / 5)
    ck.eq(t, "appeal", c["appeal"] / 10)
    ck.eq(t, "conservation", c["conservation"] / 5)
    ck.eq(t, "reputation", c["reputation"] / 5)
    ck.eq(t, "level", c["level"] / 6)
    ck.eq(t, "rock", c["enc_rock"] / 2)
    ck.eq(t, "water", c["enc_water"] / 2)
    ck.eq(t, "no_standard", animal and c["enc_standard"] < 0)
    special = [0.0] * 6
    for k, name in enumerate(e["special"]):
        if name in c["special"]:
            special[2 * k], special[2 * k + 1] = 1.0, c["special"][name] / 5
    ck.eq(t, "special", special)
    ck.eq(t, "tags", [c["tags"][i] / 2 for i in e["icons"]])
    req = []
    for name in e["reqs"]:
        n = c["req"][name]
        req.append(0 if not n else c["min_reputation"] / 15 if name == "kReqReputation"
                   else c["max_appeal"] / 50 if name == "kReqAppeal" else n / 2)
    ck.eq(t, "req", req)
    ability = [0.0] * 44
    for kw, value in c["abilities"].items():
        if isinstance(value, str):  # Iconic Animal of a continent: the card's own continent icon
            assert kw == "ICONIC_ANIMAL" and c["tags"][value] > 0
            value = -1
        ability[e["keywords"].index(kw)] = (value + 1) / 6 if value >= 0 else 1.0
    ck.eq(t, "ability", ability)
    b = c["building"]
    ck.eq(t, "building", [1, b["size"] / 5, b["rock"] / 2, b["water"] / 2, b["min_border"] / 2, b["needs_adjacency"]]
          if b else [0] * 6)
    scoring = [0.0] * 8
    for i, (threshold, points) in c["scoring"].items():
        scoring[2 * i], scoring[2 * i + 1] = threshold / 24, points / 4
    ck.eq(t, "scoring", scoring)
    on_card = [pl["card_tokens"].get(c["id_str"]) for pl in pls]
    n = next((v for v in on_card if v), 0)
    ck.eq(t, "tokens", {"small": 1, "large": 2}.get(n, n) / 5)
    ck.eq(t, "unmet", unmet_requirements(holder, c) / 3 if animal or sponsor else 0)
    held = loc in ("my_zoo", "opp_zoo", "my_final")
    if not held and (animal or sponsor):
        money = folder + (animal_cost(holder, c) if animal else 0)
        ck.eq(t, "money_now", money / 36)
        ck.eq(t, "affordable", holder["money"] >= money)
        ck.eq(t, "strength_short", strength_short(holder, c, core["map"]))
    else:
        for name in ("money_now", "affordable", "req_ok", "has_place", "strength_short"):
            ck.eq(t, name, 0)
    effect = t["effect"]
    assert set(np.unique(effect)) <= {0.0, 1.0}
    assert effect[4] == c["lasting_effect"]
    if c["type"] == "endgame":
        assert effect.tolist() == [0, 0, 0, 1, 0]
    if c["id"] in EFFECT_SPOT:
        assert tuple(effect[:4]) == EFFECT_SPOT[c["id"]], f"effect flags of {c['name']}"
    ck.mark("card", "effect", effect.any())
    if c["type"] == "endgame":
        points = final_card_points(core, me, c["id"]) if loc in ("my_final", "private_list") else 0
        ck.eq(t, "yield", [points / 4, 0])
    elif sponsor and (held or loc == "private_list"):
        conservation, appeal = sponsor_endgame(holder, c, core["map"])
        ck.eq(t, "yield", [conservation / 4, appeal / 10])
    else:
        ck.eq(t, "yield", [0, 0])
    if not sponsor:
        for name, width in (("immediate", 4), ("income", 3), ("trigger_icon", 15), ("trigger_any", 1),
                            ("trigger_gain", 3)):
            ck.eq(t, name, [0] * width if width > 1 else 0)
        return

    def part(v, top):
        return min(max(v, 0), top) / top

    imm = sponsor_immediate(holder, c, core["map"]) if not held else dict(money=0, appeal=0, conservation=0, other=0)
    ck.eq(t, "immediate", [part(imm["money"], 20), part(imm["appeal"], 10), part(imm["conservation"], 5),
                           part(imm["other"], 2)])
    inc = sponsor_income(holder, c, held)
    ck.eq(t, "income", [part(inc["money"], 10), part(inc["conservation"], 2), part(inc["other"], 2)])
    icon, any_zoo, money, appeal = SPONSOR_TRIGGERS.get(c["id"], (None, False, 0, 0))
    triggered = c["id"] in SPONSOR_TRIGGERS
    ck.eq(t, "trigger_icon", [i == icon for i in e["icons"]])
    ck.eq(t, "trigger_any", any_zoo)
    ck.eq(t, "trigger_gain", [money / 3, appeal / 2, triggered and not (money or appeal)])


def check_playable(ck, toks, legal, kind, data, prefix, mine, folders, side2, key):
    """The "can I play it now" flags against the rules: in a decision where cards are played, a card
    is a legal first atom exactly when its conditions, its place and its price are all fine."""
    if prefix:
        return
    table = cards()
    legal = set(legal)

    def flags(t):
        return t["req_ok"] == 1.0 and t["has_place"] == 1.0 and t["affordable"] == 1.0

    def each(type_name):
        for t in toks:
            if t.type != "card" or t.card() is None or table[t.card()]["type"] != type_name:
                continue
            if t["my_hand"] == 1.0:
                yield t, f"hand:{t.card()}", True
            elif t["display"] == 1.0:
                i = round(t["display_pos"] * 6) - 1
                yield t, f"display:{i}", i < folders

    if kind == "act_animals" and data["phase"] == "play":
        for t, atom, reachable in each("animal"):
            in_hand = atom.startswith("hand")
            expect = flags(t) and (in_hand or (reachable and side2("animals")))
            assert (atom in legal) == expect, f"animal {atom}: legal {atom in legal}, flags say {expect}"
            for name in ("req_ok", "has_place"):
                ck.mark("card", name, t[name])
    elif kind in ("act_sponsors", "paid_sponsor"):
        for t, atom, reachable in each("sponsor"):
            level = sponsor_level(mine, table[t.card()], key)
            if kind == "paid_sponsor":  # from the hand only, for its level in money
                if not atom.startswith("hand"):
                    continue
                expect = t["req_ok"] == 1.0 and t["has_place"] == 1.0 and mine["money"] >= level
            else:
                in_hand = atom.startswith("hand")
                expect = flags(t) and level <= data["budget"] and (in_hand or (reachable and side2("sponsors")))
            assert (atom in legal) == expect, f"sponsor {atom}: legal {atom in legal}, flags say {expect}"
            for name in ("req_ok", "has_place"):
                ck.mark("card", name, t[name])
    elif kind == "act_build" and side2("build") == ("build" in mine["upgraded"]):
        # A building is a legal first atom exactly when it has a spot (player.placeable), fits the
        # budget and the money (2 per space) and was not built in this action yet.
        placeable = next(t for t in toks if t.type == "player")["placeable"]
        built = {building_type(k, size) for k, size in data["built"]}
        for i, (name, size, _) in enumerate(BUILD_ACTION):
            expect = (placeable[i] == 1.0 and not data["normal_done"] and size <= data["budget"]
                      and 2 * size <= mine["money"] and i not in built)
            assert (f"build:{name}" in legal) == expect, f"build:{name}: legal {f'build:{name}' in legal}"
            ck.mark("player", "placeable", placeable[i])
    elif kind == "act_association":
        # The project task is open exactly when some project can be chosen; then every project that
        # can be supported at some level is a legal first atom.
        projects = [t for t in toks if t.type == "project"]
        atoms = {}
        for t in projects:
            if t["in_play"] == 1.0:
                atoms[f"project:{t.anchor - layout().anchors['project'][0]}"] = (t, True)
            elif t["my_hand"] == 1.0:
                atoms[f"hand:{t.card()}"] = (t, True)
            elif t["display"] == 1.0:
                i = round(t["display_pos"] * 6) - 1
                atoms[f"display:{i}"] = (t, i < folders and data["side2"] and t["affordable"] == 1.0)
        if any(a in legal for a in atoms):
            for atom, (t, reachable) in atoms.items():
                expect = bool(t["can_me"].any()) and reachable
                assert (atom in legal) == expect, f"project {atom}: legal {atom in legal}, can_me {t['can_me']}"
                ck.mark("project", "can_me", t["can_me"].any())


class Episode:
    """A heuristic episode (the selfplay sample path: features come from the per-seat states) replayed
    move by move in a ground-truth session with the same seed."""

    def __init__(self, seed, selector):
        self.seed, self.selector = seed, selector
        self.ep = engine.run_heuristic_episode(game_id=GAME, seed=seed, temperature=0.0, max_game_plies=3000,
                                               heuristic_selector=selector)
        self.gs = engine.GameSession(GAME, seed=seed)
        self.own = {p: own_slots(self.gs, p) for p in (0, 1)}  # both players' private slots (truth)
        self.tracker = HandTracker(self.own)
        self.actions = []
        self.rng_free = True  # until a move whose outcome draws on the step rng (see steps())

    def steps(self, stride=1):
        """Yields every stride-th decision before it is played (all of them are replayed)."""
        names, table, idx = atom_names(), cards(), card_index()
        for n, sample in enumerate(self.ep["samples"]):
            gs = self.gs
            me = sample["player"]
            assert gs.current_player == me, "the replay diverged from the episode"
            state = gs.get_state_dict()
            core = json.loads(state["core_json"])
            if n % stride == 0:
                legal_ids = gs.get_legal_actions()
                assert sorted(np.nonzero(np.asarray(sample["legal_mask"]))[0].tolist()) == sorted(legal_ids)
                yield n, sample, me, state, core, [names[a] for a in legal_ids]
            action = sample["action_id"]
            name = names[action]
            zoo_before = set(core["players"][me]["zoo_cards"])
            self.actions.append((me, action))
            gs.apply_action(action)
            self.own = {p: own_slots(gs, p) for p in (0, 1)}
            after = json.loads(gs.get_state_dict()["core_json"])
            revealed = core["stack"][-1]["kind"] == "waza_kind" or hollywood_reveals(core, after, me)
            self.tracker.update(self.own, core, me, name, revealed, harbor=state["prefix"].split()[-1:] == ["harbor"])
            # The replay in encode_state_for_perspective uses another step rng: it stays in step
            # until a random outcome (Pilfering a card, Scavenging's shuffle, a deck reshuffle, a
            # reveal from the deck).
            if not gs.is_terminal:
                played = set(after["players"][me]["zoo_cards"]) - zoo_before
                if name == "give_card" or not after["deck"] or len(after["deck"]) > len(core["deck"]) or any(
                        "SCAVENGING" in table[idx[c]]["abilities"] for c in played) or revealed:
                    self.rng_free = False


def check_episode(ck: Checker, seed, selector, perspective_every=20, stride=1):
    """Checks every stride-th decision of one episode; returns the number of decisions of the
    episode. Every `perspective_every` checked decisions (while the truth replay is rng-free) also
    encodes the same state from the ground truth for both players: the mover's must equal the
    selfplay sample bit for bit (so the stale hidden slots of the per-seat state never matter), the
    other player's is checked in full as well, and the two views must mirror each other."""
    episode = Episode(seed, selector)
    pending = None  # (decision, the levels the token of the project chosen there promised)
    for count, (n, sample, me, state, core, legal) in enumerate(episode.steps(stride)):
        own = episode.own
        toks = decode(sample["features"])
        ck.context = f"seed {seed} {selector} decision {n} frame {core['stack'][-1]['kind']}"
        check_state(ck, toks, me, core, own, state, episode.tracker, legal)
        prefix = state["prefix"].split()
        if pending is not None and pending[0] == n - 1:
            # A project was chosen as the first atom: the levels now on offer are those its token promised.
            offered = None  # stays unknown when the whole option was applied without asking
            if len(prefix) == 1:
                offered = {int(a.split(":")[1]) for a in legal if a.startswith("position:")}
            elif len(prefix) >= 2:
                offered = {int(prefix[1].split(":")[1])}  # a single level: chosen without asking
            if offered is not None:
                assert offered == pending[1], f"seed {seed} decision {n}: levels {offered}, token said {pending[1]}"
                ck.mark("project", "can_me", True)
        pending = None
        chosen = atom_names()[sample["action_id"]]
        if core["stack"][-1]["kind"] == "act_association" and not prefix:
            token = None
            for t in toks:
                if t.type != "project":
                    continue
                if (chosen == f"project:{t.anchor - layout().anchors['project'][0]}" and t["in_play"] == 1.0) or (
                        chosen == f"hand:{t.card()}" and t["my_hand"] == 1.0) or (
                        t["display"] == 1.0 and chosen == f"display:{round(t['display_pos'] * 6) - 1}"):
                    token = t
            if token is not None:
                pending = (n, {k + 1 for k in range(3) if token["can_me"][k] == 1.0})
        if episode.rng_free and count % perspective_every == 0:
            check_perspectives(ck, episode, sample, me, core, own, state, legal)
    return len(episode.ep["samples"])


def check_perspectives(ck, episode, sample, me, core, own, state, legal):
    views = {}
    for p in (0, 1):
        out = engine.encode_state_for_perspective(GAME, episode.seed, p, -1, list(episode.actions))
        views[p] = np.asarray(out["features"], dtype=np.float32)
    mover = np.asarray(sample["features"], dtype=np.float32)
    assert views[me].shape == mover.shape and np.array_equal(views[me], mover), (
        "the selfplay sample differs from the encoding of the ground-truth state")
    other = decode(views[1 - me])
    ck.context += " (other player's view)"
    check_state(ck, other, 1 - me, core, own, state, episode.tracker, legal)
    mine = decode(mover)

    def group(toks, type_name):
        return [t for t in toks if t.type == type_name]

    def same(a, b, skip):
        for name in layout().features[a.type]:
            if name not in skip:
                assert np.array_equal(a[name], b[name]), f"{a.type}.{name} differs between the two views"

    for a, b in zip(group(mine, "player"), reversed(group(other, "player"))):
        same(a, b, {"me", "opponent"})
    # An opponent hand card I saw says what it would do for the opponent: the same as that card in
    # the opponent's own view of its hand.
    held_there = {t.card(): t for t in other if t.type in ("card", "project") and t["my_hand"] == 1.0}
    for a in mine:
        if a.type not in ("card", "project") or a["opp_hand"] != 1.0 or a.card() is None:
            continue
        skip = {"my_hand", "opp_hand", "chosen", "chosen_discard"}
        if a.type == "project":
            skip |= {"can_me", "can_opp", "count_me", "count_opp"}
        same(a, held_there[a.card()], skip)
        ck.mark(a.type, "opp_hand", True)
    acts_a, acts_b = group(mine, "action_card"), group(other, "action_card")
    for a, b in zip(acts_a, acts_b[5:] + acts_b[:5]):
        same(a, b, {"mine", "opponent"})
    enc_a, enc_b = group(mine, "enclosure"), group(other, "enclosure")
    for flag_a, flag_b in (("mine", "opponent"), ("opponent", "mine")):
        for a, b in zip([t for t in enc_a if t[flag_a]], [t for t in enc_b if t[flag_b]]):
            same(a, b, {"mine", "opponent"})
    proj_a = {t.card(): t for t in group(mine, "project") if t["in_play"] or t["display"]}
    proj_b = {t.card(): t for t in group(other, "project") if t["in_play"] or t["display"]}
    assert proj_a.keys() == proj_b.keys()
    for c, a in proj_a.items():
        b = proj_b[c]
        for x, y in (("can_me", "can_opp"), ("can_opp", "can_me"), ("count_me", "count_opp"),
                     ("occupied_me", "occupied_opp"), ("occupied_opp", "occupied_me")):
            assert np.array_equal(a[x], b[y]), f"project {c}: my {x} is not the other view's {y}"
        ck.mark("project", "can_opp", a["can_opp"].any())
