"""Independent reference data for the Ark Nova encoder tests.

Nothing here goes through the encoder: card facts are parsed from the generated card table
(arknova_cards_data.inc), enum orders from the C++ headers, the zoo maps from arknova_maps.json,
and the game facts come from the UI serialization (core_json) or are recomputed in Python from the
rulebook. tests/arknova decode every token of an encoding and compare it with these.
"""
from __future__ import annotations

import collections
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

import dinoboard_engine as engine

GAME = "arknova"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "games" / "arknova"
ACTIONS = ["animals", "build", "cards", "association", "sponsors"]
TASKS = ["reputation", "partner_zoo", "university", "project"]


# ---- enums and tables parsed from the C++ sources -----------------------------------------------

def _enum(text: str, name: str) -> list[str]:
    body = re.search(r"enum (?:class )?" + name + r"\b[^{]*\{(.*?)\}", text, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return [item.split("=")[0].strip() for item in body.split(",") if item.strip()]


@lru_cache(maxsize=1)
def enums() -> dict:
    cards_h = (SRC / "arknova_cards.h").read_text()
    map_h = (SRC / "arknova_map.h").read_text()
    core_h = (SRC / "arknova_core.h").read_text()
    frames = re.search(r"#define ARKNOVA_FRAME_KINDS\(X\)(.*?)\n\n", core_h, re.S).group(1)
    out = {
        "icons": [n[len("kIcon_"):] for n in _enum(cards_h, "Icon") if n != "kNumIcons"],
        "continents": [n[len("kCont_"):] for n in _enum(cards_h, "Continent") if n != "kNumContinents"],
        "universities": re.search(r"kUniversityNames\[\] = \{(.*?)\}", cards_h).group(1).replace('"', "").split(", "),
        "req_extra": [n for n in _enum(cards_h, "ReqExtra") if n != "kNumReqs"],
        "keywords": [n[1:] for n in _enum(cards_h, "Keyword") if n != "kNone"],
        "proj_icons": [n for n in _enum(cards_h, "ProjIcon") if n != "kNone"],
        "proj_categories": [n for n in _enum(cards_h, "ProjCategory") if n != "kNone"],
        "special": [n for n in _enum(cards_h, "SpecialKind") if n != "kNone"],
        "bonus_tiles": re.search(r"kBonusTileNames\[\] = \{(.*?)\}", cards_h, re.S).group(1).replace('"', "").replace("\n", " ").split(","),
        "placement_bonus": [t.strip() for t in re.search(r"kPlacementBonusNames\[\] = \{(.*?)\}", cards_h, re.S)
                            .group(1).replace('"', "").replace("\n", " ").split(",")],
        "effect_kinds": [t.strip() for t in re.search(r"kEffectKindNames\[\] = \{(.*?)\}", cards_h, re.S)
                         .group(1).replace('"', "").replace("\n", " ").split(",")],
        "building_kinds": re.search(r"kBuildingKindNames\[\] = \{(.*?)\}", map_h, re.S).group(1).replace('"', "").replace("\n", " ").split(","),
        "frames": ["break" if f == "break_" else f for f in re.findall(r"X\((\w+)\)", frames)],
    }
    out["bonus_tiles"] = [t.strip() for t in out["bonus_tiles"]]
    out["building_kinds"] = [t.strip() for t in out["building_kinds"]]
    out["reqs"] = out["icons"] + out["req_extra"]
    assert len(out["icons"]) == 15 and len(out["frames"]) == 42 and len(out["keywords"]) == 44
    return out


BUILDING_TYPES = ["enclosure1", "enclosure2", "enclosure3", "enclosure4", "enclosure5", "kiosk", "pavilion",
                  "petting_zoo", "reptile_house", "aviary", "unique"]


def building_type(kind: str, size: int) -> int:
    return BUILDING_TYPES.index(f"enclosure{size}" if kind == "enclosure" else kind)


# ---- the card table, parsed from the generated data ---------------------------------------------

@lru_cache(maxsize=1)
def cards() -> list[dict]:
    text = (SRC / "arknova_cards_data.inc").read_text()
    n = int(re.search(r"kNumCards = (\d+)", text).group(1))

    def blank():
        return dict(tags=collections.Counter(), req=collections.Counter(), continent_icons=collections.Counter(),
                    continents=set(), abilities={}, special={}, slots=[dict(rewards={}) for _ in range(3)],
                    scoring={}, play_bonus={}, building=None, lasting_effect=False, enc_standard=-1,
                    cost=0, size=0, appeal=0, conservation=0, reputation=0, level=0, enc_rock=0, enc_water=0,
                    min_reputation=-1, max_appeal=-1, proj_category=None, proj_icon=None)

    out = [blank() for _ in range(n)]
    for m in re.finditer(r"(\+\+)?c\[(\d+)\]\.([\w\.\[\]]+)(?: (\|?=) (.*?))?;(?=\s|$)", text):
        inc, idx, field, _, value = m.groups()
        c = out[int(idx)]
        if inc:
            name, key = re.match(r"(\w+)\[(\w+)\]", field).groups()
            c[name][key.split("_", 1)[1] if key.startswith(("kIcon_", "kCont_")) else key] += 1
        elif field == "continents":
            c["continents"].add(value.split("kCont_")[1])
        elif field in ("id", "cost", "size", "appeal", "conservation", "reputation", "level", "enc_standard",
                       "enc_rock", "enc_water", "min_reputation", "max_appeal", "n_abilities", "n_special", "n_slots",
                       "n_scoring", "n_play_bonus"):
            c[field] = int(value)
        elif field == "type":
            c["type"] = value.split("::k")[1].lower()
        elif field == "name":
            c["name"] = value.strip('"')
        elif field in ("human", "lasting_effect"):
            c[field] = value == "true"
        elif field in ("proj_category", "proj_icon"):
            c[field] = value.split("::")[1]
        elif field == "building":
            head, shape, tail = re.match(r"\{true, (\d+), \{(.*)\}, (.*)\}", value).groups()
            size, rock, water, min_border, adjacency = [t.strip() for t in tail.split(",")]
            c["building"] = dict(n=int(head), size=int(size), rock=int(rock), water=int(water),
                                 min_border=int(min_border), needs_adjacency=adjacency == "true",
                                 shape=[tuple(int(x) for x in p.split(",")) for p in re.findall(r"\{(-?\d+, -?\d+)\}", shape)])
        elif field.startswith("abilities["):
            kw, v = re.match(r"\{Keyword::k(\w+), (-?\w+)\}", value).groups()
            # The number X of "keyword X" (-1: none); Iconic Animal names a continent instead.
            c["abilities"][kw] = v.split("kCont_")[1] if v.startswith("kCont_") else int(v)
        elif field.startswith("special["):
            kind, v = re.match(r"\{SpecialKind::(\w+), (\d+)\}", value).groups()
            c["special"][kind] = int(v)
        elif field.startswith("play_bonus["):
            kind, v = re.match(r"\{EffectKind::kEff(\w+), (\d+)\}", value).groups()
            c["play_bonus"][kind.lower()] = c["play_bonus"].get(kind.lower(), 0) + int(v)
        elif field.startswith("scoring["):
            t, v = re.match(r"\{(\d+), (\d+)\}", value).groups()
            c["scoring"][int(field[8])] = (int(t), int(v))
        elif field.startswith("slots["):
            k = int(field[6])
            sub = field.split(".", 1)[1]
            if sub.startswith("rewards["):
                kind, v, _ = re.match(r"\{Reward::kReward(\w+), (\d+), RewardReq::(\w+)\}", value).groups()
                c["slots"][k]["rewards"][kind.lower()] = c["slots"][k]["rewards"].get(kind.lower(), 0) + int(v)
            else:
                c["slots"][k][sub] = int(value)
        else:
            raise AssertionError(f"card table statement not understood: {m.group(0)}")
    for c in out:
        # Icons a card adds to a zoo: its tags, plus its rock / water requirement icons (rulebook p. 15).
        c["icons"] = collections.Counter(c["tags"])
        for t in ("rock", "water"):
            if c["enc_" + t] and not c["tags"][t]:
                c["icons"][t] += c["enc_" + t]
        c["id_str"] = str(c["id"])
    return out


@lru_cache(maxsize=1)
def card_index() -> dict:
    return {c["id_str"]: i for i, c in enumerate(cards())}


def is_small(c): return c["type"] == "animal" and c["size"] <= 2
def is_large(c): return c["type"] == "animal" and c["size"] >= 4


# ---- zoo maps --------------------------------------------------------------------------------------

DIRS = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]


@lru_cache(maxsize=1)
def zoo_maps() -> dict:
    """The maps by key, from arknova_maps.json (written with the C++ table by tools/gen_maps.py from
    ark-nova-ai). Cells in engine order (sorted by axial (q, r)); every map has the same 58 cells."""
    raw = json.loads((SRC / "arknova_maps.json").read_text())["maps"]
    out = {}
    for m in raw:
        cells = {(q, r): dict(terrain=t, bonus=b or "", build2=b2) for q, r, t, b, b2 in m["cells"]}
        order = sorted(cells)
        assert len(order) == 58
        for (q, r), c in cells.items():
            c["nbr"] = [(q + dq, r + dr) for dq, dr in DIRS if (q + dq, r + dr) in cells]
            c["border"] = len(c["nbr"]) < 6
            c["rock"] = sum(cells[n]["terrain"] == "rock" for n in c["nbr"])
            c["water"] = sum(cells[n]["terrain"] == "water" for n in c["nbr"])
        out[m["key"]] = dict(m, cells=cells, order=order, index={qr: i for i, qr in enumerate(order)},
                             rule_cells={tuple(c) for c in m["rule_cells"]})
    assert len({tuple(m["order"]) for m in out.values()}) == 1
    return out


def zoo_map(key: str) -> dict:
    return zoo_maps()[key]


def neighbors(cellset) -> set:
    m = zoo_map("map0")["cells"]  # the grid, the same on every map
    return {n for qr in cellset for n in m[qr]["nbr"]} - set(cellset)


def adjacent_terrain(cellset, terrain: str, key: str) -> int:
    m = zoo_map(key)["cells"]
    return sum(m[n]["terrain"] == terrain for n in neighbors(cellset))


# What a placement bonus gives (the C++ kPlacementBonusNames; rulebook p. 9 and the map sheets).
PLACEMENT_BONUS_EFFECT = {
    "money5": ("money", 5), "money10": ("money", 10), "money2": ("money", 2), "xtoken": ("xtoken", 1),
    "rep2": ("reputation", 2), "rep1": ("reputation", 1), "card": ("card", 1), "clever": ("clever", 1),
    "kiosk": ("kiosk", 1), "worker1": ("worker", 1), "partner_zoo": ("partner_zoo", 1),
    "university": ("university", 1), "multiplier": ("multiplier", 1), "sponsor_for_money": ("sponsor_for_money", 1),
}


def map_rule_state(pl, key: str) -> dict:
    """Where the map's special rule stands for a zoo (the map sheets): the break income from the
    park restaurant (1 per covered space next to it) or the ice cream parlours (1 per kiosk once all
    three are covered), the harbour (discard a card for 3 money once a turn), the research institute
    (one condition may be unmet), Hollywood (sponsor levels 1 lower once all H spaces are covered)."""
    m = zoo_map(key)
    covered = {tuple(c) for b in pl["buildings"] for c in b["cells"]}
    special = m["rule_cells"]
    rule = m["rule"]
    out = dict(map_income=0, can_trade=False, ignore_conditions=0, sponsor_discount=0, group_left=0)
    if rule == "parkrestaurant":
        out["map_income"] = sum(n in covered for r in special for n in m["cells"][r]["nbr"])
    if rule == "icecream" and special <= covered:
        out["map_income"] = sum(b["kind"] == "kiosk" for b in pl["buildings"])
    out["can_trade"] = rule == "harbor" and bool(special & covered)
    out["ignore_conditions"] = int(rule == "researchinstitute" and bool(special & covered))
    out["sponsor_discount"] = int(rule == "hollywood" and special <= covered)
    if rule in ("icecream", "hollywood"):
        out["group_left"] = len(special - covered)
    return out


def effect_vector(effect) -> list:
    """The encoder's effect block: per effect kind its amount, money / 10, enclosure size / 5, else / 3."""
    kinds = enums()["effect_kinds"]
    v = [0.0] * len(kinds)
    if effect:
        kind, amount = effect
        v[kinds.index(kind)] = amount / (10 if kind == "money" else 5 if kind == "enclosure" else 3)
    return v


def building_cells(b) -> list:
    return [tuple(c) for c in b["cells"]]


def kiosk_income_of(buildings, k) -> int:
    """Rulebook p. 18: 1 money per adjacent unique building, special enclosure, occupied standard
    enclosure and pavilion."""
    near = neighbors(building_cells(buildings[k]))
    n = 0
    for b in buildings:
        if b is buildings[k] or not near & set(building_cells(b)):
            continue
        n += b["kind"] in ("pavilion", "unique", "petting_zoo", "reptile_house", "aviary") or (
            b["kind"] == "enclosure" and b["occupied"])
    return n


# ---- rulebook numbers ------------------------------------------------------------------------------

def target_appeal(conservation: int) -> int:
    c = max(conservation, 0)
    return 100 - (2 * c - 14 if c <= 10 else 6 + 3 * (c - 10))


APPEAL_INCOME = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (7, 11), (9, 12), (11, 13), (13, 14), (15, 15),
                 (17, 16), (20, 17), (23, 18), (26, 19), (29, 20), (32, 21), (36, 22), (40, 23), (44, 24), (48, 25),
                 (52, 26), (56, 27), (61, 28), (66, 29), (71, 30), (76, 31), (81, 32), (86, 33), (91, 34), (96, 35),
                 (102, 36), (108, 37)]


def appeal_income(appeal: int) -> int:
    return max(v for t, v in APPEAL_INCOME if appeal >= t or t == 0)


def reputation_folders(reputation: int) -> int:
    return next((i + 1 for i, top in enumerate([1, 3, 6, 9, 12, 15]) if reputation <= top), 6)


DONATION_COSTS = [2, 5, 7, 10]


def zoo_icons(pl) -> collections.Counter:
    """Icons of a zoo: its cards, 1 per partner zoo continent, the universities' science icons."""
    table, idx = cards(), card_index()
    icons = collections.Counter()
    for cid in pl["zoo_cards"]:
        icons.update(table[idx[cid]]["icons"])
    for cont in pl["partner_zoos"]:
        icons[cont] += 1
    icons["science"] += 2 * ("science2" in pl["universities"]) + ("science_rep" in pl["universities"])
    return icons


def unmet_requirements(pl, c) -> int:
    """Number of a card's conditions (left side of the card) the zoo does not meet."""
    icons = zoo_icons(pl)
    unmet = 0
    for req, n in c["req"].items():
        if req == "kReqAnimalsII":
            ok = "animals" in pl["upgraded"]
        elif req == "kReqSponsorsII":
            ok = "sponsors" in pl["upgraded"]
        elif req == "kReqPartnerZoo":
            ok = any(z in c["continents"] for z in pl["partner_zoos"]) if c["continents"] else len(pl["partner_zoos"]) >= n
        elif req == "kReqReputation":
            ok = pl["reputation"] >= c["min_reputation"]
        elif req == "kReqAppeal":
            ok = pl["appeal"] <= c["max_appeal"]
        else:
            ok = icons[req] >= n
        unmet += not ok
    return unmet


def animal_cost(pl, c) -> int:
    """Printed cost, less 3 per continent icon with a partner zoo there (rulebook p. 11), less the
    Expert in Small / Large Animals discounts (sponsors 229 / 230)."""
    cost = c["cost"] - 3 * sum(n for cont, n in c["continent_icons"].items() if cont in pl["partner_zoos"])
    cost -= 3 * (is_small(c) and "229" in pl["zoo_cards"]) + 4 * (is_large(c) and "230" in pl["zoo_cards"])
    return max(0, cost)


def action_strength(pl, action: str) -> int:
    """Strength an action card would be used with now: its slot, less 2 under a Constriction token."""
    return pl["slots"].index(action) + 1 - 2 * ("constriction" in pl["action_tokens"].get(action, []))


def sponsor_level(pl, c, key: str) -> int:
    """A sponsor's level for pl: 1 lower on Hollywood Hills once all its H spaces are covered."""
    return max(0, c["level"] - map_rule_state(pl, key)["sponsor_discount"])


def strength_short(pl, c, key: str) -> float:
    """Strength points pl's action card lacks for card c now, X-tokens aside, / 5 (capped at 5).
    Animals I plays an animal from strength 2, Animals II from 1; Sponsors plays levels up to its
    strength, plus 1 on side II; supporting a project takes Association 5 (4 with the Veterinarian,
    sponsor 203)."""
    if c["type"] == "animal":
        action, needed = "animals", 1 if "animals" in pl["upgraded"] else 2
    elif c["type"] == "sponsor":
        action, needed = "sponsors", max(1, sponsor_level(pl, c, key) - ("sponsors" in pl["upgraded"]))
    elif c["type"] == "project":
        action, needed = "association", 4 if "203" in pl["zoo_cards"] else 5
    else:
        return 0.0
    return min(max(needed - action_strength(pl, action), 0), 5) / 5


# Buildings of the Build action (rulebook p. 9): name, size, side II only.
BUILD_ACTION = [("enclosure1", 1, False), ("enclosure2", 2, False), ("enclosure3", 3, False),
                ("enclosure4", 4, False), ("enclosure5", 5, False), ("kiosk", 1, False), ("pavilion", 1, False),
                ("petting_zoo", 3, False), ("reptile_house", 5, True), ("aviary", 5, True)]


def project_count(pl, icon: str) -> int:
    table, idx = cards(), card_index()
    icons = zoo_icons(pl)
    e = enums()
    if icon == "kPIconAllAnimals":
        return sum(icons[i] > 0 for i in e["icons"][:7])
    if icon == "kPIconAllContinents":
        return sum(icons[i] > 0 for i in e["continents"])
    if icon in ("kPIconAnimalSize2", "kPIconAnimalSize4"):
        zoo = [table[idx[cid]] for cid in pl["zoo_cards"]]
        return sum(is_small(c) if icon.endswith("2") else is_large(c) for c in zoo)
    return icons[icon[len("kPIcon_"):]]


def map_facts(pl, key: str) -> dict:
    """What the end-of-game and sponsor effects count on a zoo map. A space is connected when a
    building covers a space next to it; "empty" spaces are plain spaces without a building."""
    cells = zoo_map(key)["cells"]
    covered = {tuple(c) for b in pl["buildings"] for c in b["cells"]}

    def near(qr):
        return any(n in covered for n in cells[qr]["nbr"])

    empty = [qr for qr, c in cells.items() if c["terrain"] == "plain" and qr not in covered]
    out = dict(empty=len(empty), fully_covered=not empty,
               border_connected=sum(cells[qr]["border"] and near(qr) for qr in empty),
               all_border_connected=all(near(qr) for qr in empty if cells[qr]["border"]))
    for t in ("water", "rock"):
        spaces = [qr for qr, c in cells.items() if c["terrain"] == t]
        out[t] = sum(near(qr) for qr in spaces)
        out[t + "_unconnected"] = len(spaces) - out[t]
    bonus = [qr for qr in empty if cells[qr]["bonus"]]
    out["bonus"] = sum(near(qr) for qr in bonus)
    out["bonus_unconnected"] = len(bonus) - out["bonus"]
    return out


def final_card_points(core, p: int, card_id: int) -> int:
    """Conservation points a final scoring card would give player p now (at most 4)."""
    table, idx = cards(), card_index()
    pl = core["players"][p]
    zoo = [table[idx[cid]] for cid in pl["zoo_cards"]]
    icons = zoo_icons(pl)
    facts = map_facts(pl, core["map"])
    count = {
        1: sum(is_large(c) for c in zoo), 2: sum(is_small(c) for c in zoo), 3: icons["science"],
        5: sum(not t for t in pl["tokens"]), 6: facts["empty"],
        7: pl["reputation"], 8: sum(c["type"] == "sponsor" for c in zoo), 10: icons["rock"], 11: icons["water"],
    }
    if card_id == 4:  # Architectural Zoo: 1 point per condition met
        return (facts["water_unconnected"] == 0) + (facts["rock_unconnected"] == 0) + facts["all_border_connected"] + \
            facts["fully_covered"]
    if card_id == 9:
        theirs = zoo_icons(core["players"][1 - p])
        return min(4, sum(icons[i] > theirs[i] for i in enums()["icons"][:7]))
    points = 0
    for threshold, value in table[idx[str(card_id)]]["scoring"].values():
        if count[card_id] >= threshold:
            points = value
    return min(points, 4)


# ---- sponsor effects, typed from the printed card texts (not from the engine's effect code) ------

# Icon triggers: printed id -> (icon, plays into any zoo count, money per icon, appeal per icon); a
# trigger with neither gives something else per icon (a track step, an X-token, a free building, a
# card, an ability). Explorer (262) counts icons new to the zoo, so it has no icon.
SPONSOR_TRIGGERS = {
    208: ("science", True, 2, 0), 236: ("primate", True, 3, 0), 237: ("reptile", True, 3, 0),
    238: ("bird", True, 3, 0), 239: ("predator", True, 3, 0), 240: ("herbivore", True, 3, 0),
    251: ("bear", True, 0, 2), 243: ("herbivore", False, 0, 2), 244: ("bird", False, 0, 2),
    245: ("water", False, 0, 2), 246: ("rock", False, 0, 2), 247: ("primate", False, 0, 2),
    202: ("science", False, 0, 0), 204: ("science", False, 0, 0), 248: ("primate", False, 0, 0),
    210: ("americas", False, 0, 0), 211: ("europe", False, 0, 0), 212: ("australia", False, 0, 0),
    213: ("asia", False, 0, 0), 214: ("africa", False, 0, 0), 249: ("bird", False, 0, 0),
    250: ("reptile", False, 0, 0), 252: ("predator", False, 0, 0), 253: ("herbivore", False, 0, 0),
    262: (None, False, 0, 0),
}
CATEGORY_SPONSORS = {231: "primate", 232: "reptile", 233: "bird", 234: "predator", 235: "herbivore"}


def _distinct(icons, continents=True, categories=True) -> int:
    e = enums()
    return sum(icons[i] > 0 for i in (e["icons"][:7] if categories else []) + (e["continents"] if continents else []))


def sponsor_immediate(pl, c, key: str) -> dict:
    """What sponsor c's immediate effect gives pl when played now (the card already counts as in
    the zoo): money, appeal, conservation and other gains (cards to take, workers, free buildings)."""
    icons = zoo_icons(pl) + c["icons"]
    facts = map_facts(pl, key)
    out = dict(money=0, appeal=0, conservation=0, other=0)
    cid = c["id"]
    if cid in CATEGORY_SPONSORS:
        out["appeal"] = icons[CATEGORY_SPONSORS[cid]]
    elif cid in (201, 254):
        out["other"] = 1
    elif cid == 216 and pl["workers"] < 4:  # the 4th worker brings what the map prints next to it
        out["other"] = 1
        last = zoo_map(key)["last_worker"]
        assert last is None or last[0] == "conservation"
        out["conservation"] = last[1] if last and pl["workers"] == 3 else 0
    elif cid == 207:
        out["conservation"] = _distinct(icons) // 2
    elif cid == 222:
        out["conservation"] = min(3, icons["science"])
    elif cid == 208:
        out["appeal"] = icons["science"]
    elif cid == 241:
        out["appeal"] = icons["water"]
    elif cid == 242:
        out["appeal"] = 3 * (icons["rock"] // 2)
    elif cid in (258, 259, 260, 264):
        out["appeal"] = facts[{258: "water", 259: "rock", 260: "border_connected", 264: "bonus"}[cid]]
    elif cid == 262:
        out["money"] = 2 * _distinct(icons)
    elif cid == 219:
        out["money"] = 2 * (icons["water"] + icons["rock"])
    elif cid == 263:
        out["other"] = 1
    return out


def sponsor_income(pl, c, in_zoo: bool) -> dict:
    """What sponsor c pays pl at a break (as if it were in pl's zoo when it is not): money,
    conservation and other gains (X-tokens, cards)."""
    icons = zoo_icons(pl) + (collections.Counter() if in_zoo else c["icons"])
    out = dict(money=0, conservation=0, other=0)
    cid = c["id"]
    if cid in CATEGORY_SPONSORS:
        n = icons[CATEGORY_SPONSORS[cid]]
        out["money"] = 9 if n >= 5 else 6 if n >= 3 else 3 if n >= 1 else 0
    elif cid == 201:
        out["other"] = 1
    elif cid == 206:
        out["conservation"] = 1
    elif cid == 209:
        out["other"] = int(pl["xtokens"] < 5)
    elif cid == 220:
        out["money"] = 3
    elif cid == 257:  # 2 money per building next to the side entrance, except empty standard enclosures
        entrance = next((b for b in pl["buildings"] if b["name"] == "257"), None)
        if entrance is not None:
            near = neighbors(building_cells(entrance))
            out["money"] = 2 * sum(b is not entrance and bool(near & set(building_cells(b))) and
                                   not (b["kind"] == "enclosure" and not b["occupied"]) for b in pl["buildings"])
    return out


def sponsor_endgame(pl, c, key: str) -> tuple:
    """(conservation, appeal) sponsor c's end-of-game effect gives pl now."""
    icons, facts, cid = zoo_icons(pl), map_facts(pl, key), c["id"]
    steps = {201: "science", 251: "bear"}
    if cid in steps:
        n = icons[steps[cid]]
        return (2 if n >= 6 else 1 if n >= 3 else 0), 0
    own = {243: "herbivore", 244: "bird", 245: "water", 246: "rock", 247: "primate"}
    if cid in own:
        return int(icons[own[cid]] >= 6), 0
    if cid in (208, 261):
        return int(_distinct(icons, continents=False) >= 5), 0
    if cid in (225, 226):
        return int(_distinct(icons, categories=False) >= 5), 0
    if cid in (215, 218):
        return int(sum(not t for t in pl["tokens"]) >= 5), 0
    if cid in (241, 242):
        return int(facts[("water" if cid == 241 else "rock") + "_unconnected"] == 0), 0
    if cid == 257:
        return 0, 5 * facts["fully_covered"]
    if cid in (258, 259, 264):
        return facts[{258: "water", 259: "rock", 264: "bonus"}[cid] + "_unconnected"] // 2, 0
    if cid == 260:
        return facts["empty"] // 6, 0
    return 0, 0


# ---- encoder layout and token decoding -------------------------------------------------------------

class Layout:
    def __init__(self, text: str):
        self.raw = json.loads(text)
        self.header = self.raw["header"]
        self.token_dim = self.raw["token_dim"]
        self.type_names = [t["name"] for t in self.raw["types"]]
        self.features = {t["name"]: {n: (o, w) for n, o, w in t["features"]} for t in self.raw["types"]}
        self.anchors = {n: (base, count) for n, base, count in self.raw["anchors"]}
        self.entities = {n: (base, count) for n, base, count in self.raw["entities"]}

    def anchor_name(self, anchor_id: int):
        for name, (base, count) in self.anchors.items():
            if base <= anchor_id < base + count:
                return name, anchor_id - base
        raise AssertionError(f"anchor id {anchor_id} outside every range")


@lru_cache(maxsize=1)
def layout() -> Layout:
    return Layout(engine.GameSession(GAME, seed=1).get_state_dict()["encoder_layout"])


class Token:
    """One decoded token. tok["name"] is the feature's values (a float for width 1)."""

    def __init__(self, row: np.ndarray, lay: Layout):
        assert row[0] == 1.0
        self.entity, self.anchor, self.type_id = int(row[1]), int(row[2]), int(row[3])
        self.type = lay.type_names[self.type_id]
        self.pos = (int(row[5]), int(row[6])) if row[4] == 1.0 else None
        assert row[4] in (0.0, 1.0) and (row[4] == 1.0 or (row[5] == 0 and row[6] == 0))
        self.f = row[lay.header:]
        self._features = lay.features[self.type]
        used = max(o + w for o, w in self._features.values())
        assert not self.f[used:].any(), f"{self.type} token has values beyond its {used} features"

    def __getitem__(self, name: str):
        o, w = self._features[name]
        return float(self.f[o]) if w == 1 else self.f[o:o + w]

    def card(self):
        base, count = layout().entities["card"]
        return self.entity - base if base <= self.entity < base + count else None


def decode(features) -> list[Token]:
    lay = layout()
    rows = np.asarray(features, dtype=np.float32).reshape(-1, lay.token_dim)
    return [Token(r, lay) for r in rows]


@lru_cache(maxsize=1)
def atom_names() -> list[str]:
    gs = engine.GameSession(GAME, seed=1)
    return [gs.get_action_info(a)["name"] for a in range(engine.game_metadata(GAME)["action_space"])]


@lru_cache(maxsize=1)
def atom_index() -> dict:
    return {n: i for i, n in enumerate(atom_names())}


def own_slots(gs, p: int) -> dict:
    """Player p's private cards (hand, final cards, tucked, private frame lists, final cards it put
    back under the final deck), as card indices."""
    v = gs.extract_initial_observation(p)["tracker_init"]["own_slots"]
    n = len(cards())
    assert len(v) == 298
    return dict(hand={c for c in range(n) if v[c] == 1}, final=[c for c in v[240:246] if c >= 0],
                tucked=[c for c in v[246:270] if c >= 0],
                private=[[c for c in v[270 + 6 * k:276 + 6 * k] if c >= 0] for k in range(4)],
                final_known=[c for c in v[294:298] if c >= 0])


# ---- card pools: the discard pile and the cards a player has not seen -----------------------------

POOL_FEATURES = {"type": 3, "animal_size": 5, "icons": 15, "sponsor_level": 4, "project_category": 4, "appeal": 1,
                 "conservation": 1}


def _pool_counts(c) -> dict:
    """What one regular card counts for in each pool column: type, size of an animal, the icons it
    adds to a zoo, level of a sponsor (3 or less, 4, 5, 6 or more), category of a project, and an
    animal's appeal and conservation points."""
    e = enums()
    out = {name: [0] * width for name, width in POOL_FEATURES.items()}
    out["icons"] = [c["icons"][i] for i in e["icons"]]
    if c["type"] == "animal":
        out["type"][0] = 1
        out["animal_size"][c["size"] - 1] = 1
        out["appeal"], out["conservation"] = [c["appeal"]], [c["conservation"]]
    elif c["type"] == "sponsor":
        out["type"][1] = 1
        out["sponsor_level"][min(max(c["level"], 3), 6) - 3] = 1
    else:
        out["type"][2] = 1
        out["project_category"][e["proj_categories"].index(c["proj_category"])] = 1
    return out


def regular_cards() -> list[int]:
    """Animals, sponsors and projects (everything but the final scoring cards)."""
    return [i for i, c in enumerate(cards()) if c["type"] != "endgame"]


def final_scoring_cards() -> list[int]:
    return [i for i, c in enumerate(cards()) if c["type"] == "endgame"]


def _pool_sum(indices) -> dict:
    table = cards()
    out = {name: np.zeros(width) for name, width in POOL_FEATURES.items()}
    for i in indices:
        assert table[i]["type"] != "endgame"
        for name, v in _pool_counts(table[i]).items():
            out[name] += v
    return out


def pool_composition(indices) -> dict:
    """Pool column -> its values for a set of regular cards: each count (or sum) over the same
    count among all regular cards of the game."""
    total = _pool_sum(regular_cards())
    out = {name: np.divide(v, total[name], out=np.zeros_like(v), where=total[name] > 0)
           for name, v in _pool_sum(indices).items()}
    return {name: float(v[0]) if len(v) == 1 else v for name, v in out.items()}


def known_to(me: int, core, own, shown_opp) -> set:
    """Card indices whose place player `me` knows: the public zones of the serialized state, its own
    private slots and the opponent's hand cards it has seen."""
    idx = card_index()
    public = list(core["discard"]) + [c for c in core["display"] if c is not None] + core["base_projects"] + \
        core["base_project_deck"] + core["projects"]
    for pl in core["players"]:
        public += pl["zoo_cards"]
    for f in core["stack"]:
        if f["kind"] == "ab_hunter":
            public += f["data"]["cards"]
    mine = own[me]
    return ({idx[c] for c in public} | mine["hand"] | set(mine["final"]) | set(mine["tucked"]) |
            {c for frame in mine["private"] for c in frame} | set(mine["final_known"]) | set(shown_opp))
