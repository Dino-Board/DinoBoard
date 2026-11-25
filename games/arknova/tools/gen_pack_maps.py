"""Add maps 9-14 (Zoo Map Packs 1 and 2) to arknova_maps.json and arknova_maps_data.inc.

Usage: python games/arknova/tools/gen_pack_maps.py   (after gen_maps.py and gen_alt_maps.py)

The spaces and side panels come from games/arknova/bga_pack_maps.json, Board Game Arena's MAPS_DATA,
as (column x, doubled row) spaces: axial = (x, row // 2 - (x + 1) // 2), as in gen_alt_maps.py.
Board Game Arena keys a side-panel bonus by the count reached: universities and partner zoos
("2": the second), association workers ("1": the second worker), and the linked bonuses of Zoo Map
Pack 2 ("1": once both 1 partner zoo and 1 university). What the maps' pictures show and MAPS_DATA
does not:
  * Geographical Zoo (9): the continent areas, read off BGA's map-9 picture (each area in the
    colour BGA gives the continent's marker slot: africa #fdf202, europe #8fdcfa, asia #b1e3ae,
    americas #f4a33b, australia #f03c55). Its 5 bonuses are the rule's (kGeographical).
  * Drawing Board (13): the size-2 enclosure at the centre (BGA's map-13 picture), and its 4
    areas, the plain spaces the rock and water spaces separate, ordered clockwise from the top
    as BGA lists their bonuses (3 money, 2 appeal, 1 reputation, Hunter 4).
  * Lagoon (14): every new worker finds a person sponsor (the rule), so its worker bonuses keep only
    the conservation point of the 4th.
The maps are not dealt in self-play (selfplay false) until their rules are complete.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_alt_maps  # noqa: E402
import gen_maps as G  # noqa: E402

PACK = Path(__file__).resolve().parent.parent / "bga_pack_maps.json"
RULES = {"9": "kGeographical", "10": "kRescue", "11": "kCaves", "12": "kAI", "13": "kDrawingBoard",
         "14": "kLagoon"}
# BGA placement bonus -> DinoBoard's name.
BONUS = {("xtoken", 1): "xtoken", ("money", 5): "money5", ("money", 10): "money10", ("money", 2): "money2",
         ("take-in-range-or-deck", 1): "card", ("Clever", 1): "clever", ("reputation", 1): "rep1",
         ("reputation", 2): "rep2", ("Partner-Zoo", 1): "partner_zoo", ("bonus-sponsor", 1): "sponsor_for_money",
         ("kiosk", 1): "kiosk", ("Fac", 1): "university", ("Multiplier", 1): "multiplier", ("add-worker", 1): "worker1",
         ("map10", 1): "rescue", ("STORE", 1): "store", ("ExtraShift", 1): "extra_shift", ("strength", 1): "strength",
         ("Mark", 1): "mark", ("appeal", 1): "appeal1", ("wave", 1): "wave", ("SharkAttack", 1): "shark_attack",
         ("Adapt", 1): "adapt"}
# BGA side-panel bonus -> DinoBoard effect.
EFFECT = {"money": "money", "reputation": "reputation", "xtoken": "xtoken", "conservation": "conservation",
          "upgrade-card": "upgrade", "add-worker": "worker", "size-2": "enclosure", "bonus-sponsor": "sponsor_for_money",
          "take-in-range-or-deck": "card", "strength": "strength", "ExtraShift": "extra_shift", "Adapt": "adapt",
          "CutDown": "cut_down", "AnimalMagnet": "animal_magnet", "map9": "continent_marker",
          "sponsor-person-card": "person_sponsor"}
AREAS9 = {  # BGA x_row spaces of each continent area, in Continent order
    "africa": ["3_8", "2_9", "1_10", "3_10", "2_11", "3_12"],
    "americas": ["0_3", "1_4", "0_5", "2_5", "1_6", "0_7"],
    "asia": ["8_3", "7_4", "6_5", "8_5", "7_6", "8_7"],
    "australia": ["5_8", "6_9", "5_10", "7_10", "6_11", "5_12"],
    "europe": ["3_0", "5_0", "4_1", "3_2", "5_2", "4_3"],
}
PREBUILT = {"13": [["enclosure", 2, ["4_5", "4_7"]]]}


def axial(uid):
    x, row = map(int, uid.split("_"))
    return (x, row // 2 - (x + 1) // 2)


def effect(spec, where):
    (name, amount), = spec.items()
    G.need(name in EFFECT, f"{where}: unknown bonus {spec}")
    kind = EFFECT[name]
    return [kind, 2 if kind == "enclosure" else int(amount)]


def slots(panel, n, where, skip=()):
    out = [None] * n
    for k, spec in (panel or {}).items():
        spec = {a: b for a, b in spec.items() if a not in skip}
        if not spec:
            continue
        G.need(1 <= int(k) <= n, f"{where}: slot {k}")
        out[int(k) - 1] = effect(spec, f"{where} {k}")
    return out


def quarters(grid, terrain, prebuilt):
    """The plain spaces split by rock and water (without the prebuilt ones), clockwise from the top."""
    plain = {c for c in grid if terrain.get(c, "plain") == "plain"} - prebuilt
    seen, parts = set(), []
    for c in sorted(plain):
        if c in seen:
            continue
        part, todo = set(), [c]
        while todo:
            q, r = todo.pop()
            if (q, r) in part:
                continue
            part.add((q, r))
            for dq, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)):
                n = (q + dq, r + dr)
                if n in plain and n not in part:
                    todo.append(n)
        seen |= part
        parts.append(part)
    G.need(len(parts) == 4, f"Drawing Board: {len(parts)} areas")
    centre = (sum(q for q, _ in prebuilt) / 2, sum(r for _, r in prebuilt) / 2)

    def angle(part):  # clockwise from the top in pixel space (pointy columns: x = 1.5 q, y = r + q / 2)
        q = sum(c[0] for c in part) / len(part) - centre[0]
        r = sum(c[1] for c in part) / len(part) - centre[1]
        return math.atan2(1.5 * q, -(r + q / 2)) % (2 * math.pi)
    return [sorted(p) for p in sorted(parts, key=angle)]


def main():
    data = json.loads(G.OUT_JSON.read_text())["maps"]
    maps = [m for m in data if m["key"] not in {f"plan{k}" for k in RULES}]
    grid = [(c[0], c[1]) for c in next(m for m in maps if m["key"] == "map0")["cells"]]
    index = {c: i for i, c in enumerate(grid)}
    for m in maps:
        m["_cpp"] = dict(rule=gen_alt_maps.RULE_CPP[m["rule"]],
                         rule_mask=G.mask([tuple(c) for c in m["rule_cells"]], index))
    std_tokens = next(m for m in maps if m["key"] == "plan1")["token_abilities"]
    pack = json.loads(PACK.read_text())["maps"]
    for k in ("9", "10", "11", "12", "13", "14"):
        b = pack[k]
        terrain = {axial(u): "rock" for u in b["terrains"].get("Rock", [])}
        terrain.update({axial(u): "water" for u in b["terrains"].get("Water", [])})
        build2 = {axial(u) for u in b["upgradeNeeded"]}
        bonus = {}
        for uid, spec in b["bonuses"].items():
            (name, amount), = spec.items()
            G.need((name, int(amount)) in BONUS, f"map {k}: unknown placement bonus {spec}")
            bonus[axial(uid)] = BONUS[(name, int(amount))]
        for c in list(terrain) + list(build2) + list(bonus):
            G.need(c in index, f"map {k}: {c} off the grid")
        cells = [[q, r, terrain.get((q, r), "plain"), bonus.get((q, r)), (q, r) in build2] for q, r in grid]
        tokens = []
        abilities = dict(std_tokens)
        for i, space in enumerate(b["bonusSpaces"]):
            repeats = space["type"] == "income"
            if "Snapping" in space["bonus"]:
                G.need(i == 0, f"map {k}: Snapping on token {i}")
                tokens.append([["card", 1], repeats])  # the encoder's view; the ability resolves (token_abilities)
            else:
                tokens.append([effect(space["bonus"], f"map {k} token {i}"), repeats])
        G.need(len(tokens) == 7, f"map {k}: {len(tokens)} left-edge spaces")
        prebuilt = [[kind, size, sorted(list(axial(u)) for u in cs)] for kind, size, cs in PREBUILT.get(k, [])]
        if k == "9":
            areas = [[list(axial(u)) for u in AREAS9[c]] for c in ("africa", "americas", "asia", "australia", "europe")]
            for area in areas:
                G.need(len(area) == 6 and all(terrain.get(tuple(c), "plain") == "plain" for c in area),
                       f"map 9: area {area} is not 6 plain spaces")
        elif k == "13":
            areas = [[list(c) for c in part] for part in
                     quarters(grid, terrain, {tuple(c) for _, _, cs in prebuilt for c in cs})]
        else:
            areas = []
        if k == "14":  # the 4th worker: 1 conservation point and a person sponsor (the rule finds it)
            G.need(b["workersBonuses"]["3"]["multiple"][0] == {"conservation": 1}, "map 14: the 4th worker")
            workers = [None, None, ["conservation", 1]]
        else:
            workers = slots(b.get("workersBonuses"), 3, f"map {k} workers")
        m = dict(
            key=f"plan{k}", name=b["name"], rule=RULES[k][1:].lower(), cells=cells, rule_cells=[],
            partner_zoo_slots=slots(b.get("partnerZooBonuses"), 4, f"map {k} partner zoo"),
            university_slots=slots(b.get("facBonuses"), 3, f"map {k} university"),
            worker_slots=[None] + workers, linked_slots=slots(b.get("facPartnerZooLinkedBonuses"), 3, f"map {k} linked"),
            tokens=tokens, token_abilities=abilities, prebuilt=prebuilt, areas=areas, selfplay=False,
            _cpp=dict(rule=RULES[k], rule_mask=0))
        maps.append(m)
    G.write(maps, index, source="ark-nova-ai (hexmap.MAP0, data/maps) and BGA (bga_alt_maps.json, "
                                 "bga_pack_maps.json)")


if __name__ == "__main__":
    main()
