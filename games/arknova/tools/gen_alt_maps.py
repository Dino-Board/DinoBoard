"""Add the alternative sides 1a-8a of maps 1-8 to arknova_maps.json and arknova_maps_data.inc.

Usage: python games/arknova/tools/gen_alt_maps.py   (after gen_maps.py)

An alternative side keeps its map's rule and side panels (left-edge tokens, partner zoo, university
and last-worker bonuses: identical on Board Game Arena); only the spaces differ. They come from
games/arknova/bga_alt_maps.json, Board Game Arena's MAPS_DATA, as (column x, doubled row) spaces:
index in the column = row // 2, axial = (x, index - (x + 1) // 2) (checked cell by cell against maps
0, 1-8 and A). Where a side moves its rule's site, the site was read off BGA's map pictures: 1a's
observation tower and the third Hollywood space of 8a; 7a's sites are its kiosk bonus spaces. 4a and
6a use the pictures (and sites) of 4 and 6.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_maps as G  # noqa: E402

ALT = Path(__file__).resolve().parent.parent / "bga_alt_maps.json"
RULE_CPP = {"none": "kNone", "observationtower": "kObservationTower", "outdoorarea": "kOutdoorArea",
            "harbor": "kHarbor", "parkrestaurant": "kParkRestaurant", "researchinstitute": "kResearchInstitute",
            "icecream": "kIceCream", "hollywood": "kHollywood", "geographical": "kGeographical",
            "rescue": "kRescue", "caves": "kCaves", "ai": "kAI", "drawingboard": "kDrawingBoard", "lagoon": "kLagoon"}
# BGA placement bonus -> DinoBoard's name.
BONUS = {("xtoken", 1): "xtoken", ("money", 5): "money5", ("money", 10): "money10", ("money", 2): "money2",
         ("take-in-range-or-deck", 1): "card", ("Clever", 1): "clever", ("reputation", 1): "rep1",
         ("reputation", 2): "rep2", ("Partner-Zoo", 1): "partner_zoo", ("bonus-sponsor", 1): "sponsor_for_money",
         ("kiosk", 1): "kiosk", ("Fac", 1): "university", ("Multiplier", 1): "multiplier", ("add-worker", 1): "worker1"}
# Rule sites that move on the alternative side (axial); None keeps the standard side's.
SITES = {"1a": [(7, -1)], "2a": None, "3a": [], "4a": None, "5a": None, "6a": None, "7a": "kiosk",
         "8a": [(1, 4), (4, 3), (7, 0)]}


def axial(uid):
    x, row = map(int, uid.split("_"))
    return (x, row // 2 - (x + 1) // 2)


def main():
    data = json.loads(G.OUT_JSON.read_text())["maps"]
    maps = [m for m in data if not (m["key"].startswith("plan") and m["key"][-1] == "a" and m["key"] != "plana")]
    by_key = {m["key"]: m for m in maps}
    grid = [(c[0], c[1]) for c in by_key["map0"]["cells"]]
    index = {c: i for i, c in enumerate(grid)}
    for m in maps:
        m["_cpp"] = dict(rule=RULE_CPP[m["rule"]], rule_mask=G.mask([tuple(c) for c in m["rule_cells"]], index))

    alt = json.loads(ALT.read_text())["maps"]
    for side in sorted(alt):
        b, std = alt[side], by_key[f"plan{side[0]}"]
        terrain = {axial(u): "rock" for u in b["terrains"].get("Rock", [])}
        terrain.update({axial(u): "water" for u in b["terrains"].get("Water", [])})
        build2 = {axial(u) for u in b["upgradeNeeded"]}
        bonus = {}
        for uid, spec in b["bonuses"].items():
            (name, amount), = spec.items()
            G.need((name, int(amount)) in BONUS, f"{side}: unknown bonus {spec}")
            bonus[axial(uid)] = BONUS[(name, int(amount))]
        for c in list(terrain) + list(build2) + list(bonus):
            G.need(c in index, f"{side}: {c} off the grid")
        cells = [[q, r, terrain.get((q, r), "plain"), bonus.get((q, r)), (q, r) in build2] for q, r in grid]
        site = SITES[side]
        if site is None:
            sites = [tuple(c) for c in std["rule_cells"]]
            std_terrain = {(c[0], c[1]): c[2] for c in std["cells"]}
            for c in sites:
                G.need(terrain.get(c, "plain") == std_terrain[c], f"{side}: the site {c} changed terrain")
        elif site == "kiosk":
            sites = sorted(c for c, n in bonus.items() if n == "kiosk")
            G.need(len(sites) == len(std["rule_cells"]), f"{side}: {len(sites)} kiosk bonus spaces")
        else:
            sites = list(site)
        if std["rule"] == "observationtower":
            G.need(all(terrain.get(c) == "rock" for c in sites), f"{side}: the tower is not on rock")
        if std["rule"] == "hollywood":
            G.need(all(terrain.get(c, "plain") == "plain" for c in sites), f"{side}: a Hollywood space is not plain")
        G.need(bool(sites) == bool(std["rule_cells"]), f"{side}: rule sites {sites} vs {std['rule_cells']}")
        m = dict(std, key=f"plan{side}", name=f"{std['name']} (alternative side)", cells=cells,
                 rule_cells=[list(c) for c in sorted(sites)])
        m["_cpp"] = dict(rule=RULE_CPP[m["rule"]], rule_mask=G.mask(sorted(sites), index))
        maps.append(m)
    G.write(maps, index, source="ark-nova-ai (hexmap.MAP0, data/maps) and BGA (bga_alt_maps.json)")


if __name__ == "__main__":
    main()
