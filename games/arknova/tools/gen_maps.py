"""Generate games/arknova/arknova_maps_data.inc and arknova_maps.json from ark-nova-ai's zoo maps.

Usage: python games/arknova/tools/gen_maps.py <path to ark-nova-ai>

Map 0 is ark-nova-ai's hexmap.MAP0 with the side panels of tables.MAPS["map0"]; maps 1-8 and A are
its transcriptions in data/maps, read with its own loader (arknova/maps.py). Every map has the same
58-space grid; cells are listed in sorted axial (q, r) order, the engine's cell index. The JSON
holds the same data for the tests' independent reference (tests/arknova/arknova_reference.py).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT_INC = HERE / "arknova_maps_data.inc"
OUT_JSON = HERE / "arknova_maps.json"

ORDER = ["map0", "plan1", "plan2", "plan3", "plan4", "plan5", "plan6", "plan7", "plan8", "plana"]
# The special rule of each map and the cells it reads (tables.MAPS[map]["special"] names).
RULES = {
    "plan1": ("kObservationTower", "observation_tower"),
    "plan2": ("kOutdoorArea", "outdoor_area"),
    "plan4": ("kHarbor", "harbor_connecting_building_space"),
    "plan5": ("kParkRestaurant", "park_restaurant"),
    "plan6": ("kResearchInstitute", "institute_connecting_building_space"),
    "plan7": ("kIceCream", "kiosk_bonus"),
    "plan8": ("kHollywood", "H"),
}
EFFECTS = {"money": "kEffMoney", "appeal": "kEffAppeal", "xtoken": "kEffXtoken", "conservation": "kEffConservation",
           "reputation": "kEffReputation", "worker": "kEffWorker", "upgrade": "kEffUpgrade", "card": "kEffCard",
           "clever": "kEffClever", "enclosure": "kEffEnclosure", "kiosk": "kEffKiosk",
           "multiplier": "kEffMultiplier", "partner_zoo": "kEffPartnerZoo", "university": "kEffUniversity",
           "sponsor_for_money": "kEffSponsorForMoney", "special_enclosure": "kEffSpecialEnclosure"}
BONUSES = {None: "kPBNone", "money5": "kPBMoney5", "money10": "kPBMoney10", "xtoken": "kPBXtoken", "rep2": "kPBRep2",
           "card": "kPBCard", "clever": "kPBClever", "money2": "kPBMoney2", "rep1": "kPBRep1", "kiosk": "kPBKiosk",
           "worker1": "kPBWorker1", "partner_zoo": "kPBPartnerZoo", "university": "kPBUniversity",
           "multiplier": "kPBMultiplier", "sponsor_for_money": "kPBSponsorForMoney"}
KINDS = {"enclosure": "kEnclosure", "kiosk": "kKiosk"}


def need(cond, msg):
    if not cond:
        raise SystemExit(f"gen_maps: {msg}")


def one_effect(effects, where):
    need(len(effects) <= 1, f"{where}: more than one effect {effects!r}")
    if not effects:
        return None
    kind, amount = effects[0]
    need(kind in EFFECTS and 0 < amount < 128, f"{where}: effect {effects!r}")
    return [kind, amount]


def cpp_effect(e):
    return "{EffectKind::kEffMoney, 0}" if e is None else f"{{EffectKind::{EFFECTS[e[0]]}, {e[1]}}}"


def mask(cells, index):
    m = 0
    for c in cells:
        need(c in index, f"cell {c} not on the grid")
        m |= 1 << index[c]
    return m


def main():
    need(len(sys.argv) == 2, "usage: gen_maps.py <ark-nova-ai>")
    sys.path.insert(0, sys.argv[1])
    import arknova.engine as G  # noqa: F401  registers maps 1-8 and A
    import arknova.tables as T

    grid = sorted(G.MAP_DEFS["map0"].cells)
    need(len(grid) == 58, "grid size")
    index = {c: i for i, c in enumerate(grid)}
    maps = []
    for key in ORDER:
        mapdef, board = G.MAP_DEFS[key], T.MAPS[key]
        need(sorted(mapdef.cells) == grid, f"{key}: grid differs from Map 0")
        cells = [mapdef.cells[c] for c in grid]
        for c in cells:
            need(c.bonus in BONUSES and c.terrain in ("plain", "water", "rock"), f"{key}: cell {c}")
        rule, site = RULES.get(key, ("kNone", None))
        rule_cells = sorted(board.get("special", {}).get(site, [])) if site else []
        need(not site or rule_cells, f"{key}: no {site} cells")
        maps.append({
            "key": key,
            "name": board.get("name", mapdef.name),
            "rule": rule[1:].lower() if rule != "kNone" else "none",
            "cells": [[q, r, c.terrain, c.bonus, c.build2_only] for (q, r), c in zip(grid, cells)],
            "rule_cells": [list(c) for c in rule_cells],
            "partner_zoo_slots": [one_effect(e, f"{key} partner zoo") for e in board["partner_zoo_slots"]],
            "university_slots": [one_effect(e, f"{key} university") for e in board["university_slots"]],
            "last_worker": one_effect(board["last_worker"], f"{key} last worker"),
            "tokens": [[one_effect(e, f"{key} token"), bool(rep)] for e, rep in board["tokens"]],
            "prebuilt": [[kind, size, sorted(list(c) for c in cs)] for kind, size, cs in board.get("prebuilt", []) if cs],
            "_cpp": dict(rule=rule, rule_mask=mask(rule_cells, index)),
        })
        need(len(maps[-1]["partner_zoo_slots"]) == 4 and len(maps[-1]["university_slots"]) == 3
             and len(maps[-1]["tokens"]) == 7, f"{key}: panel sizes")

    lines = ["// Generated by games/arknova/tools/gen_maps.py from ark-nova-ai (hexmap.MAP0, data/maps). Do not edit.",
             f"constexpr int kNumMaps = {len(maps)};",
             "const MapSource kMapSources[kNumMaps] = {"]
    for m in maps:
        terrain = "".join({"plain": ".", "water": "W", "rock": "R"}[c[2]] for c in m["cells"])
        bonus = ", ".join(BONUSES[c[3]] for c in m["cells"])
        build2 = mask([(c[0], c[1]) for c in m["cells"] if c[4]], index)
        prebuilt = [f"{{BuildingKind::{KINDS[k]}, {s}, 0x{mask([tuple(c) for c in cs], index):x}ull}}"
                    for k, s, cs in m["prebuilt"]]
        lines += [
            f"    {{{json.dumps(m['key'])}, {json.dumps(m['name'])}, MapRule::{m['_cpp']['rule']},",
            f"     \"{terrain}\",",
            f"     {{{bonus}}},",
            f"     0x{build2:x}ull, 0x{m['_cpp']['rule_mask']:x}ull,",
            "     {" + ", ".join(cpp_effect(e) for e in m["partner_zoo_slots"]) + "},",
            "     {" + ", ".join(cpp_effect(e) for e in m["university_slots"]) + "},",
            f"     {cpp_effect(m['last_worker'])},",
            "     {" + ", ".join(f"{{{cpp_effect(e)}, {'true' if rep else 'false'}}}" for e, rep in m["tokens"]) + "},",
            f"     {len(prebuilt)}, {{{', '.join(prebuilt)}}}}},",
        ]
    lines.append("};")
    OUT_INC.write_text("\n".join(lines) + "\n")
    for m in maps:
        del m["_cpp"]
    OUT_JSON.write_text('{"_doc": "Generated by games/arknova/tools/gen_maps.py. Do not edit.",\n "maps": [\n'
                        + ",\n".join("  " + json.dumps(m) for m in maps) + "\n ]}\n")
    print(f"wrote {OUT_INC} and {OUT_JSON} ({len(maps)} maps)")


if __name__ == "__main__":
    main()
