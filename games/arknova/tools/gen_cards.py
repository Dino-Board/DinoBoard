"""Generate games/arknova/arknova_cards_data.inc from ark-nova-ai's data/cards.json.

Usage: python games/arknova/tools/gen_cards.py <path to ark-nova-ai>/data/cards.json

The input is the card list that ark-nova-ai's tools/build_cards.py builds from the fan-made card
site (Ender-Wiggin2019/Next-Ark-Nova-Cards) plus its hand-transcribed overrides. Only base-game
cards are emitted, in printed-id order; a card's index in the table is its CardIdx. Every field
the C++ engine reads is checked here, so an unknown value fails generation instead of silently
becoming a default.
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "arknova_cards_data.inc"

TYPES = {"endgame": "kEndgame", "project": "kProject", "sponsor": "kSponsor", "animal": "kAnimal"}
ICONS = ["bird", "herbivore", "predator", "primate", "reptile", "bear", "pet",
         "africa", "americas", "asia", "australia", "europe", "science", "water", "rock"]
CONTINENTS = ["africa", "americas", "asia", "australia", "europe"]
REQ_EXTRA = {"Partner Zoo": "kReqPartnerZoo", "animalsII": "kReqAnimalsII", "sponsorsII": "kReqSponsorsII",
             "appeal": "kReqAppeal", "reputation": "kReqReputation"}
SPECIAL = {"Reptile House": "kReptileHouse", "Large Bird Aviary": "kAviary", "Petting Zoo": "kPettingZoo"}
PROJECT_CATEGORY = {"base": "kProjBase", "release": "kProjRelease", "breed": "kProjBreed", "normal": "kProjNormal"}
PROJECT_ICON = {**{i: f"kPIcon_{i}" for i in ICONS}, "all-animals": "kPIconAllAnimals",
                "all-continents": "kPIconAllContinents", "animal-size-2": "kPIconAnimalSize2",
                "animal-size-4": "kPIconAnimalSize4"}
REWARD = {"conservation": "kRewardConservation", "reputation": "kRewardReputation"}
REWARD_REQ = {None: "kRReqNone", "release": "kRReqRelease", "all-animals": "kRReqAllAnimals",
              "water": "kRReqWater", "rock": "kRReqRock", "science": "kRReqScience",
              "animal-size-2": "kRReqAnimalSize2",
              **{i: f"kRReq_{i}" for i in ICONS if i not in ("water", "rock", "science")}}
PLAY_BONUS = {"reputation": "kEffReputation"}
KEYWORDS = sorted({"ACTION_ASSOCIATION", "ACTION_BUILDING", "ACTION_CARDS", "ACTION_SPONSORS", "ASSERTION",
                   "BOOST_ANIMAL", "BOOST_ASSOCIATION", "BOOST_BUILDING", "BOOST_CARD", "BOOST_SPONSORS",
                   "CLEVER", "CONSTRICTION", "DETERMINATION", "DIGGING", "DOMINANCE", "FLOCK_ANIMAL",
                   "FULL_THROATED", "HUNTER", "HYPNOSIS", "ICONIC_ANIMAL", "INVENTIVE", "INVENTIVE_BEAR",
                   "INVENTIVE_PRIMARY", "JUMPING", "MULTIPLIER_ASSOCIATION", "MULTIPLIER_BUILDING",
                   "MULTIPLIER_CARD", "MULTIPLIER_SPONSORS", "PACK", "PEACOCKING", "PERCEPTION_4",
                   "PETTING_ZOO_ANIMAL", "PILFERING_1", "PILFERING_2", "POSTURING", "POUCH", "RESISTANCE",
                   "SCAVENGING", "SNAPPING_1", "SNAPPING_2", "SPONSOR_MAGNET", "SPRINT", "SUN_BATHING", "VENOM"})


def need(cond, msg):
    if not cond:
        raise SystemExit(f"gen_cards: {msg}")


def i8(v, what):
    need(isinstance(v, int) and not isinstance(v, bool) and -128 <= v <= 127, f"{what}: {v!r} is not an int8")
    return v


def emit_card(c, i):
    cid = c["id"]
    t = f"c[{i}]"
    out = [f"  // {cid} {c['name']}",
           f"  {t}.id = {int(cid)}; {t}.type = CardType::{TYPES[c['type']]}; {t}.name = {json.dumps(c['name'])};"]
    for key in ("cost", "size", "appeal", "conservation", "reputation", "level"):
        if key in c:
            out.append(f"  {t}.{key} = {i8(c[key], f'{cid}.{key}')};")
    if "human" in c:
        out.append(f"  {t}.human = {'true' if c['human'] else 'false'};")
    enc = c.get("enclosure")
    if enc is not None:
        if enc.get("standard_size") is not None:
            out.append(f"  {t}.enc_standard = {i8(enc['standard_size'], cid)};")
        for terrain in ("rock", "water"):
            v = enc[terrain]
            need(isinstance(v, int), f"{cid}.enclosure.{terrain} = {v!r}")
            if v:
                out.append(f"  {t}.enc_{terrain} = {v};")
        for k, s in enumerate(enc.get("special") or []):
            need(k < 2, f"{cid}: more than 2 special enclosures")
            out.append(f"  {t}.special[{k}] = {{SpecialKind::{SPECIAL[s['type']]}, {i8(s['size'], cid)}}};")
        if enc.get("special"):
            out.append(f"  {t}.n_special = {len(enc['special'])};")
    for tag in c.get("tags", []):
        need(tag in ICONS, f"{cid}: unknown tag {tag!r}")
        out.append(f"  ++{t}.tags[kIcon_{tag}];")
    for cont in c.get("continents", []):
        need(cont in CONTINENTS, f"{cid}: unknown continent {cont!r}")
        out.append(f"  {t}.continents |= 1u << kCont_{cont};")
        out.append(f"  ++{t}.continent_icons[kCont_{cont}];")
    for r in c.get("requirements", []):
        if r in REQ_EXTRA:
            out.append(f"  ++{t}.req[{REQ_EXTRA[r]}];")
        else:
            need(r in ICONS, f"{cid}: unknown requirement {r!r}")
            out.append(f"  ++{t}.req[kIcon_{r}];")
    if "min_reputation" in c:
        out.append(f"  {t}.min_reputation = {i8(c['min_reputation'], cid)};")
    if "max_appeal" in c:
        out.append(f"  {t}.max_appeal = {i8(c['max_appeal'], cid)};")
    for k, a in enumerate(c.get("abilities", [])):
        need(k < 2 and a["keyword"] in KEYWORDS, f"{cid}: ability {a!r}")
        v = a["value"]
        if a["keyword"] == "ICONIC_ANIMAL":
            need(isinstance(v, str) and v.lower() in CONTINENTS, f"{cid}: ICONIC value {v!r}")
            val = f"kCont_{v.lower()}"
        elif v is None:
            val = "-1"
        else:
            val = str(i8(v, cid))
        out.append(f"  {t}.abilities[{k}] = {{Keyword::k{a['keyword']}, {val}}};")
    if c.get("abilities"):
        out.append(f"  {t}.n_abilities = {len(c['abilities'])};")
    if any(e["timing"] in ("income", "passive", "endgame") for e in c.get("effects", [])):
        out.append(f"  {t}.lasting_effect = true;")
    b = c.get("building")
    if b is not None:
        need(set(b) == {"shape", "size", "rock", "water", "min_border", "needs_adjacency"}, f"{cid}: building {b!r}")
        need(len(b["shape"]) <= kMaxShape, f"{cid}: shape too large")
        cells = ", ".join(f"{{{q}, {r}}}" for q, r in b["shape"])
        out.append(f"  {t}.building = {{true, {len(b['shape'])}, {{{cells}}}, {b['size']}, {b['rock']}, "
                   f"{b['water']}, {b['min_border']}, {'true' if b['needs_adjacency'] else 'false'}}};")
    if c["type"] == "project":
        out.append(f"  {t}.proj_category = ProjCategory::{PROJECT_CATEGORY[c['category']]};")
        out.append(f"  {t}.proj_icon = ProjIcon::{PROJECT_ICON[c['icon']]};")
        need(len(c["slots"]) <= 3, f"{cid}: more than 3 slots")
        for k, s in enumerate(c["slots"]):
            need(s["position"] == k + 1, f"{cid}: slot positions out of order")
            req = -1 if s["required"] is None else i8(s["required"], cid)
            size = s.get("animal_size") or {}
            need(set(size) <= {"min", "max"}, f"{cid}: animal_size {size!r}")
            out.append(f"  {t}.slots[{k}].required = {req}; {t}.slots[{k}].size_min = {size.get('min', -1)}; "
                       f"{t}.slots[{k}].size_max = {size.get('max', -1)};")
            need(1 <= len(s["rewards"]) <= 2, f"{cid}: rewards {s['rewards']!r}")
            for j, rw in enumerate(s["rewards"]):
                out.append(f"  {t}.slots[{k}].rewards[{j}] = {{Reward::{REWARD[rw['reward']]}, "
                           f"{i8(rw['amount'], cid)}, RewardReq::{REWARD_REQ[rw['requirement']]}}};")
            out.append(f"  {t}.slots[{k}].n_rewards = {len(s['rewards'])};")
        out.append(f"  {t}.n_slots = {len(c['slots'])};")
        need(not c.get("place_bonuses"), f"{cid}: place_bonuses not supported")
        for k, (kind, amount) in enumerate(c.get("play_bonus") or []):
            need(k < 2, f"{cid}: play bonus")
            out.append(f"  {t}.play_bonus[{k}] = {{EffectKind::{PLAY_BONUS[kind]}, {i8(amount, cid)}}};")
        if c.get("play_bonus"):
            out.append(f"  {t}.n_play_bonus = {len(c['play_bonus'])};")
    if c["type"] == "endgame":
        steps = [s for s in c["scoring"] if "threshold" in s]
        need(len(steps) in (0, 4), f"{cid}: scoring {c['scoring']!r}")
        for k, s in enumerate(steps):
            out.append(f"  {t}.scoring[{k}] = {{{i8(s['threshold'], cid)}, {i8(s['conservation'], cid)}}};")
        out.append(f"  {t}.n_scoring = {len(steps)};")
    return out


kMaxShape = 6


def main():
    need(len(sys.argv) == 2, "usage: gen_cards.py <ark-nova-ai>/data/cards.json")
    data = json.loads(Path(sys.argv[1]).read_text())
    cards = sorted((c for c in data["cards"] if c["base_game"]), key=lambda c: int(c["id"]))
    lines = ["// Generated by games/arknova/tools/gen_cards.py from ark-nova-ai data/cards.json. Do not edit.",
             f"// Source: {data['source']}" if isinstance(data.get("source"), str) else "// Source: see gen_cards.py",
             f"constexpr int kNumCards = {len(cards)};",
             "inline void fill_cards(CardDef* c) {"]
    for i, c in enumerate(cards):
        lines += emit_card(c, i)
    lines.append("}")
    lines.append("constexpr const char* kKeywordNames[] = {" + ", ".join(json.dumps(k) for k in KEYWORDS) + "};")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
