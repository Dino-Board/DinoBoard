"""Rulebook and official FAQ scenarios for the C++ engine (see g_upstream_rules in arknova_core.h).

Each scenario builds a position with ark-nova-ai, loads it into arknova_diff_cli, plays it there and
returns whether the engine did what the rulebook or the official FAQ (Feuerland, Ark Nova FAQ V2)
says. Started with --rulebook the engine must always do so. Without it the engine plays
ark-nova-ai's rules, which depart from them in the scenarios listed in DEPARTURES and agree in the
others.

    python games/arknova/tools/rule_scenarios.py --ref ~/ark-nova-fork --cli build/arknova_diff_cli

Not checked here, as they follow from how the engine models the board rather than from a
position: water and rock printed on building tiles never count as terrain (terrain is a map
property); a border space is a space with fewer than 6 neighbours (hexmap.MapDef.is_border); Map A's
printed kiosk is one of the zoo's buildings from the start (MapDef prebuilt); the Map 8 Hollywood
Hills partner zoo obeys the same Association II condition as every partner zoo
(third_partner_zoo_needs_association_ii); the solo game and the optional interactive abilities.
"""

import argparse
import json
import os
import subprocess
import sys

ANIMALS_STRENGTH5 = ["cards", "build", "association", "sponsors", "animals"]
ASSOCIATION_STRENGTH5 = ["animals", "build", "cards", "sponsors", "association"]
SPONSORS_STRENGTH5 = ["animals", "build", "cards", "association", "sponsors"]
ACTIONS = ["animals", "build", "cards", "association", "sponsors"]
CONTINENTS = ("africa", "americas", "asia", "australia", "europe")


class Cli:
    def __init__(self, path, rulebook):
        self.proc = subprocess.Popen([path] + (["--rulebook"] if rulebook else []), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        out = self.proc.stdout.readline().rstrip("\n")
        if out.startswith("err "):
            raise RuntimeError(f"arknova_diff_cli: {out[4:]}")
        return out

    def load(self, state):
        self.send("S " + dumps(export_state(state)))

    def options(self):
        return json.loads(self.send("O"))

    def play(self, key):
        return json.loads(self.send("P " + dumps(canon(key))))


# ---- positions ---------------------------------------------------------------------------------

def by_name(name):
    return next(cid for cid, c in cards.all_cards().items() if c["name"] == name)


def plain_animals(size):
    return sorted(cid for cid, c in cards.all_cards().items() if c["type"] == "animal" and not c["requirements"]
                  and c["enclosure"]["standard_size"] == size and not c["enclosure"]["rock"]
                  and not c["enclosure"]["water"] and not c["enclosure"]["special"])


def new_game(slots, map_name="map0"):
    s = engine.new_game(2, seed=1, map_name=map_name)
    while s.stack[-1].kind == "setup_keep":
        engine.step(s, engine.legal_options(s)[0])
    p = s.players[0]
    p.hand, p.money, p.reputation, p.appeal, p.xtokens = [], 100, 1, 0, 0
    p.slots = list(slots)
    s.display = list(s.display)
    return s, p


def free(s, *ids):
    """Take the cards out of the deck, the discard pile, the display and every hand."""
    s.deck = [c for c in s.deck if c not in ids]
    s.discard = [c for c in s.discard if c not in ids]
    s.display = [c for c in s.display if c not in ids]
    s.display += [s.deck.pop() for _ in range(6 - len(s.display))]
    for pl in s.players:
        pl.hand = [c for c in pl.hand if c not in ids]


def enclosure(p, size, rock=0, water=0):
    for cells in p.zoo.legal_placements("enclosure", size):
        if p.zoo.adjacent_terrain(cells, "rock") >= rock and p.zoo.adjacent_terrain(cells, "water") >= water:
            return p.zoo.place("enclosure", size, cells)[0]
    raise AssertionError(f"no {size}-space enclosure placement")


def home_for(p, card_id):
    need = cards.card(card_id)["enclosure"]
    return enclosure(p, need["standard_size"], need["rock"] or 0, need["water"] or 0)


def house(p, card_id, building):
    building.occupied = True
    p.housing[card_id] = (building.id, 0)
    p.zoo_cards.append(card_id)
    p.appeal += cards.card(card_id)["appeal"]


def start_animals(cli, s, card_id):
    """Start an Animals action (in C++) and play `card_id` from the hand."""
    cli.load(s)
    cli.play(["act", "animals", 0])
    return cli.play(next(o for o in cli.options() if o[0] == "play" and str(o[1]) == card_id))


def first(options, *heads):
    for h in heads:
        for o in options:
            if o[0] == h:
                return o
    return options[0]


def drive(cli, pick, stop, limit=80):
    """Play pick(options) until stop(options, last state) holds; returns both."""
    state = None
    for _ in range(limit):
        options = cli.options()
        if stop(options, state):
            return options, state
        state = cli.play(pick(options))
    raise AssertionError("the scenario never reached its decision")


def until_heads(*heads):
    return lambda options, _: any(o[0] in heads for o in options)


def turn_over(options, state):
    return state is not None and state["current"] != 0


def skip_ahead(options):
    return first(options, "done", "skip")


# ---- rulebook (ark-nova-ai departs) ------------------------------------------------------------

def second_animal_after_effects(cli):
    """p. 12: the Eagle-Owl's Perception brings the Lynx, played as the second animal (strength 5)."""
    s, p = new_game(ANIMALS_STRENGTH5)
    owl, lynx = by_name("Eurasian Eagle-Owl"), by_name("Eurasian Lynx")
    p.partner_zoos = ["europe"]
    need = cards.card(lynx)["enclosure"]
    enclosure(p, 2)
    enclosure(p, max(need["standard_size"], 2) + 1, need["rock"], need["water"])
    p.hand = [owl]
    s.deck = [c for c in s.deck if c not in (owl, lynx)] + [lynx]
    engine.step(s, ("act", "animals", 0))
    cli.load(s)

    def both(pred):
        option = next(o for o in engine.legal_options(s) if pred(o.label))
        engine.step(s, option)
        cli.play(option.key)

    both(lambda label: "Eagle-Owl" in label)
    while not any("Lynx" in o.label for o in engine.legal_options(s)):
        both(lambda label: True)
    both(lambda label: "Lynx" in label)
    while s.stack[-1].kind == "ab_pick":
        both(lambda label: True)
    return any(str(o[0]).startswith("play") and lynx in map(str, o) for o in cli.options())


def animals_ii_reputation_first(cli):
    """p. 13: Animals II at strength 5 gains reputation first, bringing folder 2 into range."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.upgraded.add("animals")
    s.display[1] = plain_animals(1)[0]
    enclosure(p, 2)
    cli.load(s)
    return any(o[:2] == ["act", "animals"] for o in cli.options())


def release_flips_smallest_enclosure(cli):
    """Errata (pp. 12, 15): releasing flips the smallest occupied standard enclosure that meets the animal's needs."""
    s, p = new_game(ASSOCIATION_STRENGTH5)
    p.workers = 4
    plain = plain_animals(2)
    project, animal = next((pid, a) for pid, pc in sorted(cards.all_cards().items())
                           if pc["type"] == "project" and pc["category"] == "release"
                           for a in plain if pc["icon"] in cards.card(a).get("tags", []))
    s.projects, s.project_tokens = [project] + s.projects[1:], {project: {}}
    five, two = enclosure(p, 5), enclosure(p, 2)
    house(p, animal, five)
    house(p, next(a for a in plain if a != animal), two)
    engine.step(s, ("act", "association", 0))
    cli.load(s)
    release = next(o for o in engine.legal_options(s) if o.key[0] == "release" and o.key[-1] == animal)
    occupied = {b["id"]: b["occupied"] for b in cli.play(release.key)["players"][0]["buildings"]}
    return occupied[five.id] and not occupied[two.id]


def determination_may_take_an_xtoken(cli):
    """FAQ: the any-action of Determination (Golden Eagle) may be the X-token action."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.upgraded.add("animals")
    eagle = by_name("Golden Eagle")
    free(s, eagle)
    home_for(p, eagle)
    p.hand = [eagle]
    start_animals(cli, s, eagle)
    options, _ = drive(cli, skip_ahead, until_heads("act", "xtoken"))
    return any(o[0] == "xtoken" for o in options)


def tracks_end_at_113_and_41(cli):
    """FAQ: no appeal beyond 113 and no conservation points beyond 41."""
    s, p = new_game(ANIMALS_STRENGTH5)
    animal = next(a for a in plain_animals(2) if cards.card(a)["appeal"] >= 2)
    free(s, animal)
    enclosure(p, 2)
    p.hand, p.appeal = [animal], 112
    appeal = start_animals(cli, s, animal)["players"][0]["appeal"]
    s, p = new_game(ASSOCIATION_STRENGTH5)
    p.upgraded.add("association")
    p.conservation = 41
    engine.step(s, ("act", "association", 0))
    cli.load(s)
    cli.play(["reputation"])
    conservation = cli.play(["donate"])["players"][0]["conservation"]
    return appeal == 113 and conservation == 41


def upgrade_during_an_action_waits_for_the_next(cli):
    """FAQ: an Action card upgraded during its own action keeps its old side until the next action:
    the second animal of this Animals I action cannot come from the display."""
    s, p = new_game(ANIMALS_STRENGTH5)
    gainer = next(cid for cid, c in sorted(cards.all_cards().items()) if c["type"] == "animal" and c.get("base_game")
                  and not c["requirements"] and not c["abilities"] and c["enclosure"]["standard_size"]
                  and (c.get("reputation") or c.get("conservation")))
    shown, kept = [a for a in plain_animals(1) if a != gainer][:2]
    free(s, gainer, shown, kept)
    if cards.card(gainer).get("reputation"):
        p.reputation = 4  # reaching reputation 5 upgrades an Action card
    else:
        p.conservation = 1  # reaching 2 conservation points upgrades an Action card (or adds a worker)
    home_for(p, gainer)
    enclosure(p, 1)
    # A second animal in hand keeps the action open in both engines after the first one.
    p.hand, s.display[0] = [gainer, kept], shown
    start_animals(cli, s, gainer)
    options, _ = drive(cli, lambda o: ["upgrade", "animals"] if ["upgrade", "animals"] in o else o[0],
                       lambda o, st: st is not None and st["players"][0]["upgraded"] == ["animals"])
    return not any(o[0] == "play_display" for o in options)


DEPARTURES = {second_animal_after_effects, animals_ii_reputation_first, release_flips_smallest_enclosure,
              determination_may_take_an_xtoken, tracks_end_at_113_and_41,
              upgrade_during_an_action_waits_for_the_next}


# ---- official FAQ (both engines agree) ---------------------------------------------------------

def boost_shifts_the_cards_between(cli):
    """FAQ: a card boosted to 1 moves the cards left of it one place right; to 5, those right of it left."""
    howler = by_name("Bolivian Red Howler")  # Boost: Cards
    slots = {}
    for to in (1, 5):
        s, p = new_game(["build", "cards", "association", "sponsors", "animals"])
        free(s, howler)
        home_for(p, howler)
        p.hand = [howler]
        start_animals(cli, s, howler)
        options, _ = drive(cli, skip_ahead, until_heads("boost"))
        slots[to] = cli.play(next(o for o in options if o[0] == "boost" and o[1] == to))["players"][0]["slots"]
    # After the action the Animals card is in slot 1: animals, build, cards, association, sponsors.
    return (slots[1] == ["cards", "animals", "build", "association", "sponsors"]
            and slots[5] == ["animals", "build", "association", "sponsors", "cards"])


def hypnosis_card_tokens(cli):
    """FAQ: Constriction on the card Hypnosis borrows weakens the hypnotist and is discarded; the
    card's Multiplier is not used."""
    adder = by_name("Common European Adder")
    s, p = new_game(ANIMALS_STRENGTH5)
    free(s, adder)
    p.partner_zoos = ["europe"]
    home_for(p, adder)
    p.hand, p.xtokens = [adder], 2
    q = s.players[1]
    q.appeal = 30
    q.slots = ["cards", "build", "association", "sponsors", "animals"]
    q.action_tokens = {"cards": ["constriction", "multiplier"]}
    start_animals(cli, s, adder)
    options, _ = drive(cli, skip_ahead, until_heads("act"))
    # Cards in slot 1 is strength 1, minus 2 for Constriction: playable only with 2 X-tokens.
    borrowed = [o for o in options if o[:2] == ["act", "cards"]]
    if borrowed != [["act", "cards", 2]]:
        return False
    state = cli.play(borrowed[0])
    if state["players"][1]["action_tokens"].get("cards") != ["multiplier"]:
        return False
    seen = []

    def pick(options):
        seen.extend(options)
        return skip_ahead(options)

    drive(cli, pick, turn_over)
    return not any(o[0] == "repeat" for o in seen)


def pilfering_needs_a_conservation_point(cli):
    """FAQ: the zoo with the most conservation points must have at least 1 (Japanese Macaque)."""
    macaque = by_name("Japanese Macaque")
    paid = []
    for points in (0, 1):
        s, p = new_game(ANIMALS_STRENGTH5)
        free(s, macaque)
        p.partner_zoos = ["asia"]
        home_for(p, macaque)
        p.hand = [macaque]
        q = s.players[1]
        q.appeal, q.conservation, q.money = 30, points, 50
        start_animals(cli, s, macaque)
        _, state = drive(cli, lambda o: first(o, "money", "done", "skip"), turn_over)
        paid.append(50 - state["players"][1]["money"])
    return paid == [5, 10]


def bonus_multiplier_goes_on_any_card(cli):
    """FAQ: a multiplier from a bonus tile is placed at once, on any Action card (Animals too)."""
    s, p = new_game(ANIMALS_STRENGTH5)
    state_mod.push(s, "place_multiplier", 0)
    cli.load(s)
    return sorted(cli.options()) == sorted(["multiplier", a] for a in ACTIONS)


def third_partner_zoo_needs_association_ii(cli):
    """FAQ: a third partner zoo needs the upgraded Association card, whatever brings it."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.partner_zoos = ["africa", "asia"]
    state_mod.push(s, "gain_partner_zoo", 0)
    cli.load(s)
    before = cli.options()
    p.upgraded.add("association")
    cli.load(s)
    return before == [["skip"]] and any(o[0] == "partner_zoo" for o in cli.options())


def bonus_building_needs_build_ii_for_its_spaces(cli):
    """FAQ: a building from a bonus goes on a Build II space only once Build is upgraded."""
    offered = []
    for upgraded in (False, True):
        s, p = new_game(ANIMALS_STRENGTH5)
        mapdef = p.zoo.mapdef
        target = next(c for c, v in sorted(mapdef.cells.items()) if v.build2_only)
        distance = lambda c: max(abs(c[0] - target[0]), abs(c[1] - target[1]), abs(c[0] + c[1] - target[0] - target[1]))
        # Build towards the Build II space until it is next to a building.
        while not any(n in c for b in p.zoo.buildings for c in [b.cells] for n in hexmap.neighbors(target)):
            cells = min(p.zoo.legal_placements("enclosure", 1, build2=True), key=lambda cs: distance(next(iter(cs))))
            p.zoo.place("enclosure", 1, cells)
        if upgraded:
            p.upgraded.add("build")
        state_mod.push(s, "free_build", 0, kinds=[("enclosure", 1)])
        cli.load(s)
        offered.append(["place", "enclosure", 1, [list(target)]] in cli.options())
    return offered == [False, True]


def partner_zoos_and_universities_come_from_the_board(cli):
    """FAQ: a partner zoo or university gained is one still available on the Association board."""
    s, p = new_game(ANIMALS_STRENGTH5)
    s.partner_zoos_available = type(s.partner_zoos_available)(c for c in s.partner_zoos_available if c != "africa")
    s.universities_available = type(s.universities_available)(u for u in s.universities_available if u != "science2")
    state_mod.push(s, "gain_partner_zoo", 0)
    cli.load(s)
    zoos = cli.options()
    s.stack.pop()
    state_mod.push(s, "gain_university", 0)
    cli.load(s)
    universities = cli.options()
    return (["partner_zoo", "africa"] not in zoos and ["partner_zoo", "asia"] in zoos
            and ["university", "science2"] not in universities and len(universities) >= 2)


def sponsor_bonus_plays_from_the_hand_only(cli):
    """FAQ: the bonus that plays a sponsor plays one from the hand, never the display (Sponsors II too)."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.upgraded.add("sponsors")
    cheap = [cid for cid, c in sorted(cards.all_cards().items()) if c["type"] == "sponsor" and c.get("base_game")
             and not c["requirements"] and c["level"] <= 3 and not c.get("building")][:2]
    free(s, *cheap)
    p.hand, s.display[0] = [cheap[0]], cheap[1]
    state_mod.push(s, "paid_sponsor", 0)
    cli.load(s)
    plays = [o for o in cli.options() if o[0] != "skip"]
    return bool(plays) and all(o == ["play", cheap[0]] for o in plays)


def waza_small_takes_without_the_extra_play(cli):
    """FAQ (card 228): the small animal from the display may be taken without playing the extra one."""
    s, p = new_game(["cards", "animals", "build", "association", "sponsors"])  # strength 2: one animal
    small = plain_animals(1) + plain_animals(2)
    played, extra, shown = small[:3]
    free(s, played, extra, shown, "228")
    p.zoo_cards.append("228")
    home_for(p, played)
    home_for(p, extra)
    p.hand, s.display[0] = [played, extra], shown
    start_animals(cli, s, played)
    if not any(o[0] == "play" for o in cli.options()):  # the program's extra small animal is offered
        return False
    cli.play(["done"])
    return any(o[0] == "take_small" for o in cli.options())


def polar_bear_appeal_ends_the_game_on_its_owners_turn(cli):
    """FAQ (card 251): appeal gained in another player's turn triggers the end only at the end of your
    own turn (or at a break)."""
    bear = next(cid for cid, c in sorted(cards.all_cards().items()) if c["type"] == "animal" and c.get("base_game")
                and "bear" in c.get("tags", []) and not c["requirements"] and not c["enclosure"]["special"]
                and c["enclosure"]["standard_size"])
    s, p = new_game(ANIMALS_STRENGTH5)
    free(s, bear, "251")
    home_for(p, bear)
    p.hand = [bear]
    q = s.players[1]
    q.zoo_cards.append("251")
    q.conservation = 10
    q.appeal = T.target_appeal(q.conservation) - 1
    start_animals(cli, s, bear)
    _, state = drive(cli, skip_ahead, turn_over)
    if state["end_by"] is not None or state["players"][1]["appeal"] < T.target_appeal(q.conservation):
        return False
    _, state = drive(cli, lambda o: first(o, "xtoken", "done", "skip"),
                     lambda o, st: st is not None and st["current"] == 0)
    return state["end_by"] == 1


def native_farm_animals_count_building_spaces_only(cli):
    """FAQ (card 260): only building spaces count as empty border spaces, not rock or water."""
    s, p = new_game(SPONSORS_STRENGTH5)
    farm = "260"
    free(s, farm)
    mapdef = p.zoo.mapdef
    near = lambda cells: {n for c in cells for n in hexmap.neighbors(c) if n in mapdef.cells and n not in cells}
    cells = next(c for c in p.zoo.legal_placements("enclosure", 2)
                 if any(mapdef.is_border(n) and mapdef.cells[n].terrain != "plain" for n in near(c)))
    p.zoo.place("enclosure", 2, cells)
    covered = set(cells)
    expected = sum(1 for n in near(covered) if mapdef.is_border(n) and mapdef.cells[n].terrain == "plain")
    p.hand = [farm]
    engine.step(s, ("act", "sponsors", 0))
    cli.load(s)
    state = cli.play(next(o for o in cli.options() if o[0] == "play" and str(o[1]) == farm))
    return state["players"][0]["appeal"] - cards.card(farm)["appeal"] == expected


def bonus_during_an_action_is_used_at_once(cli):
    """FAQ: the association worker of reputation 8, gained by the reputation task, serves the next task."""
    s, p = new_game(ASSOCIATION_STRENGTH5)
    p.upgraded.add("association")
    p.reputation, p.workers = 6, 1
    engine.step(s, ("act", "association", 0))
    cli.load(s)
    cli.play(["reputation"])
    return any(o[0] == "partner_zoo" for o in cli.options())


def venom_card_takes_an_xtoken(cli):
    """FAQ: the X-token action may use a card with a Venom token, discarding it; not with 5 X-tokens."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.action_tokens = {"cards": ["venom"]}
    cli.load(s)
    state = cli.play(["xtoken", "cards"])
    p.xtokens = 5
    cli.load(s)
    return "cards" not in state["players"][0]["action_tokens"] and not any(o[0] == "xtoken" for o in cli.options())


def a_new_reptile_house_takes_all_its_animals(cli):
    """FAQ: building a Reptile House, you may move every reptile that can live in it."""
    reptiles = [cid for cid, c in sorted(cards.all_cards().items()) if c["type"] == "animal" and c.get("base_game")
                and not c["requirements"] and c["enclosure"]["standard_size"] and not c["enclosure"]["rock"]
                and not c["enclosure"]["water"]
                and any(e["type"] == "Reptile House" and e["size"] <= 2 for e in c["enclosure"]["special"])][:2]
    s, p = new_game(["animals", "cards", "association", "sponsors", "build"])
    free(s, *reptiles)
    p.upgraded.add("build")
    for r in reptiles:
        house(p, r, home_for(p, r))
    engine.step(s, ("act", "build", 0))
    cli.load(s)
    cli.play(next(o for o in cli.options() if o[0] == "build" and o[1] == "reptile_house"))
    options, _ = drive(cli, skip_ahead, until_heads("move"))  # placement bonuses come first
    moves = [o for o in options if o[0] == "move"]
    if len(moves) != 2:
        return False
    cli.play(moves[0])
    return moves[1] in cli.options()


def display_refills_after_each_card_taken_between_actions(cli):
    """FAQ: when several players take a card during the break, the display is refilled after each."""
    s, p = new_game(ANIMALS_STRENGTH5)
    state_mod.push(s, "take_card", 1)
    state_mod.push(s, "take_card", 0)
    cli.load(s)
    state = cli.play(["display", 0])
    return all(c is not None for c in state["display"])


def two_multipliers_repeat_twice_even_the_xtoken_action(cli):
    """FAQ: 2 multiplier tokens on a card give 3 actions; on the X-token action, X-tokens each time."""
    s, p = new_game(ANIMALS_STRENGTH5)
    p.action_tokens = {"cards": ["multiplier", "multiplier"]}
    cli.load(s)
    state = cli.play(["xtoken", "cards"])
    repeats = 0
    while ["repeat"] in cli.options():
        repeats += 1
        state = cli.play(["repeat"])
    return repeats == 2 and state["players"][0]["xtokens"] == 3


def action_x_is_not_an_xtoken(cli):
    """FAQ: an "Action: Sponsors" ability (Pygmy Hippopotamus) cannot be the X-token action."""
    hippo = by_name("Pygmy Hippopotamus")
    s, p = new_game(ANIMALS_STRENGTH5)
    free(s, hippo)
    p.partner_zoos = ["africa"]
    home_for(p, hippo)
    p.hand = [hippo]
    start_animals(cli, s, hippo)
    options, _ = drive(cli, skip_ahead, until_heads("act", "xtoken"))
    return any(o[0] == "act" for o in options) and not any(o[0] == "xtoken" for o in options)


def partner_zoo_discount_per_continent_icon(cli):
    """FAQ: a partner zoo takes 3 money off per matching continent icon on the card, 6 for two."""
    animal, continent = next((cid, k) for cid, c in sorted(cards.all_cards().items())
                             if c["type"] == "animal" and c.get("base_game") and not c["enclosure"]["special"]
                             and set(c["requirements"]) <= {"Partner Zoo"}
                             for k in CONTINENTS if c["tags"].count(k) == 2)
    s, p = new_game(ANIMALS_STRENGTH5)
    free(s, animal)
    p.partner_zoos = [continent]
    home_for(p, animal)
    p.hand = [animal]
    state = start_animals(cli, s, animal)
    return state["players"][0]["money"] == 100 - (cards.card(animal)["cost"] - 6)


def water_next_to_a_building_counts_by_space(cli):
    """FAQ: a building next to 1 space of a 2-space water area is next to 1 water space, not 2."""
    animal = next(cid for cid, c in sorted(cards.all_cards().items()) if c["type"] == "animal"
                  and c.get("base_game") and not c["requirements"] and c["enclosure"]["standard_size"]
                  and c["enclosure"]["water"] == 2 and not c["enclosure"]["rock"] and not c["enclosure"]["special"])
    size = cards.card(animal)["enclosure"]["standard_size"]
    s, p = new_game(ANIMALS_STRENGTH5)
    water = {c for c, v in p.zoo.mapdef.cells.items() if v.terrain == "water"}

    def water_near(cells):
        return {n for c in cells for n in hexmap.neighbors(c) if n in water}

    def area(c):
        seen, todo = {c}, [c]
        while todo:
            for n in hexmap.neighbors(todo.pop()):
                if n in water and n not in seen:
                    seen.add(n)
                    todo.append(n)
        return seen

    placements = p.zoo.legal_placements("enclosure", size)
    one = next(c for c in placements if len(w := water_near(c)) == 1 and len(area(next(iter(w)))) >= 2)
    two = next(c for c in placements if len(water_near(c)) >= 2)

    def playable(cells):  # the animal in hand is the only play an Animals action could make
        t, pl = new_game(ANIMALS_STRENGTH5)
        free(t, animal)
        pl.zoo.place("enclosure", size, cells)
        pl.hand = [animal]
        cli.load(t)
        return ["act", "animals", 0] in cli.options()

    return not playable(one) and playable(two)


def outdoor_area_enclosure_is_two_spaces_larger(cli):
    """FAQ (Map 2): a standard enclosure next to the Outdoor Areas is 2 spaces larger for all purposes."""
    s, p = new_game(["animals", "cards", "association", "sponsors", "build"], map_name="plan2")
    engine.step(s, ("act", "build", 0))
    for key in [o.key for o in engine.legal_options(s) if o.key[0] == "build" and o.key[1] == "enclosure" and o.key[2] == 2]:
        cli.load(s)
        built = cli.play(key)["players"][0]["buildings"]
        if built and built[-1].get("extra_size") == 2:
            return True
    return False


def park_restaurant_special_enclosure_without_build_ii(cli):
    """FAQ (Map 5): its bonus builds 1 of the 2 special enclosures for free, Build upgraded or not."""
    s, p = new_game(ANIMALS_STRENGTH5, map_name="plan5")
    effects.gain(s, 0, "special_enclosure")
    cli.load(s)
    kinds = {o[1] for o in cli.options() if o[0] == "place"}
    return kinds == {"reptile_house", "aviary"}


SCENARIOS = [
    second_animal_after_effects, animals_ii_reputation_first, release_flips_smallest_enclosure,
    determination_may_take_an_xtoken, tracks_end_at_113_and_41, upgrade_during_an_action_waits_for_the_next,
    boost_shifts_the_cards_between, hypnosis_card_tokens, pilfering_needs_a_conservation_point,
    bonus_multiplier_goes_on_any_card, third_partner_zoo_needs_association_ii,
    bonus_building_needs_build_ii_for_its_spaces, partner_zoos_and_universities_come_from_the_board,
    sponsor_bonus_plays_from_the_hand_only, waza_small_takes_without_the_extra_play,
    polar_bear_appeal_ends_the_game_on_its_owners_turn, native_farm_animals_count_building_spaces_only,
    bonus_during_an_action_is_used_at_once, venom_card_takes_an_xtoken, a_new_reptile_house_takes_all_its_animals,
    display_refills_after_each_card_taken_between_actions, two_multipliers_repeat_twice_even_the_xtoken_action,
    action_x_is_not_an_xtoken, partner_zoo_discount_per_continent_icon, water_next_to_a_building_counts_by_space,
    outdoor_area_enclosure_is_two_spaces_larger, park_restaurant_special_enclosure_without_build_ii,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="path of the ark-nova-ai checkout")
    ap.add_argument("--cli", required=True, help="path of the arknova_diff_cli executable")
    ap.add_argument("-k", help="run only the scenarios whose name contains this")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.ref))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    global cards, effects, engine, hexmap, state_mod, T, canon, dumps, export_state
    import arknova.sponsor_rules  # noqa: F401  registers the sponsor rules
    from arknova import cards, effects, engine, hexmap
    from arknova import state as state_mod
    from arknova import tables as T
    from diff_test import canon, dumps, export_state

    failed = 0
    for rulebook in (True, False):
        for scenario in SCENARIOS:
            if args.k and args.k not in scenario.__name__:
                continue
            try:
                got = scenario(Cli(args.cli, rulebook))
            except Exception as e:  # noqa: BLE001  report and go on
                got = f"error: {type(e).__name__}: {e}"
            want = rulebook or scenario not in DEPARTURES
            ok = got is want
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {'rulebook' if rulebook else 'upstream'} {scenario.__name__}"
                  + ("" if ok else f" (got {got})"))
    if failed:
        raise SystemExit(f"{failed} scenario(s) failed")


if __name__ == "__main__":
    main()
