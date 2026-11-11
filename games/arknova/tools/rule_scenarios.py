"""Rulebook scenarios where ark-nova-ai departs from the rules (see g_upstream_rules in arknova_core.h).

Each scenario builds a position with ark-nova-ai, loads it into arknova_diff_cli and plays it there.
Started with --rulebook the C++ engine must follow the rulebook; without it, ark-nova-ai.

    python games/arknova/tools/rule_scenarios.py --ref ~/ref/ark-nova-ai --cli build/arknova_diff_cli
"""

import argparse
import json
import os
import subprocess
import sys

ANIMALS_STRENGTH5 = ["cards", "build", "association", "sponsors", "animals"]
ASSOCIATION_STRENGTH5 = ["animals", "build", "cards", "sponsors", "association"]


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


def by_name(name):
    return next(cid for cid, c in cards.all_cards().items() if c["name"] == name)


def plain_animals(size):
    return sorted(cid for cid, c in cards.all_cards().items() if c["type"] == "animal" and not c["requirements"]
                  and c["enclosure"]["standard_size"] == size and not c["enclosure"]["rock"]
                  and not c["enclosure"]["water"] and not c["enclosure"]["special"])


def new_game(slots):
    s = engine.new_game(2, seed=1)
    while s.stack[-1].kind == "setup_keep":
        engine.step(s, engine.legal_options(s)[0])
    p = s.players[0]
    p.hand, p.money, p.reputation, p.appeal, p.xtokens = [], 100, 1, 0, 0
    p.slots = list(slots)
    s.display = list(s.display)
    return s, p


def enclosure(p, size, rock=0, water=0):
    for cells in p.zoo.legal_placements("enclosure", size):
        if p.zoo.adjacent_terrain(cells, "rock") >= rock and p.zoo.adjacent_terrain(cells, "water") >= water:
            return p.zoo.place("enclosure", size, cells)[0]
    raise AssertionError(f"no {size}-space enclosure placement")


def house(p, card_id, building):
    building.occupied = True
    p.housing[card_id] = (building.id, 0)
    p.zoo_cards.append(card_id)
    p.appeal += cards.card(card_id)["appeal"]


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
    """p. 15: releasing flips the smallest occupied standard enclosure that meets the animal's needs."""
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


SCENARIOS = [second_animal_after_effects, animals_ii_reputation_first, release_flips_smallest_enclosure]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="path of the ark-nova-ai checkout")
    ap.add_argument("--cli", required=True, help="path of the arknova_diff_cli executable")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.ref))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    global cards, engine, canon, dumps, export_state
    import arknova.sponsor_rules  # noqa: F401  registers the sponsor rules
    from arknova import cards, engine
    from diff_test import canon, dumps, export_state

    failed = 0
    for rulebook in (True, False):
        for scenario in SCENARIOS:
            got = scenario(Cli(args.cli, rulebook))
            ok = got == rulebook
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {'rulebook' if rulebook else 'upstream'} {scenario.__name__}")
    if failed:
        raise SystemExit(f"{failed} scenario(s) failed")


if __name__ == "__main__":
    main()
