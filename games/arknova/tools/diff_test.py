"""Differential test of the C++ Ark Nova engine against the Python reference (ark-nova-ai).

Plays random games with the Python engine. Before every step the Python state is loaded into the C++
engine (arknova_diff_cli); the two legal option lists must be identical, and after replaying Python's
random outcomes (shuffle results, pilfered card) the C++ state must equal the Python state.
arknova_diff_cli runs with g_upstream_rules, so the few places where ark-nova-ai departs from the
rulebook (listed at its declaration in arknova_core.h) follow ark-nova-ai here; rule_scenarios.py
checks the rulebook behaviour of those places.

    git -C ~/ark-nova-fork checkout main      # ark-nova-ai as published
    python games/arknova/tools/diff_test.py --ref ~/ark-nova-fork --cli build/.../arknova_diff_cli --games 200 [--maps all]
    git -C ~/ark-nova-fork checkout rulebook  # those places fixed
    python games/arknova/tools/diff_test.py --ref ~/ark-nova-fork --rulebook --cli build/.../arknova_diff_cli --games 200
"""
import argparse
import json
import random
import subprocess
import sys
import time

# Fields stored as sets (or as lists whose order carries no meaning) on either side.
SET_FIELDS = ("upgraded", "hand", "public_hand")
STATE_SET_FIELDS = ("discard", "base_project_deck", "partner_zoos_available", "universities_available")


def canon(v):
    """JSON-ready copy of a Python engine value: tuples become lists, sets sorted lists, keys strings."""
    if isinstance(v, (set, frozenset)):
        return sorted(canon(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in v.items()}
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    raise TypeError(f"cannot export {type(v).__name__}: {v!r}")


def frame_data(data):
    return {k: canon(x) for k, x in data.items() if x is not None}


def export_turn(turn):
    if not turn:
        return {}
    out = {"action": turn["action"], "xtoken": turn["xtoken"],
           "after": [[kind, who, frame_data(data)] for kind, who, data in turn["after"]]}
    if "strength" in turn:
        out["strength"] = turn["strength"]
    if "outer" in turn and turn["outer"]:
        out["outer"] = export_turn(turn["outer"])
    unknown = set(turn) - {"action", "xtoken", "after", "strength", "outer"}
    if unknown:
        raise ValueError(f"unknown turn keys {sorted(unknown)}")
    return out


def export_player(pl):
    hand = sorted(pl.hand)
    return {
        "appeal": pl.appeal, "money": pl.money, "conservation": pl.conservation, "reputation": pl.reputation,
        "xtokens": pl.xtokens, "upgraded": sorted(pl.upgraded), "workers": pl.workers,
        "workers_out": {k: v for k, v in pl.workers_out.items() if v},
        "partner_zoos": list(pl.partner_zoos), "universities": list(pl.universities), "tokens": list(pl.tokens),
        "hand": hand, "public_hand": sorted(set(pl.public_hand) & set(hand)),
        "zoo_cards": list(pl.zoo_cards), "final_cards": list(pl.final_cards),
        "housing": canon(pl.housing),
        "action_tokens": {k: list(v) for k, v in pl.action_tokens.items() if v},
        "card_tokens": canon(pl.card_tokens), "tucked": list(pl.tucked),
        "full_cover_scored": pl.full_cover_scored, "last_action": pl.last_action, "slots": list(pl.slots),
        "buildings": [{"id": b.id, "kind": b.kind, "size": b.size, "cells": [list(c) for c in sorted(b.cells)],
                       "occupied": b.occupied, "used": b.used, "name": b.name, "extra_size": b.extra_size}
                      for b in pl.zoo.buildings],
    }


def export_state(s):
    if s.n != 2:
        raise ValueError("only the 2-player game is supported")
    if set(s.turn_flags) - {"harbor"}:
        raise ValueError(f"turn flags are not supported: {s.turn_flags}")
    return {
        "map": s.map_name, "turn_flags": canon(s.turn_flags),
        "players": [export_player(p) for p in s.players], "current": s.current, "deck": list(s.deck),
        "discard": sorted(s.discard), "display": list(s.display), "final_deck": list(s.final_deck),
        "break_pos": s.break_pos, "break_by": s.break_by, "base_projects": list(s.base_projects),
        "base_project_deck": sorted(s.base_project_deck), "borrowed": canon(s.borrowed),
        "projects": list(s.projects), "project_tokens": canon(s.project_tokens), "donations": s.donations,
        "partner_zoos_available": sorted(s.partner_zoos_available),
        "universities_available": sorted(s.universities_available), "bonus_tiles": canon(s.bonus_tiles),
        "final_card_discarded": s.final_card_discarded, "end_by": s.end_by, "final_turns": s.final_turns,
        "over": s.over, "scores": canon(s.scores), "turn": export_turn(s.turn),
        "stack": [{"kind": f.kind, "player": f.player, "data": frame_data(f.data)} for f in s.stack],
    }


TUPLE_LIST_FIELDS = ("built", "kinds")  # frame data lists of (building kind, size)
UNEXPORTED_STATE_FIELDS = ("rng", "log", "cache_options", "cached_options", "ledger", "causes")


def import_frame_data(kind, data):
    out = {k: [tuple(x) for x in v] if k in TUPLE_LIST_FIELDS else v for k, v in data.items()}
    if kind == "ab_project" and "only" not in out:  # export drops None values; only this field is nullable
        out["only"] = None
    return out


def import_turn(turn):
    if not turn:
        return {}
    out = {"action": turn["action"], "xtoken": turn["xtoken"],
           "after": [(kind, who, import_frame_data(kind, data)) for kind, who, data in turn["after"]]}
    if "strength" in turn:
        out["strength"] = turn["strength"]
    if "outer" in turn:
        out["outer"] = import_turn(turn["outer"])
    return out


def import_player(idx, pl, mapdef):
    from arknova.hexmap import Building, Zoo
    from arknova.state import Player

    zoo = Zoo(mapdef)
    for i, b in enumerate(pl["buildings"]):
        if b["id"] != i:
            raise ValueError(f"building ids must be 0..n-1 in order, got {b['id']} at {i}")
        cells = frozenset(tuple(c) for c in b["cells"])
        zoo.buildings.append(Building(b["id"], b["kind"], b["size"], cells, occupied=b["occupied"], used=b["used"],
                                      name=b["name"], extra_size=b["extra_size"]))
        for c in cells:
            zoo.cell_owner[c] = b["id"]
    return Player(
        idx=idx, zoo=zoo, slots=list(pl["slots"]), appeal=pl["appeal"], money=pl["money"],
        conservation=pl["conservation"], reputation=pl["reputation"], xtokens=pl["xtokens"],
        upgraded=set(pl["upgraded"]), workers=pl["workers"], workers_out=dict(pl["workers_out"]),
        partner_zoos=list(pl["partner_zoos"]), universities=list(pl["universities"]), tokens=list(pl["tokens"]),
        hand=list(pl["hand"]), public_hand=list(pl["public_hand"]), zoo_cards=list(pl["zoo_cards"]),
        final_cards=list(pl["final_cards"]), housing={c: tuple(v) for c, v in pl["housing"].items()},
        action_tokens={a: list(v) for a, v in pl["action_tokens"].items()}, card_tokens=dict(pl["card_tokens"]),
        tucked=list(pl["tucked"]), full_cover_scored=pl["full_cover_scored"], last_action=pl["last_action"])


def import_state(s, rng, map_defs):
    """The inverse of export_state: an ark-nova-ai State from that layout (as both engines write it).
    `rng` becomes the state's random source; the export does not carry one. The log starts empty.
    `map_defs`: ark-nova-ai's map layouts by key (engine.MAP_DEFS)."""
    from arknova.state import Frame, State

    mapdef = map_defs[s["map"]]
    return State(
        players=[import_player(i, pl, mapdef) for i, pl in enumerate(s["players"])], rng=rng, map_name=s["map"],
        current=s["current"], deck=list(s["deck"]), discard=list(s["discard"]), display=list(s["display"]),
        final_deck=list(s["final_deck"]), break_pos=s["break_pos"], break_by=s["break_by"],
        base_projects=list(s["base_projects"]), base_project_deck=list(s["base_project_deck"]),
        borrowed=None if s["borrowed"] is None else tuple(s["borrowed"]), projects=list(s["projects"]),
        project_tokens={c: {int(k): v for k, v in slots.items()} for c, slots in s["project_tokens"].items()},
        donations=s["donations"], partner_zoos_available=set(s["partner_zoos_available"]),
        universities_available=set(s["universities_available"]),
        bonus_tiles={int(k): list(v) for k, v in s["bonus_tiles"].items()},
        final_card_discarded=s["final_card_discarded"], end_by=s["end_by"], final_turns=s["final_turns"],
        over=s["over"], scores=None if s["scores"] is None else list(s["scores"]), turn=import_turn(s["turn"]),
        turn_flags=dict(s["turn_flags"]),
        stack=[Frame(f["kind"], f["player"], import_frame_data(f["kind"], f["data"])) for f in s["stack"]])


def strict_difference(a, b, path="$"):
    """First place where two Python engine values differ, types included (tuple vs list, int vs str keys),
    or None. Set-like player and state fields compare without order."""
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, (list, tuple)):
        if path.rsplit(".", 1)[-1] in SET_FIELDS + STATE_SET_FIELDS:
            a, b = sorted(a), sorted(b)
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = strict_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(map(repr, set(a) ^ set(b)))[:6]}"
        for k in a:
            d = strict_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if hasattr(a, "__dict__") and not isinstance(a, type):
        skip = UNEXPORTED_STATE_FIELDS if type(a).__name__ == "State" else ()
        if type(a).__name__ == "Zoo":
            if a.mapdef is not b.mapdef:
                return f"{path}.mapdef: different map definitions"
            skip = ("mapdef",)
        # Underscored attributes are caches (Player._icons, Zoo._placements).
        da = {k: v for k, v in vars(a).items() if k not in skip and not k.startswith("_")}
        db = {k: v for k, v in vars(b).items() if k not in skip and not k.startswith("_")}
        return strict_difference(da, db, path)
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def normalize(state):
    """Sort the set-like fields so both engines' exports compare equal regardless of member order."""
    for p in state["players"]:
        for k in SET_FIELDS:
            p[k] = sorted(p[k])
    for k in STATE_SET_FIELDS:
        state[k] = sorted(state[k])
    frames = [(f["kind"], f["data"]) for f in state["stack"]]
    turn = state["turn"]
    while turn:
        frames += [(kind, data) for kind, _, data in turn["after"]]
        turn = turn.get("outer")
    for kind, data in frames:
        # Membership-only lists (they also order options, which are compared as sets).
        if kind == "act_association" and "used" in data:
            data["used"] = sorted(data["used"])
        if kind == "extra_action":
            data["actions"] = sorted(data["actions"])
    return state


def dumps(v):
    return json.dumps(v, separators=(",", ":"))


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'python' if k in a else 'c++'}"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}: {a!r} vs {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


class RecordingRandom:
    """Stands in for state.rng and records the outcomes the C++ engine has to replay."""

    def __init__(self, rng):
        self.rng = rng
        self.events = []

    def shuffle(self, x):
        self.rng.shuffle(x)
        self.events.append(("shuffle", list(x)))

    def randrange(self, n):
        i = self.rng.randrange(n)
        self.events.append(("randrange", i))
        return i

    def __getattr__(self, name):
        raise AttributeError(f"the differential test does not replay rng.{name}")


class Cli:
    def __init__(self, path, rulebook=False):
        self.proc = subprocess.Popen([path] + (["--rulebook"] if rulebook else []), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(self, line, reply=True):
        self.proc.stdin.write(line + "\n")
        if not reply:
            return None
        out = self.proc.stdout.readline()
        if not out:
            raise RuntimeError("arknova_diff_cli exited")
        out = out.rstrip("\n")
        if out.startswith("err "):
            raise CppError(out[4:])
        return out


class CppError(Exception):
    pass


def play(engine, cli, seed, map_name, max_steps, dump=None):
    state = engine.new_game(2, seed=seed, map_name=map_name)
    pick = random.Random(seed)
    for n in range(max_steps):
        if state.over:
            return n
        # Python keeps ids of cards that left the hand in public_hand (rules never read it), so a card
        # drawn back in private would count as public. C++ forgets visibility when a card leaves.
        for p in state.players:
            p.public_hand = [c for c in p.public_hand if c in p.hand]
        before = export_state(state)
        where = f"{map_name} seed {seed} step {n} ({state.stack[-1].kind})"
        imported = import_state(json.loads(dumps(before)), state.rng, engine.MAP_DEFS)
        diff = strict_difference(state, imported)
        if diff:
            raise AssertionError(f"{where}: import_state(export_state(s)) differs from s\n  {diff}")
        # Option order follows hand order, which the export does not keep.
        if (sorted((dumps(canon(o.key)), o.label) for o in engine.legal_options(imported))
                != sorted((dumps(canon(o.key)), o.label) for o in engine.legal_options(state))):
            raise AssertionError(f"{where}: the imported state lists other options")
        cli.send("S " + dumps(before))
        py_options = [dumps(canon(o.key)) for o in engine.legal_options(state)]
        cpp_options = [dumps(o) for o in json.loads(cli.send("O"))]
        # Python lists hand-card options in hand order, C++ in card order; actions map to fixed ids anyway.
        if sorted(py_options) != sorted(cpp_options) or len(set(cpp_options)) != len(cpp_options):
            only_py = [o for o in py_options if o not in cpp_options]
            only_cpp = [o for o in cpp_options if o not in py_options]
            raise AssertionError(f"{where}: options differ\n  python only: {only_py[:8]}\n  c++ only: {only_cpp[:8]}\n"
                                 f"  python: {py_options[:12]}\n  c++: {cpp_options[:12]}")
        option = engine.legal_options(state)[pick.randrange(len(py_options))]
        top = state.stack[-1]
        hands = [list(p.hand) for p in state.players]
        recorder = RecordingRandom(state.rng)
        state.rng = recorder
        state.ledger, state.causes, logged = [], [], len(state.log)
        engine.step(state, option)
        py_feed = {"ledger": canon(state.ledger), "log": state.log[logged:]}
        state.ledger = None
        state.rng = recorder.rng
        for kind, value in recorder.events:
            if kind == "shuffle":
                cli.send("C shuffle " + " ".join(value), reply=False)
            else:
                if top.kind != "ab_pilfer":
                    raise AssertionError(f"{where}: randrange outside a pilfer")
                cli.send(f"C pick {hands[top.player][value]}", reply=False)
        try:
            cpp_after = json.loads(cli.send("P " + dumps(canon(option.key))))
        except CppError as e:
            raise AssertionError(f"{where}: c++ step {dumps(canon(option.key))} failed: {e}") from None
        diff = first_difference(normalize(export_state(state)), normalize(cpp_after))
        if diff:
            if dump:
                with open(dump, "w") as f:
                    json.dump({"before": before, "option": canon(option.key), "python": export_state(state),
                               "cpp": cpp_after}, f)
            raise AssertionError(f"{where}: state differs after {dumps(canon(option.key))}\n  {diff}")
        cpp_feed = json.loads(cli.send("R"))
        if cpp_feed != py_feed:
            raise AssertionError(f"{where}: feed differs after {dumps(canon(option.key))}\n"
                                 f"  python: {dumps(py_feed)}\n  c++:    {dumps(cpp_feed)}")
    raise AssertionError(f"{map_name} seed {seed}: game did not finish in {max_steps} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="path of the ark-nova-ai checkout")
    ap.add_argument("--cli", required=True, help="path of the arknova_diff_cli executable")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--maps", default="all",
                    help="comma-separated map keys (map0, plan1..plan8, plana) played in turn, or all")
    ap.add_argument("--dump", help="on a state mismatch, write the state before, the option and both results here")
    ap.add_argument("--rulebook", action="store_true",
                    help="run arknova_diff_cli with --rulebook, for a reference that follows the rulebook "
                         "where ark-nova-ai departs from it (a fork with those places fixed)")
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    from arknova import engine

    maps = list(engine.MAP_DEFS) if args.maps == "all" else args.maps.split(",")
    cli = Cli(args.cli, args.rulebook)
    t0, steps = time.time(), 0
    for g in range(args.seed, args.seed + args.games):
        steps += play(engine, cli, g, maps[g % len(maps)], args.max_steps, args.dump)
    print(f"{args.games} games on {len(maps)} maps, {steps} steps identical ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
