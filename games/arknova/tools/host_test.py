"""Ark Nova on ark-nova-ai's board with the C++ engine as referee (host_table.HostTable).

Plays random games on the host. Before every step the table's state (rebuilt from the host's) must
list, in ark-nova-ai, exactly the table's options: the same keys (as ark-nova-ai's tuples) and texts.
Every few steps the board's rule questions (host_table.HostRules) about each hand and display card
must get ark-nova-ai's answers, and so must the placement preview of every placement option.
The reference must follow the rulebook (see diff_test.py --rulebook); diff_test.py also checks the
ledger and log lines each move reports.

    python games/arknova/tools/host_test.py --ref ~/ark-nova-fork --games 100 [--maps all]
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from host_table import HostRules, HostTable  # noqa: E402

ANIMAL_QUESTIONS = ("requirements_ok", "animal_allowed", "can_house", "animal_cost")
SPONSOR_QUESTIONS = ("sponsor_level", "can_play_sponsor")
TASKS = ("reputation", "partner_zoo", "university", "project")


def compare_rules(cards, theirs, ours, state, where):
    """The board's rule questions (arknova/play/rules.py) about every hand and display card."""
    for p in range(len(state.players)):
        for cid in sorted(set(state.players[p].hand) | {c for c in state.display if c is not None}):
            kind = cards.card(cid)["type"]
            for q in ANIMAL_QUESTIONS if kind == "animal" else SPONSOR_QUESTIONS if kind == "sponsor" else ():
                a, b = getattr(theirs, q)(state, p, cid), getattr(ours, q)(state, p, cid)
                if a != b:
                    raise AssertionError(f"{where}: {q}(player {p}, card {cid}): ark-nova-ai {a!r}, host {b!r}")
        for strength in range(1, 9):
            a, b = theirs.animals_count(state, p, strength), ours.animals_count(state, p, strength)
            if a != b:
                raise AssertionError(f"{where}: animals_count(player {p}, {strength}): ark-nova-ai {a}, host {b}")
        for task in TASKS:
            a, b = theirs.workers_needed(state, p, task), ours.workers_needed(state, p, task)
            if a != b:
                raise AssertionError(f"{where}: workers_needed(player {p}, {task}): ark-nova-ai {a}, host {b}")


def compare_previews(options, theirs, ours, state, where):
    """place_preview of every placement option, as the board asks it (view.option)."""
    p = state.stack[-1].player
    for o in options:
        if o.key[0] not in ("place", "build", "engineer"):
            continue
        kind = o.key[1] if len(o.key) == 4 else "unique"
        a, b = theirs.place_preview(state, p, kind, o.key[-1]), ours.place_preview(state, p, kind, o.key[-1])
        if a != b:
            raise AssertionError(f"{where}: place_preview {o.key}: ark-nova-ai {a}, host {b}")


def play(engine, seed, map_name, max_steps, rules_every):
    from arknova import cards
    from arknova.play.rules import PythonRules

    their_rules, our_rules = PythonRules(), HostRules()
    table = HostTable.new(seed, map_name)
    pick = random.Random(seed)
    for n in range(max_steps):
        state = table.state
        if table.host.is_terminal:
            if not state.over or sorted(engine.winners(state)) != sorted(table.host.winners) \
                    or sorted(our_rules.winners(state)) != sorted(engine.winners(state)):
                raise AssertionError(f"{map_name} seed {seed}: the end of the game differs")
            return n
        where = f"{map_name} seed {seed} step {n} ({state.stack[-1].kind})"
        if rules_every and n % rules_every == 0:
            compare_rules(cards, their_rules, our_rules, state, where)
        ours = table.options()
        theirs = engine.legal_options(state)
        if len(set(ours)) != len(ours) or set(ours) != set(theirs):
            raise AssertionError(f"{where}: options differ\n  host only: {sorted(map(str, set(ours) - set(theirs)))[:6]}"
                                 f"\n  ark-nova-ai only: {sorted(map(str, set(theirs) - set(ours)))[:6]}")
        if engine.to_move(state) != table.to_move():
            raise AssertionError(f"{where}: to move {table.to_move()} on the host, {engine.to_move(state)} there")
        compare_previews(ours, their_rules, our_rules, state, where)
        table.step(ours[pick.randrange(len(ours))], record=True)
    raise AssertionError(f"{map_name} seed {seed}: game did not finish in {max_steps} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="ark-nova-ai checkout that follows the rulebook")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--maps", default="all", help="comma-separated map keys played in turn, or all")
    ap.add_argument("--rules-every", type=int, default=5,
                    help="compare the board's rule questions every this many steps (0: never)")
    args = ap.parse_args()
    sys.path.insert(0, args.ref)
    from arknova import engine

    maps = list(engine.MAP_DEFS) if args.maps == "all" else args.maps.split(",")
    t0, steps = time.time(), 0
    for g in range(args.seed, args.seed + args.games):
        steps += play(engine, g, maps[g % len(maps)], args.max_steps, args.rules_every)
    print(f"{args.games} games on {len(maps)} maps, {steps} host steps agree ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
