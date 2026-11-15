"""ark-nova-ai's board refereed by the C++ engine: its game table (arknova/play/table.py) and its rule
questions (arknova/play/rules.py).

HostTable: the host (dinoboard_engine.ExternalHost) holds the truth and plays every move; `state` is an
ark-nova-ai State rebuilt from the host's state (diff_test.import_state) for the board to read. That
State never draws a random number: every chance outcome comes from the host. Its log is the game's log
lines, which the host reports per move.

HostRules: the board's questions about a State, answered by dinoboard_engine.external_rules_query.

Needs ark-nova-ai importable (its checkout on sys.path).
"""
import json
import random

import dinoboard_engine
from diff_test import canon, dumps, export_state, import_state

GAME = "arknova"


class NoRandom(random.Random):
    def random(self):
        raise AssertionError("a state rebuilt from the host drew a random number")

    def getrandbits(self, k):
        raise AssertionError("a state rebuilt from the host drew a random number")


def option_key(v):
    """An option key as ark-nova-ai builds it: JSON lists become tuples, at every depth."""
    return tuple(option_key(x) for x in v) if isinstance(v, list) else v


class HostTable:
    name = "dinoboard"

    def __init__(self, host, log):
        self.host = host
        self.log = log
        self._state = None

    @classmethod
    def new(cls, seed, map_name):
        return cls(dinoboard_engine.ExternalHost(GAME, seed, map_name), [])

    @property
    def state(self):
        if self._state is None:
            from arknova.mapdefs import MAP_DEFS

            host_json = self.host.state_json()
            self._state = import_state(json.loads(host_json), NoRandom(), MAP_DEFS)
            self._state.log = self.log
            self._state._host_json = host_json  # what HostRules asks about; the board never edits this State
        return self._state

    def options(self):
        from arknova.state import Option

        return [Option(option_key(json.loads(key)), text) for key, text in self.host.options()]

    def to_move(self):
        return self.host.to_move

    def step(self, option, record=False):
        feed = json.loads(self.host.step(dumps(canon(option.key))))
        self.log = self.log + feed["log"]
        self._state = None
        return [tuple(item) for item in feed["ledger"]] if record else None

    def clone(self):
        return HostTable(self.host.clone(), list(self.log))


class HostRules:
    def _ask(self, state, question):
        # A HostTable's State carries the host state it was rebuilt from; any other State is exported.
        state_json = state._host_json if hasattr(state, "_host_json") else dumps(export_state(state))
        return json.loads(dinoboard_engine.external_rules_query(GAME, state_json, dumps(question)))

    def requirements_ok(self, state, p, cid):
        return self._ask(state, {"q": "requirements_ok", "p": p, "card": cid})

    def animal_allowed(self, state, p, cid):
        return self._ask(state, {"q": "animal_allowed", "p": p, "card": cid})

    def can_house(self, state, p, cid):
        return self._ask(state, {"q": "can_house", "p": p, "card": cid})

    def animal_cost(self, state, p, cid):
        return self._ask(state, {"q": "animal_cost", "p": p, "card": cid})

    def sponsor_level(self, state, p, cid):
        return self._ask(state, {"q": "sponsor_level", "p": p, "card": cid})

    def can_play_sponsor(self, state, p, cid):
        return self._ask(state, {"q": "can_play_sponsor", "p": p, "card": cid})

    def animals_count(self, state, p, strength):
        return self._ask(state, {"q": "animals_count", "p": p, "strength": strength})

    def workers_needed(self, state, p, task):
        return self._ask(state, {"q": "workers_needed", "p": p, "task": task})

    def winners(self, state):
        return self._ask(state, {"q": "winners"})

    def place_preview(self, state, p, kind, cells):
        return self._ask(state, {"q": "place_preview", "p": p, "kind": kind, "cells": canon(sorted(cells))})
