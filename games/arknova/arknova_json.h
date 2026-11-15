#pragma once

// Canonical JSON of the rules core, shared with ark-nova-ai's Python engine (see
// games/arknova/tools/py_state_json.py for the Python side). Used to import Python game states,
// compare the two engines move by move, and talk to ark-nova-ai's web front end.
//
// Card ids are the printed 3-digit strings ("401"); option keys are the Python tuples as lists.
// Set-like fields (hand, public_hand, discard, base_project_deck, upgraded, available partner
// zoos and universities) are sorted; ordered fields keep their order.

#include <string>

#include "arknova_core.h"

namespace board_ai::arknova {

std::string state_to_json(const State& s);
void state_from_json(State& s, const std::string& json);
std::string card_id_string(CardIdx c);
// ark-nova-ai's board asks these about a state in its layout (its play/rules.py): question
// {"q": name, "p": player, "card": id | "strength": n}; the answer is a JSON value. Names:
// requirements_ok, animal_allowed, can_house, animal_cost, sponsor_level, can_play_sponsor (per card),
// animals_count (per strength), workers_needed ("task": the workers it would take, or null),
// place_preview ("kind": building kind, "cells": [[q, r], ...]: what
// placing it would bring), winners (no "p").
std::string rules_answer(const std::string& state_json, const std::string& question_json);

}  // namespace board_ai::arknova
