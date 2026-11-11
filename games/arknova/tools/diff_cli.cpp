// Line protocol for differential testing against the Python engine (driven by diff_test.py).
// One command per line, one reply line per command except C:
//   S <state json>         load a state                           -> "ok"
//   O                      legal options of the loaded state      -> JSON list of option keys
//   C shuffle <ids...>     queue a shuffle outcome (the resulting order)
//   C pick <id>            queue a random pick outcome
//   P <option key json>    step with that option, replaying the queued outcomes -> state json
// Any failure replies "err <message>".
// Runs with g_upstream_rules (ark-nova-ai's behaviour) unless started with --rulebook.

#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arknova_core.h"
#include "arknova_json.h"

using namespace board_ai::arknova;

namespace {

struct Outcome {
  bool shuffle = false;
  std::vector<CardIdx> cards;
};

class ReplayChance : public Chance {
 public:
  std::deque<Outcome> queue;

  void shuffle(std::vector<CardIdx>& cards) override {
    Outcome o = next(true);
    std::vector<CardIdx> a = cards, b = o.cards;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) throw std::runtime_error("replayed shuffle is not a permutation of the shuffled cards");
    cards = o.cards;
  }
  CardIdx pick(const std::vector<CardIdx>& candidates) override {
    Outcome o = next(false);
    if (o.cards.size() != 1 || std::find(candidates.begin(), candidates.end(), o.cards[0]) == candidates.end()) {
      throw std::runtime_error("replayed pick is not among the candidates");
    }
    return o.cards[0];
  }

 private:
  Outcome next(bool shuffle) {
    if (queue.empty()) throw std::runtime_error(std::string("no replayed outcome for a ") + (shuffle ? "shuffle" : "pick"));
    Outcome o = queue.front();
    queue.pop_front();
    if (o.shuffle != shuffle) throw std::runtime_error("replayed outcome kind mismatch");
    return o;
  }
};

std::string one_line(std::string s) {
  std::replace(s.begin(), s.end(), '\n', ' ');
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  g_upstream_rules = !(argc > 1 && std::string(argv[1]) == "--rulebook");
  State state;
  ReplayChance chance;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    char cmd = line[0];
    std::string arg = line.size() > 2 ? line.substr(2) : "";
    try {
      if (cmd == 'S') {
        state_from_json(state, arg);
        chance.queue.clear();
        std::cout << "ok\n";
      } else if (cmd == 'O') {
        std::string out = "[";
        for (const Option& o : legal_options(state)) {
          if (out.size() > 1) out += ',';
          out += option_to_json(state, o);
        }
        std::cout << out << "]\n";
      } else if (cmd == 'C') {
        std::istringstream in(arg);
        std::string kind, id;
        in >> kind;
        if (kind != "shuffle" && kind != "pick") throw std::runtime_error("unknown outcome kind " + kind);
        Outcome o;
        o.shuffle = kind == "shuffle";
        while (in >> id) o.cards.push_back(card_by_id(std::stoi(id)));
        chance.queue.push_back(o);
        continue;
      } else if (cmd == 'P') {
        const Option* chosen = nullptr;
        std::vector<Option> options = legal_options(state);
        for (const Option& o : options) {
          if (option_to_json(state, o) == arg) chosen = &o;
        }
        if (!chosen) throw std::runtime_error("option not legal in C++: " + arg);
        step(state, *chosen, chance);
        if (!chance.queue.empty()) throw std::runtime_error("unused replayed outcomes after the step");
        std::cout << state_to_json(state) << "\n";
      } else {
        throw std::runtime_error(std::string("unknown command ") + cmd);
      }
    } catch (const std::exception& e) {
      chance.queue.clear();
      std::cout << "err " << one_line(e.what()) << "\n";
    }
    std::cout.flush();
  }
  return 0;
}
