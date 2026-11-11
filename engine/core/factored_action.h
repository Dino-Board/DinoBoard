#pragma once

// Factored (autoregressive) decisions.
//
// A game whose native decision has many compound options (a card plus a target building, a
// shape plus a cell, four cards out of eight) can expose each option as a short sequence of
// atomic actions drawn from a fixed action space. The player chooses one atom at a time; the
// atoms chosen so far (the prefix) are part of the game state, and the native option is applied
// once the prefix spells out exactly one option. Every atom is an ordinary ActionId, so search,
// training and policy heads see a small branching factor instead of the full option product.
//
// The helpers below work on the option list of one decision, each option given as its atom
// sequence. Sequences must be non-empty, distinct and prefix-free (no sequence is a proper
// prefix of another), so a completed prefix identifies exactly one option.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"

namespace board_ai {

using AtomSequence = std::vector<ActionId>;

inline bool atom_prefix_matches(const AtomSequence& seq, const AtomSequence& prefix) {
  return seq.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), seq.begin());
}

// Throws unless the sequences are non-empty, distinct and prefix-free.
inline void check_factored_options(const std::vector<AtomSequence>& options) {
  std::vector<const AtomSequence*> sorted;
  sorted.reserve(options.size());
  for (const AtomSequence& s : options) {
    if (s.empty()) throw std::logic_error("factored action: empty atom sequence");
    sorted.push_back(&s);
  }
  std::sort(sorted.begin(), sorted.end(), [](const AtomSequence* a, const AtomSequence* b) { return *a < *b; });
  // In lexicographic order a sequence that is a prefix of another sorts right before it.
  for (std::size_t i = 1; i < sorted.size(); ++i) {
    if (atom_prefix_matches(*sorted[i], *sorted[i - 1])) {
      throw std::logic_error("factored action: an atom sequence is a prefix of another (or a duplicate)");
    }
  }
}

// Distinct next atoms of the options extending `prefix`, ascending. Throws when none does.
inline std::vector<ActionId> factored_next_atoms(const std::vector<AtomSequence>& options,
                                                 const AtomSequence& prefix) {
  std::vector<ActionId> out;
  for (const AtomSequence& s : options) {
    if (s.size() > prefix.size() && atom_prefix_matches(s, prefix)) out.push_back(s[prefix.size()]);
  }
  if (out.empty()) throw std::logic_error("factored action: no option extends the chosen prefix");
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Index of the option spelled out by `prefix`, or -1 while more atoms are needed. Throws when no
// option starts with `prefix`.
inline int factored_completed(const std::vector<AtomSequence>& options, const AtomSequence& prefix) {
  bool extendable = false;
  for (std::size_t i = 0; i < options.size(); ++i) {
    if (!atom_prefix_matches(options[i], prefix)) continue;
    if (options[i].size() == prefix.size()) return static_cast<int>(i);
    extendable = true;
  }
  if (!extendable) throw std::logic_error("factored action: no option starts with the chosen prefix");
  return -1;
}

// Appends atoms while the next atom is forced (a single possible continuation), so the player is
// only asked to choose where there is a real choice. Returns the completed option index or -1.
inline int factored_extend_forced(const std::vector<AtomSequence>& options, AtomSequence& prefix) {
  while (true) {
    int done = factored_completed(options, prefix);
    if (done >= 0) return done;
    std::vector<ActionId> next = factored_next_atoms(options, prefix);
    if (next.size() != 1) return -1;
    prefix.push_back(next[0]);
  }
}

}  // namespace board_ai
