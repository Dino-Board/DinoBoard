#include "outcome_index.h"

#include <stdexcept>
#include <vector>

namespace board_ai::liarsdice {

namespace {

std::vector<Outcome> build_table() {
  std::vector<Outcome> table;
  table.reserve(kOutcomeCount);
  for (int n1 = kDicePerPlayer; n1 >= 0; --n1) {
    for (int n2 = kDicePerPlayer - n1; n2 >= 0; --n2) {
      for (int n3 = kDicePerPlayer - n1 - n2; n3 >= 0; --n3) {
        for (int n4 = kDicePerPlayer - n1 - n2 - n3; n4 >= 0; --n4) {
          for (int n5 = kDicePerPlayer - n1 - n2 - n3 - n4; n5 >= 0; --n5) {
            int n6 = kDicePerPlayer - n1 - n2 - n3 - n4 - n5;
            Outcome o{static_cast<std::int8_t>(n1),
                      static_cast<std::int8_t>(n2),
                      static_cast<std::int8_t>(n3),
                      static_cast<std::int8_t>(n4),
                      static_cast<std::int8_t>(n5),
                      static_cast<std::int8_t>(n6)};
            table.push_back(o);
          }
        }
      }
    }
  }
  return table;
}

const std::vector<Outcome>& table() {
  static const std::vector<Outcome> t = build_table();
  return t;
}

}  // namespace

const Outcome& index_to_outcome(int index) {
  const auto& t = table();
  if (index < 0 || index >= static_cast<int>(t.size())) {
    throw std::out_of_range("outcome index out of range");
  }
  return t[static_cast<size_t>(index)];
}

int outcome_to_index(const Outcome& outcome) {
  int sum = 0;
  for (int f = 0; f < kFaceCount; ++f) {
    int n = outcome[static_cast<size_t>(f)];
    if (n < 0 || n > kDicePerPlayer) return -1;
    sum += n;
  }
  if (sum != kDicePerPlayer) return -1;

  // Inverse of the lexicographic enumeration above. Walk faces in the same
  // order; at each step compute how many compositions start with the chosen
  // prefix and skip them until we hit the actual one.
  int index = 0;
  int remaining = kDicePerPlayer;
  for (int f = 0; f < kFaceCount - 1; ++f) {
    int n_f = outcome[static_cast<size_t>(f)];
    // Compositions with the prefix fixed up to face f and (remaining - k)
    // dice on this face for k > n_f are enumerated before us. The number
    // of compositions of `r` dice across `m` remaining faces is C(r+m-1,
    // m-1). Here m = kFaceCount - 1 - f (faces strictly after f).
    int m = kFaceCount - 1 - f;
    for (int k = remaining; k > n_f; --k) {
      // C(k - 1 + (m - 1), m - 1) = number of ways to spread (remaining - k)
      // dice across m remaining faces. Wait — careful: when this face holds
      // `k` dice, faces after f hold `remaining - k` dice across m faces.
      // We want stars-and-bars C((remaining - k) + m - 1, m - 1). Compute
      // binomial inline.
      int r = remaining - k;
      int top = r + m - 1;
      int bot = m - 1;
      // Compute C(top, bot) safely for small inputs (top ≤ 9, bot ≤ 4).
      long long c = 1;
      for (int i = 0; i < bot; ++i) {
        c = c * (top - i) / (i + 1);
      }
      index += static_cast<int>(c);
    }
    remaining -= n_f;
  }
  // Final face takes whatever remains; no further branching.
  return index;
}

}  // namespace board_ai::liarsdice
