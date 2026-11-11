#pragma once

#include <array>
#include <cstdint>

#include "liarsdice_state.h"

namespace board_ai::liarsdice {

// 5 dice × 6 faces canonical multiset enumeration.
//   kDicePerPlayer = 5,  kFaceCount = 6 (defined in liarsdice_state.h).
// Total compositions (n_1..n_6) with Σ n_f = 5, n_f ≥ 0 = C(10, 5) = 252.
//
// Used by:
//   - prepare_for_root: enumerate the 252 outcomes to assemble a joint
//     posterior from the belief net's per-face marginals (∏_f p(n_f|f),
//     renormalized).
//   - randomize_unseen: sampled index → write back as dice_count.
//
// (The belief net itself is per-face — output 36 = 6×6 — so does NOT
// produce this index directly. The label extractor likewise emits a
// per-face one-hot, not a 252-index.)
//
// Lexicographic order on (n_1, n_2, n_3, n_4, n_5, n_6) — enumerated with n_1
// outermost and n_6 innermost. Single canonical convention used wherever an
// outcome is named (e.g. randomize_unseen sampling).

constexpr int kOutcomeCount = 252;

using Outcome = std::array<std::int8_t, kFaceCount>;

// 252 → (n_1..n_6). Caller indexes by 0..kOutcomeCount-1.
const Outcome& index_to_outcome(int index);

// (n_1..n_6) → 0..251. Returns -1 if outcome doesn't satisfy Σ=5 / n_f≥0.
int outcome_to_index(const Outcome& outcome);

}  // namespace board_ai::liarsdice
