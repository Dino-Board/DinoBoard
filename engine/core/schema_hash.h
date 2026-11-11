#pragma once

// Schema-driven hash.
//
// `state_hash_for_perspective(player)` walks the FULL schema (every
// declared slot, visible or hidden to `player`) in declaration order ×
// row-major data-axis indices. For each slot the framework first mixes
// in the slot's STRUCTURAL position (field index in schema + idx
// axes), then dispatches the value mix:
//
//   - visible to player (viz=1): call `hash_field_slot(h, name, idx)`,
//     letting the game mix only the slot's value.
//   - hidden from player (viz=0): mix `kHiddenHashSentinel` directly,
//     so "this slot is hidden" participates in the hash structure
//     without leaking truth.
//
// Framework owns the (field_pos, idx[]) mix so games CANNOT forget to
// distinguish e.g. {hand[1]=5, hand[3]=7} from {hand[0]=5, hand[1]=7}
// (BUG-037). Game-side `hash_field_slot` should mix only the value;
// any `idx`-mixing in game code is now redundant and should be
// removed.
//
// There is NO off-schema escape hatch. Every piece of state that
// participates in DAG node identity must be a schema slot — variable-
// length structures get expressed as fixed-shape count arrays
// (multisets), with any visual ordering reconstructed client-side from
// the action stream. See game_interfaces.h IGameState comment block
// and BUG-037 postmortem for the rationale.
//
// Visit order (pinned by `viz::for_each_slot`): schema declaration
// order × row-major data-axis indices. Two states with the same
// schema and same viz_ produce identical visit sequences, so the hash
// is a pure function of (observation history, viz_) — required for
// DAG transposition correctness (BUG-028, BUG-037).

#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"
#include "types.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {

// Out-of-class definition of IGameState::state_hash_for_perspective.
// Lives here because the schema walker depends on viz_runtime which
// depends on game_interfaces — the body cannot live inline in the
// IGameState class.
inline StateHash64 IGameState::state_hash_for_perspective(int player) const {
  Hasher h;
  h.add(step_count_);
  // Walk the FULL schema field by field. For each field dispatch one
  // virtual call (`hash_field_bulk`) that handles every slot in that
  // field — the framework default body in mask_state_impl.cpp
  // replicates the historic per-slot protocol exactly (mix
  // (fid, idx[]) then game's value or kHiddenHashSentinel), so games
  // that don't override see no behavior change. Games override
  // `hash_field_bulk` for hot fields where a tight inlined loop wins
  // (6nimmt's bullhead_pile / hand).
  //
  // Cost shape vs. the old per-slot template walker:
  //   - field count is small (~10-30); per-field virtual cost is tiny
  //   - per-slot recursion + per-slot `viz_get_by_id` lookups are gone
  const auto& schema = schema_ref();
  for (const auto& field : schema.fields) {
    const viz::VizTensor& v = viz::viz_get_by_id(*this, field.id);
    if (v.empty()) continue;
    if (player < 0 || player >= v.viewer_count()) {
      throw std::out_of_range(
          "IGameState::state_hash_for_perspective: perspective " +
          std::to_string(player) + " out of range for field '" +
          field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    hash_field_bulk(h, field.id, v, player);
  }
  return h.finalize();
}

}  // namespace board_ai
