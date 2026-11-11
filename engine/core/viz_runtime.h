#pragma once

// Runtime ops on state.viz_.
//
// Split from visibility_schema.h to break the include cycle: the schema
// header is included by game_interfaces.h (so IGameState can carry
// `std::vector<VizTensor> viz_` indexed by FieldId), and the helpers
// below need the concrete IGameState type.
//
// Golden standard I1: rules' do_action_fast is the SOLE writer of
// state.viz_, calling reveal_slot / reveal_slot_to / reset_to_base.
// Framework readers (hash walker, snapshot serializer) read state.viz_
// directly; encoders never query viz at all (they read masked-state
// kPlaceholder* sentinels, see masked_state.h).
//
// Indexing: state.viz_[fid] gives the VizTensor for the field whose
// FieldId == fid (= position in schema.fields). The schema-walker hot
// path (viz_walker.h, schema_hash.h, mask_state_impl.cpp) uses
// `viz_get_by_id` for O(1) access. The string-keyed `viz_get` overload
// is kept for the rules-side reveal_slot / reveal_slot_to /
// reset_to_base helpers, which fire orders of magnitude less often than
// the per-slot walker callback; it does a one-time linear scan of the
// schema fields per call to map name → fid.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"

namespace board_ai {
namespace viz {

// Initialize state.viz_ from a schema. Called from each game's
// reset_with_seed override. Pushes each FieldDecl::base_viz onto the
// vector in declaration order so state.viz_[field.id] == that field's
// base_viz.
inline void init_viz(IGameState& state, const VisibilitySchema& schema) {
  state.viz_.clear();
  state.viz_.reserve(schema.fields.size());
  for (const auto& f : schema.fields) {
    state.viz_.emplace_back(f.base_viz);
  }
}

// O(1) FieldId-keyed accessor. Used by the schema walker and any
// hot-path framework code. Bounds-checked because a bad fid is a
// schema/registration bug we want to surface immediately.
inline VizTensor& viz_get_by_id(IGameState& state, FieldId fid) {
  if (fid < 0 || static_cast<std::size_t>(fid) >= state.viz_.size()) {
    throw std::invalid_argument(
        "viz::viz_get_by_id: fid " + std::to_string(static_cast<int>(fid)) +
        " out of range (state.viz_.size()=" +
        std::to_string(state.viz_.size()) + ")");
  }
  return state.viz_[static_cast<std::size_t>(fid)];
}
inline const VizTensor& viz_get_by_id(const IGameState& state, FieldId fid) {
  if (fid < 0 || static_cast<std::size_t>(fid) >= state.viz_.size()) {
    throw std::invalid_argument(
        "viz::viz_get_by_id(const): fid " +
        std::to_string(static_cast<int>(fid)) +
        " out of range (state.viz_.size()=" +
        std::to_string(state.viz_.size()) + ")");
  }
  return state.viz_[static_cast<std::size_t>(fid)];
}

// String-keyed lookup — slow path. Each call walks schema.fields to
// resolve name → FieldId. Used by the rules-side reveal helpers and
// the framework snapshot helpers (apply_full_slice). A typo in the
// rules-side name surfaces here as an `invalid_argument`, same as
// before — but the lookup happens against the schema rather than a
// per-state map, so an init mismatch (state.viz_ short of schema) is
// reported separately with a clearer message.
inline VizTensor& viz_get(IGameState& state, const std::string& name) {
  const FieldId fid = schema_lookup(state.schema_ref(), name);
  if (fid == kInvalidFieldId) {
    throw std::invalid_argument(
        "viz::viz_get: schema has no field '" + name + "' — name typo");
  }
  return viz_get_by_id(state, fid);
}
inline const VizTensor& viz_get(const IGameState& state,
                                const std::string& name) {
  const FieldId fid = schema_lookup(state.schema_ref(), name);
  if (fid == kInvalidFieldId) {
    throw std::invalid_argument(
        "viz::viz_get(const): schema has no field '" + name +
        "' — name typo");
  }
  return viz_get_by_id(state, fid);
}

// reveal_slot(state, "field", {idx...}) — flip viz for the named slot
// to all-1 across viewers. `idx` indexes the data-axis slots only; the
// trailing viewer axis is set entirely. For a scalar field (data rank
// 0), pass an empty `idx` to flip the single per-viewer entry.
inline void reveal_slot(IGameState& state, const std::string& name,
                        const std::vector<int>& idx) {
  auto& v = viz_get(state, name);
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    v.data[base + static_cast<std::size_t>(p)] = 1;
  }
}

// reveal_slot_to(state, "field", {idx...}, viewer) — flip viz ON only
// for the named viewer; other viewers' bits left untouched. Used for
// Priest peek and similar targeted reveals.
inline void reveal_slot_to(IGameState& state, const std::string& name,
                           const std::vector<int>& idx, int viewer) {
  auto& v = viz_get(state, name);
  const int n_viewers = v.viewer_count();
  if (viewer < 0 || viewer >= n_viewers) {
    throw std::out_of_range(
        "viz::reveal_slot_to: viewer index out of range on field '" + name + "'");
  }
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  v.data[base + static_cast<std::size_t>(viewer)] = 1;
}

// reset_to_base(state, "field", schema, {idx...}) — restore the named
// slot's viewer bits to the schema's declared base. Used by end-of-round
// resets where rules want to drop all in-round reveals on a slot.
inline void reset_to_base(IGameState& state, const std::string& name,
                          const VisibilitySchema& schema,
                          const std::vector<int>& idx) {
  const FieldId fid = schema_lookup(schema, name);
  if (fid == kInvalidFieldId) {
    throw std::invalid_argument(
        "viz::reset_to_base: schema has no field '" + name + "'");
  }
  const FieldDecl& fd = schema.fields[static_cast<std::size_t>(fid)];
  auto& v = viz_get_by_id(state, fid);
  if (v.shape != fd.base_viz.shape) {
    throw std::runtime_error(
        "viz::reset_to_base: state.viz_ shape mismatch for '" + name + "'");
  }
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    v.data[base + static_cast<std::size_t>(p)] =
        fd.base_viz.data[base + static_cast<std::size_t>(p)];
  }
}

// apply_full_slice(state, "field", perspective, slice) — wholesale-replace
// `viz_[name][..., perspective]` byte-for-byte from a flat per-data-slot
// slice (length = product of data axes). Used ONLY by the framework
// snapshot applier (engine/core/snapshot_io.h). Lives here because it
// needs the concrete IGameState type; the I1 lint
// (test_rules_sole_viz_writer) only scopes `games/<id>/`, so framework
// callers are invisible to it.
//
// `slice[k]` (k = flat data offset) writes
// `viz_data[k * n_viewers + perspective]`. Throws if perspective is out
// of range or slice length doesn't match the data shape.
inline void apply_full_slice(IGameState& state, const std::string& name,
                             int perspective,
                             const std::vector<int>& slice) {
  auto& v = viz_get(state, name);
  const int n_viewers = v.viewer_count();
  if (perspective < 0 || perspective >= n_viewers) {
    throw std::out_of_range(
        "viz::apply_full_slice: perspective out of range on field '" +
        name + "'");
  }
  const std::size_t row_stride = static_cast<std::size_t>(n_viewers);
  const std::size_t total = v.data.size();
  const std::size_t n_data_slots = total / row_stride;
  if (slice.size() != n_data_slots) {
    throw std::invalid_argument(
        "viz::apply_full_slice: slice length " +
        std::to_string(slice.size()) + " != data-slot count " +
        std::to_string(n_data_slots) + " for field '" + name + "'");
  }
  for (std::size_t k = 0; k < n_data_slots; ++k) {
    v.data[k * row_stride + static_cast<std::size_t>(perspective)] =
        static_cast<std::uint8_t>(slice[k] != 0 ? 1 : 0);
  }
}

// apply_all_viewers(state, "field", tensor) — wholesale-replace the whole
// `viz_[name]` tensor (every viewer column) from a flat row-major copy of
// the sender's tensor. Used ONLY by the framework snapshot applier, and
// only for fields declared `VizSync::kAllViewers` (see
// visibility_schema.h) — the receiver's search needs the other seats'
// columns to mask / hash / encode their decision nodes correctly.
inline void apply_all_viewers(IGameState& state, const std::string& name,
                              const std::vector<int>& tensor) {
  auto& v = viz_get(state, name);
  if (tensor.size() != v.data.size()) {
    throw std::invalid_argument(
        "viz::apply_all_viewers: tensor length " +
        std::to_string(tensor.size()) + " != viz size " +
        std::to_string(v.data.size()) + " for field '" + name + "'");
  }
  for (std::size_t k = 0; k < tensor.size(); ++k) {
    v.data[k] = static_cast<std::uint8_t>(tensor[k] != 0 ? 1 : 0);
  }
}

// swap_slot(state, "field", {idx_a...}, {idx_b...}) — swap the entire
// viewer-axis row between two slots. After the call, viz[a, :] holds
// what was at viz[b, :] and vice versa. Used when rules swap the slot
// CONTENTS — viz must follow content so "who has seen this cid" stays
// attached to the cid (e.g. Love Letter King swap).
inline void swap_slot(IGameState& state, const std::string& name,
                      const std::vector<int>& idx_a,
                      const std::vector<int>& idx_b) {
  auto& v = viz_get(state, name);
  const std::size_t base_a = flat_offset_data_only(v.shape, idx_a);
  const std::size_t base_b = flat_offset_data_only(v.shape, idx_b);
  if (base_a == base_b) return;
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    std::swap(v.data[base_a + static_cast<std::size_t>(p)],
              v.data[base_b + static_cast<std::size_t>(p)]);
  }
}

// swap_slot_owned(state, "field", owner_a, owner_b) — convenience for
// owner_only_first_axis fields whose first (and only) data axis is the
// owner index (e.g. LL `hand[player]`). Equivalent to
// swap_slot({owner_a}, {owner_b}) followed by reveal_slot_to(owner_a)
// and reveal_slot_to(owner_b). Captures "swap two owner-held slots and
// keep each owner seeing their new content" in one call.
inline void swap_slot_owned(IGameState& state, const std::string& name,
                            int owner_a, int owner_b) {
  auto& v = viz_get(state, name);
  // Owner-only-first-axis fields have data rank 1 (data shape = {N_owners}).
  // The full viz tensor shape is {N_owners, N_viewers}.
  if (v.shape.size() != 2) {
    throw std::invalid_argument(
        "viz::swap_slot_owned: field '" + name +
        "' is not owner_only_first_axis (data rank != 1); use swap_slot + "
        "manual reveal_slot_to for higher-rank fields");
  }
  if (owner_a == owner_b) return;
  swap_slot(state, name, {owner_a}, {owner_b});
  reveal_slot_to(state, name, {owner_a}, owner_a);
  reveal_slot_to(state, name, {owner_b}, owner_b);
}

// Walker callback signature is now bound by template parameter (see
// viz_walker.h). The previous `SlotVisitor` / `SlotVisitorWithVisibility`
// std::function aliases are gone — they forced an indirect call per
// slot; templating lets the compiler inline the lambda body into the
// data-axis recursion. Callers pass plain lambdas.

}  // namespace viz
}  // namespace board_ai
