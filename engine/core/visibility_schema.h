#pragma once

// Visibility Schema: per-game declarative field visibility.
//
// A game declares, once, every field in its IGameState — its name, the
// shape of its data tensor, and a bool[shape..., NPlayers] visibility
// tensor saying which seats see which slots. There is no
// public/internal/derived flag — visibility is wholly carried by the
// viz tensor. all-1 = fully public; all-0 (or empty) = fully hidden;
// per-seat patterns sit in between. The framework derives:
//   - state_hash_for_perspective(p) (hash all slots with viz[..., p]==1)
//   - encoder feature scope via masked-state placeholders (see masked_state.h)
//   - serialize_public_snapshot / apply_public_snapshot (filter by viz)
//
// Declaration surface only: schema data structures + base-viz builder
// helpers. No "overlay" / "reveal-when" closure machinery — game-side
// viz writes belong exclusively to `do_action_fast` (golden standard I1):
// dynamic visibility changes happen via the rules-side helpers
// `reveal_slot / reveal_slot_to / reset_to_base`. Putting "what flips
// when" into the schema would create a second game-side viz writer and
// break I1. (The framework helper `viz::apply_full_slice` — invoked
// from `viz::apply_public_snapshot` on the receiver — overwrites
// `viz_[name][..., perspective]` byte-for-byte from the wire's
// `__viz__` slice. That is receiver-side reconstruction of GT-side viz,
// not a new writer, and is the only permitted exception.
// test_rules_sole_viz_writer enforces I1 game-side.)
//
// The viz tensor is rank N+1 where N is the data tensor's rank
// (rank=0 for a scalar). Trailing axis is the viewer (NPlayers wide).

#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace board_ai {

namespace viz {

// Static visibility tensor stored as a flat row-major bool vector with
// shape metadata. Last dim is always NPlayers. For a rank-0 (scalar)
// data field with NPlayers=N, shape={N} and data.size()=N. For a 1D
// data field of length L, shape={L, N} and data.size()=L*N.
struct VizTensor {
  std::vector<int> shape;       // includes trailing NPlayers axis
  std::vector<std::uint8_t> data;  // row-major, 0/1 bytes (not bit-packed yet)

  bool empty() const { return data.empty(); }
  int rank() const { return static_cast<int>(shape.size()); }
  int viewer_count() const { return shape.empty() ? 0 : shape.back(); }
};

// Builder helpers — small set of patterns that cover ~95% of fields.

// All viewers see all slots. Useful for fully-public fields like
// scores, board layout, factories.
inline VizTensor all_public(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 1);
  return v;
}

// No viewer sees any slot. For framework-managed bookkeeping that no
// player observes (e.g. step_count when explicitly modeled as a viz
// field). Hash / encoder / walker all naturally skip slots with viz=0
// for the active perspective, so no separate flag is needed.
inline VizTensor all_hidden(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 0);
  return v;
}

// Owner-only along the FIRST data axis: data shape [N, ...] where the
// first axis indexes owner; viewer p sees only slot[p, ...]. Common for
// per-player private fields like hands, influence cards, reserved
// stacks indexed by owner.
//
// data_shape MUST start with N (owner axis). Resulting viz shape is
// data_shape with trailing N (viewer axis); viz[owner, ..., viewer]=1
// iff owner == viewer.
inline VizTensor owner_only_first_axis(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 0);

  // Compute strides for v.shape.
  const int rank = static_cast<int>(v.shape.size());
  std::vector<std::size_t> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * static_cast<std::size_t>(v.shape[i + 1]);
  }

  // For each (owner, viewer) combination set v[owner, *, viewer]=1 iff
  // owner == viewer. Iterate by setting all (owner, *, owner=viewer)
  // slots and leaving others 0.
  std::vector<int> idx(rank, 0);
  while (true) {
    if (idx[0] == idx[rank - 1]) {  // owner axis == viewer axis
      std::size_t off = 0;
      for (int i = 0; i < rank; ++i) {
        off += static_cast<std::size_t>(idx[i]) * strides[i];
      }
      v.data[off] = 1;
    }
    // Increment idx in row-major order.
    int dim = rank - 1;
    while (dim >= 0) {
      ++idx[dim];
      if (idx[dim] < v.shape[dim]) break;
      idx[dim] = 0;
      --dim;
    }
    if (dim < 0) break;
  }
  return v;
}

// FieldId — compact integer index into VisibilitySchema::fields. The
// per-slot virtuals (`hash_field_slot` / `mask_field_slot` /
// `read_field_slot` / `write_field_slot`) take FieldId instead of the
// field's string name so the game-side dispatch can be a single switch
// instead of a chain of string compares (the string-compare chain was
// the dominant cost in the schema-walker hot path on fields with many
// slots like 6nimmt's bullhead_pile / hand). Wire format stays
// string-keyed (I15) — FieldId is purely an internal dispatch handle.
using FieldId = std::int16_t;
constexpr FieldId kInvalidFieldId = -1;

// Which viewer columns of a field's runtime viz a seat's session receives
// on the snapshot wire.
//
//   kReceiverOnly (default) — only the receiving seat's own column
//       (`__viz__` slice). Other viewers' columns keep whatever the
//       session had (schema base + rules run inside a search).
//   kAllViewers — additionally the whole viewer axis, so that a search
//       rooted at this seat masks / hashes / encodes opponent decision
//       nodes with what those opponents can really see (a face-up card,
//       a completed peek).
//
// Declare kAllViewers ONLY when the field's who-sees-which-slot pattern
// is derivable by every seat from public information alone. The pattern
// is shipped verbatim, so a pattern that depends on hidden values would
// leak them. Counter-examples that must stay kReceiverOnly: Love Letter
// `hand` with 3+ players (whether a peek survives a play depends on
// whether the peeked card was the one played — unknown to a third seat);
// Coup `exchange_drawn` (which drawn slot was returned first is private
// to the exchanging player). Fields whose viz never changes need no sync.
enum class VizSync : std::uint8_t { kReceiverOnly = 0, kAllViewers = 1 };

// One field in the game state. `name` doubles as the path key on the
// snapshot wire; `base_viz` is what init_viz copies into state.viz_
// and what reset_to_base restores to. `id` is the field's index in
// `VisibilitySchema::fields`, set by `declare_field`.
struct FieldDecl {
  std::string name;
  VizTensor base_viz;
  FieldId id = kInvalidFieldId;
  VizSync viz_sync = VizSync::kReceiverOnly;
};

// The visibility schema of one game.
struct VisibilitySchema {
  int n_players = 0;
  std::vector<FieldDecl> fields;
};

// Look up a field by name. Returns kInvalidFieldId if not found. Used
// by the snapshot wire path (string → FieldId) and bindings that still
// take a string name from Python.
inline FieldId schema_lookup(const VisibilitySchema& schema,
                             const std::string& name) {
  for (FieldId i = 0; i < static_cast<FieldId>(schema.fields.size()); ++i) {
    if (schema.fields[static_cast<std::size_t>(i)].name == name) return i;
  }
  return kInvalidFieldId;
}

// declare_field — push a FieldDecl onto the schema with name-uniqueness
// + viewer-count consistency checks. Typical use:
//
//   declare_field(schema, "scores", viz::all_public({N}, N));
//   declare_field(schema, "influence", viz::owner_only_first_axis({N, 2}, N));

inline void declare_field(VisibilitySchema& schema, const std::string& name,
                          VizTensor base_viz,
                          VizSync viz_sync = VizSync::kReceiverOnly) {
  for (const auto& existing : schema.fields) {
    if (existing.name == name) {
      throw std::invalid_argument(
          "VisibilitySchema: duplicate field name '" + name + "'");
    }
  }
  // Empty viz is allowed and means "no viz" — the walker skips
  // empty-viz fields, so they are effectively hidden from every viewer
  // without needing a separate flag.
  if (!base_viz.empty() && schema.n_players > 0 &&
      base_viz.viewer_count() != schema.n_players) {
    throw std::invalid_argument(
        "VisibilitySchema: field '" + name + "' viz viewer_count (" +
        std::to_string(base_viz.viewer_count()) +
        ") does not match schema.n_players (" +
        std::to_string(schema.n_players) + ")");
  }

  FieldDecl f;
  f.name = name;
  f.base_viz = std::move(base_viz);
  f.viz_sync = viz_sync;
  f.id = static_cast<FieldId>(schema.fields.size());
  schema.fields.push_back(std::move(f));
}

// flat_offset_data_only — flat byte offset for a multi-index into a
// row-major tensor of shape `shape`. `shape` includes the trailing
// viewer axis; `idx` covers data axes only. Returns the offset of
// `viz[idx..., 0]`; the viewer-axis stride is 1.
inline std::size_t flat_offset_data_only(const std::vector<int>& shape,
                                         const std::vector<int>& idx) {
  if (idx.size() + 1 != shape.size()) {
    throw std::invalid_argument(
        "viz::flat_offset_data_only: index rank does not match tensor rank");
  }
  // Row-major strides for all axes; the viewer-axis stride is 1.
  const int rank = static_cast<int>(shape.size());
  std::size_t stride = 1;
  std::vector<std::size_t> strides(rank);
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<std::size_t>(shape[i]);
  }
  std::size_t off = 0;
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    if (idx[i] < 0 || idx[i] >= shape[i]) {
      throw std::out_of_range(
          "viz::flat_offset_data_only: index out of range on axis " +
          std::to_string(i));
    }
    off += static_cast<std::size_t>(idx[i]) * strides[i];
  }
  return off;
}

// init_viz / reveal_slot / reveal_slot_to / reset_to_base bodies live
// in viz_runtime.h (after game_interfaces.h to break the include cycle).

}  // namespace viz
}  // namespace board_ai
