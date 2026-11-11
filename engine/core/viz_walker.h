#pragma once

// viz_walker — single canonical traversal of viz=1 data-axis slots
// for a given perspective.
//
// The walker is a COORDINATE generator. It yields (fid, idx,
// viz_tensor_ref) for every non-internal field × data-axis slot whose
// `viz[idx..., perspective] == 1`. It does NOT read typed payload —
// the game-side caller reads its own `state.foo[idx]` using the
// yielded `idx`. Framework code does not touch typed fields here,
// consistent with the *_field_slot per-slot dispatcher design (game
// owns its layout).
//
// One traversal feeds three consumers:
//   - hash builder (schema_hash.h): hash_field_slot per visible slot
//   - snapshot serializer (snapshot_io.h): read_field_slot / write_field_slot
//   - encoder (per-game net_adapter): reads masked-state kPlaceholder* sentinels
//
// Sharing one traversal keeps "what counts as visible" defined in
// exactly one place.
//
// Walker entries are template visitors — the callback is bound by
// generic parameter so the compiler can inline the lambda body into
// the inner data-axis recursion. The callback receives the field's
// FieldId directly (= its index in `schema.fields`), so the per-slot
// dispatcher in the game can be a single switch on fid instead of a
// chain of string compares — the latter was the dominant cost on
// fields with many slots (6nimmt's bullhead_pile / hand).

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"
#include "viz_runtime.h"

namespace board_ai {
namespace viz {

namespace detail {

// Recurse over data-axis indices in row-major order, calling `fn` for
// each index whose viewer-axis bit is set at `perspective`.
template <typename F>
inline void walk_data_axes_recursive(
    FieldId fid, const VizTensor& v, int perspective,
    std::vector<int>& idx, int axis, F&& fn) {
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    if (v.data[off]) {
      fn(fid, idx, v);
    }
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_data_axes_recursive(fid, v, perspective, idx, axis + 1, fn);
  }
}

}  // namespace detail

// for_each_visible_slot — single canonical visit order:
//   for each field f in schema.fields (declaration order):
//     for each data-axis index idx in row-major order:
//       if viz_get(state, f.name)[idx..., perspective] == 1:
//         fn(f.id, idx, viz_tensor_ref)
//
// Stable across calls: order depends only on schema declaration order
// + field shape, never on hash-map iteration. Two states with the same
// schema produce identical visit sequences for the same perspective.
template <typename F>
inline void for_each_visible_slot(const IGameState& state,
                                  const VisibilitySchema& schema, int perspective,
                                  F&& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get_by_id(state, field.id);
    if (v.empty()) continue;
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_visible_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      if (v.data[off]) {
        std::vector<int> empty_idx;
        fn(field.id, empty_idx, v);
      }
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_data_axes_recursive(field.id, v, perspective, idx, 0, fn);
  }
}

namespace detail {

template <typename F>
inline void walk_hidden_data_axes_recursive(
    FieldId fid, const VizTensor& v, int perspective,
    std::vector<int>& idx, int axis, F&& fn) {
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    if (v.data[off] == 0) {
      fn(fid, idx, v);
    }
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_hidden_data_axes_recursive(fid, v, perspective, idx, axis + 1, fn);
  }
}

}  // namespace detail

// for_each_hidden_slot — dual of for_each_visible_slot.
template <typename F>
inline void for_each_hidden_slot(
    const IGameState& state, const VisibilitySchema& schema, int perspective,
    F&& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get_by_id(state, field.id);
    if (v.empty()) continue;
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_hidden_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      if (v.data[off] == 0) {
        std::vector<int> empty_idx;
        fn(field.id, empty_idx, v);
      }
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_hidden_data_axes_recursive(field.id, v, perspective, idx,
                                            0, fn);
  }
}

namespace detail {

// Full-set: visit every idx, pass through visibility bit so caller can
// route visible vs hidden uniformly. Used by the framework hash to
// pin slot structure (fid × idx) and dispatch the value mix.
template <typename F>
inline void walk_all_data_axes_recursive(
    FieldId fid, const VizTensor& v, int perspective,
    std::vector<int>& idx, int axis, F&& fn) {
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    const bool visible = v.data[off] != 0;
    fn(fid, idx, v, visible);
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_all_data_axes_recursive(fid, v, perspective, idx, axis + 1, fn);
  }
}

}  // namespace detail

// for_each_slot — schema declaration order × row-major idx, every slot
// visited (no viz filter). Callback gets the visibility bit so callers
// (currently the framework hash) can branch on visible vs hidden
// without re-querying viz.
template <typename F>
inline void for_each_slot(const IGameState& state,
                          const VisibilitySchema& schema, int perspective,
                          F&& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get_by_id(state, field.id);
    if (v.empty()) continue;
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      const bool visible = v.data[off] != 0;
      std::vector<int> empty_idx;
      fn(field.id, empty_idx, v, visible);
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_all_data_axes_recursive(field.id, v, perspective, idx, 0,
                                         fn);
  }
}

}  // namespace viz
}  // namespace board_ai
