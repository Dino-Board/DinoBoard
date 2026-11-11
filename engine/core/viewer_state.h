#pragma once

// Viewer-aware state for consumers outside the engine (the web platform).
//
// A game's `state_serializer` runs on the truth state and may emit every
// hidden value — tests and tools rely on that. Anything that leaves the
// server towards one seat must instead be a function of what that seat can
// see. This header builds the two PROBE states that make such an output
// possible without a per-game redaction list:
//
//   probe A = make_masked_state(truth, viewer)      (hidden slots = sentinel)
//   probe B = probe A with every hidden slot changed to a different value
//
// Both probes are functions of the viewer-visible slots only. A consumer
// serializes both with the game's ordinary serializer and keeps a leaf only
// where the two outputs agree; a leaf that differs depends on a hidden slot
// and is dropped. No truth value of a hidden slot is ever consulted, so no
// such value can reach the output — not even through "is the leaf equal to
// the truth" coincidences.
//
// Who-sees-what of the OTHER seats is itself only public for fields declared
// `VizSync::kAllViewers`; for every other field the other seats' viz columns
// are reset to the schema base in both probes, so a serializer that reads
// viz (e.g. "seats that have seen this hand") cannot leak a private peek.
//
// Failure modes (see docs: ALGORITHM_OVERVIEW.md §5):
//   * State that is NOT a schema slot (a deck order kept outside the
//     schema, an internal flag) is not masked. Serializers must not emit
//     hidden non-schema data on this path; `test_web_no_hidden_info.py`
//     checks it against an observer session of a different seed.
//   * A public fact that is only stored inside hidden slots (a hand size
//     as the sum of hidden per-type counts, "has committed" as a non-zero
//     hidden card) cannot be recovered: the leaf is dropped or, if the
//     serializer only tests emptiness, wrong. Serializers must take such
//     values from public slots.
//   * The serializer runs on out-of-range sentinels and must not index
//     tables with a hidden value without a range check.

#include <any>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {
namespace viz {

namespace detail {

// A value of the same wire type that differs from `v`. Throws for a type
// the snapshot wire does not use for hidden slots: better no answer than
// a probe that silently equals the other one.
inline std::any different_value(const std::any& v, const std::string& field) {
  if (v.type() == typeid(int)) {
    const int x = std::any_cast<int>(v);
    return std::any(x == std::numeric_limits<int>::max() ? x - 1 : x + 1);
  }
  if (v.type() == typeid(bool)) {
    return std::any(!std::any_cast<bool>(v));
  }
  if (v.type() == typeid(std::int8_t)) {
    const std::int8_t x = std::any_cast<std::int8_t>(v);
    return std::any(static_cast<std::int8_t>(
        x == std::numeric_limits<std::int8_t>::max() ? x - 1 : x + 1));
  }
  if (v.type() == typeid(std::int64_t)) {
    const std::int64_t x = std::any_cast<std::int64_t>(v);
    return std::any(x == std::numeric_limits<std::int64_t>::max() ? x - 1
                                                                 : x + 1);
  }
  throw std::runtime_error(
      "viz::make_viewer_probes: hidden slot of field '" + field +
      "' has a value type the probe cannot vary");
}

inline bool same_value(const std::any& a, const std::any& b) {
  if (a.type() != b.type()) return false;
  if (a.type() == typeid(int)) {
    return std::any_cast<int>(a) == std::any_cast<int>(b);
  }
  if (a.type() == typeid(bool)) {
    return std::any_cast<bool>(a) == std::any_cast<bool>(b);
  }
  if (a.type() == typeid(std::int8_t)) {
    return std::any_cast<std::int8_t>(a) == std::any_cast<std::int8_t>(b);
  }
  if (a.type() == typeid(std::int64_t)) {
    return std::any_cast<std::int64_t>(a) == std::any_cast<std::int64_t>(b);
  }
  return false;
}

}  // namespace detail

struct ViewerProbes {
  std::unique_ptr<IGameState> a;
  std::unique_ptr<IGameState> b;
};

inline ViewerProbes make_viewer_probes(const IGameState& truth, int viewer) {
  const VisibilitySchema& schema = truth.schema_ref();
  if (viewer < 0 || viewer >= truth.num_players()) {
    throw std::out_of_range("viz::make_viewer_probes: viewer out of range");
  }
  ViewerProbes out;
  out.a = make_masked_state(truth, schema, viewer);

  // Other seats' viz columns: public only for kAllViewers fields.
  for (const FieldDecl& f : schema.fields) {
    if (f.viz_sync == VizSync::kAllViewers) continue;
    VizTensor& v = viz_get_by_id(*out.a, f.id);
    if (v.empty()) continue;
    if (v.data.size() != f.base_viz.data.size()) {
      throw std::runtime_error(
          "viz::make_viewer_probes: viz shape mismatch on field '" + f.name +
          "'");
    }
    const std::size_t n = static_cast<std::size_t>(v.viewer_count());
    for (std::size_t k = 0; k < v.data.size(); ++k) {
      if (k % n != static_cast<std::size_t>(viewer)) {
        v.data[k] = f.base_viz.data[k];
      }
    }
  }

  out.b = out.a->clone_state();
  IGameState& b = *out.b;
  for_each_hidden_slot(
      *out.a, schema, viewer,
      [&](FieldId fid, const std::vector<int>& idx, const VizTensor&) {
        const std::string& name =
            schema.fields[static_cast<std::size_t>(fid)].name;
        const std::any before = b.read_field_slot(fid, idx);
        if (!before.has_value()) {
          throw std::runtime_error(
              "viz::make_viewer_probes: field '" + name +
              "' has a hidden slot that read_field_slot does not return");
        }
        b.write_field_slot(fid, idx, detail::different_value(before, name));
        if (detail::same_value(before, b.read_field_slot(fid, idx))) {
          throw std::runtime_error(
              "viz::make_viewer_probes: write_field_slot did not change a "
              "hidden slot of field '" + name + "'");
        }
      });
  return out;
}

// True iff `seat` sees at least one slot that `viewer` does not: `seat`
// holds private information from `viewer`'s point of view. When false,
// everything `seat` knows `viewer` knows too — `seat`'s legal actions and
// search statistics then reveal nothing.
inline bool seat_sees_more_than(const IGameState& state, int seat,
                                int viewer) {
  if (seat == viewer) return false;
  const VisibilitySchema& schema = state.schema_ref();
  for (const FieldDecl& f : schema.fields) {
    const VizTensor& v = viz_get_by_id(state, f.id);
    if (v.empty()) continue;
    const std::size_t n = static_cast<std::size_t>(v.viewer_count());
    if (seat < 0 || viewer < 0 || static_cast<std::size_t>(seat) >= n ||
        static_cast<std::size_t>(viewer) >= n) {
      throw std::out_of_range("viz::seat_sees_more_than: seat out of range");
    }
    for (std::size_t k = 0; k < v.data.size(); k += n) {
      if (v.data[k + static_cast<std::size_t>(seat)] != 0 &&
          v.data[k + static_cast<std::size_t>(viewer)] == 0) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace viz
}  // namespace board_ai
