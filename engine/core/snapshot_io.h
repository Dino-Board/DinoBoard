#pragma once

// Snapshot wire I/O — schema-walker driven, single unified protocol.
//
// GT writes the complete observation for a perspective in two halves on
// the same `AnyMap snap`:
//
//   1) Per-field value half: `snap[field_name]` is a `vector<any>` of
//      `(idx_path, value)` pairs — one entry pair for every data slot
//      where `viz_[name][..., perspective] == 1`. `idx_path` is a
//      `vector<int>` (empty for scalar fields). `value` is whatever
//      `state.read_field_slot(fid, idx)` returned (the framework looks
//      up `fid` from the field's name).
//
//   2) Viz-slice half: `snap[kVizSliceKey]` is an `AnyMap` keyed by field
//      name; each value is a flat `vector<int>` of length
//      `prod(data_axes)` carrying `viz_[name][..., perspective]` for
//      every data slot. Receiver wholesale-replaces its own viz slice
//      from this — no derivation, no reset-to-base.
//
//   3) All-viewers half (optional): `snap[kVizAllViewersKey]` is an
//      `AnyMap` keyed by field name, present only when the schema has
//      fields declared `VizSync::kAllViewers`. Each value is the field's
//      whole viz tensor (flat, row-major, viewer axis last). The receiver
//      replaces every viewer column from it, so opponent-perspective
//      masks / hashes / encodings inside its search see what those
//      opponents can really see. A snapshot without this key (older
//      sender, hand-built snapshot) still applies: only the receiver's
//      own column is synced, as before.
//
// Both halves are walked through the schema's full slot set. There is no
// all_public / non-all_public branching. A field declared all_public has
// every slot's viz=1, so its full row appears in the value half; a field
// declared owner_only_first_axis has only the perspective-owned rows in
// the value half. Either way, the receiver just re-applies what the wire
// says — both for values (write_field_slot) and for viz
// (viz::apply_full_slice).

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {
namespace viz {

inline constexpr const char* kVizSliceKey = "__viz__";
inline constexpr const char* kVizAllViewersKey = "__viz_all_viewers__";

namespace detail {

// Number of data slots in a viz tensor (product of all data axes; viz
// shape is data_axes + [n_viewers]).
inline std::size_t data_slot_count(const VizTensor& v) {
  if (v.data.empty()) return 0;
  const int n_viewers = v.viewer_count();
  if (n_viewers <= 0) return 0;
  return v.data.size() / static_cast<std::size_t>(n_viewers);
}

// Extract a vector<int> from std::any, accepting either native vector<int>
// or vector<any>-of-int (pybind round-trips empty Python lists this way).
inline std::vector<int> any_to_int_vec(const std::any& a,
                                       const std::string& ctx) {
  if (a.type() == typeid(std::vector<int>)) {
    return std::any_cast<const std::vector<int>&>(a);
  }
  if (a.type() == typeid(std::vector<std::any>)) {
    const auto& av = std::any_cast<const std::vector<std::any>&>(a);
    std::vector<int> out;
    out.reserve(av.size());
    for (const auto& x : av) {
      if (x.type() != typeid(int)) {
        throw std::invalid_argument(
            ctx + ": vector contains non-int element");
      }
      out.push_back(std::any_cast<int>(x));
    }
    return out;
  }
  throw std::invalid_argument(ctx + ": payload is not vector<int>");
}

}  // namespace detail

// Serialize the perspective's full observation into `snap`. See file
// header for wire format. Fields named in `skip` are written to neither
// half (used for fixed-at-game-start statics like Azul's
// `game_first_player`).
inline void serialize_public_snapshot(
    const IGameState& state, const VisibilitySchema& schema,
    int perspective, AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  if (perspective < 0) {
    throw std::invalid_argument(
        "viz::serialize_public_snapshot: perspective must be >= 0");
  }

  // Value half: walk visible slots, append (idx, value) pairs. Wire
  // keys are field names; internally we use fid to dispatch.
  for_each_visible_slot(
      state, schema, perspective,
      [&](FieldId fid, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        const std::string& name =
            schema.fields[static_cast<std::size_t>(fid)].name;
        if (skip.count(name)) return;
        auto it = snap.find(name);
        if (it == snap.end()) {
          it = snap.emplace(name, std::any(std::vector<std::any>{})).first;
        }
        auto& vec = std::any_cast<std::vector<std::any>&>(it->second);
        vec.push_back(std::any(idx));
        vec.push_back(state.read_field_slot(fid, idx));
      });

  // Viz-slice half: for every schema field, copy the perspective's
  // full data-slot viz row into a flat vector<int>.
  AnyMap viz_slices;
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (f.base_viz.empty()) continue;
    const auto& v = viz_get(const_cast<IGameState&>(state), f.name);
    if (v.shape != f.base_viz.shape) {
      throw std::runtime_error(
          "viz::serialize_public_snapshot: state.viz_ shape mismatch for '" +
          f.name + "'");
    }
    const int n_viewers = v.viewer_count();
    if (perspective >= n_viewers) {
      throw std::out_of_range(
          "viz::serialize_public_snapshot: perspective out of range on '" +
          f.name + "'");
    }
    const std::size_t n_slots = detail::data_slot_count(v);
    std::vector<int> slice;
    slice.reserve(n_slots);
    const std::size_t row_stride = static_cast<std::size_t>(n_viewers);
    for (std::size_t k = 0; k < n_slots; ++k) {
      slice.push_back(static_cast<int>(
          v.data[k * row_stride + static_cast<std::size_t>(perspective)]));
    }
    viz_slices.emplace(f.name, std::any(std::move(slice)));
  }
  snap[kVizSliceKey] = std::any(std::move(viz_slices));

  // All-viewers half: whole viz tensor of every kAllViewers field. Only
  // the who-sees-what pattern is shipped here (declared public by the
  // game); slot VALUES still ride the value half, gated on `perspective`.
  AnyMap all_viewers;
  for (const auto& f : schema.fields) {
    if (f.viz_sync != VizSync::kAllViewers) continue;
    if (skip.count(f.name)) continue;
    if (f.base_viz.empty()) continue;
    const auto& v = viz_get(state, f.name);
    std::vector<int> tensor(v.data.begin(), v.data.end());
    all_viewers.emplace(f.name, std::any(std::move(tensor)));
  }
  if (!all_viewers.empty()) {
    snap[kVizAllViewersKey] = std::any(std::move(all_viewers));
  }
}

// Apply a perspective snapshot onto `state`. Wholesale replaces the
// perspective's viz slice and writes the (idx, value) pairs back via
// `state.write_field_slot`. Fields named in `skip` are ignored on both
// halves.
inline void apply_public_snapshot(
    IGameState& state, const VisibilitySchema& schema,
    int perspective, const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  if (perspective < 0) {
    throw std::invalid_argument(
        "viz::apply_public_snapshot: perspective must be >= 0");
  }

  // Viz-slice half: replace receiver's perspective slice byte-for-byte.
  // Done first so write_field_slot below operates on a state whose viz
  // already reflects what GT shipped (downstream readers that key on
  // viz see a consistent picture).
  auto viz_it = snap.find(kVizSliceKey);
  if (viz_it == snap.end()) {
    throw std::invalid_argument(
        std::string("viz::apply_public_snapshot: snap is missing '") +
        kVizSliceKey + "' viz-slice section");
  }
  if (viz_it->second.type() != typeid(AnyMap)) {
    throw std::invalid_argument(
        std::string("viz::apply_public_snapshot: '") + kVizSliceKey +
        "' value is not an AnyMap");
  }
  // All-viewers half (optional, see file header): applied before the
  // receiver's own slice so the latter stays authoritative for its column.
  auto all_it = snap.find(kVizAllViewersKey);
  if (all_it != snap.end()) {
    if (all_it->second.type() != typeid(AnyMap)) {
      throw std::invalid_argument(
          std::string("viz::apply_public_snapshot: '") + kVizAllViewersKey +
          "' value is not an AnyMap");
    }
    for (const auto& [name, tensor] :
         std::any_cast<const AnyMap&>(all_it->second)) {
      if (skip.count(name)) continue;
      const FieldId fid = schema_lookup(schema, name);
      if (fid == kInvalidFieldId ||
          schema.fields[static_cast<std::size_t>(fid)].viz_sync !=
              VizSync::kAllViewers) {
        throw std::invalid_argument(
            std::string("viz::apply_public_snapshot: '") + kVizAllViewersKey +
            "' names field '" + name +
            "', which the schema does not declare VizSync::kAllViewers");
      }
      apply_all_viewers(
          state, name,
          detail::any_to_int_vec(
              tensor, std::string("viz::apply_public_snapshot: field '") +
                          name + "' all-viewers viz"));
    }
  }

  const auto& viz_slices = std::any_cast<const AnyMap&>(viz_it->second);
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (f.base_viz.empty()) continue;
    auto fit = viz_slices.find(f.name);
    if (fit == viz_slices.end()) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: viz-slice missing field '" +
          f.name + "'");
    }
    const std::vector<int> slice = detail::any_to_int_vec(
        fit->second,
        std::string("viz::apply_public_snapshot: field '") + f.name + "' viz");
    apply_full_slice(state, f.name, perspective, slice);
  }

  // Value half: walk every (name, value-vec) entry and write back via
  // state.write_field_slot. Each entry is a vector<any> alternating
  // idx (vector<int>/vector<any>) and value.
  std::unordered_set<std::string> field_names;
  for (const auto& f : schema.fields) field_names.insert(f.name);

  for (const auto& [name, payload] : snap) {
    if (name == kVizSliceKey || name == kVizAllViewersKey) continue;
    if (skip.count(name)) continue;
    if (!field_names.count(name)) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: snap key '" + name +
          "' is not a schema field");
    }
    if (payload.type() != typeid(std::vector<std::any>)) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: field '" + name +
          "' payload is not vector<any>");
    }
    const auto& vec = std::any_cast<const std::vector<std::any>&>(payload);
    if (vec.size() % 2 != 0) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: field '" + name +
          "' has odd entry count (idx/value pairs expected)");
    }
    const FieldId fid = schema_lookup(schema, name);
    if (fid == kInvalidFieldId) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: schema_lookup failed for '" + name +
          "' (validated against field_names but not found in schema?)");
    }
    for (std::size_t k = 0; k + 1 < vec.size(); k += 2) {
      const std::vector<int> idx = detail::any_to_int_vec(
          vec[k],
          std::string("viz::apply_public_snapshot: field '") + name +
              "' idx");
      state.write_field_slot(fid, idx, vec[k + 1]);
    }
  }
}

// Wrap a game's events-only extractor (game-specific events diff) into
// a full PublicEventExtractor (events + wholesale public_snapshot). The
// snapshot half is framework work — schema-walker driven — so games
// only have to author the diff. Registration sites should prefer this
// helper over hand-writing both extractors.
//
// `schema_provider` lets the caller hand in a per-instance schema (e.g.
// templated per NPlayers) without forcing the helper to be templated.
//
// Returned functor closes over the events-only callable and the schema
// provider; lifetime = the bundle that holds it.
template <typename EventsOnlyFn, typename SchemaProviderFn>
inline auto make_public_event_extractor_from_events_only(
    EventsOnlyFn events_only,
    SchemaProviderFn schema_provider) {
  return [events_only = std::move(events_only),
          schema_provider = std::move(schema_provider)](
      const IGameState& before, ActionId action,
      const IGameState& after, int perspective) -> PublicEventTrace {
    PublicEventTrace out;
    out.events = events_only(before, action, after, perspective);
    serialize_public_snapshot(after, schema_provider(),
                              perspective, out.public_snapshot);
    return out;
  };
}

// Default public-state applier: pure schema-walker invocation. Every
// hidden-info game's applier is identical modulo schema, so the helper
// captures the schema provider and produces a ready-to-register functor.
template <typename SchemaProviderFn>
inline auto make_public_state_applier(SchemaProviderFn schema_provider) {
  return [schema_provider = std::move(schema_provider)](
      IGameState& state, const AnyMap& snap, int receiver_seat) {
    apply_public_snapshot(state, schema_provider(), receiver_seat, snap);
  };
}

// Always-no-op events-only extractor — for games whose tracker carries
// nothing event-derived (e.g. LoveLetter, where everything the tracker
// learns flows through pack_init_payload + state.viz=1 reveals).
inline std::vector<PublicEvent> no_events_extractor(
    const IGameState& /*before*/, ActionId /*action*/,
    const IGameState& /*after*/, int /*perspective*/) {
  return {};
}

}  // namespace viz
}  // namespace board_ai
