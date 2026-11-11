#pragma once

// Perspective-masked state — encoder/extractor input contract (golden
// standard §2.5).
//
// `make_masked_state(state, schema, perspective)` clones `state` and
// overwrites every slot whose `viz[..., perspective] == 0` with the
// kPlaceholder sentinel for that slot's element type. Callers feed
// the clone — never the original truth state — into encoder /
// belief-feature-extractor `extract_features`-style methods.
//
// **There is no distinct C++ MaskedState type**, by design. The signature
// `const IGameState&` on encoder and extractor entry points is what
// games already understand; introducing a phantom type would propagate
// through every game's net_adapter, every pybind binding, and every
// optional-component interface for a marginal gain in compile-time
// safety. The structural barrier instead is:
//
//   1. Encoder/extractor public entry points (`encode`,
//      `extract_from_state`) always go through `make_masked_state`
//      before forwarding to the virtual `encode_features` /
//      `extract`. The `*_with_masked` overload exists only so an MCTS
//      sim that already masked once for hashing can reuse the result.
//   2. Every viz=0 slot is overwritten with `kPlaceholder*` — values
//      outside any legitimate range. Any code reading the masked state
//      and seeing kPlaceholder knows the slot is hidden; it can never
//      read another perspective's truth from those slots.
//   3. `test_encoder_respects_hash_scope` (changing opp private must
//      leave encoder output bit-equal) and
//      `test_public_hash_excludes_internal_rng` (60 seeds, hash must
//      not depend on hidden-side bytes) are the regression guards.
//
// In short: the discipline is "always go through `make_masked_state`",
// enforced by the only public encoder/extractor entry points doing so
// themselves, plus tests. There is no `class MaskedState` to forge.

#include <cstdint>
#include <limits>
#include <memory>

#include "game_interfaces.h"

namespace board_ai {

// kPlaceholder sentinels — encoders treat any slot whose value equals
// the corresponding sentinel as "hidden." Values are outside legitimate
// ranges:
//   - kPlaceholderInt32 (INT32_MIN): outside cid / action-id ranges.
//   - kPlaceholderInt8  (INT8_MIN, -128): distinguishable from -1
//     (which is the publicly-observable "empty/revealed" marker).
//   - kPlaceholderBool  (false): bool fields needing a "hidden" state
//     should pair with an int viz-paired marker instead.
constexpr std::int32_t kPlaceholderInt32 = std::numeric_limits<std::int32_t>::min();
constexpr std::int8_t  kPlaceholderInt8  = std::numeric_limits<std::int8_t>::min();
constexpr bool         kPlaceholderBool  = false;

// Returns a fresh clone of `state` with every viz=0 slot (relative to
// `perspective`) overwritten with kPlaceholder.
//
// Body: clone, then call the cloned state's virtual
// `mask_all_hidden_slots` — it walks the schema and dispatches each
// hidden slot back to per-game `mask_field_slot(fid, idx)` for the
// placeholder write. Games whose state shares persistent immutable
// data (e.g. Splendor's shared_ptr<const SplendorData>) override
// `mask_all_hidden_slots` to detach a writable copy first.
inline std::unique_ptr<IGameState> make_masked_state(
    const IGameState& state, const viz::VisibilitySchema& schema,
    int perspective) {
  auto cloned = state.clone_state();
  cloned->mask_all_hidden_slots(schema, perspective);
  return cloned;
}

}  // namespace board_ai
