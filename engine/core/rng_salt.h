#pragma once

#include <cstdint>
#include <string_view>

#include "types.h"

namespace board_ai::rng {

// Deterministic subseed derivation: hash a textual label into the parent
// seed. Single source of truth for any "I need an independent stream
// derived from seed S for purpose L" — replaces ad-hoc magic XOR salts
// scattered across the runtime (0xA17EBABE, 0xDEADBEEF, etc.).
//
// Properties:
//   - Same (parent_seed, label) always produces the same subseed.
//   - Different labels under the same parent_seed are uncorrelated to a
//     splitmix64 / murmur fmix64 standard.
//   - Caller-defined labels are self-documenting at the call site.
inline std::uint64_t derive_subseed(std::uint64_t parent_seed,
                                    std::string_view label) {
  std::uint64_t h = parent_seed;
  for (char c : label) {
    h = murmur3_fmix64(h ^ static_cast<std::uint64_t>(static_cast<unsigned char>(c)));
  }
  return murmur3_fmix64(h ^ kGoldenRatio64);
}

// Same as derive_subseed but mixes in a numeric index after the label —
// convenient for "per-sim N", "per-seat P" style derivations without
// allocating a label string per call.
inline std::uint64_t derive_subseed(std::uint64_t parent_seed,
                                    std::string_view label,
                                    std::uint64_t index) {
  return murmur3_fmix64(derive_subseed(parent_seed, label) ^ (index + kGoldenRatio64));
}

}  // namespace board_ai::rng
