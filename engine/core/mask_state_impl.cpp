// Out-of-line default bodies for IGameState's bulk virtuals
// (`mask_all_hidden_slots`, `mask_field_bulk`, `hash_field_bulk`).
//
// Lives in its own TU because the walker (viz_walker.h) includes
// game_interfaces.h, so the bodies cannot live in game_interfaces.h
// without an include cycle.
//
// Bulk virtuals: framework hot paths (hash, masking) dispatch one
// virtual call PER FIELD instead of per slot. Default implementations
// here walk the field's slots and dispatch back to the per-slot
// virtuals (`hash_field_slot` / `mask_field_slot`), preserving exact
// previous semantics. Games override the bulk virtuals for hot fields
// where a tight inlined loop beats the per-slot dispatch chain
// (6nimmt's bullhead_pile / hand).

#include "game_interfaces.h"

#include <cstddef>
#include <vector>

#include "types.h"  // for Hasher + kHiddenHashSentinel
#include "viz_runtime.h"

namespace board_ai {

void IGameState::mask_all_hidden_slots(
    const viz::VisibilitySchema& schema, int perspective) {
  // Per-field bulk dispatch. Games override `mask_field_bulk` per-field
  // to memset-style fill; the default body below replicates the
  // historic per-slot walk.
  for (const auto& f : schema.fields) {
    const viz::VizTensor& v = viz::viz_get_by_id(*this, f.id);
    if (v.empty()) continue;
    this->mask_field_bulk(f.id, v, perspective);
  }
}

void IGameState::mask_field_bulk(viz::FieldId fid,
                                 const viz::VizTensor& v,
                                 int perspective) {
  // Default: walk hidden slots in this single field and dispatch each
  // to `mask_field_slot`. Bit-equal to the previous schema-wide walk.
  const int data_rank = v.rank() - 1;
  if (data_rank == 0) {
    const std::size_t off = static_cast<std::size_t>(perspective);
    if (v.data[off] == 0) {
      std::vector<int> empty_idx;
      this->mask_field_slot(fid, empty_idx);
    }
    return;
  }
  std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
  std::vector<std::size_t> strides(static_cast<std::size_t>(v.rank()));
  strides.back() = 1;
  for (int i = v.rank() - 2; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] =
        strides[static_cast<std::size_t>(i + 1)] *
        static_cast<std::size_t>(v.shape[i + 1]);
  }
  const std::size_t row_stride = static_cast<std::size_t>(v.viewer_count());
  // Row-major iteration over data axes.
  while (true) {
    std::size_t base = 0;
    for (int i = 0; i < data_rank; ++i) {
      base +=
          static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) *
          strides[static_cast<std::size_t>(i)];
    }
    if (v.data[base + static_cast<std::size_t>(perspective)] == 0) {
      this->mask_field_slot(fid, idx);
    }
    int dim = data_rank - 1;
    while (dim >= 0) {
      ++idx[static_cast<std::size_t>(dim)];
      if (idx[static_cast<std::size_t>(dim)] < v.shape[dim]) break;
      idx[static_cast<std::size_t>(dim)] = 0;
      --dim;
    }
    if (dim < 0) break;
    (void)row_stride;
  }
}

void IGameState::hash_field_bulk(Hasher& h, viz::FieldId fid,
                                 const viz::VizTensor& v,
                                 int perspective) const {
  // Default: walk every slot in this single field and replicate the
  // per-slot framework hash protocol exactly. Visible → game's
  // `hash_field_slot` (value mix); hidden → kHiddenHashSentinel.
  const int data_rank = v.rank() - 1;
  if (data_rank == 0) {
    const std::size_t off = static_cast<std::size_t>(perspective);
    h.add(static_cast<std::size_t>(fid));
    if (v.data[off]) {
      this->hash_field_slot(h, fid, std::vector<int>{});
    } else {
      h.combine(kHiddenHashSentinel);
    }
    return;
  }
  std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
  std::vector<std::size_t> strides(static_cast<std::size_t>(v.rank()));
  strides.back() = 1;
  for (int i = v.rank() - 2; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] =
        strides[static_cast<std::size_t>(i + 1)] *
        static_cast<std::size_t>(v.shape[i + 1]);
  }
  while (true) {
    std::size_t base = 0;
    for (int i = 0; i < data_rank; ++i) {
      base +=
          static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) *
          strides[static_cast<std::size_t>(i)];
    }
    h.add(static_cast<std::size_t>(fid));
    for (int x : idx) h.add(x);
    if (v.data[base + static_cast<std::size_t>(perspective)]) {
      this->hash_field_slot(h, fid, idx);
    } else {
      h.combine(kHiddenHashSentinel);
    }
    int dim = data_rank - 1;
    while (dim >= 0) {
      ++idx[static_cast<std::size_t>(dim)];
      if (idx[static_cast<std::size_t>(dim)] < v.shape[dim]) break;
      idx[static_cast<std::size_t>(dim)] = 0;
      --dim;
    }
    if (dim < 0) break;
  }
}

}  // namespace board_ai
