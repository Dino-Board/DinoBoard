#pragma once

#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"

namespace board_ai {

class IBeliefTracker;

// Optional layout of the belief features: `global_dim` features of the position, then
// `entities` blocks of `entity_dim` features, one per output logit (e.g. one per card, whose
// logit says whether an opponent holds it). Lets the belief net score every entity with shared
// weights (belief.json "type": "per_entity"). entities == 0: no layout (one flat vector).
struct BeliefEntityLayout {
  int global_dim = 0;
  int entities = 0;
  int entity_dim = 0;
};

// Optional: the hidden information is, per segment (an opponent), a sorted
// hand of elements of `classes` kinds (cards), and the belief net predicts it
// one element at a time, smallest first (belief.json "type":
// "autoregressive", engine/core/belief_sequence.h). feature_dim() is then the
// width of the position features; the net's input adds the prefix of
// elements chosen so far, and output_logit_count() == classes.
// classes == 0: not autoregressive.
struct BeliefSequenceLayout {
  int classes = 0;
  int segments = 0;
};

// Belief-network feature extractor.
//
// Mirrors IFeatureEncoder's input surface (a perspective-masked state
// + tracker), but produces the input vector for the standalone belief
// network rather than the policy/value network. The belief network's
// output is opp-role posterior logits over the unseen pool — (N-1)
// opponents × K role classes. The extractor runs at MCTS root once per
// decision (cached across sims via IBeliefTracker::prepare_for_root);
// the resulting posterior pi[opp][R] is consumed inside
// `randomize_unseen` to draw the determinization sample (per-game
// shape: multiset categorical for Coup, outcome categorical for
// LiarsDice).
//
// Input contract is identical to IFeatureEncoder:
//   - Reads a perspective-masked state. Slots hidden from
//     `perspective_player` arrive as kPlaceholder*; the extractor must
//     branch on the placeholder value, never on a viz query. There is
//     no path by which truth from other perspectives can leak in.
//   - `tracker` carries public-derived statistics (claim history,
//     reshuffle counts, etc.) — public aggregates only. May be null.
//
// Output contract:
//   - `feature_dim()` is the input width into the belief net.
//   - `output_logit_count()` is (N-1) × K, the flat output dim of the
//     belief net. Per-game; the framework just plumbs it through.
class IBeliefFeatureExtractor {
 public:
  virtual ~IBeliefFeatureExtractor() = default;
  virtual int feature_dim() const = 0;
  virtual int output_logit_count() const = 0;
  // When set: feature_dim() == global_dim + entities * entity_dim and
  // output_logit_count() == entities.
  virtual BeliefEntityLayout entity_layout() const { return {}; }
  // When set: output_logit_count() == classes.
  virtual BeliefSequenceLayout sequence_layout() const { return {}; }

  virtual void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;

  // Convenience: materialize the perspective-masked state once and
  // forward to extract(). Mirrors IFeatureEncoder::encode shape.
  bool extract_from_state(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* features) const {
    auto masked = make_masked_state(state, state.schema_ref(),
                                    perspective_player);
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    extract(*masked, perspective_player, tracker, features);
    return static_cast<int>(features->size()) == feature_dim();
  }
};

}  // namespace board_ai
