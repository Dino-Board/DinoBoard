#pragma once
#include <vector>
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/game_registry.h"
#include "jaipur_state.h"
#include "jaipur_rules.h"

namespace board_ai::jaipur {

// Feature dim, perspective-rotated (self before opp). See
// docs/plans/jaipur_ai_encoding.md §3.3 for the block-by-block breakdown.
//   1. market_count                                7
//   2. hand_count[self,opp]                       12  (opp masked → 0)
//   3. camel_count[self,opp]                       2
//   4. rupees[self,opp]                            2
//   5. value_token pile sizes                      6
//   6. bonus pile sizes (3 tiers)                  3
//   7. misc public (piles_exh, deck_size, plies)   3
//   8. exchange scratch (phase + 6+6+1+1+1)       16
//   9. tracker summary (sold 12 + net_in 12)      24
//                                              ---- 75
constexpr int kJaipurFeatureDim = 75;

class JaipurFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kJaipurFeatureDim; }

  void encode_features(const IGameState& state, int perspective_player,
                       const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;
};

StateSerializer make_jaipur_serializer();
ActionDescriptor make_jaipur_descriptor();

}  // namespace board_ai::jaipur
