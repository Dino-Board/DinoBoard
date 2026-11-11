#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::liarsdice {

constexpr int kDicePerPlayer = 5;
constexpr int kFaceCount = 6;
constexpr int kMinBidFace = 2;  // 1 is wild, can't be bid as Y

// Action layout (per variant): bid slots [0, 5N*5), challenge = 5N*5.
// `action_space(N) = 5N * 5 + 1`.
//   X = action_id / 5 + 1   (1 .. 5N)
//   Y = action_id % 5 + 2   (2 .. 6)
//   challenge = action_id == 5N * 5
//
// Per-variant constants live below; rules / encoder size off `Cfg::kPlayers`.

template <int NPlayers>
struct LiarsdiceConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  static constexpr int kTotalDice = kDicePerPlayer * NPlayers;
  static constexpr int kBidCount = kTotalDice * 5;     // X * Y combos
  static constexpr int kChallengeAction = kBidCount;
  static constexpr int kActionSpace = kBidCount + 1;

  // Encoder feature dim. See LIARSDICE_PLAN §1.6 — own dice count(6) +
  // current_bid_count scalar(1) + current_bid_face one-hot(6) +
  // per-seat last 5 bids [N][5][6 = X dense + Y one-hot 5]. The trailing
  // term scales with N.
  static constexpr int kPerBidWidth = 6;
  static constexpr int kBidsPerSeat = 5;
  static constexpr int kBidHistoryFeatures =
      NPlayers * kBidsPerSeat * kPerBidWidth;
  static constexpr int kFeatureDim =
      6 /*own dice count*/ + 1 /*current_bid_count*/ +
      6 /*current_bid_face one-hot*/ + kBidHistoryFeatures;

  // Belief net input dim: per-seat last 5 bids slice only. See PLAN §1.5.4.
  static constexpr int kBeliefFeatureDim = kBidHistoryFeatures;
};

template <int NPlayers>
struct LiarsdiceData {
  using Cfg = LiarsdiceConfig<NPlayers>;

  // Per-player face counts. dice_count[p][f] holds the number of dice on
  // face (f+1). Σ_f dice_count[p][f] = kDicePerPlayer (single-round; no
  // dice loss). Schema visibility is owner_only_first_axis.
  std::array<std::array<std::int8_t, kFaceCount>, Cfg::kPlayers> dice_count{};

  // Current outstanding bid. (0, 0) sentinel = no bid yet (first move).
  std::int8_t current_bid_count = 0;  // 0..5N
  std::int8_t current_bid_face = 0;   // 0 = no bid; otherwise 2..6

  // Whose turn to act. The last bidder is `current_player_ - 1 (mod N)`
  // when current_bid_count > 0; the framework's challenge resolution
  // reads it back via (current_player + N - 1) % N.
  int current_player_ = 0;

  // Terminal flags.
  bool terminal = false;
  int winner = -1;
};

template <int NPlayers>
struct LiarsdiceState final : public CloneableState<LiarsdiceState<NPlayers>> {
  using Cfg = LiarsdiceConfig<NPlayers>;
  LiarsdiceData<NPlayers> data;

  LiarsdiceState();

  // Visibility schema — see PLAN §1.3 for field-by-field rationale.
  //   - all_public: current_bid_count, current_bid_face, current_player,
  //                 is_terminal, winner.
  //   - owner_only_first_axis: dice_count[N][6]. base viz[p, *, p] = 1.
  //
  // No bid history slot — the bid sequence lives in the tracker (see
  // LiarsdiceBeliefTracker; PLAN §1.5). Schema only carries the current
  // outstanding bid that legal_actions needs to enforce strict
  // monotonicity.
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override { return 0; }
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  std::vector<int> losers() const override;
  std::vector<int> winners() const override;
  bool is_turn_start() const override { return true; }
};

extern template struct LiarsdiceData<2>;
extern template struct LiarsdiceData<3>;
extern template struct LiarsdiceData<4>;
extern template struct LiarsdiceState<2>;
extern template struct LiarsdiceState<3>;
extern template struct LiarsdiceState<4>;

}  // namespace board_ai::liarsdice
