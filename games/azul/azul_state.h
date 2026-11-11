#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::azul {

constexpr int kRows = 5;
constexpr int kColors = 5;
constexpr int kTargetsPerColor = 6;  // line[0..4] + floor

template <int NPlayers>
struct AzulConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  static constexpr int kFactories = 2 * NPlayers + 1;
  static constexpr int kCenterSource = kFactories;
  static constexpr int kActionSpace = (kFactories + 1) * kColors * kTargetsPerColor;
  static constexpr int kFeatureDim = NPlayers * 61 + NPlayers + kFactories * kColors + kColors + kColors + 4;
};

struct PlayerState {
  std::array<std::uint8_t, kRows> line_len{};
  std::array<std::int8_t, kRows> line_color{{-1, -1, -1, -1, -1}};
  std::array<std::uint8_t, kRows> wall_mask{};
  std::array<std::int8_t, 7> floor{{-1, -1, -1, -1, -1, -1, -1}};
  std::uint8_t floor_count = 0;
  int score = 0;
};

template <int NPlayers>
struct AzulSnapshot {
  using Cfg = AzulConfig<NPlayers>;
  int current_player = 0;
  int first_player_next_round = 0;
  int winner = -1;
  int round_index = 0;
  bool terminal = false;
  bool first_player_token_in_center = true;
  bool shared_victory = false;
  std::array<int, Cfg::kPlayers> scores{};
  std::array<std::array<std::uint8_t, kColors>, Cfg::kFactories> factories{};
  std::array<std::uint8_t, kColors> center{};
  std::array<int, kColors> bag_counts{};
  std::array<int, kColors> box_lid_counts{};
  std::array<PlayerState, Cfg::kPlayers> players{};
};

template <int NPlayers>
struct UndoRecord {
  using Cfg = AzulConfig<NPlayers>;
  bool use_full_snapshot = false;
  AzulSnapshot<NPlayers> full_before{};
  int prev_current_player = 0;
  int prev_first_player_next_round = 0;
  int prev_winner = -1;
  int prev_round_index = 0;
  bool prev_terminal = false;
  bool prev_first_player_token_in_center = true;
  bool prev_shared_victory = false;
  std::array<int, Cfg::kPlayers> prev_scores{};
  std::array<std::uint8_t, kColors> prev_center{};
  bool has_factory_source = false;
  int source_factory_idx = -1;
  std::array<std::uint8_t, kColors> prev_factory_source{};
  PlayerState prev_player{};
};

struct PersistentTreeCache {
  std::vector<int> tree{};
  std::vector<int> chance_buckets{};
  std::unordered_map<StateHash64, int> sig_to_node{};
};

template <int NPlayers>
class AzulState final : public CloneableState<AzulState<NPlayers>> {
 public:
  using Cfg = AzulConfig<NPlayers>;

  AzulState();

  // Visibility schema. Azul partitions:
  //   - all_public: every game-facing field. Factories, center pile,
  //     each player's pattern lines / wall / floor / score, round meta,
  //     the first-player marker, AND bag_counts / box_lid_counts.
  //     Tiles within a color are interchangeable; only counts are real,
  //     and counts are publicly derivable (start total minus what's been
  //     placed/discarded), so every viewer sees them.
  // PlayerState's sub-fields are flattened into top-level schema names
  // (player_line_len, player_wall_mask, ...) since the schema doesn't
  // model nested structs — every consumer addresses fields by string.
  static const viz::VisibilitySchema& schema();

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  void hash_field_bulk(Hasher& h, viz::FieldId fid,
                       const viz::VizTensor& v,
                       int perspective) const override;
  // Walker-driven snapshot wire I/O — typed read/write of one schema
  // slot. All Azul fields are all_public, so every slot reaches these
  // on the visible path; mask_field_slot is unused.
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override { return current_player_; }
  int first_player() const override { return game_first_player_; }
  bool is_terminal() const override { return terminal; }
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override { return winner_; }
  // winner() stays -1 whenever there is no single winner. On a shared
  // victory (still tied after the completed-rows tie-break) winners() is
  // every player tied for first and losers() everyone else, so with 3-4
  // players the ones who are NOT part of the tie still lose (zero-sum:
  // +1/nw each winner, -1/nl each loser). A tie between ALL players — and
  // every 2-player tie — stays a plain draw: both lists empty, all-zero.
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;

  int current_player_ = 0;
  int game_first_player_ = 0;
  int first_player_next_round = 0;
  int winner_ = -1;
  int round_index = 0;
  bool terminal = false;
  bool first_player_token_in_center = true;
  bool shared_victory = false;
  std::array<int, Cfg::kPlayers> scores{};

  std::array<std::array<std::uint8_t, kColors>, Cfg::kFactories> factories{};
  std::array<std::uint8_t, kColors> center{};
  std::array<int, kColors> bag_counts{};
  std::array<int, kColors> box_lid_counts{};
  std::array<PlayerState, Cfg::kPlayers> players{};

  std::vector<UndoRecord<NPlayers>> undo_stack{};
  PersistentTreeCache persistent_tree_cache{};

  void reset_with_seed(std::uint64_t seed) override;
  bool all_sources_empty() const;
  void refill_factories_from_rng(std::mt19937_64& rng);
  int draw_one_tile(std::mt19937_64& rng);
  StateHash64 state_signature() const { return state_hash(); }
  bool is_tree_cache_consistent() const;
};

extern template class AzulState<2>;
extern template class AzulState<3>;
extern template class AzulState<4>;

}  // namespace board_ai::azul
