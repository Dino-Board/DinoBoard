#include "maskmen_state.h"

#include <algorithm>
#include <cstdint>
#include <random>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::maskmen {

namespace {
enum : viz::FieldId {
  kFieldCurrentPlayer = 0,
  kFieldIsTerminal,
  kFieldWinner,
  kFieldLastPlayType,
  kFieldLastPlaySize,
  kFieldIsFirstPlayOfRound,
  kFieldLastPlayerToPlay,
  kFieldPassedThisRound,
  kFieldStrengthAbove,
  kFieldHandCount,
  kFieldHandSize,
};
}  // namespace

template <int NPlayers>
const viz::VisibilitySchema& MaskmenState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "is_terminal",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "last_play_type",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "last_play_size",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "is_first_play_of_round",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "last_player_to_play",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "passed_this_round",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "strength_above",
                       viz::all_public({kTypeCount, kTypeCount},
                                       Cfg::kPlayers));
    viz::declare_field(
        schema, "hand_count",
        viz::owner_only_first_axis({Cfg::kPlayers, kTypeCount},
                                   Cfg::kPlayers));
    viz::declare_field(schema, "hand_size",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    return schema;
  }();
  return s;
}

template <int NPlayers>
MaskmenState<NPlayers>::MaskmenState() = default;

template <int NPlayers>
void MaskmenState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  data = MaskmenData<NPlayers>{};

  // Build the 60-card deck (10 of each type) and shuffle, then deal the
  // first 15*N into per-player count buckets. The surplus 60-15*N cards
  // are silently discarded — they are never read and never enter the
  // hash scope (see PLAN §1.1).
  std::mt19937_64 rng(seed);
  std::array<std::int8_t, kTotalCards> deck{};
  int idx = 0;
  for (int t = 0; t < kTypeCount; ++t) {
    for (int k = 0; k < kCardsPerType; ++k) {
      deck[idx++] = static_cast<std::int8_t>(t);
    }
  }
  for (int i = kTotalCards - 1; i > 0; --i) {
    const int j = static_cast<int>(rng() % static_cast<std::uint64_t>(i + 1));
    std::swap(deck[i], deck[j]);
  }

  const int dealt = NPlayers * kStartingHand;
  for (int i = 0; i < dealt; ++i) {
    const int owner = i / kStartingHand;
    const int t = deck[i];
    data.hand_count[static_cast<size_t>(owner)][static_cast<size_t>(t)] += 1;
  }
  data.hand_size.fill(static_cast<std::int8_t>(kStartingHand));

  data.current_player_ = 0;
  data.terminal = false;
  data.winner = -1;
  data.last_play_type = -1;
  data.last_play_size = 0;
  data.is_first_play_of_round = 1;
  data.last_player_to_play = -1;
  for (auto& p : data.passed_this_round) p = 0;
  for (auto& row : data.strength_above) row.fill(0);

  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 MaskmenState<NPlayers>::state_hash() const {
  return this->state_hash_for_perspective(data.current_player_);
}

template <int NPlayers>
void MaskmenState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kFieldCurrentPlayer:      h.add(d.current_player_); return;
    case kFieldIsTerminal:         h.add(d.terminal ? 1 : 0); return;
    case kFieldWinner:             h.add(d.winner + 1); return;
    // last_play_type is -1 (sentinel) or 0..5; shift by 1 so it stays
    // non-negative for the hash mix.
    case kFieldLastPlayType:       h.add(static_cast<int>(d.last_play_type) + 1); return;
    case kFieldLastPlaySize:       h.add(static_cast<int>(d.last_play_size)); return;
    case kFieldIsFirstPlayOfRound: h.add(static_cast<int>(d.is_first_play_of_round)); return;
    case kFieldLastPlayerToPlay:   h.add(d.last_player_to_play + 1); return;
    case kFieldPassedThisRound:
      h.add(static_cast<int>(d.passed_this_round[static_cast<size_t>(idx[0])]));
      return;
    case kFieldStrengthAbove:
      h.add(static_cast<int>(
          d.strength_above[static_cast<size_t>(idx[0])]
                          [static_cast<size_t>(idx[1])]));
      return;
    case kFieldHandCount:
      // Shift by 1 to keep the hash input non-negative even when the
      // mask path writes placeholder INT8_MIN (in practice we never
      // reach this branch for hidden slots — the walker shortcuts to
      // kHiddenHashSentinel — but be defensive).
      h.add(static_cast<int>(
          d.hand_count[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])]) + 1);
      return;
    case kFieldHandSize:
      h.add(static_cast<int>(d.hand_size[static_cast<size_t>(idx[0])]));
      return;
  }
}

template <int NPlayers>
std::any MaskmenState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kFieldCurrentPlayer:      return std::any(d.current_player_);
    case kFieldIsTerminal:         return std::any(static_cast<bool>(d.terminal));
    case kFieldWinner:             return std::any(d.winner);
    case kFieldLastPlayType:       return std::any(static_cast<int>(d.last_play_type));
    case kFieldLastPlaySize:       return std::any(static_cast<int>(d.last_play_size));
    case kFieldIsFirstPlayOfRound: return std::any(static_cast<int>(d.is_first_play_of_round));
    case kFieldLastPlayerToPlay:   return std::any(d.last_player_to_play);
    case kFieldPassedThisRound:
      return std::any(static_cast<int>(
          d.passed_this_round[static_cast<size_t>(idx[0])]));
    case kFieldStrengthAbove:
      return std::any(static_cast<int>(
          d.strength_above[static_cast<size_t>(idx[0])]
                          [static_cast<size_t>(idx[1])]));
    case kFieldHandCount:
      return std::any(static_cast<int>(
          d.hand_count[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])]));
    case kFieldHandSize:
      return std::any(static_cast<int>(d.hand_size[static_cast<size_t>(idx[0])]));
  }
  return {};
}

template <int NPlayers>
void MaskmenState<NPlayers>::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx,
    const std::any& value) {
  auto& d = data;
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    return 0;
  };
  auto as_bool = [&]() -> bool {
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value);
    if (value.type() == typeid(int)) return std::any_cast<int>(value) != 0;
    return false;
  };
  switch (fid) {
    case kFieldCurrentPlayer:      d.current_player_ = as_int(); return;
    case kFieldIsTerminal:         d.terminal = as_bool(); return;
    case kFieldWinner:             d.winner = as_int(); return;
    case kFieldLastPlayType:
      d.last_play_type = static_cast<std::int8_t>(as_int()); return;
    case kFieldLastPlaySize:
      d.last_play_size = static_cast<std::int8_t>(as_int()); return;
    case kFieldIsFirstPlayOfRound:
      d.is_first_play_of_round = static_cast<std::int8_t>(as_int()); return;
    case kFieldLastPlayerToPlay:
      d.last_player_to_play = as_int(); return;
    case kFieldPassedThisRound:
      d.passed_this_round[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kFieldStrengthAbove:
      d.strength_above[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kFieldHandCount:
      d.hand_count[static_cast<size_t>(idx[0])]
                  [static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kFieldHandSize:
      d.hand_size[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
  }
}

template <int NPlayers>
void MaskmenState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  // Only hand_count[opp][t] is ever hidden for the perspective.
  if (fid == kFieldHandCount) {
    data.hand_count[static_cast<size_t>(idx[0])]
                   [static_cast<size_t>(idx[1])] = kPlaceholderInt8;
  }
}

template <int NPlayers>
int MaskmenState<NPlayers>::current_player() const {
  return data.current_player_;
}

template <int NPlayers>
bool MaskmenState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int MaskmenState<NPlayers>::winner() const {
  return data.winner;
}

template struct MaskmenData<2>;
template struct MaskmenData<3>;
template struct MaskmenData<4>;
template struct MaskmenState<2>;
template struct MaskmenState<3>;
template struct MaskmenState<4>;

}  // namespace board_ai::maskmen
