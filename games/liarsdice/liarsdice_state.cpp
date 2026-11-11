#include "liarsdice_state.h"

#include <cstdint>
#include <random>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::liarsdice {

template <int NPlayers>
const viz::VisibilitySchema& LiarsdiceState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // Public scalars: current outstanding bid + current_player + termination.
    viz::declare_field(schema, "current_bid_count",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "current_bid_face",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "is_terminal",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner",
                       viz::all_public({}, Cfg::kPlayers));

    // Per-player face-count multiset. Owner sees their own counts; opponents
    // see viz=0 until terminal, when rules.do_action_fast (challenge
    // resolution) calls reveal_slot to flip everyone's view to public.
    viz::declare_field(
        schema, "dice_count",
        viz::owner_only_first_axis({Cfg::kPlayers, kFaceCount},
                                   Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
LiarsdiceState<NPlayers>::LiarsdiceState() = default;

template <int NPlayers>
void LiarsdiceState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  data = LiarsdiceData<NPlayers>{};

  // Roll all dice up front. Per PLAN §1.2 — all players simultaneously
  // roll 5 d6 at game start, no decision content. The framework's
  // initialize-from-seed contract owns this RNG; do_action_fast never
  // runs another physical roll because there's no re-roll mechanic in
  // the single-round variant.
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dice(1, kFaceCount);
  for (int p = 0; p < NPlayers; ++p) {
    for (int d = 0; d < kDicePerPlayer; ++d) {
      int face = dice(rng);
      data.dice_count[p][static_cast<size_t>(face - 1)] += 1;
    }
  }

  data.current_bid_count = 0;
  data.current_bid_face = 0;
  data.current_player_ = 0;
  data.terminal = false;
  data.winner = -1;

  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 LiarsdiceState<NPlayers>::state_hash() const {
  return this->state_hash_for_perspective(data.current_player_);
}

namespace {
enum : viz::FieldId {
  kCurrentBidCount = 0,
  kCurrentBidFace,
  kCurrentPlayer,
  kIsTerminal,
  kWinner,
  kDiceCount,
};
}  // namespace

template <int NPlayers>
void LiarsdiceState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentBidCount: h.add(static_cast<int>(d.current_bid_count)); return;
    case kCurrentBidFace:  h.add(static_cast<int>(d.current_bid_face));  return;
    case kCurrentPlayer:   h.add(d.current_player_); return;
    case kIsTerminal:      h.add(d.terminal ? 1 : 0); return;
    case kWinner:          h.add(d.winner + 1); return;
    case kDiceCount:
      h.add(static_cast<int>(
          d.dice_count[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])]) + 7);
      return;
  }
}

template <int NPlayers>
std::any LiarsdiceState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentBidCount: return std::any(static_cast<int>(d.current_bid_count));
    case kCurrentBidFace:  return std::any(static_cast<int>(d.current_bid_face));
    case kCurrentPlayer:   return std::any(static_cast<int>(d.current_player_));
    case kIsTerminal:      return std::any(static_cast<bool>(d.terminal));
    case kWinner:          return std::any(static_cast<int>(d.winner));
    case kDiceCount:
      return std::any(static_cast<int>(
          d.dice_count[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])]));
  }
  return {};
}

template <int NPlayers>
void LiarsdiceState<NPlayers>::write_field_slot(
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
    case kCurrentBidCount:
      d.current_bid_count = static_cast<std::int8_t>(as_int()); return;
    case kCurrentBidFace:
      d.current_bid_face = static_cast<std::int8_t>(as_int()); return;
    case kCurrentPlayer:
      d.current_player_ = as_int(); return;
    case kIsTerminal:
      d.terminal = as_bool(); return;
    case kWinner:
      d.winner = as_int(); return;
    case kDiceCount:
      d.dice_count[static_cast<size_t>(idx[0])]
                  [static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
  }
}

template <int NPlayers>
void LiarsdiceState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  // Walker reaches here only for slots whose viz is 0 for the perspective.
  // Liarsdice's only such slot is dice_count[opp][face] when opp ≠
  // perspective and the game is not yet over.
  if (fid == kDiceCount) {
    data.dice_count[static_cast<size_t>(idx[0])]
                   [static_cast<size_t>(idx[1])] = kPlaceholderInt8;
  }
}

template <int NPlayers>
int LiarsdiceState<NPlayers>::current_player() const {
  return data.current_player_;
}

template <int NPlayers>
bool LiarsdiceState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int LiarsdiceState<NPlayers>::winner() const {
  return data.winner;
}

template <int NPlayers>
std::vector<int> LiarsdiceState<NPlayers>::losers() const {
  if (!data.terminal || data.winner < 0) return {};
  const int challenger = data.current_player_;
  const int bidder = (challenger + NPlayers - 1) % NPlayers;
  const int loser = (data.winner == challenger) ? bidder : challenger;
  return {loser};
}

template <int NPlayers>
std::vector<int> LiarsdiceState<NPlayers>::winners() const {
  if (!data.terminal || data.winner < 0) return {};
  const int challenger = data.current_player_;
  const int bidder = (challenger + NPlayers - 1) % NPlayers;
  const int loser = (data.winner == challenger) ? bidder : challenger;
  std::vector<int> result;
  for (int p = 0; p < NPlayers; ++p) {
    if (p != loser) result.push_back(p);
  }
  return result;
}

template struct LiarsdiceData<2>;
template struct LiarsdiceData<3>;
template struct LiarsdiceData<4>;
template struct LiarsdiceState<2>;
template struct LiarsdiceState<3>;
template struct LiarsdiceState<4>;

}  // namespace board_ai::liarsdice
