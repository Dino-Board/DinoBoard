#include "azul_state.h"

#include <algorithm>
#include <functional>

#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::azul {

template <int NPlayers>
const viz::VisibilitySchema& AzulState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "game_first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player_next_round",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "round_index",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player_token_in_center",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "shared_victory",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public 1D / 2D ----
    viz::declare_field(schema, "scores",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(
        schema, "factories",
        viz::all_public({Cfg::kFactories, kColors}, Cfg::kPlayers));
    viz::declare_field(schema, "center",
                       viz::all_public({kColors}, Cfg::kPlayers));

    // ---- per-player public boards (PlayerState sub-fields, flattened) ----
    viz::declare_field(
        schema, "player_line_len",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_line_color",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_wall_mask",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_floor",
        viz::all_public({Cfg::kPlayers, 7}, Cfg::kPlayers));
    viz::declare_field(schema, "player_floor_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_score",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));

    // bag_counts / box_lid_counts: tiles within a color are
    // interchangeable; only counts are real, and counts are publicly
    // derivable (start total minus what's been placed/discarded). Both
    // are first-class all_public schema fields — there is no order to
    // hide.
    viz::declare_field(schema, "bag_counts",
                       viz::all_public({kColors}, Cfg::kPlayers));
    viz::declare_field(schema, "box_lid_counts",
                       viz::all_public({kColors}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
AzulState<NPlayers>::AzulState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void AzulState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  current_player_ = 0;
  game_first_player_ = 0;
  first_player_next_round = 0;
  winner_ = -1;
  round_index = 0;
  terminal = false;
  first_player_token_in_center = true;
  shared_victory = false;
  scores = {};
  factories = {};
  center = {};
  bag_counts = {};
  box_lid_counts = {};
  players = {};
  undo_stack.clear();
  persistent_tree_cache.tree.clear();
  persistent_tree_cache.chance_buckets.clear();
  persistent_tree_cache.sig_to_node.clear();
  // Bag starts with 20 of each color — a public fact.
  for (int c = 0; c < kColors; ++c) {
    bag_counts[c] = 20;
  }
  // Initial draw: build a one-shot rng from the seed (rng is not stored
  // on state — caller of do_action_fast supplies its own rng for
  // subsequent draws). Counts are the only real public fact about the
  // bag; draw_one_tile samples a color weighted by remaining counts.
  std::mt19937_64 init_rng(seed);
  refill_factories_from_rng(init_rng);
  viz::init_viz(*this, schema());
}

template <int NPlayers>
bool AzulState<NPlayers>::all_sources_empty() const {
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      if (c > 0) {
        return false;
      }
    }
  }
  for (std::uint8_t c : center) {
    if (c > 0) {
      return false;
    }
  }
  // The round ends when no TILES are left. The first-player marker is not a
  // tile: if nobody took from the centre this round (every pick emptied a
  // single-colour factory) it simply stays there, nobody is penalised, and
  // first_player_next_round keeps its value — the player who started this
  // round starts the next one too. Requiring the marker to be gone left a
  // non-terminal state with no legal action.
  return true;
}

template <int NPlayers>
std::vector<int> AzulState<NPlayers>::winners() const {
  if (!terminal) return {};
  if (winner_ >= 0) return {winner_};
  // winner_ == -1 without shared_victory is the tail solver's cut-off at a
  // round boundary (do_action_deterministic): not a game result, scored as
  // a draw.
  if (!shared_victory) return {};
  // Same key the rules rank by: score, then completed wall rows.
  auto rows_of = [&](int pid) {
    int rows = 0;
    for (int r = 0; r < kRows; ++r) {
      if (players[static_cast<size_t>(pid)].wall_mask[static_cast<size_t>(r)] == 0b11111U) ++rows;
    }
    return rows;
  };
  int best_score = -1;
  int best_rows = -1;
  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    const int rows = rows_of(pid);
    if (scores[static_cast<size_t>(pid)] > best_score ||
        (scores[static_cast<size_t>(pid)] == best_score && rows > best_rows)) {
      best_score = scores[static_cast<size_t>(pid)];
      best_rows = rows;
    }
  }
  std::vector<int> out;
  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    if (scores[static_cast<size_t>(pid)] == best_score && rows_of(pid) == best_rows) {
      out.push_back(pid);
    }
  }
  if (static_cast<int>(out.size()) == Cfg::kPlayers) return {};  // everyone tied: draw
  return out;
}

template <int NPlayers>
std::vector<int> AzulState<NPlayers>::losers() const {
  const std::vector<int> w = winners();
  if (w.empty()) return {};
  std::vector<int> out;
  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    if (std::find(w.begin(), w.end(), pid) == w.end()) out.push_back(pid);
  }
  return out;
}

template <int NPlayers>
int AzulState<NPlayers>::draw_one_tile(std::mt19937_64& rng) {
  int total = 0;
  for (int c : bag_counts) total += c;
  if (total == 0) {
    // Bag empty: refill from box lid (counts copy, then zero box).
    for (int c = 0; c < kColors; ++c) {
      bag_counts[c] = box_lid_counts[c];
      box_lid_counts[c] = 0;
      total += bag_counts[c];
    }
    if (total == 0) return -1;
  }
  // Weighted uniform sample over remaining tiles. Modulo on a 64-bit
  // mt19937_64 with total <= 100 has bias < 5e-18 — irrelevant.
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int c = 0; c < kColors; ++c) {
    if (pick < bag_counts[c]) {
      --bag_counts[c];
      return c;
    }
    pick -= bag_counts[c];
  }
  return -1;  // unreachable
}

template <int NPlayers>
void AzulState<NPlayers>::refill_factories_from_rng(std::mt19937_64& rng) {
  factories = {};
  center = {};
  for (int f = 0; f < Cfg::kFactories; ++f) {
    for (int i = 0; i < 4; ++i) {
      const int color = draw_one_tile(rng);
      if (color < 0 || color >= kColors) {
        continue;
      }
      factories[f][color] = static_cast<std::uint8_t>(factories[f][color] + 1);
    }
  }
}

template <int NPlayers>
bool AzulState<NPlayers>::is_tree_cache_consistent() const {
  if (persistent_tree_cache.tree.size() < persistent_tree_cache.chance_buckets.size()) {
    return false;
  }
  for (const auto& kv : persistent_tree_cache.sig_to_node) {
    const int idx = kv.second;
    if (idx < 0 || idx >= static_cast<int>(persistent_tree_cache.tree.size())) {
      return false;
    }
  }
  return true;
}

template <int NPlayers>
StateHash64 AzulState<NPlayers>::state_hash() const {
  std::size_t h = 0;
  hash_combine(h,static_cast<std::size_t>(current_player_));
  hash_combine(h,static_cast<std::size_t>(first_player_next_round));
  hash_combine(h,static_cast<std::size_t>(winner_ + 1));
  hash_combine(h,static_cast<std::size_t>(round_index));
  hash_combine(h,static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(first_player_token_in_center ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(shared_victory ? 1 : 0));
  for (int s : scores) {
    hash_combine(h,static_cast<std::size_t>(s));
  }
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      hash_combine(h,static_cast<std::size_t>(c));
    }
  }
  for (std::uint8_t c : center) {
    hash_combine(h,static_cast<std::size_t>(c));
  }
  // bag/box_lid are stored as per-color counts — the only public fact
  // about them. No order to hash.
  for (int count : bag_counts) hash_combine(h, static_cast<std::size_t>(count));
  for (int count : box_lid_counts) hash_combine(h, static_cast<std::size_t>(count));
  for (const auto& p : players) {
    for (std::uint8_t len : p.line_len) {
      hash_combine(h,static_cast<std::size_t>(len));
    }
    for (std::int8_t color : p.line_color) {
      hash_combine(h,static_cast<std::size_t>(color + 1));
    }
    for (std::uint8_t m : p.wall_mask) {
      hash_combine(h,static_cast<std::size_t>(m));
    }
    hash_combine(h,static_cast<std::size_t>(p.floor_count));
    for (std::int8_t f : p.floor) {
      hash_combine(h,static_cast<std::size_t>(f + 1));
    }
    hash_combine(h,static_cast<std::size_t>(p.score));
  }
  return static_cast<StateHash64>(h);
}

// Field IDs match schema declaration order in schema().
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kGameFirstPlayer,
  kFirstPlayerNextRound,
  kWinner,
  kRoundIndex,
  kTerminal,
  kFirstPlayerTokenInCenter,
  kSharedVictory,
  kScores,
  kFactories,
  kCenter,
  kPlayerLineLen,
  kPlayerLineColor,
  kPlayerWallMask,
  kPlayerFloor,
  kPlayerFloorCount,
  kPlayerScore,
  kBagCounts,
  kBoxLidCounts,
};
}  // namespace

// Per-field bulk hash override for Azul's array fields. All Azul fields
// are all_public (viz=1 everywhere), so no visibility check is needed —
// but the per-slot virtual dispatch and idx vector construction in the
// framework default is still overhead worth eliminating for the larger
// arrays. Scalar fields fall through to the default path.
template <int NPlayers>
void AzulState<NPlayers>::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  switch (fid) {
    case kFactories: {
      // 2D: [kFactories][kColors], all_public.
      for (int f = 0; f < Cfg::kFactories; ++f) {
        for (int c = 0; c < kColors; ++c) {
          h.add(static_cast<std::size_t>(fid));
          h.add(f);
          h.add(c);
          h.add(static_cast<int>(
              factories[static_cast<size_t>(f)][static_cast<size_t>(c)]));
        }
      }
      return;
    }
    case kCenter: {
      // 1D: [kColors], all_public.
      for (int c = 0; c < kColors; ++c) {
        h.add(static_cast<std::size_t>(fid));
        h.add(c);
        h.add(static_cast<int>(center[static_cast<size_t>(c)]));
      }
      return;
    }
    case kScores: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(scores[static_cast<size_t>(p)]);
      }
      return;
    }
    case kPlayerLineLen: {
      // 2D: [kPlayers][kRows], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int r = 0; r < kRows; ++r) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(r);
          h.add(static_cast<int>(
              players[static_cast<size_t>(p)].line_len[static_cast<size_t>(r)]));
        }
      }
      return;
    }
    case kPlayerLineColor: {
      // 2D: [kPlayers][kRows], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int r = 0; r < kRows; ++r) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(r);
          h.add(static_cast<int>(
              players[static_cast<size_t>(p)].line_color[static_cast<size_t>(r)]) + 1);
        }
      }
      return;
    }
    case kPlayerWallMask: {
      // 2D: [kPlayers][kRows], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int r = 0; r < kRows; ++r) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(r);
          h.add(static_cast<int>(
              players[static_cast<size_t>(p)].wall_mask[static_cast<size_t>(r)]));
        }
      }
      return;
    }
    case kPlayerFloor: {
      // 2D: [kPlayers][7], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int i = 0; i < 7; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          h.add(static_cast<int>(
              players[static_cast<size_t>(p)].floor[static_cast<size_t>(i)]) + 1);
        }
      }
      return;
    }
    case kPlayerFloorCount: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(static_cast<int>(players[static_cast<size_t>(p)].floor_count));
      }
      return;
    }
    case kPlayerScore: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(players[static_cast<size_t>(p)].score);
      }
      return;
    }
    case kBagCounts: {
      // 1D: [kColors], all_public.
      for (int c = 0; c < kColors; ++c) {
        h.add(static_cast<std::size_t>(fid));
        h.add(c);
        h.add(bag_counts[static_cast<size_t>(c)]);
      }
      return;
    }
    case kBoxLidCounts: {
      // 1D: [kColors], all_public.
      for (int c = 0; c < kColors; ++c) {
        h.add(static_cast<std::size_t>(fid));
        h.add(c);
        h.add(box_lid_counts[static_cast<size_t>(c)]);
      }
      return;
    }
    default:
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers>
void AzulState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer:               h.add(current_player_); return;
    case kGameFirstPlayer:             h.add(game_first_player_); return;
    case kFirstPlayerNextRound:        h.add(first_player_next_round); return;
    case kWinner:                      h.add(winner_ + 1); return;
    case kRoundIndex:                  h.add(round_index); return;
    case kTerminal:                    h.add(terminal ? 1 : 0); return;
    case kFirstPlayerTokenInCenter:    h.add(first_player_token_in_center ? 1 : 0); return;
    case kSharedVictory:               h.add(shared_victory ? 1 : 0); return;
    case kScores:
      h.add(scores[static_cast<size_t>(idx[0])]); return;
    case kCenter:
      h.add(static_cast<int>(center[static_cast<size_t>(idx[0])])); return;
    case kBagCounts:
      h.add(bag_counts[static_cast<size_t>(idx[0])]); return;
    case kBoxLidCounts:
      h.add(box_lid_counts[static_cast<size_t>(idx[0])]); return;
    case kPlayerFloorCount:
      h.add(static_cast<int>(players[static_cast<size_t>(idx[0])].floor_count));
      return;
    case kPlayerScore:
      h.add(players[static_cast<size_t>(idx[0])].score); return;
    case kFactories:
      h.add(static_cast<int>(
          factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
      return;
    case kPlayerLineLen:
      h.add(static_cast<int>(
          players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])]));
      return;
    case kPlayerLineColor:
      h.add(static_cast<int>(
          players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])]) + 1);
      return;
    case kPlayerWallMask:
      h.add(static_cast<int>(
          players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])]));
      return;
    case kPlayerFloor:
      h.add(static_cast<int>(
          players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])]) + 1);
      return;
  }
}

template <int NPlayers>
std::any AzulState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer:               return std::any(current_player_);
    case kGameFirstPlayer:             return std::any(game_first_player_);
    case kFirstPlayerNextRound:        return std::any(first_player_next_round);
    case kWinner:                      return std::any(winner_);
    case kRoundIndex:                  return std::any(round_index);
    case kTerminal:                    return std::any(terminal);
    case kFirstPlayerTokenInCenter:    return std::any(first_player_token_in_center);
    case kSharedVictory:               return std::any(shared_victory);
    case kScores:                      return std::any(scores[static_cast<size_t>(idx[0])]);
    case kCenter:                      return std::any(static_cast<int>(center[static_cast<size_t>(idx[0])]));
    case kBagCounts:                   return std::any(bag_counts[static_cast<size_t>(idx[0])]);
    case kBoxLidCounts:                return std::any(box_lid_counts[static_cast<size_t>(idx[0])]);
    case kPlayerFloorCount:
      return std::any(static_cast<int>(players[static_cast<size_t>(idx[0])].floor_count));
    case kPlayerScore:
      return std::any(players[static_cast<size_t>(idx[0])].score);
    case kFactories:
      return std::any(static_cast<int>(
          factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kPlayerLineLen:
      return std::any(static_cast<int>(
          players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])]));
    case kPlayerLineColor:
      return std::any(static_cast<int>(
          players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])]));
    case kPlayerWallMask:
      return std::any(static_cast<int>(
          players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])]));
    case kPlayerFloor:
      return std::any(static_cast<int>(
          players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])]));
  }
  return {};
}

template <int NPlayers>
void AzulState<NPlayers>::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx,
    const std::any& value) {
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
    case kCurrentPlayer:               current_player_ = as_int(); return;
    case kGameFirstPlayer:             game_first_player_ = as_int(); return;
    case kFirstPlayerNextRound:        first_player_next_round = as_int(); return;
    case kWinner:                      winner_ = as_int(); return;
    case kRoundIndex:                  round_index = as_int(); return;
    case kTerminal:                    terminal = as_bool(); return;
    case kFirstPlayerTokenInCenter:    first_player_token_in_center = as_bool(); return;
    case kSharedVictory:               shared_victory = as_bool(); return;
    case kScores:                      scores[static_cast<size_t>(idx[0])] = as_int(); return;
    case kCenter:
      center[static_cast<size_t>(idx[0])] = static_cast<std::uint8_t>(as_int()); return;
    case kBagCounts:                   bag_counts[static_cast<size_t>(idx[0])] = as_int(); return;
    case kBoxLidCounts:                box_lid_counts[static_cast<size_t>(idx[0])] = as_int(); return;
    case kPlayerFloorCount:
      players[static_cast<size_t>(idx[0])].floor_count = static_cast<std::uint8_t>(as_int());
      return;
    case kPlayerScore:
      players[static_cast<size_t>(idx[0])].score = as_int(); return;
    case kFactories:
      factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::uint8_t>(as_int());
      return;
    case kPlayerLineLen:
      players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])] =
          static_cast<std::uint8_t>(as_int());
      return;
    case kPlayerLineColor:
      players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kPlayerWallMask:
      players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])] =
          static_cast<std::uint8_t>(as_int());
      return;
    case kPlayerFloor:
      players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
  }
}

template class AzulState<2>;
template class AzulState<3>;
template class AzulState<4>;

}  // namespace board_ai::azul
