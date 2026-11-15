#include "azul_net_adapter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

#include "azul_rules.h"

namespace board_ai::azul {

namespace {
constexpr int kFirstPlayerToken = -2;

void append_wall_features(const PlayerState& p, std::vector<float>* out) {
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kColors; ++c) {
      out->push_back(((p.wall_mask[r] >> c) & 1U) ? 1.0f : 0.0f);
    }
  }
}

void append_pattern_features(const PlayerState& p, std::vector<float>* out) {
  for (int row = 0; row < kRows; ++row) {
    std::array<float, kColors> counts{};
    if (p.line_color[row] >= 0) {
      counts[static_cast<size_t>(p.line_color[row])] = static_cast<float>(p.line_len[row]);
    }
    const float cap = static_cast<float>(row + 1);
    for (float v : counts) {
      out->push_back(v / cap);
    }
    out->push_back(static_cast<float>(p.line_len[row]) / cap);
  }
}

void append_floor_features(const PlayerState& p, std::vector<float>* out) {
  std::array<float, 6> counts{};
  for (int i = 0; i < p.floor_count && i < 7; ++i) {
    const int t = p.floor[i];
    if (t >= 0 && t < kColors) {
      counts[static_cast<size_t>(t)] += 1.0f;
    } else if (t == kFirstPlayerToken) {
      counts[5] = 1.0f;
    }
  }
  for (float& v : counts) {
    v /= 7.0f;
    out->push_back(v);
  }
}

}  // namespace

template <int NPlayers>
void AzulFeatureEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const AzulState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 || perspective_player >= Cfg::kPlayers) {
    return;
  }

  // Walls for all players (me first)
  for (int i = 0; i < Cfg::kPlayers; ++i) {
    const int pid = (perspective_player + i) % Cfg::kPlayers;
    append_wall_features(s->players[pid], out);
  }

  // Pattern lines for all players
  for (int i = 0; i < Cfg::kPlayers; ++i) {
    const int pid = (perspective_player + i) % Cfg::kPlayers;
    append_pattern_features(s->players[pid], out);
  }

  // Floors for all players
  for (int i = 0; i < Cfg::kPlayers; ++i) {
    const int pid = (perspective_player + i) % Cfg::kPlayers;
    append_floor_features(s->players[pid], out);
  }

  // Scores
  for (int i = 0; i < Cfg::kPlayers; ++i) {
    const int pid = (perspective_player + i) % Cfg::kPlayers;
    out->push_back(std::min(s->players[pid].score, 200) / 200.0f);
  }

  // Factories
  for (int f = 0; f < Cfg::kFactories; ++f) {
    for (int c = 0; c < kColors; ++c) {
      out->push_back(static_cast<float>(s->factories[f][c]) / 4.0f);
    }
  }

  // Center
  for (int c = 0; c < kColors; ++c) {
    out->push_back(static_cast<float>(s->center[c]) / 20.0f);
  }

  // Bag composition (aggregate count per color — observer-visible:
  // all players know how many of each color remain in the bag).
  for (int c = 0; c < kColors; ++c) {
    out->push_back(static_cast<float>(s->bag_counts[static_cast<size_t>(c)]) / 20.0f);
  }

  // Metadata
  out->push_back(s->first_player_token_in_center ? 1.0f : 0.0f);
  out->push_back(s->current_player_ == perspective_player ? 1.0f : 0.0f);
  int bag_total = 0;
  for (int c : s->bag_counts) bag_total += c;
  out->push_back(std::min(s->round_index, 20) / 20.0f);
  out->push_back(static_cast<float>(bag_total) / 100.0f);
}

template <int NPlayers>
AzulTokenEncoder<NPlayers>::AzulTokenEncoder() {
  static_assert(Cfg::kPlayers <= kSeatSlots);
  static_assert(kGlobalFeatures <= kPileFeatures && kPlayerFeatures <= kPileFeatures &&
                kRowFeatures <= kPileFeatures);
  static_assert(Cfg::kActionSpace == (Cfg::kFactories + 1) * kColors * kTargetsPerColor);
  layout_.num_tokens = kNumTokens;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = 1;  // tiles of a colour are interchangeable: no entity ids
  layout_.num_pointer_heads = kTargetsPerColor;

  layout_.token_types.assign(kNumTokens, kPileToken);
  layout_.token_types[0] = kGlobalToken;
  for (int p = 0; p < Cfg::kPlayers; ++p) layout_.token_types[kPlayerBase + p] = kPlayerToken;
  for (int i = 0; i < Cfg::kPlayers * kRows; ++i) layout_.token_types[kRowBase + i] = kRowToken;

  // Action (source, colour, target) = pointer head `target` on the pile
  // (source, colour); every action is bound to a token.
  layout_.action_token.resize(Cfg::kActionSpace);
  layout_.action_head.resize(Cfg::kActionSpace);
  for (int a = 0; a < Cfg::kActionSpace; ++a) {
    layout_.action_token[static_cast<size_t>(a)] = kPileBase + a / kTargetsPerColor;
    layout_.action_head[static_cast<size_t>(a)] = a % kTargetsPerColor;
  }
}

template <int NPlayers>
void AzulTokenEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  using Rules = AzulRules<NPlayers>;
  const auto* s = dynamic_cast<const AzulState<NPlayers>*>(&state);
  if (!s) throw std::runtime_error("AzulTokenEncoder: state is not an AzulState");
  if (!out) throw std::runtime_error("AzulTokenEncoder: out is null");
  if (perspective_player < 0 || perspective_player >= Cfg::kPlayers) {
    throw std::runtime_error("AzulTokenEncoder: perspective_player out of range");
  }
  const int me = perspective_player;
  auto rel = [me](int pid) { return (pid - me + Cfg::kPlayers) % Cfg::kPlayers; };

  const std::size_t base = out->size();
  out->resize(base + static_cast<std::size_t>(kFeatureDim), 0.0f);
  auto token = [&](int i) { return out->data() + base + static_cast<std::size_t>(i) * kTokenDim; };

  // What this round's settlement will do to a player's wall: full pattern
  // lines are placed top to bottom, so the wall a row is scored against
  // already holds the tiles of the full lines above it.
  struct Settlement {
    std::array<PlayerState, kRows> wall_at_row{};
    std::array<int, kRows> gain{};
    int total_gain = 0;
    bool completes_wall_row = false;
  };
  auto settle = [](const PlayerState& p) {
    Settlement st;
    PlayerState w = p;
    for (int row = 0; row < kRows; ++row) {
      st.wall_at_row[static_cast<size_t>(row)] = w;
      if (w.line_len[row] != row + 1 || w.line_color[row] < 0) continue;
      const int col = Rules::wall_col_for_color(row, w.line_color[row]);
      const auto bit = static_cast<std::uint8_t>(1U << col);
      if ((w.wall_mask[row] & bit) != 0) continue;
      w.wall_mask[row] = static_cast<std::uint8_t>(w.wall_mask[row] | bit);
      st.gain[static_cast<size_t>(row)] = Rules::score_wall_placement(w, row, col);
      st.total_gain += st.gain[static_cast<size_t>(row)];
    }
    for (int row = 0; row < kRows; ++row) {
      if (w.wall_mask[row] == 0b11111U) st.completes_wall_row = true;
    }
    return st;
  };
  std::array<Settlement, Cfg::kPlayers> settlement;
  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    settlement[static_cast<size_t>(pid)] = settle(s->players[pid]);
  }
  auto on_wall = [](const PlayerState& p, int row, int col) {
    return ((p.wall_mask[row] >> col) & 1U) != 0U;
  };
  // The rule of legal_actions for one pattern line.
  auto fits = [&](const PlayerState& p, int row, int color) {
    if (p.line_len[row] >= row + 1) return false;
    if (p.line_len[row] > 0 && p.line_color[row] != color) return false;
    return !on_wall(p, row, Rules::wall_col_for_color(row, color));
  };

  // Global: [0] round, [1..5] bag by colour, [6..10] box lid by colour,
  // [11] bag total, [12] first-player marker in the centre, [13] perspective
  // is to move, [14] tiles left on factories and centre, [15] piles left
  // (= picks left this round at most).
  {
    float* t = token(0);
    t[0] = 1.0f;
    float* f = t + 2;
    f[0] = static_cast<float>(std::min(s->round_index, 10)) / 10.0f;
    int bag_total = 0;
    for (int c = 0; c < kColors; ++c) {
      f[1 + c] = static_cast<float>(s->bag_counts[static_cast<size_t>(c)]) / 20.0f;
      f[6 + c] = static_cast<float>(s->box_lid_counts[static_cast<size_t>(c)]) / 20.0f;
      bag_total += s->bag_counts[static_cast<size_t>(c)];
    }
    f[11] = static_cast<float>(bag_total) / 100.0f;
    f[12] = s->first_player_token_in_center ? 1.0f : 0.0f;
    f[13] = s->current_player_ == me ? 1.0f : 0.0f;
    int tiles = 0;
    int piles = 0;
    for (int c = 0; c < kColors; ++c) {
      for (int fi = 0; fi < Cfg::kFactories; ++fi) {
        tiles += s->factories[fi][c];
        piles += s->factories[fi][c] > 0 ? 1 : 0;
      }
      tiles += s->center[c];
      piles += s->center[c] > 0 ? 1 : 0;
    }
    f[14] = static_cast<float>(tiles) / static_cast<float>(4 * Cfg::kFactories);
    f[15] = static_cast<float>(piles) / 20.0f;
  }

  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    const PlayerState& p = s->players[pid];
    const Settlement& st = settlement[static_cast<size_t>(pid)];

    // Player: [0..3] seat relative to the perspective, [4] to move, [5] took
    // the first-player marker (starts next round), [6] score, [7] floor
    // tiles, [8] floor penalty so far, [9] penalty of the next floor tile,
    // [10..14] floor tiles by colour, [15] points the full lines will score
    // at settlement, [16] settlement completes a wall row (game ends),
    // [17] end-of-game bonus of the wall as it is, [18..22] wall tiles per
    // column, [23..27] wall tiles per colour.
    {
      float* t = token(kPlayerBase + rel(pid));
      t[0] = 1.0f;
      float* f = t + 2;
      f[rel(pid)] = 1.0f;
      f[4] = s->current_player_ == pid ? 1.0f : 0.0f;
      f[5] = (!s->first_player_token_in_center && s->first_player_next_round == pid) ? 1.0f : 0.0f;
      f[6] = static_cast<float>(p.score) / 100.0f;
      f[7] = static_cast<float>(p.floor_count) / 7.0f;
      f[8] = static_cast<float>(Rules::floor_penalty(0, p.floor_count)) / 14.0f;
      f[9] = static_cast<float>(Rules::floor_penalty(p.floor_count, 1)) / 3.0f;
      for (int i = 0; i < p.floor_count && i < 7; ++i) {
        if (p.floor[i] >= 0 && p.floor[i] < kColors) f[10 + p.floor[i]] += 1.0f / 7.0f;
      }
      f[15] = static_cast<float>(st.total_gain) / 20.0f;
      f[16] = st.completes_wall_row ? 1.0f : 0.0f;
      f[17] = static_cast<float>(Rules::final_bonus(p)) / 50.0f;
      for (int row = 0; row < kRows; ++row) {
        for (int c = 0; c < kColors; ++c) {
          if (on_wall(p, row, c)) f[18 + c] += 0.2f;
          if (on_wall(p, row, Rules::wall_col_for_color(row, c))) f[23 + c] += 0.2f;
        }
      }
    }

    // Pattern row: [0..3] owner seat, [4..8] row, [9..13] colour of the
    // line, [14] filled share, [15] free places, [16] full, [17..21] colour
    // already on this wall row, [22..26] wall row by column, [27..31] points
    // a tile of each colour would score in this row at settlement (0 where
    // the colour is on the wall), [32] points the line will score (0 unless
    // full).
    for (int row = 0; row < kRows; ++row) {
      float* t = token(kRowBase + rel(pid) * kRows + row);
      t[0] = 1.0f;
      float* f = t + 2;
      const int cap = row + 1;
      const int len = p.line_len[row];
      f[rel(pid)] = 1.0f;
      f[4 + row] = 1.0f;
      if (len > 0 && p.line_color[row] >= 0 && p.line_color[row] < kColors) f[9 + p.line_color[row]] = 1.0f;
      f[14] = static_cast<float>(len) / static_cast<float>(cap);
      f[15] = static_cast<float>(cap - len) / 5.0f;
      f[16] = len == cap ? 1.0f : 0.0f;
      const PlayerState& wall = st.wall_at_row[static_cast<size_t>(row)];
      for (int c = 0; c < kColors; ++c) {
        const int col = Rules::wall_col_for_color(row, c);
        f[22 + c] = on_wall(p, row, c) ? 1.0f : 0.0f;
        if (on_wall(p, row, col)) {
          f[17 + c] = 1.0f;
        } else {
          f[27 + c] = static_cast<float>(Rules::score_wall_placement(wall, row, col)) / 10.0f;
        }
      }
      f[32] = static_cast<float>(st.gain[static_cast<size_t>(row)]) / 10.0f;
    }
  }

  // Pile: [0] in the centre, [1] taking it also takes the first-player
  // marker, [2..6] colour, [7] tiles, [8..12] the other tiles on its factory
  // by colour (they go to the centre), then for the perspective player and
  // each pattern line r, [13 + 5r ..]: line accepts the colour, tiles that
  // fit, line becomes full, points it then scores at settlement, floor
  // points lost to the overflow (and the marker); [38] floor points lost by
  // taking it all to the floor; for the player after the perspective:
  // [39] some line accepts the colour, [40] the pile fills such a line.
  const PlayerState& mine = s->players[me];
  const Settlement& my_settlement = settlement[static_cast<size_t>(me)];
  const PlayerState& next = s->players[(me + 1) % Cfg::kPlayers];
  auto write_pile = [&](int source, int color, int count, const std::array<std::uint8_t, kColors>* factory) {
    float* t = token(kPileBase + source * kColors + color);
    t[0] = 1.0f;
    float* f = t + 2;
    const int marker = (!factory && s->first_player_token_in_center) ? 1 : 0;
    f[0] = factory ? 0.0f : 1.0f;
    f[1] = static_cast<float>(marker);
    f[2 + color] = 1.0f;
    f[7] = static_cast<float>(std::min(count, 16)) / 8.0f;
    if (factory) {
      for (int c = 0; c < kColors; ++c) {
        if (c != color) f[8 + c] = static_cast<float>((*factory)[static_cast<size_t>(c)]) / 4.0f;
      }
    }
    for (int row = 0; row < kRows; ++row) {
      if (!fits(mine, row, color)) continue;
      float* b = f + 13 + 5 * row;
      const int free_places = row + 1 - mine.line_len[row];
      const int to_line = std::min(free_places, count);
      b[0] = 1.0f;
      b[1] = static_cast<float>(to_line) / 5.0f;
      if (to_line == free_places) {
        b[2] = 1.0f;
        b[3] = static_cast<float>(Rules::score_wall_placement(
                   my_settlement.wall_at_row[static_cast<size_t>(row)], row,
                   Rules::wall_col_for_color(row, color))) / 10.0f;
      }
      b[4] = static_cast<float>(Rules::floor_penalty(mine.floor_count, count - to_line + marker)) / 14.0f;
    }
    f[38] = static_cast<float>(Rules::floor_penalty(mine.floor_count, count + marker)) / 14.0f;
    for (int row = 0; row < kRows; ++row) {
      if (!fits(next, row, color)) continue;
      f[39] = 1.0f;
      if (count >= row + 1 - next.line_len[row]) f[40] = 1.0f;
    }
  };
  for (int source = 0; source < Cfg::kFactories; ++source) {
    for (int c = 0; c < kColors; ++c) {
      const int count = s->factories[source][c];
      if (count > 0) write_pile(source, c, count, &s->factories[source]);
    }
  }
  for (int c = 0; c < kColors; ++c) {
    if (s->center[c] > 0) write_pile(Cfg::kCenterSource, c, s->center[c], nullptr);
  }
}

template class AzulFeatureEncoder<2>;
template class AzulFeatureEncoder<3>;
template class AzulFeatureEncoder<4>;
template class AzulTokenEncoder<2>;
template class AzulTokenEncoder<3>;
template class AzulTokenEncoder<4>;

}  // namespace board_ai::azul
