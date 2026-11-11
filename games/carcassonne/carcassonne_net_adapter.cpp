#include "carcassonne_net_adapter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "carcassonne_rules.h"

namespace board_ai::carcassonne {

namespace {

float norm_coord(int coord, int max_coord) {
  if (coord < 0) return 0.0f;
  return (static_cast<float>(coord) - static_cast<float>(max_coord) * 0.5f) /
         (static_cast<float>(max_coord) * 0.5f);
}

float norm_score(int score) {
  return static_cast<float>(score) / 120.0f;
}

}  // namespace

void CarcassonneFeatureEncoder::encode_features(
    const IGameState& masked_state, int perspective_player,
    const IBeliefTracker* /*tracker*/, std::vector<float>* out) const {
  const auto& s = checked_cast<CarcassonneState>(masked_state);
  if (perspective_player < 0 || perspective_player >= kPlayers) {
    throw std::invalid_argument("carcassonne encoder perspective out of range");
  }
  if (out == nullptr) {
    throw std::invalid_argument("carcassonne encoder output is null");
  }

  const int opp = 1 - perspective_player;

  for (int i = 0; i < kHistorySlots; ++i) {
    const int tile = s.history_tile[static_cast<size_t>(i)];
    for (int t = 0; t < kTileTypes; ++t) {
      out->push_back(tile == t ? 1.0f : 0.0f);
    }

    const int rotation = s.history_rotation[static_cast<size_t>(i)];
    for (int r = 0; r < 4; ++r) {
      out->push_back(tile >= 0 && rotation == r ? 1.0f : 0.0f);
    }

    const int row = s.history_row[static_cast<size_t>(i)];
    const int col = s.history_col[static_cast<size_t>(i)];
    out->push_back(tile >= 0 ? norm_coord(row, kBoardRows - 1) : 0.0f);
    out->push_back(tile >= 0 ? norm_coord(col, kBoardCols - 1) : 0.0f);

    float meeple_value = 0.0f;
    int meeple_pos = kMeepleNoPos;
    if (tile >= 0 && row >= 0 && col >= 0) {
      const int cell = s.index(row, col);
      const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
      meeple_pos = s.board_meeple_pos[static_cast<size_t>(cell)];
      if (owner == perspective_player) {
        meeple_value = 1.0f;
      } else if (owner == opp) {
        meeple_value = -1.0f;
      }
    }
    for (int p = 0; p < kMeeplePositions; ++p) {
      out->push_back(meeple_pos == p ? meeple_value : 0.0f);
    }
  }

  out->push_back(norm_score(s.scores[static_cast<size_t>(perspective_player)]));
  out->push_back(norm_score(s.scores[static_cast<size_t>(opp)]));
  out->push_back(static_cast<float>(
                     s.meeples_remaining[static_cast<size_t>(perspective_player)]) /
                 static_cast<float>(kMeeplesPerPlayer));
  out->push_back(static_cast<float>(s.meeples_remaining[static_cast<size_t>(opp)]) /
                 static_cast<float>(kMeeplesPerPlayer));

  for (int t = 0; t < kTileTypes; ++t) {
    out->push_back(s.current_tile == t ? 1.0f : 0.0f);
  }

  if (static_cast<int>(out->size()) != kFeatureDim) {
    throw std::runtime_error("carcassonne encoder feature_dim mismatch");
  }
}

namespace {

using Enc = CarcassonneTokenEncoder;

constexpr std::array<int, kSides> kDr{{-1, 0, 1, 0}};
constexpr std::array<int, kSides> kDc{{0, 1, 0, -1}};

// A tile type's geometry with one rotation applied, in board sides / halves.
struct RotTile {
  std::array<std::int8_t, kSides> edge{};
  std::array<std::int8_t, kSides> city_of_side{};
  std::array<std::int8_t, kSides> road_of_side{};
  std::array<std::int8_t, kFarmSides> farm_of_half{};
  std::array<std::uint8_t, kSides> city_sides{};
  std::array<std::uint8_t, kSides> road_sides{};
  std::array<std::uint8_t, kSides> farm_halves{};
  std::array<std::uint8_t, kSides> farm_city_sides{};
  int city_groups = 0;
  int road_groups = 0;
  int farm_groups = 0;
};

const RotTile& rot_tile(int tile, int rot) {
  static const auto table = [] {
    std::array<std::array<RotTile, 4>, kTileTypes> t{};
    for (int type = 0; type < kTileTypes; ++type) {
      const TileDef& d = tile_def(type);
      for (int rot = 0; rot < 4; ++rot) {
        RotTile& r = t[static_cast<size_t>(type)][static_cast<size_t>(rot)];
        r.city_of_side.fill(-1);
        r.road_of_side.fill(-1);
        r.farm_of_half.fill(-1);
        for (int side = 0; side < kSides; ++side) {
          r.edge[static_cast<size_t>(side)] = static_cast<std::int8_t>(edge_terrain_at(type, rot, side));
        }
        r.city_groups = d.city_group_count;
        for (int g = 0; g < d.city_group_count; ++g) {
          for (int s : rotated_group_sides(d.city_groups[static_cast<size_t>(g)], rot)) {
            r.city_of_side[static_cast<size_t>(s)] = static_cast<std::int8_t>(g);
            r.city_sides[static_cast<size_t>(g)] |= static_cast<std::uint8_t>(1u << s);
          }
        }
        r.road_groups = d.road_group_count;
        for (int g = 0; g < d.road_group_count; ++g) {
          for (int s : rotated_group_sides(d.road_groups[static_cast<size_t>(g)], rot)) {
            r.road_of_side[static_cast<size_t>(s)] = static_cast<std::int8_t>(g);
            r.road_sides[static_cast<size_t>(g)] |= static_cast<std::uint8_t>(1u << s);
          }
        }
        r.farm_groups = d.farm_group_count;
        for (int g = 0; g < d.farm_group_count; ++g) {
          const FarmGroup& fg = d.farm_groups[static_cast<size_t>(g)];
          for (int h : rotated_farm_connections(fg, rot)) {
            r.farm_of_half[static_cast<size_t>(h)] = static_cast<std::int8_t>(g);
            r.farm_halves[static_cast<size_t>(g)] |= static_cast<std::uint8_t>(1u << h);
          }
          for (int s : rotated_farm_city_sides(fg, rot)) {
            r.farm_city_sides[static_cast<size_t>(g)] |= static_cast<std::uint8_t>(1u << s);
          }
        }
      }
    }
    return t;
  }();
  return table[static_cast<size_t>(tile)][static_cast<size_t>(rot)];
}

int lowest_bit(unsigned mask) {
  for (int i = 0; i < 8; ++i) {
    if (mask & (1u << i)) return i;
  }
  throw std::runtime_error("carcassonne token encoder: empty group mask");
}

// A connected city or road.
struct Feature {
  int tiles = 0;
  int open = 0;
  int shields = 0;
  bool off_board = false;
  std::array<int, kPlayers> meeples{};
  bool complete() const { return open == 0 && !off_board; }
};

struct Farm {
  std::array<int, kPlayers> meeples{};
  int complete_cities = 0;
  int open_cities = 0;
};

// Connected components of every city / road / farm on the board, keyed by
// node = cell * 4 + group index on that cell's tile.
struct BoardAnalysis {
  std::vector<int> cells;
  std::array<std::int16_t, kBoardCells * 4> city{};
  std::array<std::int16_t, kBoardCells * 4> road{};
  std::array<std::int16_t, kBoardCells * 4> farm{};
  std::vector<Feature> cities;
  std::vector<Feature> roads;
  std::vector<Farm> farms;
  std::array<int, kPlayers> projected{};
};

const RotTile& rot_at(const CarcassonneState& s, int cell) {
  return rot_tile(s.board_tile[static_cast<size_t>(cell)], s.board_rotation[static_cast<size_t>(cell)]);
}

void label_features(const CarcassonneState& s, const std::vector<int>& cells, bool city,
                    std::array<std::int16_t, kBoardCells * 4>* label, std::vector<Feature>* out) {
  label->fill(-1);
  std::array<std::int16_t, kBoardCells> counted{};
  counted.fill(-1);
  std::vector<int> stack;
  for (int cell : cells) {
    const RotTile& rt = rot_at(s, cell);
    const int groups = city ? rt.city_groups : rt.road_groups;
    for (int g = 0; g < groups; ++g) {
      if ((*label)[static_cast<size_t>(cell * 4 + g)] >= 0) continue;
      const auto id = static_cast<std::int16_t>(out->size());
      out->emplace_back();
      Feature& f = out->back();
      (*label)[static_cast<size_t>(cell * 4 + g)] = id;
      stack.push_back(cell * 4 + g);
      while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        const int c = node / 4;
        const RotTile& r = rot_at(s, c);
        if (counted[static_cast<size_t>(c)] != id) {
          counted[static_cast<size_t>(c)] = id;
          f.tiles += 1;
          if (city && tile_def(s.board_tile[static_cast<size_t>(c)]).shield) f.shields += 1;
        }
        const unsigned mask = city ? r.city_sides[static_cast<size_t>(node % 4)] : r.road_sides[static_cast<size_t>(node % 4)];
        for (int side = 0; side < kSides; ++side) {
          if (!(mask & (1u << side))) continue;
          const int nr = c / kBoardCols + kDr[static_cast<size_t>(side)];
          const int nc = c % kBoardCols + kDc[static_cast<size_t>(side)];
          if (!s.in_bounds(nr, nc)) {
            f.off_board = true;
            continue;
          }
          const int ncell = s.index(nr, nc);
          if (s.board_tile[static_cast<size_t>(ncell)] == kNoTile) {
            f.open += 1;
            continue;
          }
          const RotTile& nrt = rot_at(s, ncell);
          const int opp = opposite_side(side);
          const int ng = city ? nrt.city_of_side[static_cast<size_t>(opp)] : nrt.road_of_side[static_cast<size_t>(opp)];
          if (ng < 0) throw std::runtime_error("carcassonne token encoder: feature edge mismatch");
          auto& nl = (*label)[static_cast<size_t>(ncell * 4 + ng)];
          if (nl < 0) {
            nl = id;
            stack.push_back(ncell * 4 + ng);
          }
        }
      }
    }
  }
}

void label_farms(const CarcassonneState& s, BoardAnalysis* b) {
  b->farm.fill(-1);
  std::vector<int> stack;
  std::vector<int> adjacent_cities;
  for (int cell : b->cells) {
    const RotTile& rt = rot_at(s, cell);
    for (int g = 0; g < rt.farm_groups; ++g) {
      if (b->farm[static_cast<size_t>(cell * 4 + g)] >= 0) continue;
      const auto id = static_cast<std::int16_t>(b->farms.size());
      b->farms.emplace_back();
      adjacent_cities.clear();
      b->farm[static_cast<size_t>(cell * 4 + g)] = id;
      stack.push_back(cell * 4 + g);
      while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        const int c = node / 4;
        const int gi = node % 4;
        const RotTile& r = rot_at(s, c);
        for (int side = 0; side < kSides; ++side) {
          if (!(r.farm_city_sides[static_cast<size_t>(gi)] & (1u << side))) continue;
          if (r.edge[static_cast<size_t>(side)] != kCity) continue;
          adjacent_cities.push_back(b->city[static_cast<size_t>(c * 4 + r.city_of_side[static_cast<size_t>(side)])]);
        }
        for (int h = 0; h < kFarmSides; ++h) {
          if (!(r.farm_halves[static_cast<size_t>(gi)] & (1u << h))) continue;
          const int edge = farmer_side_edge(h);
          const int nr = c / kBoardCols + kDr[static_cast<size_t>(edge)];
          const int nc = c % kBoardCols + kDc[static_cast<size_t>(edge)];
          if (!s.in_bounds(nr, nc)) continue;
          const int ncell = s.index(nr, nc);
          if (s.board_tile[static_cast<size_t>(ncell)] == kNoTile) continue;
          const int ng = rot_at(s, ncell).farm_of_half[static_cast<size_t>(opposite_farmer_side(h))];
          if (ng < 0) continue;
          auto& nl = b->farm[static_cast<size_t>(ncell * 4 + ng)];
          if (nl < 0) {
            nl = id;
            stack.push_back(ncell * 4 + ng);
          }
        }
      }
      std::sort(adjacent_cities.begin(), adjacent_cities.end());
      adjacent_cities.erase(std::unique(adjacent_cities.begin(), adjacent_cities.end()), adjacent_cities.end());
      Farm& farm = b->farms[static_cast<size_t>(id)];
      for (int cid : adjacent_cities) {
        if (b->cities[static_cast<size_t>(cid)].complete()) {
          farm.complete_cities += 1;
        } else {
          farm.open_cities += 1;
        }
      }
    }
  }
}

int cloister_occupancy(const CarcassonneState& s, int cell) { return CarcassonneRules::cloister_occupancy(s, cell); }

void award(std::array<int, kPlayers>* scores, const std::array<int, kPlayers>& meeples, int points) {
  const int best = std::max(meeples[0], meeples[1]);
  if (best <= 0) return;
  for (int p = 0; p < kPlayers; ++p) {
    if (meeples[static_cast<size_t>(p)] == best) (*scores)[static_cast<size_t>(p)] += points;
  }
}

BoardAnalysis analyze(const CarcassonneState& s) {
  BoardAnalysis b;
  for (int cell = 0; cell < kBoardCells; ++cell) {
    if (s.board_tile[static_cast<size_t>(cell)] != kNoTile) b.cells.push_back(cell);
  }
  label_features(s, b.cells, /*city=*/true, &b.city, &b.cities);
  label_features(s, b.cells, /*city=*/false, &b.road, &b.roads);
  label_farms(s, &b);

  for (int cell : b.cells) {
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (owner < 0) continue;
    const int pos = s.board_meeple_pos[static_cast<size_t>(cell)];
    const RotTile& rt = rot_at(s, cell);
    if (pos >= 0 && pos < kSides) {
      const int terrain = rt.edge[static_cast<size_t>(pos)];
      if (terrain == kCity) {
        b.cities[static_cast<size_t>(b.city[static_cast<size_t>(cell * 4 + rt.city_of_side[static_cast<size_t>(pos)])])]
            .meeples[static_cast<size_t>(owner)] += 1;
      } else if (terrain == kRoad) {
        b.roads[static_cast<size_t>(b.road[static_cast<size_t>(cell * 4 + rt.road_of_side[static_cast<size_t>(pos)])])]
            .meeples[static_cast<size_t>(owner)] += 1;
      }
    } else if (pos >= kSides && pos < kSides + kFarmSides) {
      const int g = rt.farm_of_half[static_cast<size_t>(pos - kSides)];
      if (g >= 0) {
        b.farms[static_cast<size_t>(b.farm[static_cast<size_t>(cell * 4 + g)])].meeples[static_cast<size_t>(owner)] += 1;
      }
    }
  }

  // Score if the game ended now, counting features the last placement
  // completed but that are only resolved after the meeple decision.
  b.projected = s.scores;
  for (const Feature& f : b.cities) {
    award(&b.projected, f.meeples, (f.complete() ? 2 : 1) * (f.tiles + f.shields));
  }
  for (const Feature& f : b.roads) award(&b.projected, f.meeples, f.tiles);
  for (const Farm& f : b.farms) award(&b.projected, f.meeples, 3 * f.complete_cities);
  for (int cell : b.cells) {
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (owner >= 0 && s.board_meeple_pos[static_cast<size_t>(cell)] == kMeepleCenterPos) {
      b.projected[static_cast<size_t>(owner)] += cloister_occupancy(s, cell);
    }
  }
  return b;
}

float coord(int v) { return (static_cast<float>(v) - 9.5f) / 10.0f; }

// Terrain a tile placed at `cell` must show on each side (-1 = no neighbour).
std::array<int, kSides> edge_needs(const CarcassonneState& s, int cell) {
  std::array<int, kSides> need{-1, -1, -1, -1};
  for (int side = 0; side < kSides; ++side) {
    const int nr = cell / kBoardCols + kDr[static_cast<size_t>(side)];
    const int nc = cell % kBoardCols + kDc[static_cast<size_t>(side)];
    if (!s.in_bounds(nr, nc)) continue;
    const int ncell = s.index(nr, nc);
    if (s.board_tile[static_cast<size_t>(ncell)] == kNoTile) continue;
    need[static_cast<size_t>(side)] = rot_at(s, ncell).edge[static_cast<size_t>(opposite_side(side))];
  }
  return need;
}

bool fits(const std::array<int, kSides>& need, int tile, int rot) {
  const RotTile& rt = rot_tile(tile, rot);
  for (int side = 0; side < kSides; ++side) {
    if (need[static_cast<size_t>(side)] >= 0 && need[static_cast<size_t>(side)] != rt.edge[static_cast<size_t>(side)]) {
      return false;
    }
  }
  return true;
}

// [is_city, is_road, tiles, open edges, complete, my meeples, opp meeples, shields]
void write_feature(float* f, const Feature& feat, bool city, int me) {
  f[0] = city ? 1.0f : 0.0f;
  f[1] = city ? 0.0f : 1.0f;
  f[2] = static_cast<float>(feat.tiles) / 10.0f;
  f[3] = static_cast<float>(feat.open) / 10.0f;
  f[4] = feat.complete() ? 1.0f : 0.0f;
  f[5] = static_cast<float>(feat.meeples[static_cast<size_t>(me)]) / 3.0f;
  f[6] = static_cast<float>(feat.meeples[static_cast<size_t>(1 - me)]) / 3.0f;
  f[7] = static_cast<float>(feat.shields) / 5.0f;
}

// Offsets into a token's type-specific features (see the layout comment
// above encode_features).
constexpr int kCurrentSlots = 9;      // 13 meeple positions x kMeepleSlotFeatures
constexpr int kTileSameFeature = 71;  // 6 side pairs
constexpr int kFrontierSide = 2;      // 4 sides x kFrontierSideDim
constexpr int kFrontierSideDim = 8;
constexpr int kFrontierFarm = 34;     // 8 halves x 2
constexpr int kFrontierOccupied = 50;
constexpr int kFrontierFits = 51;     // 4 rotations
constexpr int kFrontierBagShare = 55;
constexpr int kFrontierDead = 56;
constexpr int kFrontierOutcome = 57;  // 4 rotations x kFrontierOutcomeDim
constexpr int kFrontierOutcomeDim = 7;
static_assert(kFrontierOutcome + 4 * kFrontierOutcomeDim == Enc::kFrontierFeatures);
static_assert(kTileSameFeature + 6 == Enc::kTileFeatures);

// The 13 meeple-position slots of the tile at `cell`: what the city / road /
// farm / cloister at each position looks like on board `b`. `can_place` is
// whether the mover may put a meeple on the tile at all.
void write_position_slots(float* f, const CarcassonneState& s, const BoardAnalysis& b, int cell, bool can_place) {
  const int tile = s.board_tile[static_cast<size_t>(cell)];
  const RotTile& rt = rot_at(s, cell);
  for (int side = 0; side < kSides; ++side) {
    float* fp = f + side * Enc::kMeepleSlotFeatures;
    const int terrain = rt.edge[static_cast<size_t>(side)];
    if (terrain != kCity && terrain != kRoad) continue;
    const bool city = terrain == kCity;
    const int g = city ? rt.city_of_side[static_cast<size_t>(side)] : rt.road_of_side[static_cast<size_t>(side)];
    const Feature& feat = city ? b.cities[static_cast<size_t>(b.city[static_cast<size_t>(cell * 4 + g)])]
                               : b.roads[static_cast<size_t>(b.road[static_cast<size_t>(cell * 4 + g)])];
    const unsigned mask = city ? rt.city_sides[static_cast<size_t>(g)] : rt.road_sides[static_cast<size_t>(g)];
    fp[0] = can_place && lowest_bit(mask) == side && feat.meeples[0] == 0 && feat.meeples[1] == 0 ? 1.0f : 0.0f;
    fp[1] = city ? 1.0f : 0.0f;
    fp[2] = static_cast<float>(feat.tiles) / 10.0f;
    fp[3] = static_cast<float>(feat.open) / 10.0f;
    fp[4] = feat.complete() ? 1.0f : 0.0f;
    fp[5] = static_cast<float>(feat.shields) / 5.0f;
  }
  for (int h = 0; h < kFarmSides; ++h) {
    float* fp = f + (kSides + h) * Enc::kMeepleSlotFeatures;
    const int g = rt.farm_of_half[static_cast<size_t>(h)];
    if (g < 0) continue;
    const Farm& farm = b.farms[static_cast<size_t>(b.farm[static_cast<size_t>(cell * 4 + g)])];
    const int lowest = lowest_bit(rt.farm_halves[static_cast<size_t>(g)]);
    fp[0] = can_place && lowest == h && farm.meeples[0] == 0 && farm.meeples[1] == 0 ? 1.0f : 0.0f;
    // Halves of one farm of this tile share this value.
    fp[1] = static_cast<float>(lowest + 1) / static_cast<float>(kFarmSides);
    fp[2] = static_cast<float>(farm.open_cities) / 5.0f;
    fp[5] = static_cast<float>(farm.complete_cities) / 5.0f;
  }
  if (tile_def(tile).chapel) {
    float* fp = f + kMeepleCenterPos * Enc::kMeepleSlotFeatures;
    fp[0] = can_place ? 1.0f : 0.0f;
    const int occ = cloister_occupancy(s, cell);
    fp[4] = occ == 9 ? 1.0f : 0.0f;
    fp[5] = static_cast<float>(occ) / 9.0f;
  }
}

using PositionSlots = std::array<float, kMeeplePositions * Enc::kMeepleSlotFeatures>;

// Position slots of each tile type standing alone and unrotated: the
// geometry of the tile still to be placed, in the same terms the slots use
// once it is on the board.
const PositionSlots& lone_tile_slots(int tile) {
  static const auto table = [] {
    std::array<PositionSlots, kTileTypes> t{};
    CarcassonneState s;
    s.board_tile.fill(static_cast<std::int8_t>(kNoTile));
    s.board_rotation.fill(0);
    s.board_meeple_owner.fill(static_cast<std::int8_t>(kNoOwner));
    s.board_meeple_pos.fill(static_cast<std::int8_t>(kMeepleNoPos));
    const int cell = s.index(kBoardRows / 2, kBoardCols / 2);
    for (int type = 0; type < kTileTypes; ++type) {
      s.board_tile[static_cast<size_t>(cell)] = static_cast<std::int8_t>(type);
      write_position_slots(t[static_cast<size_t>(type)].data(), s, analyze(s), cell, /*can_place=*/false);
    }
    return t;
  }();
  return table[static_cast<size_t>(tile)];
}

}  // namespace

CarcassonneTokenEncoder::CarcassonneTokenEncoder() {
  static_assert(kGlobalFeatures <= kTokenFeatures && kTileFeatures <= kTokenFeatures &&
                kFrontierFeatures <= kTokenFeatures);
  static_assert(kMeepleCenterAction - kMeepleSideActionBase + 1 == kMeeplePositions);
  layout_.variable_tokens = true;
  layout_.token_positions = true;
  layout_.token_dim = kVarTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = 1 + kTileTypes;
  layout_.num_pointer_heads = kPointerHeadCount;
  layout_.action_token.assign(kActionSpace, -1);
  layout_.action_head.assign(kActionSpace, -1);
  layout_.num_anchor_ids = kCurrentAnchor;
  layout_.anchor_action.assign(static_cast<size_t>(kCurrentAnchor * kPointerHeadCount), -1);
  for (int cell = 0; cell < kBoardCells; ++cell) {
    for (int rot = 0; rot < 4; ++rot) {
      layout_.anchor_action[static_cast<size_t>(cell * kPointerHeadCount + rot)] =
          CarcassonneRules::encode_place_action(cell / kBoardCols, cell % kBoardCols, rot);
    }
  }
  for (ActionId a = kMeepleSideActionBase; a <= kMeepleCenterAction; ++a) {
    layout_.anchor_action[static_cast<size_t>((kCurrentAnchor - 1) * kPointerHeadCount + kMeepleHeadBase +
                                              CarcassonneRules::decode_meeple_position(a))] = a;
  }
}

// Token features (offsets into the type-specific block, after
// [valid, entity, anchor]):
//   global:   [0..2] my / opp score, margin; [3..5] same if the game ended
//             now; [6,7] my / opp meeples left; [8] tiles left in bag;
//             [9,10] stage place / meeple; [11] perspective to move;
//             [12..35] bag count per tile type; [36] tiles discarded.
//   current:  [0,1] stage place / meeple; [2..5] rotation and [6,7] cell of
//             the tile just placed (meeple stage); [8] legal placements;
//             [9..86] per meeple position 6 values: legal, side is city
//             (farm: which farm of the tile, shared by its halves), size
//             (farm: adjacent open cities), open edges, complete, extra
//             (city shields / farm adjacent complete cities / cloister
//             occupancy). In the placement stage the slots describe the
//             tile to place as if it stood alone, unrotated (never legal).
//   tile:     [0..3] rotation; [4,5] cell; [6] most recent placement;
//             [7,8] my / opp meeple; [9..12] meeple on city / road / farm /
//             cloister; [13..44] per side the feature it belongs to
//             (write_feature); [45..68] per farm half my / opp farmers and
//             adjacent complete cities; [69] cloister occupancy; [70]
//             occupied neighbours; [71..76] sides (0,1) (0,2) (0,3) (1,2)
//             (1,3) (2,3) are the same city / road.
//   frontier: [0,1] cell; [2..33] per side the required terrain (field /
//             road / city, all zero without a neighbour) and the
//             neighbouring feature's size, open edges, my / opp meeples,
//             shields; [34..49] per farm half my / opp farmers across it;
//             [50] occupied cells in the 3x3 block; [51..54] placing the
//             current tile here at rotation r is legal; [55] share of bag
//             tiles that fit; [56] nothing in the bag fits; [57..84] per
//             legal rotation what that placement does (PlacementPreview):
//             features completed, my / opp points scored at once, best
//             city and best road the mover could claim (0.5 open, 1
//             completed), my / opp meeples on the cities / roads joined.
void CarcassonneTokenEncoder::encode_features(const IGameState& masked_state, int perspective_player,
                                              const IBeliefTracker* /*tracker*/,
                                              std::vector<float>* out) const {
  if (perspective_player < 0 || perspective_player >= kPlayers) {
    throw std::invalid_argument("CarcassonneTokenEncoder: perspective out of range");
  }
  if (out == nullptr) throw std::invalid_argument("CarcassonneTokenEncoder: output is null");
  thread_local std::vector<float> slots;
  slots.clear();
  encode_slots(masked_state, perspective_player, &slots);
  compact(masked_state, slots.data(), out);
}

void CarcassonneTokenEncoder::compact(const IGameState& masked_state, const float* slots,
                                      std::vector<float>* out) const {
  const auto& s = checked_cast<CarcassonneState>(masked_state);
  const int last_cell = s.last_row >= 0 && s.last_col >= 0 ? s.index(s.last_row, s.last_col) : -1;
  for (int i = 0; i < kNumTokens; ++i) {
    const float* t = slots + static_cast<std::size_t>(i) * kTokenDim;
    if (t[0] == 0.0f) continue;
    const std::size_t base = out->size();
    out->resize(base + static_cast<std::size_t>(kVarTokenDim), 0.0f);
    float* v = out->data() + base;
    std::copy(t + 3, t + 3 + kTokenFeatures, v + kVarHeader);
    float* f = v + kVarHeader;
    v[0] = 1.0f;
    v[1] = t[1];
    int cell = -1;
    if (i == 0) {
      v[3] = kGlobalToken;
    } else if (i == kCurrentSlot) {
      v[3] = kCurrentToken;
      v[2] = kCurrentAnchor;
      if (s.stage == kStageMeeple) cell = last_cell;
      f[6] = f[7] = 0.0f;
    } else {
      v[3] = i < kFrontierBase ? kTileToken : kFrontierToken;
      v[2] = t[2];
      cell = static_cast<int>(t[2]) - 1;
      if (i < kFrontierBase) {
        f[4] = f[5] = 0.0f;
      } else {
        f[0] = f[1] = 0.0f;
      }
    }
    if (cell >= 0) {
      v[4] = 1.0f;
      v[5] = static_cast<float>(cell % kBoardCols);
      v[6] = static_cast<float>(cell / kBoardCols);
    }
  }
}

void CarcassonneTokenEncoder::encode_slots(const IGameState& masked_state, int perspective_player,
                                           std::vector<float>* out) const {
  const auto& s = checked_cast<CarcassonneState>(masked_state);
  const int me = perspective_player;
  const int opp = 1 - me;
  const BoardAnalysis b = analyze(s);
  if (static_cast<int>(b.cells.size()) > kMaxPlacedTiles) {
    throw std::runtime_error("CarcassonneTokenEncoder: more placed tiles than token slots");
  }

  const std::size_t base = out->size();
  out->resize(base + static_cast<std::size_t>(kNumTokens * kTokenDim), 0.0f);
  auto token = [&](int i) { return out->data() + base + static_cast<std::size_t>(i) * kTokenDim; };
  constexpr int kF = 3;

  const bool meeple_stage = s.stage == kStageMeeple;
  const int last_cell = s.last_row >= 0 && s.last_col >= 0 ? s.index(s.last_row, s.last_col) : -1;
  int bag_total = 0;
  for (int c : s.bag_counts) bag_total += c;

  {
    float* t = token(0);
    t[0] = 1.0f;
    float* f = t + kF;
    f[0] = static_cast<float>(s.scores[static_cast<size_t>(me)]) / 100.0f;
    f[1] = static_cast<float>(s.scores[static_cast<size_t>(opp)]) / 100.0f;
    f[2] = static_cast<float>(s.scores[static_cast<size_t>(me)] - s.scores[static_cast<size_t>(opp)]) / 50.0f;
    f[3] = static_cast<float>(b.projected[static_cast<size_t>(me)]) / 100.0f;
    f[4] = static_cast<float>(b.projected[static_cast<size_t>(opp)]) / 100.0f;
    f[5] = static_cast<float>(b.projected[static_cast<size_t>(me)] - b.projected[static_cast<size_t>(opp)]) / 50.0f;
    f[6] = static_cast<float>(s.meeples_remaining[static_cast<size_t>(me)]) / static_cast<float>(kMeeplesPerPlayer);
    f[7] = static_cast<float>(s.meeples_remaining[static_cast<size_t>(opp)]) / static_cast<float>(kMeeplesPerPlayer);
    f[8] = static_cast<float>(bag_total) / static_cast<float>(kDeckSlots);
    f[9] = meeple_stage ? 0.0f : 1.0f;
    f[10] = meeple_stage ? 1.0f : 0.0f;
    f[11] = s.current_player_ == me ? 1.0f : 0.0f;
    int discarded = 0;
    for (int t2 = 0; t2 < kTileTypes; ++t2) {
      f[12 + t2] = static_cast<float>(s.bag_counts[static_cast<size_t>(t2)]) / 9.0f;
      discarded += s.discarded_tile_counts[static_cast<size_t>(t2)];
    }
    f[36] = static_cast<float>(discarded) / 5.0f;
  }

  for (std::size_t i = 0; i < b.cells.size(); ++i) {
    const int cell = b.cells[i];
    const int tile = s.board_tile[static_cast<size_t>(cell)];
    const int rot = s.board_rotation[static_cast<size_t>(cell)];
    const RotTile& rt = rot_tile(tile, rot);
    float* t = token(kTileBase + static_cast<int>(i));
    t[0] = 1.0f;
    t[1] = static_cast<float>(1 + tile);
    t[2] = static_cast<float>(1 + cell);
    float* f = t + kF;
    f[rot] = 1.0f;
    f[4] = coord(cell / kBoardCols);
    f[5] = coord(cell % kBoardCols);
    f[6] = cell == last_cell ? 1.0f : 0.0f;
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (owner >= 0) {
      f[owner == me ? 7 : 8] = 1.0f;
      const int pos = s.board_meeple_pos[static_cast<size_t>(cell)];
      if (pos == kMeepleCenterPos) {
        f[12] = 1.0f;
      } else if (pos >= kSides) {
        f[11] = 1.0f;
      } else {
        f[rt.edge[static_cast<size_t>(pos)] == kCity ? 9 : 10] = 1.0f;
      }
    }
    for (int side = 0; side < kSides; ++side) {
      const int terrain = rt.edge[static_cast<size_t>(side)];
      float* fs = f + 13 + side * 8;
      if (terrain == kCity) {
        write_feature(fs, b.cities[static_cast<size_t>(b.city[static_cast<size_t>(cell * 4 + rt.city_of_side[static_cast<size_t>(side)])])], true, me);
      } else if (terrain == kRoad) {
        write_feature(fs, b.roads[static_cast<size_t>(b.road[static_cast<size_t>(cell * 4 + rt.road_of_side[static_cast<size_t>(side)])])], false, me);
      }
    }
    for (int h = 0; h < kFarmSides; ++h) {
      const int g = rt.farm_of_half[static_cast<size_t>(h)];
      if (g < 0) continue;
      const Farm& farm = b.farms[static_cast<size_t>(b.farm[static_cast<size_t>(cell * 4 + g)])];
      float* fh = f + 45 + h * 3;
      fh[0] = static_cast<float>(farm.meeples[static_cast<size_t>(me)]) / 3.0f;
      fh[1] = static_cast<float>(farm.meeples[static_cast<size_t>(opp)]) / 3.0f;
      fh[2] = static_cast<float>(farm.complete_cities) / 5.0f;
    }
    if (tile_def(tile).chapel) f[69] = static_cast<float>(cloister_occupancy(s, cell)) / 9.0f;
    int pair = 0;
    for (int a = 0; a < kSides; ++a) {
      for (int c = a + 1; c < kSides; ++c, ++pair) {
        const int terrain = rt.edge[static_cast<size_t>(a)];
        if (terrain == kField || terrain != rt.edge[static_cast<size_t>(c)]) continue;
        const auto& label = terrain == kCity ? b.city : b.road;
        const auto& group = terrain == kCity ? rt.city_of_side : rt.road_of_side;
        if (label[static_cast<size_t>(cell * 4 + group[static_cast<size_t>(a)])] ==
            label[static_cast<size_t>(cell * 4 + group[static_cast<size_t>(c)])]) {
          f[kTileSameFeature + pair] = 1.0f;
        }
      }
    }
    int neighbours = 0;
    for (int side = 0; side < kSides; ++side) {
      const int nr = cell / kBoardCols + kDr[static_cast<size_t>(side)];
      const int nc = cell % kBoardCols + kDc[static_cast<size_t>(side)];
      if (s.in_bounds(nr, nc) && s.board_tile[static_cast<size_t>(s.index(nr, nc))] != kNoTile) ++neighbours;
    }
    f[70] = static_cast<float>(neighbours) / 4.0f;
  }

  const bool place_stage = !meeple_stage && s.current_tile != kNoTile;
  int frontier = 0;
  int legal_placements = 0;
  for (int cell = 0; cell < kBoardCells; ++cell) {
    if (s.board_tile[static_cast<size_t>(cell)] != kNoTile) continue;
    const auto need = edge_needs(s, cell);
    if (std::all_of(need.begin(), need.end(), [](int n) { return n < 0; })) continue;
    if (frontier >= kMaxFrontierCells) {
      throw std::runtime_error("CarcassonneTokenEncoder: more frontier cells than token slots");
    }
    float* t = token(kFrontierBase + frontier++);
    t[0] = 1.0f;
    t[2] = static_cast<float>(1 + cell);
    float* f = t + kF;
    f[0] = coord(cell / kBoardCols);
    f[1] = coord(cell % kBoardCols);
    for (int side = 0; side < kSides; ++side) {
      float* fs = f + kFrontierSide + side * kFrontierSideDim;
      const int n = need[static_cast<size_t>(side)];
      if (n < 0) continue;
      fs[n] = 1.0f;
      if (n != kCity && n != kRoad) continue;
      const int ncell = s.index(cell / kBoardCols + kDr[static_cast<size_t>(side)], cell % kBoardCols + kDc[static_cast<size_t>(side)]);
      const RotTile& nrt = rot_at(s, ncell);
      const int os = opposite_side(side);
      const Feature& feat = n == kCity
          ? b.cities[static_cast<size_t>(b.city[static_cast<size_t>(ncell * 4 + nrt.city_of_side[static_cast<size_t>(os)])])]
          : b.roads[static_cast<size_t>(b.road[static_cast<size_t>(ncell * 4 + nrt.road_of_side[static_cast<size_t>(os)])])];
      fs[3] = static_cast<float>(feat.tiles) / 10.0f;
      fs[4] = static_cast<float>(feat.open) / 10.0f;
      fs[5] = static_cast<float>(feat.meeples[static_cast<size_t>(me)]) / 3.0f;
      fs[6] = static_cast<float>(feat.meeples[static_cast<size_t>(opp)]) / 3.0f;
      fs[7] = static_cast<float>(feat.shields) / 5.0f;
    }
    for (int h = 0; h < kFarmSides; ++h) {
      const int edge = farmer_side_edge(h);
      if (need[static_cast<size_t>(edge)] < 0) continue;
      const int ncell = s.index(cell / kBoardCols + kDr[static_cast<size_t>(edge)], cell % kBoardCols + kDc[static_cast<size_t>(edge)]);
      const int g = rot_at(s, ncell).farm_of_half[static_cast<size_t>(opposite_farmer_side(h))];
      if (g < 0) continue;
      const Farm& farm = b.farms[static_cast<size_t>(b.farm[static_cast<size_t>(ncell * 4 + g)])];
      f[kFrontierFarm + h * 2] = static_cast<float>(farm.meeples[static_cast<size_t>(me)]) / 3.0f;
      f[kFrontierFarm + h * 2 + 1] = static_cast<float>(farm.meeples[static_cast<size_t>(opp)]) / 3.0f;
    }
    int occupied = 0;
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        const int rr = cell / kBoardCols + dr;
        const int cc = cell % kBoardCols + dc;
        if (s.in_bounds(rr, cc) && s.board_tile[static_cast<size_t>(s.index(rr, cc))] != kNoTile) ++occupied;
      }
    }
    f[kFrontierOccupied] = static_cast<float>(occupied) / 8.0f;
    if (place_stage) {
      // Rotations at or past the tile's period repeat an earlier one and are
      // not actions (CarcassonneRules::can_place_tile_at).
      for (int rot = 0; rot < rotation_period(s.current_tile); ++rot) {
        if (!fits(need, s.current_tile, rot)) continue;
        f[kFrontierFits + rot] = 1.0f;
        ++legal_placements;
        const PlacementPreview pv =
            CarcassonneRules::preview_placement(s, cell / kBoardCols, cell % kBoardCols, rot);
        float* fo = f + kFrontierOutcome + rot * kFrontierOutcomeDim;
        fo[0] = static_cast<float>(pv.completed) / 4.0f;
        fo[1] = static_cast<float>(pv.score_delta[static_cast<size_t>(me)]) / 20.0f;
        fo[2] = static_cast<float>(pv.score_delta[static_cast<size_t>(opp)]) / 20.0f;
        fo[3] = static_cast<float>(pv.city_claim) / 2.0f;
        fo[4] = static_cast<float>(pv.road_claim) / 2.0f;
        fo[5] = static_cast<float>(pv.meeples[static_cast<size_t>(me)]) / 3.0f;
        fo[6] = static_cast<float>(pv.meeples[static_cast<size_t>(opp)]) / 3.0f;
      }
    }
    int fitting = 0;
    for (int type = 0; type < kTileTypes; ++type) {
      const int count = s.bag_counts[static_cast<size_t>(type)];
      if (count == 0) continue;
      for (int rot = 0; rot < 4; ++rot) {
        if (fits(need, type, rot)) {
          fitting += count;
          break;
        }
      }
    }
    if (bag_total > 0) {
      f[kFrontierBagShare] = static_cast<float>(fitting) / static_cast<float>(bag_total);
      f[kFrontierDead] = fitting == 0 ? 1.0f : 0.0f;
    }
  }

  {
    float* t = token(kCurrentSlot);
    t[0] = 1.0f;
    float* f = t + kF;
    f[0] = place_stage ? 1.0f : 0.0f;
    f[1] = meeple_stage ? 1.0f : 0.0f;
    if (place_stage) {
      t[1] = static_cast<float>(1 + s.current_tile);
      f[8] = static_cast<float>(legal_placements) / 40.0f;
      const PositionSlots& slots = lone_tile_slots(s.current_tile);
      std::copy(slots.begin(), slots.end(), f + kCurrentSlots);
    }
    if (meeple_stage) {
      if (last_cell < 0) throw std::runtime_error("CarcassonneTokenEncoder: meeple stage without a placed tile");
      t[1] = static_cast<float>(1 + s.board_tile[static_cast<size_t>(last_cell)]);
      f[2 + s.board_rotation[static_cast<size_t>(last_cell)]] = 1.0f;
      f[6] = coord(last_cell / kBoardCols);
      f[7] = coord(last_cell % kBoardCols);
      const bool can_place = s.meeples_remaining[static_cast<size_t>(s.current_player_)] > 0 &&
                             s.board_meeple_owner[static_cast<size_t>(last_cell)] == kNoOwner;
      write_position_slots(f + kCurrentSlots, s, b, last_cell, can_place);
    }
  }
}

}  // namespace board_ai::carcassonne
