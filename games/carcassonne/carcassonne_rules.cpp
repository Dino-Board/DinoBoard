#include "carcassonne_rules.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace board_ai::carcassonne {

namespace {

constexpr std::array<int, kSides> kDr{{-1, 0, 1, 0}};
constexpr std::array<int, kSides> kDc{{0, 1, 0, -1}};

struct FeatureCollect {
  FeatureScoreInfo info{};
  std::vector<std::pair<int, int>> nodes{};
};

struct FarmCollect {
  FarmScoreInfo info{};
  std::vector<std::pair<int, int>> nodes{};
};

int group_for_side(int tile_type, int rotation, Terrain terrain, int side) {
  const auto& def = tile_def(tile_type);
  const auto& groups = terrain == kCity ? def.city_groups : def.road_groups;
  const int count = terrain == kCity ? def.city_group_count : def.road_group_count;
  for (int gi = 0; gi < count; ++gi) {
    const auto sides = rotated_group_sides(groups[static_cast<size_t>(gi)], rotation);
    for (int s : sides) {
      if (s == side) return gi;
    }
  }
  return -1;
}

int farm_group_for_connection(int tile_type, int rotation, int farmer_side) {
  const auto& def = tile_def(tile_type);
  for (int gi = 0; gi < def.farm_group_count; ++gi) {
    const auto sides =
        rotated_farm_connections(def.farm_groups[static_cast<size_t>(gi)],
                                 rotation);
    for (int s : sides) {
      if (s == farmer_side) return gi;
    }
  }
  return -1;
}

int canonical_farm_connection(const TileDef& def, int rotation, int group_idx) {
  if (group_idx < 0 || group_idx >= def.farm_group_count) {
    throw std::runtime_error("carcassonne farm group out of range");
  }
  const auto sides =
      rotated_farm_connections(def.farm_groups[static_cast<size_t>(group_idx)],
                               rotation);
  if (sides.empty()) {
    throw std::runtime_error("carcassonne farm group has no connections");
  }
  return *std::min_element(sides.begin(), sides.end());
}

std::vector<int> sides_for_group(int tile_type, int rotation, Terrain terrain,
                                 int group_idx) {
  const auto& def = tile_def(tile_type);
  if (terrain == kCity) {
    if (group_idx < 0 || group_idx >= def.city_group_count) {
      throw std::runtime_error("carcassonne city group out of range");
    }
    return rotated_group_sides(def.city_groups[static_cast<size_t>(group_idx)],
                               rotation);
  }
  if (terrain == kRoad) {
    if (group_idx < 0 || group_idx >= def.road_group_count) {
      throw std::runtime_error("carcassonne road group out of range");
    }
    return rotated_group_sides(def.road_groups[static_cast<size_t>(group_idx)],
                               rotation);
  }
  throw std::runtime_error("carcassonne unsupported feature terrain");
}

bool meeple_on_feature_node(const CarcassonneState& s, int cell, Terrain terrain,
                            int group_idx) {
  const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
  if (owner < 0) return false;
  const int pos = s.board_meeple_pos[static_cast<size_t>(cell)];
  if (pos < 0 || pos >= kSides) return false;
  const int tile = s.board_tile[static_cast<size_t>(cell)];
  const int rot = s.board_rotation[static_cast<size_t>(cell)];
  if (tile < 0) return false;
  if (edge_terrain_at(tile, rot, pos) != terrain) return false;
  return group_for_side(tile, rot, terrain, pos) == group_idx;
}

FeatureCollect collect_feature(const CarcassonneState& s, int start_cell,
                               Terrain terrain, int start_side) {
  if (start_cell < 0 || start_cell >= kBoardCells) {
    throw std::runtime_error("carcassonne collect start cell out of range");
  }
  const int start_tile = s.board_tile[static_cast<size_t>(start_cell)];
  if (start_tile < 0) {
    throw std::runtime_error("carcassonne collect start cell empty");
  }
  const int start_rot = s.board_rotation[static_cast<size_t>(start_cell)];
  if (edge_terrain_at(start_tile, start_rot, start_side) != terrain) {
    throw std::runtime_error("carcassonne collect start terrain mismatch");
  }
  const int start_group = group_for_side(start_tile, start_rot, terrain, start_side);
  if (start_group < 0) {
    throw std::runtime_error("carcassonne collect missing start group");
  }

  FeatureCollect out;
  out.info.complete = true;
  std::array<std::uint8_t, kBoardCells * kSides> visited_nodes{};
  std::array<std::uint8_t, kBoardCells> seen_tile{};
  std::queue<std::pair<int, int>> q;
  q.push({start_cell, start_group});

  while (!q.empty()) {
    const auto [cell, group_idx] = q.front();
    q.pop();
    const int node_key = cell * kSides + group_idx;
    if (visited_nodes[static_cast<size_t>(node_key)] != 0) continue;
    visited_nodes[static_cast<size_t>(node_key)] = 1;
    out.nodes.push_back({cell, group_idx});

    const int tile_type = s.board_tile[static_cast<size_t>(cell)];
    const int rotation = s.board_rotation[static_cast<size_t>(cell)];
    const int row = cell / kBoardCols;
    const int col = cell % kBoardCols;
    if (seen_tile[static_cast<size_t>(cell)] == 0) {
      seen_tile[static_cast<size_t>(cell)] = 1;
      out.info.tiles += 1;
      if (terrain == kCity && tile_def(tile_type).shield) out.info.shields += 1;
    }
    if (meeple_on_feature_node(s, cell, terrain, group_idx)) {
      const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
      out.info.owners[static_cast<size_t>(owner)] += 1;
    }

    for (int side : sides_for_group(tile_type, rotation, terrain, group_idx)) {
      const int nr = row + kDr[static_cast<size_t>(side)];
      const int nc = col + kDc[static_cast<size_t>(side)];
      if (!s.in_bounds(nr, nc)) {
        out.info.complete = false;
        continue;
      }
      const int ncell = s.index(nr, nc);
      const int ntile = s.board_tile[static_cast<size_t>(ncell)];
      if (ntile < 0) {
        out.info.complete = false;
        out.info.open_edges += 1;
        continue;
      }
      const int nrot = s.board_rotation[static_cast<size_t>(ncell)];
      const int opp = opposite_side(side);
      if (edge_terrain_at(ntile, nrot, opp) != terrain) {
        out.info.complete = false;
        continue;
      }
      const int ngroup = group_for_side(ntile, nrot, terrain, opp);
      if (ngroup < 0) {
        throw std::runtime_error("carcassonne collect missing neighbor group");
      }
      q.push({ncell, ngroup});
    }
  }

  return out;
}

bool farm_node_has_meeple(const CarcassonneState& s, int cell, int group_idx) {
  const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
  if (owner < 0) return false;
  const int pos = s.board_meeple_pos[static_cast<size_t>(cell)];
  if (pos < kSides || pos >= kSides + kFarmSides) return false;
  const int tile = s.board_tile[static_cast<size_t>(cell)];
  if (tile < 0) return false;
  const int rot = s.board_rotation[static_cast<size_t>(cell)];
  return farm_group_for_connection(tile, rot, pos - kSides) == group_idx;
}

std::pair<int, int> canonical_feature_node(
    const std::vector<std::pair<int, int>>& nodes) {
  if (nodes.empty()) {
    throw std::runtime_error("carcassonne feature has no nodes");
  }
  return *std::min_element(nodes.begin(), nodes.end());
}

FarmCollect collect_farm_impl(const CarcassonneState& s, int start_cell,
                              int farmer_side) {
  const int start_tile = s.board_tile[static_cast<size_t>(start_cell)];
  if (start_tile < 0) {
    throw std::runtime_error("carcassonne farm collect start empty");
  }
  const int start_rot = s.board_rotation[static_cast<size_t>(start_cell)];
  const int start_group =
      farm_group_for_connection(start_tile, start_rot, farmer_side);
  if (start_group < 0) {
    throw std::runtime_error("carcassonne farm collect missing start group");
  }

  FarmCollect out;
  std::array<std::uint8_t, kBoardCells * kSides> visited{};
  std::set<std::pair<int, int>> completed_cities;
  std::set<std::pair<int, int>> incomplete_cities;
  std::queue<std::pair<int, int>> q;
  q.push({start_cell, start_group});

  while (!q.empty()) {
    const auto [cell, group_idx] = q.front();
    q.pop();
    const int key = cell * kSides + group_idx;
    if (visited[static_cast<size_t>(key)] != 0) continue;
    visited[static_cast<size_t>(key)] = 1;
    out.nodes.push_back({cell, group_idx});

    const int tile_type = s.board_tile[static_cast<size_t>(cell)];
    const int rotation = s.board_rotation[static_cast<size_t>(cell)];
    const auto& farm_group =
        tile_def(tile_type).farm_groups[static_cast<size_t>(group_idx)];
    const int row = cell / kBoardCols;
    const int col = cell % kBoardCols;

    if (farm_node_has_meeple(s, cell, group_idx)) {
      const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
      out.info.owners[static_cast<size_t>(owner)] += 1;
    }

    for (int city_side : rotated_farm_city_sides(farm_group, rotation)) {
      if (edge_terrain_at(tile_type, rotation, city_side) != kCity) continue;
      const int city_group = group_for_side(tile_type, rotation, kCity, city_side);
      if (city_group < 0) continue;
      FeatureCollect city = collect_feature(s, cell, kCity, city_side);
      (void)city_group;
      if (city.info.complete) {
        completed_cities.insert(canonical_feature_node(city.nodes));
      } else {
        incomplete_cities.insert(canonical_feature_node(city.nodes));
      }
    }

    for (int fs : rotated_farm_connections(farm_group, rotation)) {
      const int edge = farmer_side_edge(fs);
      const int nr = row + kDr[static_cast<size_t>(edge)];
      const int nc = col + kDc[static_cast<size_t>(edge)];
      if (!s.in_bounds(nr, nc)) continue;
      const int ncell = s.index(nr, nc);
      if (s.board_tile[static_cast<size_t>(ncell)] == kNoTile) continue;
      const int nrot = s.board_rotation[static_cast<size_t>(ncell)];
      const int opp_fs = opposite_farmer_side(fs);
      const int ngroup = farm_group_for_connection(
          s.board_tile[static_cast<size_t>(ncell)], nrot, opp_fs);
      if (ngroup >= 0) q.push({ncell, ngroup});
    }
  }

  out.info.completed_cities = static_cast<int>(completed_cities.size());
  out.info.incomplete_cities = static_cast<int>(incomplete_cities.size());
  return out;
}

void award_to_majority(CarcassonneState& s,
                       const std::array<int, kPlayers>& owners,
                       int points) {
  int best = 0;
  for (int count : owners) best = std::max(best, count);
  if (best <= 0 || points <= 0) return;
  for (int p = 0; p < kPlayers; ++p) {
    if (owners[static_cast<size_t>(p)] == best) {
      s.scores[static_cast<size_t>(p)] += points;
      if (p == s.current_player_) s.last_score_delta += points;
    }
  }
}

void return_feature_meeples(CarcassonneState& s,
                            const std::vector<std::pair<int, int>>& nodes,
                            Terrain terrain) {
  for (const auto& node : nodes) {
    const int cell = node.first;
    const int group_idx = node.second;
    if (!meeple_on_feature_node(s, cell, terrain, group_idx)) continue;
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    s.board_meeple_owner[static_cast<size_t>(cell)] =
        static_cast<std::int8_t>(kNoOwner);
    s.board_meeple_pos[static_cast<size_t>(cell)] =
        static_cast<std::int8_t>(kMeepleNoPos);
    s.meeples_remaining[static_cast<size_t>(owner)] =
        static_cast<std::int8_t>(s.meeples_remaining[static_cast<size_t>(owner)] + 1);
  }
}

void return_cloister_meeple(CarcassonneState& s, int cell) {
  const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
  if (owner < 0) return;
  if (s.board_meeple_pos[static_cast<size_t>(cell)] != kMeepleCenterPos) return;
  s.board_meeple_owner[static_cast<size_t>(cell)] =
      static_cast<std::int8_t>(kNoOwner);
  s.board_meeple_pos[static_cast<size_t>(cell)] =
      static_cast<std::int8_t>(kMeepleNoPos);
  s.meeples_remaining[static_cast<size_t>(owner)] =
      static_cast<std::int8_t>(s.meeples_remaining[static_cast<size_t>(owner)] + 1);
}

int majority_count(const std::array<int, kPlayers>& owners) {
  int best = 0;
  for (int count : owners) best = std::max(best, count);
  return best;
}

void record_feature_final_meeple_scores(
    CarcassonneState& s, const FeatureCollect& fc, Terrain terrain,
    int points) {
  const int best = majority_count(fc.info.owners);
  if (best <= 0 || points <= 0) return;
  std::array<std::uint8_t, kPlayers> recorded{};
  for (const auto& node : fc.nodes) {
    const int cell = node.first;
    const int group_idx = node.second;
    if (!meeple_on_feature_node(s, cell, terrain, group_idx)) continue;
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (fc.info.owners[static_cast<size_t>(owner)] == best &&
        recorded[static_cast<size_t>(owner)] == 0) {
      s.final_meeple_scores[static_cast<size_t>(cell)] += points;
      recorded[static_cast<size_t>(owner)] = 1;
    }
  }
}

void record_farm_final_meeple_scores(CarcassonneState& s,
                                     const FarmCollect& fc, int points) {
  const int best = majority_count(fc.info.owners);
  if (best <= 0 || points <= 0) return;
  std::array<std::uint8_t, kPlayers> recorded{};
  for (const auto& node : fc.nodes) {
    const int cell = node.first;
    const int group_idx = node.second;
    if (!farm_node_has_meeple(s, cell, group_idx)) continue;
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (fc.info.owners[static_cast<size_t>(owner)] == best &&
        recorded[static_cast<size_t>(owner)] == 0) {
      s.final_meeple_scores[static_cast<size_t>(cell)] += points;
      recorded[static_cast<size_t>(owner)] = 1;
    }
  }
}

bool validate_action_impl(const CarcassonneState& s, ActionId action) {
  if (s.terminal || action < 0 || action >= kActionSpace) return false;
  if (s.stage == kStagePlaceTile) {
    if (action >= kPlaceActionCount || s.current_tile == kNoTile) return false;
    return CarcassonneRules::can_place_tile_at(
        s, s.current_tile, CarcassonneRules::decode_row(action),
        CarcassonneRules::decode_col(action),
        CarcassonneRules::decode_rotation(action));
  }
  if (s.stage == kStageMeeple) {
    if (action == kMeeplePassAction) return true;
    if (action < kMeepleSideActionBase || action >= kActionSpace) return false;
    return CarcassonneRules::can_place_meeple(
        s, CarcassonneRules::decode_meeple_position(action));
  }
  throw std::runtime_error("carcassonne unknown stage");
}

}  // namespace

ActionId CarcassonneRules::encode_place_action(int row, int col, int rotation) {
  if (row < 0 || row >= kBoardRows || col < 0 || col >= kBoardCols ||
      rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne placement action coordinate out of range");
  }
  return (row * kBoardCols + col) * 4 + rotation;
}

ActionId CarcassonneRules::encode_meeple_pass_action() {
  return kMeeplePassAction;
}

ActionId CarcassonneRules::encode_meeple_side_action(int side) {
  if (side < 0 || side >= kSides) {
    throw std::out_of_range("carcassonne meeple side out of range");
  }
  return kMeepleSideActionBase + side;
}

ActionId CarcassonneRules::encode_meeple_farm_action(int farmer_side) {
  if (farmer_side < 0 || farmer_side >= kFarmSides) {
    throw std::out_of_range("carcassonne meeple farm side out of range");
  }
  return kMeepleFarmActionBase + farmer_side;
}

ActionId CarcassonneRules::encode_meeple_center_action() {
  return kMeepleCenterAction;
}

int CarcassonneRules::decode_row(ActionId action) {
  if (action < 0 || action >= kPlaceActionCount) {
    throw std::out_of_range("carcassonne placement action out of range");
  }
  return static_cast<int>(action / 4) / kBoardCols;
}

int CarcassonneRules::decode_col(ActionId action) {
  if (action < 0 || action >= kPlaceActionCount) {
    throw std::out_of_range("carcassonne placement action out of range");
  }
  return static_cast<int>(action / 4) % kBoardCols;
}

int CarcassonneRules::decode_rotation(ActionId action) {
  if (action < 0 || action >= kPlaceActionCount) {
    throw std::out_of_range("carcassonne placement action out of range");
  }
  return static_cast<int>(action % 4);
}

int CarcassonneRules::decode_meeple_position(ActionId action) {
  if (action >= kMeepleSideActionBase &&
      action < kMeepleSideActionBase + kSides) {
    return static_cast<int>(action - kMeepleSideActionBase);
  }
  if (action >= kMeepleFarmActionBase &&
      action < kMeepleFarmActionBase + kFarmSides) {
    return kSides + static_cast<int>(action - kMeepleFarmActionBase);
  }
  if (action == kMeepleCenterAction) return kMeepleCenterPos;
  throw std::out_of_range("carcassonne meeple action out of range");
}

bool CarcassonneRules::can_place_tile_at(const CarcassonneState& s,
                                         int tile_type, int row, int col,
                                         int rotation) {
  if (!s.in_bounds(row, col) || tile_type < 0 || tile_type >= kTileTypes ||
      rotation < 0 || rotation >= 4) {
    return false;
  }
  // A symmetric tile looks the same every `period` quarter turns. Only the
  // lowest of those rotations is an action, so equal placements are one
  // action and one state (board_rotation is always below the period).
  if (rotation >= rotation_period(tile_type)) return false;
  const int cell = s.index(row, col);
  if (s.board_tile[static_cast<size_t>(cell)] != kNoTile) return false;

  bool adjacent = false;
  for (int side = 0; side < kSides; ++side) {
    const int nr = row + kDr[static_cast<size_t>(side)];
    const int nc = col + kDc[static_cast<size_t>(side)];
    if (!s.in_bounds(nr, nc)) continue;
    const int ncell = s.index(nr, nc);
    const int ntile = s.board_tile[static_cast<size_t>(ncell)];
    if (ntile == kNoTile) continue;
    adjacent = true;
    const int nrot = s.board_rotation[static_cast<size_t>(ncell)];
    if (edge_terrain_at(tile_type, rotation, side) !=
        edge_terrain_at(ntile, nrot, opposite_side(side))) {
      return false;
    }
  }
  return adjacent;
}

bool CarcassonneRules::can_place_meeple(const CarcassonneState& s, int pos) {
  if (s.stage != kStageMeeple) return false;
  const int player = s.current_player_;
  if (s.meeples_remaining[static_cast<size_t>(player)] <= 0) return false;
  if (s.last_row < 0 || s.last_col < 0) return false;
  const int cell = s.index(s.last_row, s.last_col);
  if (s.board_tile[static_cast<size_t>(cell)] == kNoTile) return false;
  if (s.board_meeple_owner[static_cast<size_t>(cell)] != kNoOwner) return false;

  const int tile_type = s.board_tile[static_cast<size_t>(cell)];
  const int rotation = s.board_rotation[static_cast<size_t>(cell)];
  if (pos >= 0 && pos < kSides) {
    const Terrain terrain = edge_terrain_at(tile_type, rotation, pos);
    if (terrain != kCity && terrain != kRoad) return false;
    const int group = group_for_side(tile_type, rotation, terrain, pos);
    if (group < 0) return false;
    const auto group_sides = sides_for_group(tile_type, rotation, terrain, group);
    const int canonical = *std::min_element(group_sides.begin(),
                                            group_sides.end());
    if (pos != canonical) return false;
    const auto fc = collect_feature(s, cell, terrain, pos);
    return fc.info.owners[0] == 0 && fc.info.owners[1] == 0;
  }
  if (pos >= kSides && pos < kSides + kFarmSides) {
    const int farmer_side = pos - kSides;
    const int group = farm_group_for_connection(tile_type, rotation, farmer_side);
    if (group < 0) return false;
    const int canonical = canonical_farm_connection(tile_def(tile_type), rotation, group);
    if (farmer_side != canonical) return false;
    const auto farm = collect_farm_impl(s, cell, farmer_side);
    return farm.info.owners[0] == 0 && farm.info.owners[1] == 0;
  }
  if (pos == kMeepleCenterPos) {
    return tile_def(tile_type).chapel;
  }
  return false;
}

FeatureScoreInfo CarcassonneRules::collect_city(const CarcassonneState& s,
                                                int start_cell,
                                                int start_side) {
  return collect_feature(s, start_cell, kCity, start_side).info;
}

FeatureScoreInfo CarcassonneRules::collect_road(const CarcassonneState& s,
                                                int start_cell,
                                                int start_side) {
  return collect_feature(s, start_cell, kRoad, start_side).info;
}

FarmScoreInfo CarcassonneRules::collect_farm(const CarcassonneState& s,
                                             int start_cell,
                                             int farmer_side) {
  return collect_farm_impl(s, start_cell, farmer_side).info;
}

int CarcassonneRules::feature_group_for_side(int tile_type, int rotation,
                                             Terrain terrain, int side) {
  return group_for_side(tile_type, rotation, terrain, side);
}

std::array<int, kPlayers> CarcassonneRules::compute_final_scores(
    const CarcassonneState& s) {
  // Project final scoring without mutating the caller's state. We need
  // a non-const copy because `apply_final_scoring` writes into scores /
  // final_meeple_scores / final_score_deltas. The clone is cheap
  // relative to one PV evaluator call (board + small arrays).
  CarcassonneState clone = s;
  // Final scoring skips complete features: by then they have always been
  // resolved. In the meeple stage the ones the last tile completed have
  // not, so resolve them first or their points vanish from the projection.
  if (clone.stage == kStageMeeple && clone.last_row >= 0 && clone.last_col >= 0) {
    resolve_completed_features(clone, clone.index(clone.last_row, clone.last_col));
  }
  apply_final_scoring(clone);
  return clone.scores;
}

PlacementPreview CarcassonneRules::preview_placement(const CarcassonneState& s,
                                                     int row, int col,
                                                     int rotation) {
  CarcassonneState tmp = s;
  apply_tile_action(tmp, encode_place_action(row, col, rotation));
  const int cell = tmp.index(row, col);
  const int tile_type = tmp.board_tile[static_cast<size_t>(cell)];

  PlacementPreview out;
  bool scores = false;
  // Two groups of the tile can be one feature (joined elsewhere).
  std::set<std::pair<int, std::pair<int, int>>> seen;
  for (int side = 0; side < kSides; ++side) {
    const Terrain terrain = edge_terrain_at(tile_type, rotation, side);
    if (terrain != kCity && terrain != kRoad) continue;
    const int group = group_for_side(tile_type, rotation, terrain, side);
    const auto group_sides = sides_for_group(tile_type, rotation, terrain, group);
    if (side != *std::min_element(group_sides.begin(), group_sides.end())) continue;
    const FeatureCollect fc = collect_feature(tmp, cell, terrain, side);
    if (can_place_meeple(tmp, side)) {
      int& claim = terrain == kCity ? out.city_claim : out.road_claim;
      claim = std::max(claim, fc.info.complete ? 2 : 1);
    }
    if (!seen.insert({static_cast<int>(terrain), canonical_feature_node(fc.nodes)}).second) continue;
    for (int p = 0; p < kPlayers; ++p) {
      out.meeples[static_cast<size_t>(p)] += fc.info.owners[static_cast<size_t>(p)];
    }
    if (!fc.info.complete) continue;
    out.completed += 1;
    if (majority_count(fc.info.owners) > 0) scores = true;
  }
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      if (!tmp.in_bounds(row + dr, col + dc)) continue;
      const int c = tmp.index(row + dr, col + dc);
      const int tile = tmp.board_tile[static_cast<size_t>(c)];
      if (tile == kNoTile || !tile_def(tile).chapel || cloister_occupancy(tmp, c) != 9) continue;
      const bool monk = tmp.board_meeple_owner[static_cast<size_t>(c)] != kNoOwner &&
                        tmp.board_meeple_pos[static_cast<size_t>(c)] == kMeepleCenterPos;
      if (!monk && c != cell) continue;
      out.completed += 1;
      if (monk) scores = true;
    }
  }
  if (scores) {
    resolve_completed_features(tmp, cell);
    for (int p = 0; p < kPlayers; ++p) {
      out.score_delta[static_cast<size_t>(p)] =
          tmp.scores[static_cast<size_t>(p)] - s.scores[static_cast<size_t>(p)];
    }
  }
  return out;
}

int CarcassonneRules::cloister_occupancy(const CarcassonneState& s, int cell) {
  if (cell < 0 || cell >= kBoardCells) {
    throw std::out_of_range("carcassonne cloister cell out of range");
  }
  const int row = cell / kBoardCols;
  const int col = cell % kBoardCols;
  int count = 0;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int rr = row + dr;
      const int cc = col + dc;
      if (s.in_bounds(rr, cc) &&
          s.board_tile[static_cast<size_t>(s.index(rr, cc))] != kNoTile) {
        count += 1;
      }
    }
  }
  return count;
}

bool CarcassonneRules::validate_action(const IGameState& state,
                                       ActionId action) const {
  const auto& s = checked_cast<CarcassonneState>(state);
  return validate_action_impl(s, action);
}

std::vector<ActionId> CarcassonneRules::legal_actions(
    const IGameState& state) const {
  const auto& s = checked_cast<CarcassonneState>(state);
  std::vector<ActionId> out;
  if (s.terminal) return out;
  if (s.stage == kStagePlaceTile) {
    if (s.current_tile == kNoTile) return out;
    out.reserve(96);
    for (int row = 0; row < kBoardRows; ++row) {
      for (int col = 0; col < kBoardCols; ++col) {
        for (int rot = 0; rot < 4; ++rot) {
          const ActionId action = encode_place_action(row, col, rot);
          if (can_place_tile_at(s, s.current_tile, row, col, rot)) {
            out.push_back(action);
          }
        }
      }
    }
    return out;
  }
  if (s.stage == kStageMeeple) {
    out.push_back(kMeeplePassAction);
    for (int side = 0; side < kSides; ++side) {
      if (can_place_meeple(s, side)) {
        out.push_back(encode_meeple_side_action(side));
      }
    }
    for (int fs = 0; fs < kFarmSides; ++fs) {
      if (can_place_meeple(s, kSides + fs)) {
        out.push_back(encode_meeple_farm_action(fs));
      }
    }
    if (can_place_meeple(s, kMeepleCenterPos)) {
      out.push_back(kMeepleCenterAction);
    }
    return out;
  }
  throw std::runtime_error("carcassonne unknown stage in legal_actions");
}

bool CarcassonneRules::any_legal_tile_placement(const CarcassonneState& s,
                                                int tile_type) {
  for (int row = 0; row < kBoardRows; ++row) {
    for (int col = 0; col < kBoardCols; ++col) {
      for (int rot = 0; rot < 4; ++rot) {
        if (can_place_tile_at(s, tile_type, row, col, rot)) return true;
      }
    }
  }
  return false;
}

int CarcassonneRules::draw_next_tile(CarcassonneState& s, std::mt19937_64* rng) {
  // Bag is a count-only multiset (game-rules/bag-draw recipe). Each draw
  // is a weighted-by-count sample from `bag_counts` consuming the
  // caller's rng. If the drawn tile cannot be placed anywhere on the
  // current board, increment discarded_tile_counts and re-draw —
  // matches both the rulebook intent and how Python ref impl handles
  // unplaceable tiles.
  if (rng == nullptr) {
    throw std::runtime_error("carcassonne draw_next_tile requires rng");
  }
  while (s.deck_count > 0) {
    const int tile_type = s.draw_one_tile_from_bag(*rng);
    if (tile_type == kNoTile) break;
    if (tile_type < 0 || tile_type >= kTileTypes) {
      throw std::runtime_error("carcassonne draw_next_tile invalid tile");
    }
    if (any_legal_tile_placement(s, tile_type)) return tile_type;
    s.discarded_tile_counts[static_cast<size_t>(tile_type)] =
        static_cast<std::int8_t>(
            s.discarded_tile_counts[static_cast<size_t>(tile_type)] + 1);
  }
  return kNoTile;
}

void CarcassonneRules::resolve_completed_features(CarcassonneState& s,
                                                  int placed_cell) {
  s.last_score_delta = 0;
  const int tile_type = s.board_tile[static_cast<size_t>(placed_cell)];
  const int rotation = s.board_rotation[static_cast<size_t>(placed_cell)];
  std::array<std::uint8_t, kSides * 2> processed{};

  for (int side = 0; side < kSides; ++side) {
    const Terrain terrain = edge_terrain_at(tile_type, rotation, side);
    if (terrain != kCity && terrain != kRoad) continue;
    const int group_idx = group_for_side(tile_type, rotation, terrain, side);
    const int terrain_offset = terrain == kCity ? 0 : kSides;
    const int proc_key = terrain_offset + group_idx;
    if (processed[static_cast<size_t>(proc_key)] != 0) continue;
    processed[static_cast<size_t>(proc_key)] = 1;

    FeatureCollect fc = collect_feature(s, placed_cell, terrain, side);
    if (!fc.info.complete) continue;
    if (terrain == kCity) {
      award_to_majority(s, fc.info.owners,
                        2 * (fc.info.tiles + fc.info.shields));
    } else {
      award_to_majority(s, fc.info.owners, fc.info.tiles);
    }
    return_feature_meeples(s, fc.nodes, terrain);
  }

  const int placed_row = placed_cell / kBoardCols;
  const int placed_col = placed_cell % kBoardCols;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int row = placed_row + dr;
      const int col = placed_col + dc;
      if (!s.in_bounds(row, col)) continue;
      const int cell = s.index(row, col);
      const int tile = s.board_tile[static_cast<size_t>(cell)];
      if (tile == kNoTile || !tile_def(tile).chapel) continue;
      const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
      if (owner < 0) continue;
      if (s.board_meeple_pos[static_cast<size_t>(cell)] != kMeepleCenterPos) {
        continue;
      }
      if (cloister_occupancy(s, cell) == 9) {
        s.scores[static_cast<size_t>(owner)] += 9;
        if (owner == s.current_player_) s.last_score_delta += 9;
        return_cloister_meeple(s, cell);
      }
    }
  }
}

void CarcassonneRules::apply_final_scoring(CarcassonneState& s) {
  const auto scores_before_final = s.scores;
  s.final_meeple_scores.fill(0);
  s.final_score_deltas = {0, 0};
  std::array<std::uint8_t, kBoardCells * kSides * 2> processed{};
  for (int cell = 0; cell < kBoardCells; ++cell) {
    const int tile_type = s.board_tile[static_cast<size_t>(cell)];
    if (tile_type == kNoTile) continue;
    const int rotation = s.board_rotation[static_cast<size_t>(cell)];
    for (int side = 0; side < kSides; ++side) {
      const Terrain terrain = edge_terrain_at(tile_type, rotation, side);
      if (terrain != kCity && terrain != kRoad) continue;
      const int group_idx = group_for_side(tile_type, rotation, terrain, side);
      const int terrain_offset = terrain == kCity ? 0 : kBoardCells * kSides;
      const int key = terrain_offset + cell * kSides + group_idx;
      if (processed[static_cast<size_t>(key)] != 0) continue;
      FeatureCollect fc = collect_feature(s, cell, terrain, side);
      for (const auto& node : fc.nodes) {
        const int node_key =
            terrain_offset + node.first * kSides + node.second;
        processed[static_cast<size_t>(node_key)] = 1;
      }
      if (fc.info.complete) continue;
      if (terrain == kCity) {
        const int points = fc.info.tiles + fc.info.shields;
        record_feature_final_meeple_scores(s, fc, terrain, points);
        award_to_majority(s, fc.info.owners, points);
      } else {
        record_feature_final_meeple_scores(s, fc, terrain, fc.info.tiles);
        award_to_majority(s, fc.info.owners, fc.info.tiles);
      }
    }
  }

  for (int cell = 0; cell < kBoardCells; ++cell) {
    const int tile_type = s.board_tile[static_cast<size_t>(cell)];
    if (tile_type == kNoTile || !tile_def(tile_type).chapel) continue;
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (owner < 0) continue;
    if (s.board_meeple_pos[static_cast<size_t>(cell)] != kMeepleCenterPos) {
      continue;
    }
    const int points = cloister_occupancy(s, cell);
    if (points == 9) continue;
    s.final_meeple_scores[static_cast<size_t>(cell)] += points;
    s.scores[static_cast<size_t>(owner)] += points;
    if (owner == s.current_player_) s.last_score_delta += points;
  }

  std::array<std::uint8_t, kBoardCells * kSides> farm_processed{};
  for (int cell = 0; cell < kBoardCells; ++cell) {
    const int owner = s.board_meeple_owner[static_cast<size_t>(cell)];
    if (owner < 0) continue;
    const int pos = s.board_meeple_pos[static_cast<size_t>(cell)];
    if (pos < kSides || pos >= kSides + kFarmSides) continue;
    const int tile_type = s.board_tile[static_cast<size_t>(cell)];
    const int rotation = s.board_rotation[static_cast<size_t>(cell)];
    const int group = farm_group_for_connection(tile_type, rotation, pos - kSides);
    if (group < 0) continue;
    const int key = cell * kSides + group;
    if (farm_processed[static_cast<size_t>(key)] != 0) continue;
    FarmCollect fc = collect_farm_impl(s, cell, pos - kSides);
    for (const auto& node : fc.nodes) {
      farm_processed[static_cast<size_t>(node.first * kSides + node.second)] = 1;
    }
    const int points = fc.info.completed_cities * 3;
    record_farm_final_meeple_scores(s, fc, points);
    award_to_majority(s, fc.info.owners, points);
  }

  for (int p = 0; p < kPlayers; ++p) {
    s.final_score_deltas[static_cast<size_t>(p)] =
        s.scores[static_cast<size_t>(p)] -
        scores_before_final[static_cast<size_t>(p)];
  }
}

void CarcassonneRules::award_to_owner_majority(
    CarcassonneState& s, const std::array<int, kPlayers>& owners, int points) {
  award_to_majority(s, owners, points);
}

void CarcassonneRules::set_winner(CarcassonneState& s) {
  int best = *std::max_element(s.scores.begin(), s.scores.end());
  int winner = -1;
  bool tied = false;
  for (int p = 0; p < kPlayers; ++p) {
    if (s.scores[static_cast<size_t>(p)] == best) {
      if (winner < 0) {
        winner = p;
      } else {
        tied = true;
      }
    }
  }
  s.winner_ = tied ? -1 : winner;
  s.shared_victory = tied;
}

void CarcassonneRules::apply_tile_action(CarcassonneState& s, ActionId action) {
  if (s.stage != kStagePlaceTile || !validate_action_impl(s, action)) {
    throw std::invalid_argument(
        "illegal carcassonne tile action " + std::to_string(action));
  }
  s.last_score_delta = 0;
  const int row = decode_row(action);
  const int col = decode_col(action);
  const int rotation = decode_rotation(action);
  const int placed_cell = s.index(row, col);
  const int placed_tile = s.current_tile;
  const int history_idx = s.plies + 1;
  if (history_idx < 0 || history_idx >= kHistorySlots) {
    throw std::runtime_error("carcassonne placement history overflow");
  }
  s.board_tile[static_cast<size_t>(placed_cell)] =
      static_cast<std::int8_t>(placed_tile);
  s.board_rotation[static_cast<size_t>(placed_cell)] =
      static_cast<std::int8_t>(rotation);
  s.history_tile[static_cast<size_t>(history_idx)] =
      static_cast<std::int8_t>(placed_tile);
  s.history_rotation[static_cast<size_t>(history_idx)] =
      static_cast<std::int8_t>(rotation);
  s.history_row[static_cast<size_t>(history_idx)] =
      static_cast<std::int8_t>(row);
  s.history_col[static_cast<size_t>(history_idx)] =
      static_cast<std::int8_t>(col);
  s.current_tile = static_cast<std::int8_t>(kNoTile);
  s.last_row = static_cast<std::int8_t>(row);
  s.last_col = static_cast<std::int8_t>(col);
  s.last_rotation = static_cast<std::int8_t>(rotation);
  s.stage = static_cast<std::int8_t>(kStageMeeple);
}

void CarcassonneRules::complete_turn_after_meeple(CarcassonneState& s,
                                                  std::mt19937_64* rng) {
  const int placed_cell = s.index(s.last_row, s.last_col);
  resolve_completed_features(s, placed_cell);
  s.plies += 1;

  const int next_tile = draw_next_tile(s, rng);
  if (next_tile == kNoTile) {
    apply_final_scoring(s);
    set_winner(s);
    s.current_tile = static_cast<std::int8_t>(kNoTile);
    s.terminal = true;
    s.stage = static_cast<std::int8_t>(kStagePlaceTile);
    return;
  }

  s.current_tile = static_cast<std::int8_t>(next_tile);
  s.current_player_ = (s.current_player_ + 1) % kPlayers;
  s.stage = static_cast<std::int8_t>(kStagePlaceTile);
}

void CarcassonneRules::center_board(CarcassonneState& s) {
  int min_row = kBoardRows;
  int max_row = -1;
  int min_col = kBoardCols;
  int max_col = -1;
  for (int row = 0; row < kBoardRows; ++row) {
    for (int col = 0; col < kBoardCols; ++col) {
      const int cell = s.index(row, col);
      if (s.board_tile[static_cast<size_t>(cell)] == kNoTile) continue;
      min_row = std::min(min_row, row);
      max_row = std::max(max_row, row);
      min_col = std::min(min_col, col);
      max_col = std::max(max_col, col);
    }
  }
  if (max_row < 0) return;

  const int occupied_rows = max_row - min_row + 1;
  const int occupied_cols = max_col - min_col + 1;
  if (occupied_rows > kBoardRows || occupied_cols > kBoardCols) {
    throw std::runtime_error("carcassonne board exceeds centering bounds");
  }
  const int target_min_row = (kBoardRows - occupied_rows + 1) / 2;
  const int target_min_col = (kBoardCols - occupied_cols + 1) / 2;
  const int dr = target_min_row - min_row;
  const int dc = target_min_col - min_col;
  if (dr == 0 && dc == 0) return;

  std::array<std::int8_t, kBoardCells> next_tile{};
  std::array<std::int8_t, kBoardCells> next_rotation{};
  std::array<std::int8_t, kBoardCells> next_owner{};
  std::array<std::int8_t, kBoardCells> next_pos{};
  std::array<int, kBoardCells> next_final_scores{};
  next_tile.fill(static_cast<std::int8_t>(kNoTile));
  next_rotation.fill(0);
  next_owner.fill(static_cast<std::int8_t>(kNoOwner));
  next_pos.fill(static_cast<std::int8_t>(kMeepleNoPos));
  next_final_scores.fill(0);

  for (int row = 0; row < kBoardRows; ++row) {
    for (int col = 0; col < kBoardCols; ++col) {
      const int src = s.index(row, col);
      if (s.board_tile[static_cast<size_t>(src)] == kNoTile) continue;
      const int dst_row = row + dr;
      const int dst_col = col + dc;
      if (!s.in_bounds(dst_row, dst_col)) {
        throw std::runtime_error("carcassonne centering moved tile out of bounds");
      }
      const int dst = s.index(dst_row, dst_col);
      next_tile[static_cast<size_t>(dst)] = s.board_tile[static_cast<size_t>(src)];
      next_rotation[static_cast<size_t>(dst)] =
          s.board_rotation[static_cast<size_t>(src)];
      next_owner[static_cast<size_t>(dst)] =
          s.board_meeple_owner[static_cast<size_t>(src)];
      next_pos[static_cast<size_t>(dst)] =
          s.board_meeple_pos[static_cast<size_t>(src)];
      next_final_scores[static_cast<size_t>(dst)] =
          s.final_meeple_scores[static_cast<size_t>(src)];
    }
  }

  s.board_tile = next_tile;
  s.board_rotation = next_rotation;
  s.board_meeple_owner = next_owner;
  s.board_meeple_pos = next_pos;
  s.final_meeple_scores = next_final_scores;

  if (s.last_row >= 0 && s.last_col >= 0) {
    const int row = static_cast<int>(s.last_row) + dr;
    const int col = static_cast<int>(s.last_col) + dc;
    if (!s.in_bounds(row, col)) {
      throw std::runtime_error("carcassonne centering moved last tile out of bounds");
    }
    s.last_row = static_cast<std::int8_t>(row);
    s.last_col = static_cast<std::int8_t>(col);
  }

  for (int i = 0; i < kHistorySlots; ++i) {
    if (s.history_tile[static_cast<size_t>(i)] == kNoTile) continue;
    const int row = static_cast<int>(s.history_row[static_cast<size_t>(i)]) + dr;
    const int col = static_cast<int>(s.history_col[static_cast<size_t>(i)]) + dc;
    if (!s.in_bounds(row, col)) {
      throw std::runtime_error("carcassonne centering moved history out of bounds");
    }
    s.history_row[static_cast<size_t>(i)] = static_cast<std::int8_t>(row);
    s.history_col[static_cast<size_t>(i)] = static_cast<std::int8_t>(col);
  }
}

void CarcassonneRules::apply_meeple_action(CarcassonneState& s, ActionId action,
                                           std::mt19937_64* rng) {
  if (s.stage != kStageMeeple || !validate_action_impl(s, action)) {
    throw std::invalid_argument(
        "illegal carcassonne meeple action " + std::to_string(action));
  }
  if (action != kMeeplePassAction) {
    const int pos = decode_meeple_position(action);
    const int cell = s.index(s.last_row, s.last_col);
    s.board_meeple_owner[static_cast<size_t>(cell)] =
        static_cast<std::int8_t>(s.current_player_);
    s.board_meeple_pos[static_cast<size_t>(cell)] =
        static_cast<std::int8_t>(pos);
    s.meeples_remaining[static_cast<size_t>(s.current_player_)] =
        static_cast<std::int8_t>(
            s.meeples_remaining[static_cast<size_t>(s.current_player_)] - 1);
  }
  complete_turn_after_meeple(s, rng);
}

void CarcassonneRules::apply_action(CarcassonneState& s, ActionId action,
                                    std::mt19937_64* rng) {
  if (s.stage == kStagePlaceTile) {
    apply_tile_action(s, action);
    return;
  }
  if (s.stage == kStageMeeple) {
    apply_meeple_action(s, action, rng);
    return;
  }
  throw std::runtime_error("carcassonne apply_action unknown stage");
}

void CarcassonneRules::do_action_fast_impl(IGameState& state, ActionId action,
                                           std::mt19937_64& rng) const {
  auto& s = checked_cast<CarcassonneState>(state);
  apply_action(s, action, &rng);
  center_board(s);
}

UndoToken CarcassonneRules::do_action_deterministic_impl(
    IGameState& /*state*/, ActionId /*action*/) const {
  throw std::runtime_error(
      "carcassonne: deterministic/undo path is not supported");
}

void CarcassonneRules::undo_action_impl(IGameState& state,
                                        const UndoToken& token) const {
  (void)state;
  (void)token;
  throw std::runtime_error("carcassonne: undo is not supported");
}

}  // namespace board_ai::carcassonne
