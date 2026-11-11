#include "carcassonne_tiles.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>

namespace board_ai::carcassonne {

namespace {

FeatureGroup group(std::initializer_list<int> sides) {
  FeatureGroup g;
  g.count = static_cast<std::int8_t>(sides.size());
  int i = 0;
  for (int side : sides) {
    if (side < 0 || side >= kSides) {
      throw std::runtime_error("carcassonne tile group side out of range");
    }
    g.sides[static_cast<size_t>(i)] = static_cast<std::int8_t>(side);
    ++i;
  }
  return g;
}

FarmGroup farm(std::initializer_list<int> connections,
               std::initializer_list<int> city_sides = {}) {
  FarmGroup g;
  g.connection_count = static_cast<std::int8_t>(connections.size());
  int i = 0;
  for (int side : connections) {
    if (side < 0 || side >= kFarmSides) {
      throw std::runtime_error("carcassonne farm connection out of range");
    }
    g.connections[static_cast<size_t>(i)] = static_cast<std::int8_t>(side);
    ++i;
  }
  g.city_side_count = static_cast<std::int8_t>(city_sides.size());
  i = 0;
  for (int side : city_sides) {
    if (side < 0 || side >= kSides) {
      throw std::runtime_error("carcassonne farm city side out of range");
    }
    g.city_sides[static_cast<size_t>(i)] = static_cast<std::int8_t>(side);
    ++i;
  }
  return g;
}

TileDef make_tile(char code, const char* name, int count,
                  std::array<std::int8_t, kSides> edges,
                  std::initializer_list<FeatureGroup> cities,
                  std::initializer_list<FeatureGroup> roads,
                  std::initializer_list<FarmGroup> farms,
                  bool shield = false,
                  bool chapel = false) {
  TileDef d;
  d.code = code;
  d.name = name;
  d.initial_count = static_cast<std::int8_t>(count);
  d.image = "";
  d.edges = edges;
  d.city_group_count = static_cast<std::int8_t>(cities.size());
  int i = 0;
  for (const auto& g : cities) {
    d.city_groups[static_cast<size_t>(i)] = g;
    ++i;
  }
  d.road_group_count = static_cast<std::int8_t>(roads.size());
  i = 0;
  for (const auto& g : roads) {
    d.road_groups[static_cast<size_t>(i)] = g;
    ++i;
  }
  d.farm_group_count = static_cast<std::int8_t>(farms.size());
  i = 0;
  for (const auto& g : farms) {
    d.farm_groups[static_cast<size_t>(i)] = g;
    ++i;
  }
  d.shield = shield;
  d.chapel = chapel;
  return d;
}

// A tile's groups must agree with its edges, or features leak across them:
// a farm half-edge listed on a city edge joins two fields through the city
// wall whenever the neighbour makes the same mistake (tile D once did).
//   - every city / road side lies in exactly one group of that terrain;
//   - a half-edge lies in exactly one farm unless its edge is city, where it
//     lies in none; the two halves of a field edge share their farm;
//   - a farm's city sides are city edges.
void validate_tile(const TileDef& d) {
  const std::string where = std::string("carcassonne tile ") + d.code + ": ";
  std::array<int, kSides> city_groups{};
  std::array<int, kSides> road_groups{};
  for (int g = 0; g < d.city_group_count; ++g) {
    const FeatureGroup& fg = d.city_groups[static_cast<size_t>(g)];
    for (int i = 0; i < fg.count; ++i) city_groups[static_cast<size_t>(fg.sides[static_cast<size_t>(i)])] += 1;
  }
  for (int g = 0; g < d.road_group_count; ++g) {
    const FeatureGroup& fg = d.road_groups[static_cast<size_t>(g)];
    for (int i = 0; i < fg.count; ++i) road_groups[static_cast<size_t>(fg.sides[static_cast<size_t>(i)])] += 1;
  }
  for (int side = 0; side < kSides; ++side) {
    const int terrain = d.edges[static_cast<size_t>(side)];
    if (city_groups[static_cast<size_t>(side)] != (terrain == kCity ? 1 : 0) ||
        road_groups[static_cast<size_t>(side)] != (terrain == kRoad ? 1 : 0)) {
      throw std::runtime_error(where + "city / road groups disagree with edge " + std::to_string(side));
    }
  }
  std::array<int, kFarmSides> farm_of_half{};
  farm_of_half.fill(-1);
  for (int g = 0; g < d.farm_group_count; ++g) {
    const FarmGroup& fg = d.farm_groups[static_cast<size_t>(g)];
    if (fg.connection_count == 0) throw std::runtime_error(where + "farm without half-edges");
    for (int i = 0; i < fg.connection_count; ++i) {
      const int half = fg.connections[static_cast<size_t>(i)];
      if (farm_of_half[static_cast<size_t>(half)] >= 0) {
        throw std::runtime_error(where + "half-edge " + std::to_string(half) + " in two farms");
      }
      farm_of_half[static_cast<size_t>(half)] = g;
    }
    for (int i = 0; i < fg.city_side_count; ++i) {
      if (d.edges[static_cast<size_t>(fg.city_sides[static_cast<size_t>(i)])] != kCity) {
        throw std::runtime_error(where + "farm touches a city on a non-city edge");
      }
    }
  }
  for (int half = 0; half < kFarmSides; ++half) {
    const bool city_edge = d.edges[static_cast<size_t>(farmer_side_edge(half))] == kCity;
    if ((farm_of_half[static_cast<size_t>(half)] >= 0) == city_edge) {
      throw std::runtime_error(where + "farm half-edge " + std::to_string(half) + " disagrees with its edge terrain");
    }
  }
  constexpr std::array<std::array<int, 2>, kSides> kHalvesOfSide{{
      {{kFarmTLT, kFarmTRT}}, {{kFarmTRR, kFarmBRR}}, {{kFarmBRB, kFarmBLB}}, {{kFarmTLL, kFarmBLL}}}};
  for (int side = 0; side < kSides; ++side) {
    if (d.edges[static_cast<size_t>(side)] != kField) continue;
    const auto& halves = kHalvesOfSide[static_cast<size_t>(side)];
    if (farm_of_half[static_cast<size_t>(halves[0])] != farm_of_half[static_cast<size_t>(halves[1])]) {
      throw std::runtime_error(where + "field edge " + std::to_string(side) + " split between two farms");
    }
  }
}

std::vector<TileDef> build_defs() {
  std::vector<TileDef> defs;
  defs.reserve(kTileTypes);
  auto f = static_cast<std::int8_t>(kField);
  auto r = static_cast<std::int8_t>(kRoad);
  auto c = static_cast<std::int8_t>(kCity);

  defs.push_back(make_tile('A', "chapel_with_road", 2, {f, f, r, f},
                           {}, {group({kBottom})},
                           {farm({kFarmTLL, kFarmTLT, kFarmTRT, kFarmTRR,
                                  kFarmBRR, kFarmBRB, kFarmBLB, kFarmBLL})},
                           false, true));
  defs.push_back(make_tile('B', "chapel", 4, {f, f, f, f},
                           {}, {},
                           {farm({kFarmTLL, kFarmTLT, kFarmTRT, kFarmTRR,
                                  kFarmBRR, kFarmBRB, kFarmBLB, kFarmBLL})},
                           false, true));
  defs.push_back(make_tile('C', "full_city_with_shield", 1, {c, c, c, c},
                           {group({kTop, kRight, kBottom, kLeft})}, {},
                           {}, true, false));
  defs.push_back(make_tile('D', "city_top_straight_road", 4, {c, r, f, r},
                           {group({kTop})}, {group({kRight, kLeft})},
                           {farm({kFarmTLL, kFarmTRR}, {kTop}),
                            farm({kFarmBRR, kFarmBRB, kFarmBLB, kFarmBLL})}));
  defs.push_back(make_tile('E', "city_top", 5, {c, f, f, f},
                           {group({kTop})}, {},
                           {farm({kFarmTLL, kFarmTRR, kFarmBRR, kFarmBRB,
                                  kFarmBLB, kFarmBLL}, {kTop})}));
  defs.push_back(make_tile('F', "city_narrow_shield", 2, {f, c, f, c},
                           {group({kRight, kLeft})}, {},
                           {farm({kFarmTLT, kFarmTRT}, {kRight, kLeft}),
                            farm({kFarmBRB, kFarmBLB}, {kRight, kLeft})},
                           true, false));
  defs.push_back(make_tile('G', "city_narrow", 1, {f, c, f, c},
                           {group({kRight, kLeft})}, {},
                           {farm({kFarmTLT, kFarmTRT}, {kRight, kLeft}),
                            farm({kFarmBRB, kFarmBLB}, {kRight, kLeft})}));
  defs.push_back(make_tile('H', "city_left_right", 3, {f, c, f, c},
                           {group({kRight}), group({kLeft})}, {},
                           {farm({kFarmTLT, kFarmTRT, kFarmBRB, kFarmBLB},
                                 {kRight, kLeft})}));
  defs.push_back(make_tile('I', "city_top_right", 2, {c, c, f, f},
                           {group({kTop}), group({kRight})}, {},
                           {farm({kFarmTLL, kFarmBRB, kFarmBLB, kFarmBLL},
                                 {kTop, kRight})}));
  defs.push_back(make_tile('J', "city_top_road_bend_right", 3, {c, r, r, f},
                           {group({kTop})}, {group({kRight, kBottom})},
                           {farm({kFarmTLL, kFarmTRR, kFarmBLB, kFarmBLL}, {kTop}),
                            farm({kFarmBRR, kFarmBRB})}));
  defs.push_back(make_tile('K', "city_top_road_bend_left", 3, {c, f, r, r},
                           {group({kTop})}, {group({kBottom, kLeft})},
                           {farm({kFarmTLL, kFarmTRR, kFarmBRB, kFarmBRR}, {kTop}),
                            farm({kFarmBLL, kFarmBLB})}));
  defs.push_back(make_tile('L', "city_top_crossroads", 3, {c, r, r, r},
                           {group({kTop})},
                           {group({kRight}), group({kBottom}), group({kLeft})},
                           {farm({kFarmTLL, kFarmTRR}, {kTop}),
                            farm({kFarmBLL, kFarmBLB}),
                            farm({kFarmBRB, kFarmBRR})}));
  defs.push_back(make_tile('M', "city_diagonal_top_right_shield", 2,
                           {c, c, f, f},
                           {group({kTop, kRight})}, {},
                           {farm({kFarmTLL, kFarmBLB, kFarmBLL, kFarmBRB},
                                 {kTop, kRight})},
                           true, false));
  defs.push_back(make_tile('N', "city_diagonal_top_right", 3, {c, c, f, f},
                           {group({kTop, kRight})}, {},
                           {farm({kFarmTLL, kFarmBLB, kFarmBLL, kFarmBRB},
                                 {kTop, kRight})}));
  defs.push_back(make_tile('O', "city_diagonal_top_left_shield_road", 2,
                           {c, r, r, c},
                           {group({kTop, kLeft})}, {group({kRight, kBottom})},
                           {farm({kFarmBLB, kFarmTRR}, {kTop, kLeft}),
                            farm({kFarmBRR, kFarmBRB})},
                           true, false));
  defs.push_back(make_tile('P', "city_diagonal_top_left_road", 3,
                           {c, r, r, c},
                           {group({kTop, kLeft})}, {group({kRight, kBottom})},
                           {farm({kFarmBLB, kFarmTRR}, {kTop, kLeft}),
                            farm({kFarmBRR, kFarmBRB})}));
  defs.push_back(make_tile('Q', "city_bottom_grass_shield", 1, {c, c, f, c},
                           {group({kTop, kRight, kLeft})}, {},
                           {farm({kFarmBLB, kFarmBRB}, {kTop, kRight, kLeft})},
                           true, false));
  defs.push_back(make_tile('R', "city_bottom_grass", 3, {c, c, f, c},
                           {group({kTop, kRight, kLeft})}, {},
                           {farm({kFarmBLB, kFarmBRB}, {kTop, kRight, kLeft})}));
  defs.push_back(make_tile('S', "city_bottom_road_shield", 2, {c, c, r, c},
                           {group({kTop, kRight, kLeft})}, {group({kBottom})},
                           {farm({kFarmBLB}, {kTop, kRight, kLeft}),
                            farm({kFarmBRB}, {kTop, kRight, kLeft})},
                           true, false));
  defs.push_back(make_tile('T', "city_bottom_road", 1, {c, c, r, c},
                           {group({kTop, kRight, kLeft})}, {group({kBottom})},
                           {farm({kFarmBLB}, {kTop, kRight, kLeft}),
                            farm({kFarmBRB}, {kTop, kRight, kLeft})}));
  defs.push_back(make_tile('U', "straight_road", 8, {r, f, r, f},
                           {}, {group({kTop, kBottom})},
                           {farm({kFarmTLL, kFarmTLT, kFarmBLB, kFarmBLL}),
                            farm({kFarmTRR, kFarmTRT, kFarmBRR, kFarmBRB})}));
  defs.push_back(make_tile('V', "bent_road", 9, {f, f, r, r},
                           {}, {group({kBottom, kLeft})},
                           {farm({kFarmBLB, kFarmBLL}),
                            farm({kFarmTLL, kFarmTLT, kFarmTRR, kFarmTRT,
                                  kFarmBRB, kFarmBRR})}));
  defs.push_back(make_tile('W', "three_split_road", 4, {f, r, r, r},
                           {},
                           {group({kRight}), group({kBottom}), group({kLeft})},
                           {farm({kFarmBLB, kFarmBLL}),
                            farm({kFarmBRB, kFarmBRR}),
                            farm({kFarmTLL, kFarmTLT, kFarmTRR, kFarmTRT})}));
  defs.push_back(make_tile('X', "crossroads", 1, {r, r, r, r},
                           {},
                           {group({kTop}), group({kRight}), group({kBottom}), group({kLeft})},
                           {farm({kFarmBLB, kFarmBLL}),
                            farm({kFarmBRB, kFarmBRR}),
                            farm({kFarmTLL, kFarmTLT}),
                            farm({kFarmTRR, kFarmTRT})}));

  if (static_cast<int>(defs.size()) != kTileTypes) {
    throw std::runtime_error("carcassonne tile definition count mismatch");
  }
  for (int i = 0; i < kTileTypes; ++i) {
    if (defs[static_cast<size_t>(i)].code != static_cast<char>('A' + i)) {
      throw std::runtime_error("carcassonne tile definition order mismatch");
    }
    validate_tile(defs[static_cast<size_t>(i)]);
  }
  return defs;
}

// Side / half-edge sets of a tile's groups after `rotation`, as sorted
// bitmasks, so two rotations can be compared as pictures.
struct RotatedShape {
  std::array<std::int8_t, kSides> edges{};
  std::vector<unsigned> cities;
  std::vector<unsigned> roads;
  std::vector<unsigned> farms;  // halves | city sides << 8

  bool operator==(const RotatedShape& o) const {
    return edges == o.edges && cities == o.cities && roads == o.roads && farms == o.farms;
  }
};

RotatedShape rotated_shape(const TileDef& d, int rotation) {
  RotatedShape out;
  for (int side = 0; side < kSides; ++side) {
    out.edges[static_cast<size_t>(rotated_side(side, rotation))] = d.edges[static_cast<size_t>(side)];
  }
  auto side_mask = [&](const std::vector<int>& sides) {
    unsigned mask = 0;
    for (int s : sides) mask |= 1u << s;
    return mask;
  };
  for (int g = 0; g < d.city_group_count; ++g) {
    out.cities.push_back(side_mask(rotated_group_sides(d.city_groups[static_cast<size_t>(g)], rotation)));
  }
  for (int g = 0; g < d.road_group_count; ++g) {
    out.roads.push_back(side_mask(rotated_group_sides(d.road_groups[static_cast<size_t>(g)], rotation)));
  }
  for (int g = 0; g < d.farm_group_count; ++g) {
    const FarmGroup& fg = d.farm_groups[static_cast<size_t>(g)];
    out.farms.push_back(side_mask(rotated_farm_connections(fg, rotation)) |
                        side_mask(rotated_farm_city_sides(fg, rotation)) << 8);
  }
  std::sort(out.cities.begin(), out.cities.end());
  std::sort(out.roads.begin(), out.roads.end());
  std::sort(out.farms.begin(), out.farms.end());
  return out;
}

}  // namespace

const std::vector<TileDef>& tile_defs() {
  static const std::vector<TileDef> defs = build_defs();
  return defs;
}

const TileDef& tile_def(int tile_type) {
  if (tile_type < 0 || tile_type >= kTileTypes) {
    throw std::out_of_range("carcassonne tile_type out of range");
  }
  return tile_defs()[static_cast<size_t>(tile_type)];
}

int tile_type_from_code(char code) {
  if (code < 'A' || code > 'X') {
    throw std::out_of_range("carcassonne tile code out of range");
  }
  return static_cast<int>(code - 'A');
}

int rotation_period(int tile_type) {
  static const std::array<int, kTileTypes> periods = [] {
    std::array<int, kTileTypes> out{};
    for (int type = 0; type < kTileTypes; ++type) {
      const TileDef& d = tile_def(type);
      const RotatedShape upright = rotated_shape(d, 0);
      out[static_cast<size_t>(type)] =
          rotated_shape(d, 1) == upright ? 1 : rotated_shape(d, 2) == upright ? 2 : 4;
    }
    return out;
  }();
  if (tile_type < 0 || tile_type >= kTileTypes) {
    throw std::out_of_range("carcassonne tile_type out of range");
  }
  return periods[static_cast<size_t>(tile_type)];
}

int canonical_meeple_position(int tile_type, int rotation, int position) {
  const TileDef& d = tile_def(tile_type);
  auto lowest_if_member = [](const std::vector<int>& members, int wanted) {
    return std::find(members.begin(), members.end(), wanted) == members.end()
        ? -1 : *std::min_element(members.begin(), members.end());
  };
  if (position >= 0 && position < kSides) {
    for (int g = 0; g < d.city_group_count + d.road_group_count; ++g) {
      const FeatureGroup& group = g < d.city_group_count
          ? d.city_groups[static_cast<size_t>(g)]
          : d.road_groups[static_cast<size_t>(g - d.city_group_count)];
      const int lowest = lowest_if_member(rotated_group_sides(group, rotation), position);
      if (lowest >= 0) return lowest;
    }
  } else if (position >= kSides && position < kSides + kFarmSides) {
    for (int g = 0; g < d.farm_group_count; ++g) {
      const int lowest = lowest_if_member(
          rotated_farm_connections(d.farm_groups[static_cast<size_t>(g)], rotation), position - kSides);
      if (lowest >= 0) return kSides + lowest;
    }
  }
  return position;
}

int rotated_side(int side, int rotation) {
  if (side < 0 || side >= kSides) {
    throw std::out_of_range("carcassonne side out of range");
  }
  return (side + rotation + kSides) % kSides;
}

int opposite_side(int side) {
  if (side < 0 || side >= kSides) {
    throw std::out_of_range("carcassonne side out of range");
  }
  return (side + 2) % kSides;
}

int rotated_farmer_side(int farmer_side, int rotation) {
  if (farmer_side < 0 || farmer_side >= kFarmSides) {
    throw std::out_of_range("carcassonne farmer side out of range");
  }
  if (rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne rotation out of range");
  }
  static constexpr std::array<int, kFarmSides> kTurn{
      kFarmTRT, kFarmTRR, kFarmBRR, kFarmBRB,
      kFarmBLB, kFarmBLL, kFarmTLL, kFarmTLT};
  int out = farmer_side;
  for (int i = 0; i < rotation; ++i) {
    out = kTurn[static_cast<size_t>(out)];
  }
  return out;
}

int opposite_farmer_side(int farmer_side) {
  if (farmer_side < 0 || farmer_side >= kFarmSides) {
    throw std::out_of_range("carcassonne farmer side out of range");
  }
  static constexpr std::array<int, kFarmSides> kOpp{
      kFarmTRR, kFarmBLB, kFarmBRB, kFarmTLL,
      kFarmBLL, kFarmTRT, kFarmTLT, kFarmBRR};
  return kOpp[static_cast<size_t>(farmer_side)];
}

int farmer_side_edge(int farmer_side) {
  if (farmer_side < 0 || farmer_side >= kFarmSides) {
    throw std::out_of_range("carcassonne farmer side out of range");
  }
  switch (farmer_side) {
    case kFarmTLT:
    case kFarmTRT:
      return kTop;
    case kFarmTRR:
    case kFarmBRR:
      return kRight;
    case kFarmBRB:
    case kFarmBLB:
      return kBottom;
    case kFarmTLL:
    case kFarmBLL:
      return kLeft;
  }
  throw std::runtime_error("carcassonne farmer side edge missing");
}

Terrain edge_terrain_at(int tile_type, int rotation, int side) {
  const auto& d = tile_def(tile_type);
  if (rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne rotation out of range");
  }
  if (side < 0 || side >= kSides) {
    throw std::out_of_range("carcassonne side out of range");
  }
  const int original_side = (side - rotation + kSides * 4) % kSides;
  return static_cast<Terrain>(d.edges[static_cast<size_t>(original_side)]);
}

std::vector<int> rotated_group_sides(const FeatureGroup& group, int rotation) {
  if (rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne rotation out of range");
  }
  std::vector<int> out;
  out.reserve(static_cast<size_t>(group.count));
  for (int i = 0; i < group.count; ++i) {
    out.push_back(rotated_side(group.sides[static_cast<size_t>(i)], rotation));
  }
  return out;
}

std::vector<int> rotated_farm_connections(const FarmGroup& group, int rotation) {
  if (rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne rotation out of range");
  }
  std::vector<int> out;
  out.reserve(static_cast<size_t>(group.connection_count));
  for (int i = 0; i < group.connection_count; ++i) {
    out.push_back(rotated_farmer_side(group.connections[static_cast<size_t>(i)],
                                      rotation));
  }
  return out;
}

std::vector<int> rotated_farm_city_sides(const FarmGroup& group, int rotation) {
  if (rotation < 0 || rotation >= 4) {
    throw std::out_of_range("carcassonne rotation out of range");
  }
  std::vector<int> out;
  out.reserve(static_cast<size_t>(group.city_side_count));
  for (int i = 0; i < group.city_side_count; ++i) {
    out.push_back(rotated_side(group.city_sides[static_cast<size_t>(i)],
                               rotation));
  }
  return out;
}

std::string tile_code_string(int tile_type) {
  return std::string(1, tile_def(tile_type).code);
}

}  // namespace board_ai::carcassonne
