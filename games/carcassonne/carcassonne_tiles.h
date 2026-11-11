#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace board_ai::carcassonne {

constexpr int kTileTypes = 24;
constexpr int kSides = 4;
constexpr int kFarmSides = 8;

enum Terrain : std::int8_t {
  kField = 0,
  kRoad = 1,
  kCity = 2,
};

enum Side : int {
  kTop = 0,
  kRight = 1,
  kBottom = 2,
  kLeft = 3,
};

enum FarmerSide : int {
  kFarmTLL = 0,
  kFarmTLT = 1,
  kFarmTRT = 2,
  kFarmTRR = 3,
  kFarmBRR = 4,
  kFarmBRB = 5,
  kFarmBLB = 6,
  kFarmBLL = 7,
};

struct FeatureGroup {
  std::int8_t count = 0;
  std::array<std::int8_t, kSides> sides{{-1, -1, -1, -1}};
};

struct FarmGroup {
  std::int8_t connection_count = 0;
  std::array<std::int8_t, kFarmSides> connections{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::int8_t city_side_count = 0;
  std::array<std::int8_t, kSides> city_sides{{-1, -1, -1, -1}};
};

struct TileDef {
  char code = 'A';
  const char* name = "";
  const char* image = "";
  std::int8_t initial_count = 0;
  std::array<std::int8_t, kSides> edges{{kField, kField, kField, kField}};
  std::int8_t city_group_count = 0;
  std::array<FeatureGroup, kSides> city_groups{};
  std::int8_t road_group_count = 0;
  std::array<FeatureGroup, kSides> road_groups{};
  std::int8_t farm_group_count = 0;
  std::array<FarmGroup, kSides> farm_groups{};
  bool shield = false;
  bool chapel = false;
};

const std::vector<TileDef>& tile_defs();
const TileDef& tile_def(int tile_type);
int tile_type_from_code(char code);
// Quarter turns after which the tile looks the same again: 1 (B, C, X),
// 2 (F, G, H, U) or 4. Rotation r and r % period give the same placement,
// so only rotations below the period are legal actions.
int rotation_period(int tile_type);
// Meeple position (0..3 board side, 4..11 board half-edge) -> the lowest
// position of the same city, road or farm on that tile, which is the one the
// rules store and offer as an action. Any other value is returned unchanged.
int canonical_meeple_position(int tile_type, int rotation, int position);
int rotated_side(int side, int rotation);
int opposite_side(int side);
int rotated_farmer_side(int farmer_side, int rotation);
int opposite_farmer_side(int farmer_side);
int farmer_side_edge(int farmer_side);
Terrain edge_terrain_at(int tile_type, int rotation, int side);
std::vector<int> rotated_group_sides(const FeatureGroup& group, int rotation);
std::vector<int> rotated_farm_connections(const FarmGroup& group, int rotation);
std::vector<int> rotated_farm_city_sides(const FarmGroup& group, int rotation);
std::string tile_code_string(int tile_type);

}  // namespace board_ai::carcassonne
