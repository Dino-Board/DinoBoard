#pragma once

// Hex geometry, the zoo maps and per-player zoo state (ark-nova-ai arknova/hexmap.py, maps.py).
// Every map has the same 58-space grid; cells are indexed 0..kMapCells-1 in sorted axial (q, r)
// order, so "sorted(cells)" in the Python engine is ascending cell index here. A set of cells is a
// CellMask (58 cells fit in 64 bits). What a cell is (terrain, placement bonus, Build II space)
// depends on the map: MapDef.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "arknova_cards.h"

namespace board_ai::arknova {

constexpr int kMapCells = 58;
using CellMask = std::uint64_t;

enum class Terrain : std::uint8_t { kPlain, kWater, kRock };

// The grid: position and neighbours, the same on every map.
struct MapCell {
  std::int8_t q = 0, r = 0;
  bool border = false;
  std::int8_t nbr[6] = {-1, -1, -1, -1, -1, -1};  // cell index or -1 (off the map)
};

const MapCell& map_cell(int i);
int cell_index(int q, int r);  // -1 when (q, r) is not on the map
CellMask neighbor_mask(CellMask cells);  // map cells adjacent to `cells`, excluding `cells`
int hex_distance(int a, int b);
inline int popcount(CellMask m) { return __builtin_popcountll(m); }

enum class BuildingKind : std::uint8_t {
  kEnclosure, kKiosk, kPavilion, kPettingZoo, kReptileHouse, kAviary, kUnique, kNumKinds
};
constexpr const char* kBuildingKindNames[] = {"enclosure", "kiosk", "pavilion", "petting_zoo",
                                              "reptile_house", "aviary", "unique"};
inline bool is_special(BuildingKind k) {
  return k == BuildingKind::kPettingZoo || k == BuildingKind::kReptileHouse || k == BuildingKind::kAviary;
}

// ---- zoo maps (map_rules.py: the special rule of each map reads its rule_cells) -------------

enum class MapRule : std::uint8_t {
  kNone,
  kObservationTower,   // +2 appeal when an empty standard enclosure next to a rule cell gets an animal
  kOutdoorArea,        // a standard enclosure next to a rule cell holds 2 more spaces
  kHarbor,             // once a rule cell is covered: once per turn, discard a hand card for 3 money
  kParkRestaurant,     // break income: 1 money per covered space next to a rule cell
  kResearchInstitute,  // once a rule cell is covered: ignore 1 condition when playing an animal
  kIceCream,           // all rule cells covered: break income of 1 money per kiosk
  kHollywood,          // covering a rule cell reveals cards until a sponsor (to the hand); all covered:
                       // sponsors cost 1 level less
};
constexpr int kObservationAppeal = 2;
constexpr int kOutdoorExtraSize = 2;
constexpr int kHarborMoney = 3;

struct MapTerrain {
  Terrain terrain = Terrain::kPlain;
  PlacementBonus bonus = kPBNone;
  bool build2_only = false;
};
struct MapToken {
  Effect effect;  // amount 0: nothing
  bool repeats;   // paid again in every break once taken
};
struct Prebuilt {
  BuildingKind kind;
  std::int8_t size;
  CellMask cells;
};
// Slot bonuses are indexed by the slot being filled (0-based); an entry with amount 0 is "nothing".
struct MapDef {
  const char* key = "";   // the Python map name: "map0", "plan1" ... "plan8", "plana"
  const char* name = "";
  std::array<MapTerrain, kMapCells> cells{};
  CellMask buildable = 0;  // plain cells
  MapRule rule = MapRule::kNone;
  CellMask rule_cells = 0;
  Effect partner_zoo_slots[kMaxPartnerZoos];
  Effect university_slots[kNumUniversities];
  Effect last_worker;
  MapToken tokens[kNumMapTokens];
  std::int8_t n_prebuilt = 0;  // printed buildings every zoo starts with (Map A)
  Prebuilt prebuilt[2];
};
int num_maps();
const MapDef& map_def(int m);
int map_by_key(const std::string& key);  // -1 when unknown

// A building shape as axial offsets. Shapes are identified by their sorted offset list.
struct Shape {
  std::int8_t n = 0;
  std::int8_t cells[kMaxShape][2] = {};
};
// SHAPES[(kind, size)]; throws for a pair with no standard shape.
const Shape& standard_shape(BuildingKind kind, int size);
bool has_standard_shape(BuildingKind kind, int size);
Shape card_shape(CardIdx sponsor);

// One distinct position of a shape on map `map` (hexmap._shape_positions).
struct ShapePosition {
  CellMask cells = 0;
  bool plain = false;
  bool has_build2 = false;
  std::int8_t on_border = 0;
  std::int8_t rock = 0;   // rock cells adjacent to the position
  std::int8_t water = 0;  // water cells adjacent to the position
};
const std::vector<ShapePosition>& shape_positions(const Shape& base, int map);
// Shapes turn but are never mirrored, so a shape has at most this many distinct orientations.
constexpr int kMaxOrientations = 6;
// Which distinct rotation of `base` the placement `cells` is (0 <= index < kMaxOrientations).
// Together with the lowest covered cell it identifies the placement. Throws when the cells are
// not a rotation of the shape.
int placement_orientation(const Shape& base, CellMask cells);

constexpr int kMaxBuildings = kMapCells;  // every building covers at least one cell
constexpr std::int8_t kNoBuilding = -1;

struct Building {
  BuildingKind kind = BuildingKind::kEnclosure;
  std::int8_t size = 0;
  CellMask cells = 0;
  bool occupied = false;
  std::int8_t used = 0;
  CardIdx name = kNoCard;  // the sponsor card of a unique building
  std::int8_t extra_size = 0;
  int capacity() const { return size + extra_size; }
};

struct PlacementQuery {
  BuildingKind kind = BuildingKind::kEnclosure;
  int size = 0;
  bool build2 = false;
  const Shape* shape = nullptr;  // nullptr = the standard shape of (kind, size)
  int need_rock = 0, need_water = 0, min_border = 0;
  bool adjacent = true;
  bool any_terrain = false;
};

struct Zoo {
  std::int8_t map = 0;  // map_def index
  std::int8_t n_buildings = 0;
  Building buildings[kMaxBuildings];
  std::int8_t cell_owner[kMapCells];

  Zoo();
  const MapDef& def() const { return map_def(map); }
  const MapTerrain& cell(int c) const { return def().cells[c]; }
  CellMask covered() const;
  // Legal positions sorted like the Python engine (by sorted cell list).
  std::vector<CellMask> legal_placements(const PlacementQuery& q) const;
  // Places a building and returns its id; `bonuses` receives the placement bonuses of the covered
  // cells in ascending cell order.
  int place(BuildingKind kind, int size, CellMask cells, CardIdx name, std::vector<PlacementBonus>* bonuses);
  int adjacent_terrain(CellMask cells, Terrain t) const;
  // Ids of buildings adjacent to building b, ascending.
  std::vector<int> adjacent_buildings(int b) const;
  int kiosk_income() const;
  // Break income of kiosk k alone: the buildings next to it that pay (see kiosk_income).
  int kiosk_income_of(int k) const;
  bool is_fully_covered() const;
  int count(BuildingKind kind) const;
};

// Sort key of a cell set: the ascending cell list, compared lexicographically.
bool cells_less(CellMask a, CellMask b);

}  // namespace board_ai::arknova
