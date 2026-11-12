#include "arknova_map.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace board_ai::arknova {
namespace {

constexpr int kDirs[6][2] = {{0, -1}, {1, -1}, {1, 0}, {0, 1}, {-1, 1}, {-1, 0}};  // N NE SE S SW NW

// The grid by columns (hexmap.MapDef.from_columns): column col, row k is axial (col, k - (col + 1) / 2).
constexpr int kColumnLengths[] = {6, 7, 6, 7, 6, 7, 6, 7, 6};

struct MapData {
  std::array<MapCell, kMapCells> cells{};
  std::map<std::pair<int, int>, int> index;
};

MapData build_grid() {
  std::vector<std::pair<int, int>> qr;
  for (int col = 0; col < static_cast<int>(std::size(kColumnLengths)); ++col) {
    for (int row = 0; row < kColumnLengths[col]; ++row) qr.emplace_back(col, row - (col + 1) / 2);
  }
  if (static_cast<int>(qr.size()) != kMapCells) throw std::runtime_error("arknova map: cell count");
  std::sort(qr.begin(), qr.end());
  MapData m;
  for (int i = 0; i < kMapCells; ++i) {
    m.cells[i].q = static_cast<std::int8_t>(qr[i].first);
    m.cells[i].r = static_cast<std::int8_t>(qr[i].second);
    m.index[qr[i]] = i;
  }
  for (int i = 0; i < kMapCells; ++i) {
    MapCell& c = m.cells[i];
    for (int d = 0; d < 6; ++d) {
      auto it = m.index.find({c.q + kDirs[d][0], c.r + kDirs[d][1]});
      c.nbr[d] = it == m.index.end() ? -1 : static_cast<std::int8_t>(it->second);
      if (c.nbr[d] < 0) c.border = true;
    }
  }
  return m;
}

const MapData& map_data() {
  static const MapData m = build_grid();
  return m;
}

struct MapSource {
  const char* key;
  const char* name;
  MapRule rule;
  const char* terrain;  // per cell: '.' plain, 'W' water, 'R' rock
  PlacementBonus bonus[kMapCells];
  CellMask build2;
  CellMask rule_cells;
  Effect partner_zoo_slots[kMaxPartnerZoos];
  Effect university_slots[kNumUniversities];
  Effect last_worker;
  MapToken tokens[kNumMapTokens];
  std::int8_t n_prebuilt;
  Prebuilt prebuilt[2];
};

#include "arknova_maps_data.inc"

std::vector<MapDef> build_maps() {
  std::vector<MapDef> out;
  for (const MapSource& src : kMapSources) {
    MapDef m;
    m.key = src.key;
    m.name = src.name;
    if (std::string(src.terrain).size() != kMapCells) throw std::runtime_error(std::string("arknova map: ") + src.key);
    for (int c = 0; c < kMapCells; ++c) {
      MapTerrain& t = m.cells[c];
      t.terrain = src.terrain[c] == 'W' ? Terrain::kWater : src.terrain[c] == 'R' ? Terrain::kRock : Terrain::kPlain;
      t.bonus = src.bonus[c];
      t.build2_only = src.build2 >> c & 1;
      if (t.terrain == Terrain::kPlain) m.buildable |= CellMask{1} << c;
    }
    m.rule = src.rule;
    m.rule_cells = src.rule_cells;
    std::copy(std::begin(src.partner_zoo_slots), std::end(src.partner_zoo_slots), m.partner_zoo_slots);
    std::copy(std::begin(src.university_slots), std::end(src.university_slots), m.university_slots);
    m.last_worker = src.last_worker;
    std::copy(std::begin(src.tokens), std::end(src.tokens), m.tokens);
    m.n_prebuilt = src.n_prebuilt;
    std::copy(src.prebuilt, src.prebuilt + src.n_prebuilt, m.prebuilt);
    out.push_back(m);
  }
  return out;
}

const std::vector<MapDef>& all_maps() {
  static const std::vector<MapDef> maps = build_maps();
  return maps;
}

using ShapeKey = std::vector<std::pair<int, int>>;

ShapeKey shape_key(const Shape& s) {
  ShapeKey k;
  for (int i = 0; i < s.n; ++i) k.emplace_back(s.cells[i][0], s.cells[i][1]);
  std::sort(k.begin(), k.end());
  return k;
}

ShapeKey normalized(ShapeKey cells) {
  std::sort(cells.begin(), cells.end());
  auto m = cells.front();
  for (auto& c : cells) c = {c.first - m.first, c.second - m.second};
  return cells;
}

std::vector<ShapeKey> orientations(const ShapeKey& base) {
  std::vector<ShapeKey> out;
  ShapeKey cur = base;
  for (int i = 0; i < 6; ++i) {
    ShapeKey n = normalized(cur);
    if (std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
    for (auto& c : cur) c = {-c.second, c.first + c.second};
  }
  return out;
}

std::vector<ShapePosition> compute_positions(const ShapeKey& base, const MapDef& md) {
  const MapData& m = map_data();
  std::vector<ShapePosition> out;
  std::vector<CellMask> seen;
  for (const ShapeKey& o : orientations(base)) {
    for (int a = 0; a < kMapCells; ++a) {
      CellMask fs = 0;
      bool on_map = true;
      for (auto [dq, dr] : o) {
        auto it = m.index.find({m.cells[a].q + dq, m.cells[a].r + dr});
        if (it == m.index.end()) {
          on_map = false;
          break;
        }
        fs |= CellMask{1} << it->second;
      }
      if (!on_map || std::find(seen.begin(), seen.end(), fs) != seen.end()) continue;
      seen.push_back(fs);
      ShapePosition p;
      p.cells = fs;
      p.plain = (fs & md.buildable) == fs;
      CellMask near = neighbor_mask(fs);
      for (int c = 0; c < kMapCells; ++c) {
        if (fs >> c & 1) {
          p.has_build2 |= md.cells[c].build2_only;
          p.on_border += m.cells[c].border;
        }
        if (near >> c & 1) {
          p.rock += md.cells[c].terrain == Terrain::kRock;
          p.water += md.cells[c].terrain == Terrain::kWater;
        }
      }
      out.push_back(p);
    }
  }
  return out;
}

Shape make_shape(std::initializer_list<std::pair<int, int>> cells) {
  Shape s;
  for (auto [q, r] : cells) {
    s.cells[s.n][0] = static_cast<std::int8_t>(q);
    s.cells[s.n][1] = static_cast<std::int8_t>(r);
    ++s.n;
  }
  return s;
}

struct ShapeTables {
  std::map<std::pair<int, int>, Shape> standard;  // (kind, size) -> shape
  std::map<ShapeKey, std::vector<std::vector<ShapePosition>>> positions;  // per map
};

const ShapeTables& shape_tables() {
  static const ShapeTables t = [] {
    ShapeTables t;
    auto put = [&](BuildingKind k, int size, Shape s) { t.standard[{static_cast<int>(k), size}] = s; };
    put(BuildingKind::kEnclosure, 1, make_shape({{0, 0}}));
    put(BuildingKind::kEnclosure, 2, make_shape({{0, 0}, {1, 0}}));
    put(BuildingKind::kEnclosure, 3, make_shape({{0, 0}, {1, 0}, {0, 1}}));
    put(BuildingKind::kEnclosure, 4, make_shape({{0, 0}, {1, -1}, {1, 0}, {2, -1}}));
    put(BuildingKind::kEnclosure, 5, make_shape({{0, 0}, {1, -1}, {1, 0}, {2, -1}, {2, 0}}));
    put(BuildingKind::kKiosk, 1, make_shape({{0, 0}}));
    put(BuildingKind::kPavilion, 1, make_shape({{0, 0}}));
    put(BuildingKind::kPettingZoo, 3, make_shape({{0, 0}, {1, -1}, {2, -1}}));
    put(BuildingKind::kReptileHouse, 5, make_shape({{0, 0}, {-1, 0}, {-1, 1}, {1, -1}, {1, 0}}));
    put(BuildingKind::kAviary, 5, make_shape({{0, 0}, {-1, 0}, {1, -1}, {-1, 1}, {0, 1}}));
    auto add = [&](const Shape& s) {
      ShapeKey k = shape_key(s);
      if (t.positions.count(k)) return;
      auto& per_map = t.positions[k];
      for (const MapDef& md : all_maps()) per_map.push_back(compute_positions(k, md));
    };
    for (auto& [_, s] : t.standard) add(s);
    for (int c = 0; c < num_cards(); ++c) {
      if (card(static_cast<CardIdx>(c)).building.present) add(card_shape(static_cast<CardIdx>(c)));
    }
    return t;
  }();
  return t;
}

}  // namespace

const MapCell& map_cell(int i) { return map_data().cells.at(i); }

int num_maps() { return static_cast<int>(all_maps().size()); }

const MapDef& map_def(int m) { return all_maps().at(m); }

int map_by_key(const std::string& key) {
  for (int m = 0; m < num_maps(); ++m) {
    if (key == map_def(m).key) return m;
  }
  return -1;
}

int cell_index(int q, int r) {
  const auto& idx = map_data().index;
  auto it = idx.find({q, r});
  return it == idx.end() ? -1 : it->second;
}

CellMask neighbor_mask(CellMask cells) {
  const MapData& m = map_data();
  CellMask out = 0;
  for (int c = 0; c < kMapCells; ++c) {
    if (!(cells >> c & 1)) continue;
    for (int d = 0; d < 6; ++d) {
      if (m.cells[c].nbr[d] >= 0) out |= CellMask{1} << m.cells[c].nbr[d];
    }
  }
  return out & ~cells;
}

int hex_distance(int a, int b) {
  const MapCell& x = map_cell(a);
  const MapCell& y = map_cell(b);
  int dq = x.q - y.q, dr = x.r - y.r;
  return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

const Shape& standard_shape(BuildingKind kind, int size) {
  const auto& st = shape_tables().standard;
  auto it = st.find({static_cast<int>(kind), size});
  if (it == st.end()) {
    throw std::runtime_error(std::string("arknova: no shape for ") + kBuildingKindNames[static_cast<int>(kind)] +
                             " size " + std::to_string(size));
  }
  return it->second;
}

bool has_standard_shape(BuildingKind kind, int size) {
  return shape_tables().standard.count({static_cast<int>(kind), size}) > 0;
}

Shape card_shape(CardIdx sponsor) {
  const BuildingSpec& b = card(sponsor).building;
  if (!b.present) throw std::runtime_error(std::string("arknova: card has no building: ") + card(sponsor).name);
  Shape s;
  s.n = b.n;
  for (int i = 0; i < b.n; ++i) {
    s.cells[i][0] = b.shape[i][0];
    s.cells[i][1] = b.shape[i][1];
  }
  return s;
}

const std::vector<ShapePosition>& shape_positions(const Shape& base, int map) {
  const auto& pos = shape_tables().positions;
  auto it = pos.find(shape_key(base));
  if (it == pos.end()) throw std::runtime_error("arknova: shape positions not precomputed");
  return it->second.at(map);
}

int placement_orientation(const Shape& base, CellMask cells) {
  ShapeKey placed;
  for (int c = 0; c < kMapCells; ++c) {
    if (cells >> c & 1) placed.emplace_back(map_cell(c).q, map_cell(c).r);
  }
  if (placed.empty()) throw std::runtime_error("arknova: empty placement");
  placed = normalized(placed);
  std::vector<ShapeKey> all = orientations(shape_key(base));
  auto it = std::find(all.begin(), all.end(), placed);
  if (it == all.end()) throw std::runtime_error("arknova: placement does not match the shape");
  return static_cast<int>(it - all.begin());
}

bool cells_less(CellMask a, CellMask b) {
  while (a && b) {
    int ca = __builtin_ctzll(a), cb = __builtin_ctzll(b);
    if (ca != cb) return ca < cb;
    a &= a - 1;
    b &= b - 1;
  }
  return !a && b;
}

Zoo::Zoo() { std::fill(std::begin(cell_owner), std::end(cell_owner), kNoBuilding); }

CellMask Zoo::covered() const {
  CellMask m = 0;
  for (int i = 0; i < n_buildings; ++i) m |= buildings[i].cells;
  return m;
}

std::vector<CellMask> Zoo::legal_placements(const PlacementQuery& q) const {
  int need_rock = q.any_terrain ? 0 : q.need_rock;
  int need_water = q.any_terrain ? 0 : q.need_water;
  if (is_special(q.kind) && count(q.kind)) return {};
  const Shape& base = q.shape ? *q.shape : standard_shape(q.kind, q.size);
  CellMask owned = covered();
  CellMask occupied_adj = neighbor_mask(owned);
  CellMask kiosks = 0;
  if (q.kind == BuildingKind::kKiosk) {
    for (int i = 0; i < n_buildings; ++i) {
      if (buildings[i].kind == BuildingKind::kKiosk) kiosks |= buildings[i].cells;
    }
  }
  bool first = n_buildings == 0;
  std::vector<CellMask> found;
  for (const ShapePosition& p : shape_positions(base, map)) {
    if ((!p.plain && !q.any_terrain) || (p.has_build2 && !q.build2) || p.on_border < q.min_border) continue;
    if (need_rock > p.rock || need_water > p.water) continue;
    if (p.cells & owned) continue;
    if (first) {
      if (!p.on_border) continue;
    } else if (q.adjacent && !(p.cells & occupied_adj)) {
      continue;
    }
    if (kiosks) {
      bool near = false;
      for (int c = 0; c < kMapCells && !near; ++c) {
        if (!(p.cells >> c & 1)) continue;
        for (int k = 0; k < kMapCells; ++k) {
          if ((kiosks >> k & 1) && hex_distance(c, k) < 3) {
            near = true;
            break;
          }
        }
      }
      if (near) continue;
    }
    found.push_back(p.cells);
  }
  std::sort(found.begin(), found.end(), cells_less);
  return found;
}

int Zoo::place(BuildingKind kind, int size, CellMask cells, CardIdx name, std::vector<PlacementBonus>* bonuses) {
  if (cells & covered()) throw std::runtime_error("arknova: cell already covered");
  if (n_buildings >= kMaxBuildings) throw std::runtime_error("arknova: too many buildings");
  int id = n_buildings++;
  Building& b = buildings[id];
  b = Building{};
  b.kind = kind;
  b.size = static_cast<std::int8_t>(size);
  b.cells = cells;
  b.name = name;
  const MapDef& md = def();
  for (int c = 0; c < kMapCells; ++c) {
    if (!(cells >> c & 1)) continue;
    cell_owner[c] = static_cast<std::int8_t>(id);
    if (bonuses && md.cells[c].bonus != kPBNone) bonuses->push_back(md.cells[c].bonus);
  }
  return id;
}

int Zoo::adjacent_terrain(CellMask cells, Terrain t) const {
  CellMask near = neighbor_mask(cells);
  const MapDef& md = def();
  int n = 0;
  for (int c = 0; c < kMapCells; ++c) n += (near >> c & 1) && md.cells[c].terrain == t;
  return n;
}

std::vector<int> Zoo::adjacent_buildings(int b) const {
  CellMask near = neighbor_mask(buildings[b].cells);
  std::vector<int> ids;
  for (int c = 0; c < kMapCells; ++c) {
    if ((near >> c & 1) && cell_owner[c] != kNoBuilding && cell_owner[c] != b) {
      if (std::find(ids.begin(), ids.end(), cell_owner[c]) == ids.end()) ids.push_back(cell_owner[c]);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

int Zoo::kiosk_income_of(int k) const {
  int total = 0;
  for (int a : adjacent_buildings(k)) {
    const Building& o = buildings[a];
    if (o.kind == BuildingKind::kPavilion || o.kind == BuildingKind::kUnique || is_special(o.kind) ||
        (o.kind == BuildingKind::kEnclosure && o.occupied)) {
      ++total;
    }
  }
  return total;
}

int Zoo::kiosk_income() const {
  int total = 0;
  for (int k = 0; k < n_buildings; ++k) {
    if (buildings[k].kind == BuildingKind::kKiosk) total += kiosk_income_of(k);
  }
  return total;
}

bool Zoo::is_fully_covered() const { return (def().buildable & ~covered()) == 0; }

int Zoo::count(BuildingKind kind) const {
  int n = 0;
  for (int i = 0; i < n_buildings; ++i) n += buildings[i].kind == kind;
  return n;
}

}  // namespace board_ai::arknova
