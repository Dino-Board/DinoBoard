#include "harmonies_net_adapter.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>

#include "../../engine/core/game_interfaces.h"
#include "harmonies_card_data.h"
#include "harmonies_rules.h"

namespace board_ai::harmonies {

namespace {

template <int NPlayers, int Side>
void encode_player(const HarmoniesState<NPlayers, Side>& s, int p,
                   std::vector<float>* out) {
  using Spec = HarmoniesFeatureSpec<NPlayers, Side>;
  for (int h = 0; h < Spec::kHexes; ++h) {
    const int tt = static_cast<int>(s.tower_type[p][h]);
    for (int b = 0; b < Spec::kTowerTypeBuckets; ++b) {
      out->push_back(b == tt ? 1.0f : 0.0f);
    }
  }
  for (int h = 0; h < Spec::kHexes; ++h) {
    out->push_back(s.has_animal[p][h] ? 1.0f : 0.0f);
  }
  for (int i = 0; i < kHandSize; ++i) {
    const int cid = static_cast<int>(s.held_cards[p][i]);
    for (int c = 0; c <= kAnimalCards; ++c) {
      out->push_back(c == cid ? 1.0f : 0.0f);
    }
  }
  for (int i = 0; i < kHandSize; ++i) {
    const int rem = s.cube_remaining[p][i];
    const int init = std::max<int>(1, s.cube_initial[p][i]);
    out->push_back(static_cast<float>(rem) / static_cast<float>(init));
  }
  for (int i = 0; i < kHandSize; ++i) {
    out->push_back(static_cast<float>(s.cube_initial[p][i]) / 3.0f);
  }
  out->push_back(static_cast<float>(s.space_remaining[p]) / static_cast<float>(Spec::kHexes));
  out->push_back(static_cast<float>(s.animal_scores[p]) / 50.0f);
  out->push_back(static_cast<float>(s.landscape_scores[p]) / 50.0f);
}

}  // namespace

template <int Side>
std::vector<int> board_symmetry_action_map(int g) {
  using Geo = Board<Side>;
  std::vector<int> m(static_cast<std::size_t>(Geo::kActionSpace));
  std::iota(m.begin(), m.end(), 0);
  for (int h = 0; h < Geo::kHexes; ++h) {
    const int gh = Geo::kSymHex[static_cast<std::size_t>(g)][static_cast<std::size_t>(h)];
    for (int c = 0; c < kColors; ++c) {
      m[static_cast<std::size_t>(kActPlaceTokenBase + h * kColors + c)] = kActPlaceTokenBase + gh * kColors + c;
    }
    for (int k = 0; k < kAnimalCards; ++k) {
      m[static_cast<std::size_t>(Geo::kActPlaceCubeBase + h * kAnimalCards + k)] =
          Geo::kActPlaceCubeBase + gh * kAnimalCards + k;
    }
  }
  return m;
}

template std::vector<int> board_symmetry_action_map<kSideA>(int);
template std::vector<int> board_symmetry_action_map<kSideB>(int);

template <int NPlayers, int Side>
void HarmoniesFeatureEncoder<NPlayers, Side>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto& s = checked_cast<HarmoniesState<NPlayers, Side>>(state);
  out->clear();
  out->reserve(HarmoniesFeatureSpec<NPlayers, Side>::kFeatureDim);

  for (int k = 0; k < NPlayers; ++k) {
    const int p = (perspective_player + k) % NPlayers;
    encode_player(s, p, out);
  }

  for (int slot = 0; slot < kSlots; ++slot) {
    for (int c = 0; c < kColors; ++c) {
      out->push_back(static_cast<float>(s.central_board_counts[slot][c]) / 8.0f);
    }
  }
  for (int c = 0; c < kColors; ++c) {
    out->push_back(static_cast<float>(s.selected_tokens[c]) / 4.0f);
  }
  for (int slot = 0; slot < kSlots; ++slot) {
    const int cid = static_cast<int>(s.animal_market[slot]);
    for (int c = 0; c <= kAnimalCards; ++c) {
      out->push_back(c == cid ? 1.0f : 0.0f);
    }
  }
  for (int c = 0; c < kColors; ++c) {
    out->push_back(static_cast<float>(s.pouch_counts[c]) / 25.0f);
  }
  out->push_back(s.turn_started ? 1.0f : 0.0f);
  out->push_back(s.took_card_this_turn ? 1.0f : 0.0f);
  out->push_back(s.end_triggered ? 1.0f : 0.0f);
  const int cp = (s.current_player_ - perspective_player + NPlayers) % NPlayers;
  for (int k = 0; k < 4; ++k) {
    out->push_back(k == cp ? 1.0f : 0.0f);
  }
}

template <int NPlayers, int Side>
std::vector<EncoderSymmetry> HarmoniesFeatureEncoder<NPlayers, Side>::symmetries() const {
  using Spec = HarmoniesFeatureSpec<NPlayers, Side>;
  using Geo = Board<Side>;
  constexpr int kB = Spec::kTowerTypeBuckets;
  std::vector<EncoderSymmetry> out(static_cast<std::size_t>(Geo::kNumSymmetries));
  for (int g = 0; g < Geo::kNumSymmetries; ++g) {
    auto& fm = out[static_cast<std::size_t>(g)].feature_map;
    fm.resize(static_cast<std::size_t>(Spec::kFeatureDim));
    std::iota(fm.begin(), fm.end(), 0);
    for (int k = 0; k < NPlayers; ++k) {
      const int base = k * Spec::kPerPlayer;
      for (int h = 0; h < Spec::kHexes; ++h) {
        const int gh = Geo::kSymHex[static_cast<std::size_t>(g)][static_cast<std::size_t>(h)];
        for (int b = 0; b < kB; ++b) {
          fm[static_cast<std::size_t>(base + h * kB + b)] = base + gh * kB + b;
        }
        fm[static_cast<std::size_t>(base + Spec::kHexes * kB + h)] = base + Spec::kHexes * kB + gh;
      }
    }
    out[static_cast<std::size_t>(g)].action_map = board_symmetry_action_map<Side>(g);
  }
  return out;
}

#define HARMONIES_INST_HarmoniesFeatureEncoder(N, S) template class HarmoniesFeatureEncoder<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesFeatureEncoder)
#undef HARMONIES_INST_HarmoniesFeatureEncoder

// ---- HarmoniesBeliefTracker ----

template <int NPlayers, int Side>
void HarmoniesBeliefTracker<NPlayers, Side>::init(
    IGameState& /*state*/, int /*perspective*/, const AnyMap& /*payload*/) {}

template <int NPlayers, int Side>
void HarmoniesBeliefTracker<NPlayers, Side>::observe_public_event(
    int /*actor*/, ActionId /*action*/,
    const std::vector<PublicEvent>& /*events*/) {}

template <int NPlayers, int Side>
void HarmoniesBeliefTracker<NPlayers, Side>::randomize_unseen(
    IGameState& state, int /*observer*/, std::mt19937_64& rng) const {
  auto& s = checked_cast<HarmoniesState<NPlayers, Side>>(state);

  // Collect unseen pool: cards not in market, not in any player's hand and
  // not completed (completed cards are public and never return).
  std::vector<int> unseen;
  unseen.reserve(kAnimalCards);
  std::array<bool, kAnimalCards> in_market_or_hand{};
  for (int c = 0; c < kAnimalCards; ++c) {
    if (s.completed_cards[c]) in_market_or_hand[c] = true;
  }
  for (int slot = 0; slot < kSlots; ++slot) {
    const int cid = s.animal_market[slot];
    if (cid > 0) in_market_or_hand[cid - 1] = true;
  }
  for (int p = 0; p < NPlayers; ++p) {
    for (int i = 0; i < kHandSize; ++i) {
      const int cid = s.held_cards[p][i];
      if (cid > 0) in_market_or_hand[cid - 1] = true;
    }
  }
  for (int c = 0; c < kAnimalCards; ++c) {
    if (!in_market_or_hand[c]) unseen.push_back(c);
  }

  std::shuffle(unseen.begin(), unseen.end(), rng);

  // Every card is accounted for publicly, so the unseen pool IS the deck.
  // A size mismatch means the observer's public state is broken — report
  // it instead of sampling a deck that cannot exist.
  if (static_cast<int>(unseen.size()) != s.animal_deck_count) {
    throw std::runtime_error(
        "HarmoniesBeliefTracker::randomize_unseen: " +
        std::to_string(unseen.size()) + " cards are unaccounted for but the "
        "public deck count is " + std::to_string(s.animal_deck_count));
  }
  s.animal_deck.fill(0);
  const int deck_sz = s.animal_deck_count;
  for (int i = 0; i < deck_sz; ++i) {
    s.animal_deck[unseen[i]] = 1;
  }
}

template <int NPlayers, int Side>
std::unique_ptr<IBeliefTracker> HarmoniesBeliefTracker<NPlayers, Side>::clone() const {
  return std::make_unique<HarmoniesBeliefTracker<NPlayers, Side>>(*this);
}

template <int NPlayers, int Side>
AnyMap HarmoniesBeliefTracker<NPlayers, Side>::serialize() const {
  return {};
}

#define HARMONIES_INST_HarmoniesBeliefTracker(N, S) template class HarmoniesBeliefTracker<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesBeliefTracker)
#undef HARMONIES_INST_HarmoniesBeliefTracker

// ---- HarmoniesTokenEncoder ----

namespace {

constexpr int kInf = 1 << 20;

// Bottom-up colours of each tower type; a building counts as two reds.
constexpr int kTowerStack[kTowerTypes][3] = {
    {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}, {6, 0, 0},
    {6, 6, 0}, {3, 3, 0}, {3, 4, 0}, {2, 2, 0}, {3, 3, 4}, {2, 2, 2}};

// Fewest tokens stacked on tower type `from` to reach tower type `to`.
const std::array<std::array<int, kTowerTypes>, kTowerTypes>& tower_distance() {
  static const auto table = [] {
    std::array<std::array<int, kTowerTypes>, kTowerTypes> d{};
    for (int s = 0; s < kTowerTypes; ++s) {
      d[s].fill(kInf);
      d[s][s] = 0;
      std::array<int, kTowerTypes> queue{};
      int head = 0, tail = 0;
      queue[tail++] = s;
      while (head < tail) {
        const int u = queue[head++];
        for (int c = 0; c < kColors; ++c) {
          const int v = kTowerTransition[u][c];
          if (v >= 0 && d[s][v] == kInf) {
            d[s][v] = d[s][u] + 1;
            queue[tail++] = v;
          }
        }
      }
    }
    return d;
  }();
  return table;
}

// piece_distance()[card - 1][piece][tower] = tokens still needed on a hex of
// that tower type to satisfy the piece (kInf if it never can).
using PieceDistance = std::array<std::array<std::array<int, kTowerTypes>, kMaxPiecesPerCard>, kAnimalCards>;
const PieceDistance& piece_distance() {
  static const auto table = [] {
    PieceDistance t{};
    const auto& d = tower_distance();
    for (int c = 0; c < kAnimalCards; ++c) {
      for (int i = 0; i < kCards[c].pattern_size; ++i) {
        for (int from = 0; from < kTowerTypes; ++from) {
          int best = kInf;
          for (int to = 0; to < kTowerTypes; ++to) {
            if (kCards[c].pieces[i].tower_mask & (1u << to)) best = std::min(best, d[from][to]);
          }
          t[c][i][from] = best;
        }
      }
    }
    return t;
  }();
  return table;
}

// The tower a piece asks for when built from scratch: fewest tokens, then
// lowest tower type.
int target_tower(const CardPiece& piece) {
  const auto& d = tower_distance();
  int best = -1;
  for (int to = 0; to < kTowerTypes; ++to) {
    if (!(piece.tower_mask & (1u << to))) continue;
    if (best < 0 || d[0][to] < d[0][best]) best = to;
  }
  if (best < 0) throw std::logic_error("harmonies: animal card piece accepts no tower");
  return best;
}

// Every in-bounds way to lay a card's pattern on the board: the rules'
// anchoring (anchor piece on `anchor`, one of the 6 rotations).
struct Placement {
  int anchor = 0;
  int n = 0;
  std::array<std::int8_t, kMaxPiecesPerCard> hex{};
};

template <int Side>
const std::vector<Placement>& card_placements(int card_id) {
  using Geo = Board<Side>;
  static const auto table = [] {
    std::array<std::vector<Placement>, kAnimalCards> t;
    for (int c = 0; c < kAnimalCards; ++c) {
      const auto& card = kCards[c];
      const auto& ap = card.pieces[card.anchor_index];
      for (int a = 0; a < Geo::kHexes; ++a) {
        for (int r = 0; r < 6; ++r) {
          const auto* m = kTransforms[r];
          const int aq = m[0] * ap.dq + m[1] * ap.dr;
          const int ar = m[2] * ap.dq + m[3] * ap.dr;
          Placement p;
          p.anchor = a;
          p.n = card.pattern_size;
          bool inside = true;
          for (int i = 0; i < card.pattern_size && inside; ++i) {
            const auto& pc = card.pieces[i];
            const int h = Geo::axial_to_hex(Geo::kAxialQ[a] + m[0] * pc.dq + m[1] * pc.dr - aq,
                                            Geo::kAxialR[a] + m[2] * pc.dq + m[3] * pc.dr - ar);
            inside = h >= 0;
            p.hex[i] = static_cast<std::int8_t>(h);
          }
          if (inside) t[c].push_back(p);
        }
      }
    }
    return t;
  }();
  return table[static_cast<std::size_t>(card_id - 1)];
}

// How far one board is from a cube of one card. `miss[p]` = tokens still to
// place for placement p (kInf: off forever — its cube hex holds an animal, or
// a piece needs stacking on an animal or under an incompatible stack).
template <int Side>
struct CardFit {
  static constexpr int kH = Board<Side>::kHexes;
  int card = 0;
  int missing = kInf;  // fewest tokens to a legal cube
  int ready = 0;       // hexes where a cube can go now
  std::array<std::uint8_t, kH> part{}, missing_here{}, ready_at{};
  std::vector<int> miss;
};

template <int Side>
CardFit<Side> fit_card(const std::array<std::int8_t, Board<Side>::kHexes>& tt,
                       const std::array<std::int8_t, Board<Side>::kHexes>& ha, int card_id) {
  CardFit<Side> f;
  f.card = card_id;
  const auto& pls = card_placements<Side>(card_id);
  const auto& pd = piece_distance()[static_cast<std::size_t>(card_id - 1)];
  f.miss.assign(pls.size(), kInf);
  for (std::size_t j = 0; j < pls.size(); ++j) {
    const Placement& p = pls[j];
    if (ha[static_cast<std::size_t>(p.anchor)]) continue;
    int m = 0;
    for (int i = 0; i < p.n && m < kInf; ++i) {
      const int h = p.hex[static_cast<std::size_t>(i)];
      const int d = pd[static_cast<std::size_t>(i)][static_cast<std::size_t>(tt[static_cast<std::size_t>(h)])];
      m = d > 0 && ha[static_cast<std::size_t>(h)] ? kInf : std::min(kInf, m + d);
    }
    f.miss[j] = m;
    f.missing = std::min(f.missing, m);
  }
  for (std::size_t j = 0; j < pls.size(); ++j) {
    const Placement& p = pls[j];
    if (f.miss[j] == 0) f.ready_at[static_cast<std::size_t>(p.anchor)] = 1;
    if (f.missing >= kInf || f.miss[j] != f.missing) continue;
    for (int i = 0; i < p.n; ++i) {
      const int h = p.hex[static_cast<std::size_t>(i)];
      f.part[static_cast<std::size_t>(h)] = 1;
      if (pd[static_cast<std::size_t>(i)][static_cast<std::size_t>(tt[static_cast<std::size_t>(h)])] > 0) {
        f.missing_here[static_cast<std::size_t>(h)] = 1;
      }
    }
  }
  for (auto r : f.ready_at) f.ready += r;
  return f;
}

// The card's fewest missing tokens after placing colour c (0-based) on hex x,
// for every (x, c) with a legal stack and no animal on x; -1 elsewhere.
// Placing tokens never revives a placement that is off, so only placements
// through x change.
template <int Side>
std::vector<int> missing_after(const CardFit<Side>& f, const std::array<std::int8_t, Board<Side>::kHexes>& tt,
                               const std::array<std::int8_t, Board<Side>::kHexes>& ha) {
  constexpr int kH = Board<Side>::kHexes;
  const auto& pls = card_placements<Side>(f.card);
  const auto& pd = piece_distance()[static_cast<std::size_t>(f.card - 1)];
  std::array<int, kH> best_without{};
  best_without.fill(kInf);
  std::array<std::vector<std::pair<int, int>>, kH> through;  // (placement, piece)
  for (std::size_t j = 0; j < pls.size(); ++j) {
    if (f.miss[j] >= kInf) continue;
    std::array<bool, kH> in{};
    for (int i = 0; i < pls[j].n; ++i) {
      const int h = pls[j].hex[static_cast<std::size_t>(i)];
      in[static_cast<std::size_t>(h)] = true;
      through[static_cast<std::size_t>(h)].emplace_back(static_cast<int>(j), i);
    }
    for (int x = 0; x < kH; ++x) {
      if (!in[static_cast<std::size_t>(x)]) best_without[static_cast<std::size_t>(x)] = std::min(best_without[static_cast<std::size_t>(x)], f.miss[j]);
    }
  }
  std::vector<int> out(static_cast<std::size_t>(kH * kColors), -1);
  for (int x = 0; x < kH; ++x) {
    if (ha[static_cast<std::size_t>(x)]) continue;
    const int t = tt[static_cast<std::size_t>(x)];
    for (int c = 0; c < kColors; ++c) {
      const int nt = kTowerTransition[t][c];
      if (nt < 0) continue;
      int m = best_without[static_cast<std::size_t>(x)];
      for (const auto& [j, i] : through[static_cast<std::size_t>(x)]) {
        const int dn = pd[static_cast<std::size_t>(i)][static_cast<std::size_t>(nt)];
        if (dn >= kInf) continue;
        m = std::min(m, f.miss[static_cast<std::size_t>(j)] - pd[static_cast<std::size_t>(i)][static_cast<std::size_t>(t)] + dn);
      }
      out[static_cast<std::size_t>(x * kColors + c)] = m;
    }
  }
  return out;
}

// Landscape facts of one board.
template <int Side>
struct BoardFacts {
  static constexpr int kH = Board<Side>::kHexes;
  std::array<int, kH> island{};       // island id of a non-blue hex, -1 for blue
  std::array<int, kH> island_size{};  // per island id
  int islands = 0;
  std::array<int, kH> field_group{};  // size of the yellow group, 0 if not yellow
  std::array<int, kH> points{};       // tree / mountain / building points of the hex
  std::array<int, kH> nb_colours{}, nb_empty{}, edge{}, island_split{};
};

template <int Side>
BoardFacts<Side> board_facts(const std::array<std::int8_t, Board<Side>::kHexes>& tt) {
  using Geo = Board<Side>;
  constexpr int kH = Geo::kHexes;
  BoardFacts<Side> b;
  auto label = [&](auto member, std::array<int, kH>& id, std::array<int, kH>& size) {
    id.fill(-1);
    int n = 0;
    for (int h = 0; h < kH; ++h) {
      if (id[static_cast<std::size_t>(h)] >= 0 || !member(tt[static_cast<std::size_t>(h)])) continue;
      std::array<int, kH> stack{};
      int top = 0, count = 0;
      stack[top++] = h;
      id[static_cast<std::size_t>(h)] = n;
      while (top > 0) {
        const int cur = stack[--top];
        ++count;
        for (int d = 0; d < 6; ++d) {
          const int nb = Geo::neighbor(cur, d);
          if (nb < 0 || id[static_cast<std::size_t>(nb)] >= 0 || !member(tt[static_cast<std::size_t>(nb)])) continue;
          id[static_cast<std::size_t>(nb)] = n;
          stack[top++] = nb;
        }
      }
      size[static_cast<std::size_t>(n++)] = count;
    }
    return n;
  };
  b.islands = label([](int t) { return t != 1; }, b.island, b.island_size);
  std::array<int, kH> field_id{}, field_size{};
  label([](int t) { return t == 5; }, field_id, field_size);
  auto is_mountain = [](int t) { return t == 2 || t == 10 || t == 12; };
  for (int h = 0; h < kH; ++h) {
    const int t = tt[static_cast<std::size_t>(h)];
    if (field_id[static_cast<std::size_t>(h)] >= 0) b.field_group[static_cast<std::size_t>(h)] = field_size[static_cast<std::size_t>(field_id[static_cast<std::size_t>(h)])];
    unsigned colours = 0;
    std::array<bool, kH + 1> split_seen{};
    bool mountain_nb = false;
    for (int d = 0; d < 6; ++d) {
      const int nb = Geo::neighbor(h, d);
      if (nb < 0) {
        ++b.edge[static_cast<std::size_t>(h)];
        continue;
      }
      const int nt = tt[static_cast<std::size_t>(nb)];
      if (nt == 0) ++b.nb_empty[static_cast<std::size_t>(h)];
      if (kTowerTopColor[nt] > 0) colours |= 1u << kTowerTopColor[nt];
      if (is_mountain(nt)) mountain_nb = true;
      const int isl = b.island[static_cast<std::size_t>(nb)];
      if (t == 1 && isl >= 0 && !split_seen[static_cast<std::size_t>(isl)]) {
        split_seen[static_cast<std::size_t>(isl)] = true;
        ++b.island_split[static_cast<std::size_t>(h)];
      }
    }
    for (unsigned bits = colours; bits; bits &= bits - 1) ++b.nb_colours[static_cast<std::size_t>(h)];
    int pts = 0;
    if (t == 4) pts = 1;
    else if (t == 9) pts = 3;
    else if (t == 11) pts = 7;
    else if (is_mountain(t) && mountain_nb) pts = t == 2 ? 1 : t == 10 ? 3 : 7;
    else if (t == 7 && b.nb_colours[static_cast<std::size_t>(h)] >= 3) pts = 5;
    b.points[static_cast<std::size_t>(h)] = pts;
  }
  return b;
}

// Landscape points gained by placing colour c (0-based) on hex x, for every
// legal stack on a hex without an animal; 0 elsewhere.
template <int NPlayers, int Side>
std::vector<int> landscape_deltas(const std::array<std::int8_t, Board<Side>::kHexes>& tt,
                                  const std::array<std::int8_t, Board<Side>::kHexes>& ha) {
  using Rules = HarmoniesRules<NPlayers, Side>;
  constexpr int kH = Board<Side>::kHexes;
  const int base = Rules::landscape_breakdown(tt).total();
  std::vector<int> out(static_cast<std::size_t>(kH * kColors), 0);
  auto board = tt;
  for (int x = 0; x < kH; ++x) {
    if (ha[static_cast<std::size_t>(x)]) continue;
    const int t = tt[static_cast<std::size_t>(x)];
    for (int c = 0; c < kColors; ++c) {
      const int nt = kTowerTransition[t][c];
      if (nt < 0) continue;
      board[static_cast<std::size_t>(x)] = static_cast<std::int8_t>(nt);
      out[static_cast<std::size_t>(x * kColors + c)] = Rules::landscape_breakdown(board).total() - base;
    }
    board[static_cast<std::size_t>(x)] = static_cast<std::int8_t>(t);
  }
  return out;
}

float clip_missing(int m) { return m >= kInf ? 1.0f : static_cast<float>(std::min(m, 6)) / 6.0f; }

struct FeatureDesc {
  const char* name;
  int offset;
  int width;
};
#define HARMONIES_DESC_TABLE(table, ns, FEATURES) const FeatureDesc table[] = {FEATURES(HARMONIES_DESC_##ns)};
#define HARMONIES_DESC_hgf(name, width) {#name, hgf::name, width},
#define HARMONIES_DESC_hpf(name, width) {#name, hpf::name, width},
#define HARMONIES_DESC_hsf(name, width) {#name, hsf::name, width},
#define HARMONIES_DESC_hkf(name, width) {#name, hkf::name, width},
#define HARMONIES_DESC_hcf(name, width) {#name, hcf::name, width},
#define HARMONIES_DESC_hof(name, width) {#name, hof::name, width},
#define HARMONIES_DESC_hqf(name, width) {#name, hqf::name, width},
HARMONIES_DESC_TABLE(kGlobalDesc, hgf, HARMONIES_GLOBAL_FEATURES)
HARMONIES_DESC_TABLE(kPlayerDesc, hpf, HARMONIES_PLAYER_FEATURES)
HARMONIES_DESC_TABLE(kSlotDesc, hsf, HARMONIES_SLOT_FEATURES)
HARMONIES_DESC_TABLE(kCardDesc, hkf, HARMONIES_CARD_FEATURES)
HARMONIES_DESC_TABLE(kCellDesc, hcf, HARMONIES_CELL_FEATURES)
HARMONIES_DESC_TABLE(kOppCellDesc, hof, HARMONIES_OPP_CELL_FEATURES)
HARMONIES_DESC_TABLE(kCubeDesc, hqf, HARMONIES_CUBE_FEATURES)
#undef HARMONIES_DESC_TABLE

template <std::size_t N>
void append_type(std::string& out, const char* name, const FeatureDesc (&table)[N]) {
  out += std::string("{\"name\":\"") + name + "\",\"features\":[";
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out += ',';
    out += std::string("[\"") + table[i].name + "\"," + std::to_string(table[i].offset) + "," +
           std::to_string(table[i].width) + "]";
  }
  out += "]}";
}

void append_range(std::string& out, const char* name, int base, int count, bool last = false) {
  out += std::string("[\"") + name + "\"," + std::to_string(base) + "," + std::to_string(count) + "]";
  if (!last) out += ',';
}

}  // namespace

template <int NPlayers, int Side>
HarmoniesTokenEncoder<NPlayers, Side>::HarmoniesTokenEncoder() {
  static_assert(hgf::kCount <= kFeatures && hpf::kCount <= kFeatures && hsf::kCount <= kFeatures &&
                hkf::kCount <= kFeatures && hcf::kCount <= kFeatures && hqf::kCount <= kFeatures);
  layout_.variable_tokens = true;
  layout_.token_positions = true;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = kNumEntities;
  layout_.num_pointer_heads = kNumHeads;
  layout_.action_token.assign(static_cast<std::size_t>(Geo::kActionSpace), -1);
  layout_.action_head.assign(static_cast<std::size_t>(Geo::kActionSpace), -1);
  layout_.num_anchor_ids = kNumAnchors;
  layout_.anchor_action.assign(static_cast<std::size_t>(kNumAnchors * kNumHeads), -1);
  auto bind = [&](int anchor_index, int head, int action) {
    layout_.anchor_action[static_cast<std::size_t>(anchor_index * kNumHeads + head)] = action;
  };
  for (int h = 0; h < kHexes; ++h) {
    for (int c = 0; c < kColors; ++c) bind(kAnchorCell + h, c, kActPlaceTokenBase + h * kColors + c);
    for (int k = 0; k < kAnimalCards; ++k) {
      bind(kAnchorCube + h * kAnimalCards + k, 0, Geo::kActPlaceCubeBase + h * kAnimalCards + k);
    }
  }
  for (int s = 0; s < kSlots; ++s) bind(kAnchorSlot + s, 0, kActSelectSlotBase + s);
  for (int k = 0; k < kAnimalCards; ++k) bind(kAnchorCard + k, 0, kActTakeCardBase + k);
}

template <int NPlayers, int Side>
std::vector<EncoderSymmetry> HarmoniesTokenEncoder<NPlayers, Side>::symmetries() const {
  std::vector<EncoderSymmetry> out(static_cast<std::size_t>(Geo::kNumSymmetries));
  for (int g = 0; g < Geo::kNumSymmetries; ++g) {
    const auto gi = static_cast<std::size_t>(g);
    EncoderSymmetry& sym = out[gi];
    sym.anchor_map.resize(static_cast<std::size_t>(kNumAnchors + 1));
    std::iota(sym.anchor_map.begin(), sym.anchor_map.end(), 0);
    for (int h = 0; h < kHexes; ++h) {
      const int gh = Geo::kSymHex[gi][static_cast<std::size_t>(h)];
      sym.anchor_map[static_cast<std::size_t>(1 + kAnchorCell + h)] = 1 + kAnchorCell + gh;
      for (int k = 0; k < kAnimalCards; ++k) {
        sym.anchor_map[static_cast<std::size_t>(1 + kAnchorCube + h * kAnimalCards + k)] =
            1 + kAnchorCube + gh * kAnimalCards + k;
      }
    }
    for (int i = 0; i < 6; ++i) sym.position_affine[static_cast<std::size_t>(i)] = Geo::kSymAxial[gi][static_cast<std::size_t>(i)];
    sym.column_map.assign(kTokenTypeCount, {});
    auto& opp = sym.column_map[kOppCellToken];
    opp.resize(kTokenDim);
    std::iota(opp.begin(), opp.end(), 0);
    for (int d = 0; d < 6; ++d) {
      const int gd = Geo::kSymDir[gi][static_cast<std::size_t>(d)];
      for (int j = 0; j < 8; ++j) {
        opp[static_cast<std::size_t>(kHeader + hof::neighbour + d * 8 + j)] = kHeader + hof::neighbour + gd * 8 + j;
      }
    }
    sym.action_map = board_symmetry_action_map<Side>(g);
  }
  return out;
}

template <int NPlayers, int Side>
std::string HarmoniesTokenEncoder<NPlayers, Side>::layout_json() {
  std::string out = "{\"token_dim\":" + std::to_string(kTokenDim) + ",\"header\":" + std::to_string(kHeader) +
                    ",\"header_fields\":[\"valid\",\"entity\",\"anchor\",\"type\",\"has_position\",\"q\",\"r\"]," +
                    "\"num_heads\":" + std::to_string(kNumHeads) + ",\"types\":[";
  append_type(out, "global", kGlobalDesc);
  out += ',';
  append_type(out, "player", kPlayerDesc);
  out += ',';
  append_type(out, "slot", kSlotDesc);
  out += ',';
  append_type(out, "card", kCardDesc);
  out += ',';
  append_type(out, "cell", kCellDesc);
  out += ',';
  append_type(out, "opp_cell", kOppCellDesc);
  out += ',';
  append_type(out, "cube", kCubeDesc);
  out += "],\"entities\":[";
  append_range(out, "card", 1, kAnimalCards, true);
  out += "],\"anchors\":[";
  append_range(out, "cell", 1 + kAnchorCell, kHexes);
  append_range(out, "slot", 1 + kAnchorSlot, kSlots);
  append_range(out, "market_card", 1 + kAnchorCard, kAnimalCards);
  append_range(out, "cube", 1 + kAnchorCube, kHexes * kAnimalCards, true);
  out += "]}";
  return out;
}

template <int NPlayers, int Side>
void HarmoniesTokenEncoder<NPlayers, Side>::encode_features(const IGameState& masked_state, int perspective_player,
                                                            const IBeliefTracker* /*tracker*/,
                                                            std::vector<float>* out) const {
  using State = HarmoniesState<NPlayers, Side>;
  using Rules = HarmoniesRules<NPlayers, Side>;
  using Hexes = std::array<std::int8_t, kHexes>;
  const auto& s = checked_cast<State>(masked_state);
  if (perspective_player < 0 || perspective_player >= NPlayers) {
    throw std::invalid_argument("HarmoniesTokenEncoder: perspective out of range");
  }
  if (out == nullptr) throw std::invalid_argument("HarmoniesTokenEncoder: output is null");
  const int me = perspective_player;
  const bool my_move = s.current_player_ == me && !s.terminal;
  const bool placing = my_move && s.turn_started;
  auto token = [&](int type, int entity, int anchor_id) {
    const std::size_t base = out->size();
    out->resize(base + static_cast<std::size_t>(kTokenDim), 0.0f);
    float* t = out->data() + base;
    t[0] = 1.0f;
    t[1] = static_cast<float>(entity);
    t[2] = static_cast<float>(anchor_id);
    t[3] = static_cast<float>(type);
    return base;
  };
  auto at = [&](std::size_t base) { return out->data() + base + kHeader; };
  auto position = [&](std::size_t base, int hex) {
    float* t = out->data() + base;
    t[4] = 1.0f;
    t[5] = static_cast<float>(Geo::kAxialQ[static_cast<std::size_t>(hex)]);
    t[6] = static_cast<float>(Geo::kAxialR[static_cast<std::size_t>(hex)]);
  };

  // Per-board analysis.
  std::array<BoardFacts<Side>, NPlayers> facts;
  std::array<std::vector<int>, NPlayers> deltas;
  std::array<typename Rules::LandscapeBreakdown, NPlayers> breakdown;
  for (int p = 0; p < NPlayers; ++p) {
    facts[static_cast<std::size_t>(p)] = board_facts<Side>(s.tower_type[static_cast<std::size_t>(p)]);
    deltas[static_cast<std::size_t>(p)] = landscape_deltas<NPlayers, Side>(s.tower_type[static_cast<std::size_t>(p)],
                                                                           s.has_animal[static_cast<std::size_t>(p)]);
    breakdown[static_cast<std::size_t>(p)] = Rules::landscape_breakdown(s.tower_type[static_cast<std::size_t>(p)]);
  }
  const Hexes& my_tt = s.tower_type[static_cast<std::size_t>(me)];
  const Hexes& my_ha = s.has_animal[static_cast<std::size_t>(me)];

  // My held cards by rank of card id, with their fits and after-placement minima.
  std::vector<int> my_slots;
  for (int i = 0; i < kHandSize; ++i) {
    if (s.held_cards[static_cast<std::size_t>(me)][static_cast<std::size_t>(i)] > 0) my_slots.push_back(i);
  }
  std::sort(my_slots.begin(), my_slots.end(), [&](int a, int b) {
    return s.held_cards[static_cast<std::size_t>(me)][static_cast<std::size_t>(a)] <
           s.held_cards[static_cast<std::size_t>(me)][static_cast<std::size_t>(b)];
  });
  std::vector<CardFit<Side>> my_fit;
  std::vector<std::vector<int>> my_after;
  for (int slot : my_slots) {
    my_fit.push_back(fit_card<Side>(my_tt, my_ha, s.held_cards[static_cast<std::size_t>(me)][static_cast<std::size_t>(slot)]));
    my_after.push_back(missing_after<Side>(my_fit.back(), my_tt, my_ha));
  }
  // Colour c advances (completes) some held card at some hex.
  std::array<bool, kColors> advances{}, completes{};
  for (std::size_t k = 0; k < my_fit.size(); ++k) {
    for (int x = 0; x < kHexes; ++x) {
      for (int c = 0; c < kColors; ++c) {
        const int m = my_after[k][static_cast<std::size_t>(x * kColors + c)];
        if (m < 0 || my_fit[k].missing >= kInf) continue;
        if (m < my_fit[k].missing) advances[static_cast<std::size_t>(c)] = true;
        if (m == 0 && my_fit[k].missing > 0) completes[static_cast<std::size_t>(c)] = true;
      }
    }
  }

  // ---- global ----
  {
    float* f = at(token(kGlobalToken, 0, 0));
    f[hgf::turn_started] = s.turn_started ? 1.0f : 0.0f;
    f[hgf::took_card] = s.took_card_this_turn ? 1.0f : 0.0f;
    f[hgf::end_triggered] = s.end_triggered ? 1.0f : 0.0f;
    f[hgf::my_turn] = s.current_player_ == me ? 1.0f : 0.0f;
    int pool = 0, pouch = 0, market = 0, min_space = kHexes;
    for (int c = 0; c < kColors; ++c) {
      f[hgf::pool + c] = static_cast<float>(s.selected_tokens[static_cast<std::size_t>(c)]) / 3.0f;
      f[hgf::pouch + c] = static_cast<float>(s.pouch_counts[static_cast<std::size_t>(c)]) / 23.0f;
      pool += s.selected_tokens[static_cast<std::size_t>(c)];
      pouch += s.pouch_counts[static_cast<std::size_t>(c)];
    }
    for (int i = 0; i < kSlots; ++i) market += s.animal_market[static_cast<std::size_t>(i)] > 0;
    for (int p = 0; p < NPlayers; ++p) min_space = std::min<int>(min_space, s.space_remaining[static_cast<std::size_t>(p)]);
    f[hgf::pool_total] = static_cast<float>(pool) / 3.0f;
    f[hgf::pouch_total] = static_cast<float>(pouch) / 120.0f;
    f[hgf::deck] = static_cast<float>(s.animal_deck_count) / 27.0f;
    f[hgf::market_cards] = static_cast<float>(market) / 5.0f;
    f[hgf::refills_left] = static_cast<float>(pouch / kMaxTokensPerSlot) / 36.0f;
    f[hgf::min_spaces] = static_cast<float>(min_space) / static_cast<float>(kHexes);
  }

  // ---- players ----
  for (int k = 0; k < NPlayers; ++k) {
    const int p = (me + k) % NPlayers;
    const auto pi = static_cast<std::size_t>(p);
    float* f = at(token(kPlayerToken, 0, 0));
    f[hpf::seat + k] = 1.0f;
    f[hpf::to_move] = s.current_player_ == p ? 1.0f : 0.0f;
    f[hpf::first_player] = s.game_first_player_ == p ? 1.0f : 0.0f;
    const int total = s.animal_scores[pi] + s.landscape_scores[pi];
    int best_other = -kInf;
    for (int o = 0; o < NPlayers; ++o) {
      if (o != p) best_other = std::max(best_other, s.animal_scores[static_cast<std::size_t>(o)] + s.landscape_scores[static_cast<std::size_t>(o)]);
    }
    f[hpf::animal_score] = static_cast<float>(s.animal_scores[pi]) / 50.0f;
    f[hpf::landscape_score] = static_cast<float>(s.landscape_scores[pi]) / 50.0f;
    f[hpf::total_score] = static_cast<float>(total) / 100.0f;
    f[hpf::margin] = static_cast<float>(total - best_other) / 50.0f;
    const auto& bd = breakdown[pi];
    f[hpf::trees] = static_cast<float>(bd.trees) / 20.0f;
    f[hpf::mountains] = static_cast<float>(bd.mountains) / 20.0f;
    f[hpf::fields] = static_cast<float>(bd.fields) / 20.0f;
    f[hpf::buildings] = static_cast<float>(bd.buildings) / 20.0f;
    f[hpf::water] = static_cast<float>(bd.water) / 20.0f;
    f[hpf::islands] = static_cast<float>(facts[pi].islands) / 8.0f;
    f[hpf::spaces] = static_cast<float>(s.space_remaining[pi]) / static_cast<float>(kHexes);
    int cubes = 0, held = 0, open = 0;
    for (int h = 0; h < kHexes; ++h) cubes += s.has_animal[pi][static_cast<std::size_t>(h)];
    for (int i = 0; i < kHandSize; ++i) {
      const int cid = s.held_cards[pi][static_cast<std::size_t>(i)];
      if (cid <= 0) continue;
      ++held;
      const auto& card = kCards[cid - 1];
      const int placed = s.cube_initial[pi][static_cast<std::size_t>(i)] - s.cube_remaining[pi][static_cast<std::size_t>(i)];
      open += card.scores[card.num_cubes - 1] - (placed > 0 ? card.scores[placed - 1] : 0);
    }
    f[hpf::cubes_placed] = static_cast<float>(cubes) / static_cast<float>(kHexes);
    f[hpf::cards_held] = static_cast<float>(held) / 4.0f;
    f[hpf::card_points_open] = static_cast<float>(open) / 40.0f;
  }

  // ---- central-board slots ----
  for (int sl = 0; sl < kSlots; ++sl) {
    float* f = at(token(kSlotToken, 0, 1 + kAnchorSlot + sl));
    int total = 0, adv = 0, comp = 0;
    float best = 0.0f;
    bool any_best = false;
    for (int c = 0; c < kColors; ++c) {
      const int n = s.central_board_counts[static_cast<std::size_t>(sl)][static_cast<std::size_t>(c)];
      f[hsf::tokens + c] = static_cast<float>(n) / 3.0f;
      total += n;
      if (n == 0) continue;
      if (advances[static_cast<std::size_t>(c)]) adv += n;
      if (completes[static_cast<std::size_t>(c)]) comp += n;
      for (int x = 0; x < kHexes; ++x) {
        if (my_ha[static_cast<std::size_t>(x)] || kTowerTransition[my_tt[static_cast<std::size_t>(x)]][c] < 0) continue;
        const float d = static_cast<float>(deltas[static_cast<std::size_t>(me)][static_cast<std::size_t>(x * kColors + c)]);
        if (!any_best || d > best) best = d;
        any_best = true;
      }
    }
    f[hsf::total] = static_cast<float>(total) / 3.0f;
    f[hsf::best_landscape] = best / 10.0f;
    f[hsf::advancing] = static_cast<float>(adv) / 3.0f;
    f[hsf::completing] = static_cast<float>(comp) / 3.0f;
  }

  // ---- cards ----
  auto write_card = [&](float* f, int cid) {
    const auto& card = kCards[cid - 1];
    f[hkf::pattern_size] = static_cast<float>(card.pattern_size) / 4.0f;
    for (int i = 0; i < card.pattern_size; ++i) {
      const int tower = target_tower(card.pieces[i]);
      for (int lv = 0; lv < 3 && kTowerStack[tower][lv] > 0; ++lv) f[hkf::need + kTowerStack[tower][lv] - 1] += 0.25f;
      if (i == card.anchor_index) f[hkf::habitat + kTowerTopColor[tower] - 1] = 1.0f;
    }
    f[hkf::cubes_total] = static_cast<float>(card.num_cubes) / 5.0f;
    f[hkf::full_points] = static_cast<float>(card.scores[card.num_cubes - 1]) / 20.0f;
  };
  auto write_progress = [&](float* f, int cid, int placed, int remaining) {
    const auto& card = kCards[cid - 1];
    f[hkf::cubes_left] = static_cast<float>(remaining) / 5.0f;
    f[hkf::points_now] = static_cast<float>(placed > 0 ? card.scores[placed - 1] : 0) / 20.0f;
    if (placed < card.num_cubes) {
      f[hkf::next_points] = static_cast<float>(card.scores[placed] - (placed > 0 ? card.scores[placed - 1] : 0)) / 10.0f;
    }
  };
  auto write_fit = [&](float* f, const CardFit<Side>& fit) {
    f[hkf::missing] = clip_missing(fit.missing);
    f[hkf::no_fit] = fit.missing >= kInf ? 1.0f : 0.0f;
    f[hkf::ready] = static_cast<float>(fit.ready) / static_cast<float>(kHexes);
  };
  for (int sl = 0; sl < kSlots; ++sl) {
    const int cid = s.animal_market[static_cast<std::size_t>(sl)];
    if (cid <= 0) continue;
    float* f = at(token(kCardToken, cid, 1 + kAnchorCard + cid - 1));
    f[hkf::market] = 1.0f;
    write_card(f, cid);
    write_progress(f, cid, 0, kCards[cid - 1].num_cubes);
    write_fit(f, fit_card<Side>(my_tt, my_ha, cid));
    int opp_best = kInf;
    for (int o = 0; o < NPlayers; ++o) {
      if (o == me) continue;
      opp_best = std::min(opp_best, fit_card<Side>(s.tower_type[static_cast<std::size_t>(o)],
                                                   s.has_animal[static_cast<std::size_t>(o)], cid).missing);
    }
    f[hkf::opp_missing] = clip_missing(opp_best);
  }
  for (int k = 0; k < NPlayers; ++k) {
    const int p = (me + k) % NPlayers;
    const auto pi = static_cast<std::size_t>(p);
    for (int i = 0; i < kHandSize; ++i) {
      const int cid = s.held_cards[pi][static_cast<std::size_t>(i)];
      if (cid <= 0) continue;
      float* f = at(token(kCardToken, cid, 0));
      f[k == 0 ? hkf::my_hand : hkf::opp_hand] = 1.0f;
      f[hkf::holder_seat + k] = 1.0f;
      write_card(f, cid);
      const int remaining = s.cube_remaining[pi][static_cast<std::size_t>(i)];
      write_progress(f, cid, s.cube_initial[pi][static_cast<std::size_t>(i)] - remaining, remaining);
      if (k == 0) {
        const auto rank = static_cast<std::size_t>(std::find(my_slots.begin(), my_slots.end(), i) - my_slots.begin());
        f[hkf::rank + static_cast<int>(rank)] = 1.0f;
        write_fit(f, my_fit[rank]);
      } else {
        write_fit(f, fit_card<Side>(s.tower_type[pi], s.has_animal[pi], cid));
      }
    }
  }

  // ---- my cells ----
  const auto& mf = facts[static_cast<std::size_t>(me)];
  auto write_landscape = [&](float* f, const BoardFacts<Side>& b, const Hexes& tt, const Hexes& ha, int h,
                             int off_tower) {
    const auto hi = static_cast<std::size_t>(h);
    const int t = tt[hi];
    f[off_tower + t] = 1.0f;
    f[off_tower + 13] = static_cast<float>(State::tower_height(t)) / 3.0f;            // height
    f[off_tower + 14] = ha[hi] ? 1.0f : 0.0f;                                         // animal
    f[off_tower + 15] = static_cast<float>(b.points[hi]) / 7.0f;                      // points
    f[off_tower + 16] = static_cast<float>(std::min(b.field_group[hi], 5)) / 5.0f;    // field_group
    f[off_tower + 17] = b.island[hi] >= 0
        ? static_cast<float>(b.island_size[static_cast<std::size_t>(b.island[hi])]) / static_cast<float>(kHexes)
        : 0.0f;                                                                       // island
    f[off_tower + 18] = static_cast<float>(b.island_split[hi]) / 3.0f;                // island_split
    f[off_tower + 19] = static_cast<float>(b.nb_colours[hi]) / 6.0f;                  // nb_colours
    f[off_tower + 20] = static_cast<float>(b.nb_empty[hi]) / 6.0f;                    // nb_empty
    f[off_tower + 21] = static_cast<float>(b.edge[hi]) / 6.0f;                        // edge
  };
  static_assert(hcf::height == hcf::tower + 13 && hcf::edge == hcf::tower + 21);
  static_assert(hof::height == hof::tower + 13 && hof::edge == hof::tower + 21);
  for (int h = 0; h < kHexes; ++h) {
    const auto hi = static_cast<std::size_t>(h);
    const std::size_t base = token(kCellToken, 0, 1 + kAnchorCell + h);
    position(base, h);
    float* f = at(base);
    write_landscape(f, mf, my_tt, my_ha, h, hcf::tower);
    for (int c = 0; c < kColors; ++c) {
      if (my_ha[hi] || kTowerTransition[my_tt[hi]][c] < 0) continue;
      f[hcf::can_place + c] = 1.0f;
      f[hcf::legal + c] = placing && s.selected_tokens[static_cast<std::size_t>(c)] > 0 ? 1.0f : 0.0f;
      f[hcf::delta + c] = static_cast<float>(deltas[static_cast<std::size_t>(me)][hi * kColors + static_cast<std::size_t>(c)]) / 10.0f;
    }
    for (std::size_t k = 0; k < my_fit.size(); ++k) {
      const int ki = static_cast<int>(k);
      f[hcf::card_part + ki] = my_fit[k].part[hi];
      f[hcf::card_missing + ki] = my_fit[k].missing_here[hi];
      f[hcf::card_ready + ki] = my_fit[k].ready_at[hi];
      if (my_fit[k].missing >= kInf) continue;
      for (int c = 0; c < kColors; ++c) {
        int m = my_after[k][hi * kColors + static_cast<std::size_t>(c)];
        if (m < 0) continue;
        if (m >= kInf) m = my_fit[k].missing + 3;
        f[hcf::card_advance + ki * kColors + c] =
            static_cast<float>(std::clamp(my_fit[k].missing - m, -3, 3)) / 3.0f;
      }
    }
  }

  // ---- opponent cells ----
  for (int k = 1; k < NPlayers; ++k) {
    const int p = (me + k) % NPlayers;
    const auto pi = static_cast<std::size_t>(p);
    const Hexes& tt = s.tower_type[pi];
    const Hexes& ha = s.has_animal[pi];
    for (int h = 0; h < kHexes; ++h) {
      const auto hi = static_cast<std::size_t>(h);
      float* f = at(token(kOppCellToken, 0, 0));
      f[hof::owner_seat + k - 1] = 1.0f;
      write_landscape(f, facts[pi], tt, ha, h, hof::tower);
      for (int c = 0; c < kColors; ++c) {
        f[hof::delta + c] = static_cast<float>(deltas[pi][hi * kColors + static_cast<std::size_t>(c)]) / 10.0f;
      }
      for (int d = 0; d < 6; ++d) {
        const int nb = Geo::neighbor(h, d);
        // 0 = off the board, 1 = empty, 2..7 = top colour blue..red.
        const int cls = nb < 0 ? 0 : 1 + kTowerTopColor[tt[static_cast<std::size_t>(nb)]];
        f[hof::neighbour + d * 8 + cls] = 1.0f;
      }
    }
  }

  // ---- legal cube placements ----
  if (placing) {
    for (std::size_t k = 0; k < my_slots.size(); ++k) {
      const auto slot = static_cast<std::size_t>(my_slots[k]);
      const int cid = s.held_cards[static_cast<std::size_t>(me)][slot];
      const int remaining = s.cube_remaining[static_cast<std::size_t>(me)][slot];
      if (remaining <= 0) continue;
      const auto& card = kCards[cid - 1];
      const int placed = s.cube_initial[static_cast<std::size_t>(me)][slot] - remaining;
      for (int h = 0; h < kHexes; ++h) {
        if (!my_fit[k].ready_at[static_cast<std::size_t>(h)]) continue;
        const std::size_t base = token(kCubeToken, cid, 1 + kAnchorCube + h * kAnimalCards + cid - 1);
        position(base, h);
        float* f = at(base);
        f[hqf::rank + static_cast<int>(k)] = 1.0f;
        f[hqf::points] = static_cast<float>(card.scores[placed] - (placed > 0 ? card.scores[placed - 1] : 0)) / 10.0f;
        f[hqf::completes] = remaining == 1 ? 1.0f : 0.0f;
        f[hqf::cubes_left] = static_cast<float>(remaining - 1) / 5.0f;
        Hexes ha_after = my_ha;
        ha_after[static_cast<std::size_t>(h)] = 1;
        int blocked = 0;
        for (std::size_t j = 0; j < my_fit.size(); ++j) {
          if (j == k) continue;
          if (fit_card<Side>(my_tt, ha_after, my_fit[j].card).missing > my_fit[j].missing) ++blocked;
        }
        f[hqf::blocks] = static_cast<float>(blocked) / 4.0f;
      }
    }
  }
}

template class HarmoniesTokenEncoder<2, kSideB>;
template class HarmoniesTokenEncoder<3, kSideB>;
template class HarmoniesTokenEncoder<4, kSideB>;

}  // namespace board_ai::harmonies
