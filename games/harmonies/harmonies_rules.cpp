#include "harmonies_rules.h"

#include <algorithm>
#include <array>
#include <queue>
#include <stdexcept>

#include "harmonies_card_data.h"

namespace board_ai::harmonies {

namespace {

// Apply a (a,b,c,d) 2x2 transform to (dq, dr): returns (a*dq+b*dr, c*dq+d*dr).
inline std::pair<int, int> apply_transform(int t_idx, int dq, int dr) {
  const int a = kTransforms[t_idx][0];
  const int b = kTransforms[t_idx][1];
  const int c = kTransforms[t_idx][2];
  const int d = kTransforms[t_idx][3];
  return {a * dq + b * dr, c * dq + d * dr};
}

}  // namespace

template <int NPlayers, int Side>
int HarmoniesRules<NPlayers, Side>::draw_one_token(State& s, std::mt19937_64& rng) {
  int total = 0;
  for (int c = 0; c < kColors; ++c) total += s.pouch_counts[c];
  if (total == 0) return -1;
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int c = 0; c < kColors; ++c) {
    if (pick < s.pouch_counts[c]) {
      --s.pouch_counts[c];
      return c;  // 0-indexed color (Blue=0..Red=5)
    }
    pick -= s.pouch_counts[c];
  }
  return -1;
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::refill_central_board(State& s, std::mt19937_64& rng) {
  for (int slot = 0; slot < kSlots; ++slot) {
    int total_in_slot = 0;
    for (int c = 0; c < kColors; ++c) total_in_slot += s.central_board_counts[slot][c];
    if (total_in_slot > 0) continue;
    // Draw 3 tokens (or as many as the pouch allows).
    int total_pouch = 0;
    for (int c = 0; c < kColors; ++c) total_pouch += s.pouch_counts[c];
    if (total_pouch < kMaxTokensPerSlot) {
      // Pouch can't refill a full slot — game-end trigger.
      s.end_triggered = true;
      // Still draw what we can (the rules: pouch insufficient ends the game,
      // but state should be consistent).
      // Decision: draw what's available, the game will terminate at the
      // end of the next first-player turn boundary.
    }
    for (int t = 0; t < kMaxTokensPerSlot; ++t) {
      int c = draw_one_token(s, rng);
      if (c < 0) break;
      ++s.central_board_counts[slot][c];
    }
  }
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::apply_select_slot(State& s, int slot) {
  for (int c = 0; c < kColors; ++c) {
    s.selected_tokens[c] = static_cast<std::int8_t>(
        s.selected_tokens[c] + s.central_board_counts[slot][c]);
    s.central_board_counts[slot][c] = 0;
  }
  s.turn_started = true;
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::apply_place_token(State& s, int hex, int color_1based) {
  const int p = s.current_player_;
  const int tt = static_cast<int>(s.tower_type[p][hex]);
  const int col_idx = color_1based - 1;
  const int new_tt = kTowerTransition[tt][col_idx];
  if (new_tt < 0) {
    throw std::runtime_error("harmonies: illegal PLACE_TOKEN transition");
  }
  s.tower_type[p][hex] = static_cast<std::int8_t>(new_tt);
  --s.selected_tokens[col_idx];
  if (tt == 0) {
    --s.space_remaining[p];
    if (s.space_remaining[p] <= 2) s.end_triggered = true;
  }
  recompute_landscape_scores(s);
}

template <int NPlayers, int Side>
bool HarmoniesRules<NPlayers, Side>::card_matches_at(
    const State& s, int player, int card_id, int anchor_hex) {
  if (s.has_animal[player][anchor_hex]) return false;
  const auto& card = kCards[card_id - 1];
  const int ax_q = Geo::kAxialQ[anchor_hex];
  const int ax_r = Geo::kAxialR[anchor_hex];
  const auto& anchor_piece = card.pieces[card.anchor_index];
  for (int t = 0; t < 6; ++t) {
    const auto anchor_rot = apply_transform(t, anchor_piece.dq, anchor_piece.dr);
    bool ok = true;
    for (int i = 0; i < card.pattern_size; ++i) {
      const auto& piece = card.pieces[i];
      const auto rel = apply_transform(t, piece.dq, piece.dr);
      const int tq = ax_q + rel.first - anchor_rot.first;
      const int tr = ax_r + rel.second - anchor_rot.second;
      const int th = Geo::axial_to_hex(tq, tr);
      if (th < 0) { ok = false; break; }
      const int tt = static_cast<int>(s.tower_type[player][th]);
      const std::uint16_t bit = static_cast<std::uint16_t>(1u << tt);
      if ((piece.tower_mask & bit) == 0) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::apply_take_card(
    State& s, int card_id, std::mt19937_64& rng) {
  const int p = s.current_player_;
  // Find market slot containing this card.
  int slot = -1;
  for (int i = 0; i < kSlots; ++i) {
    if (s.animal_market[i] == card_id) { slot = i; break; }
  }
  if (slot < 0) throw std::runtime_error("harmonies: TAKE_CARD card not in market");
  // Find free hand slot.
  int hand_slot = -1;
  for (int i = 0; i < kHandSize; ++i) {
    if (s.held_cards[p][i] == 0) { hand_slot = i; break; }
  }
  if (hand_slot < 0) throw std::runtime_error("harmonies: TAKE_CARD hand full");
  const auto& card = kCards[card_id - 1];
  s.held_cards[p][hand_slot] = static_cast<std::int8_t>(card_id);
  s.cube_initial[p][hand_slot] = static_cast<std::int8_t>(card.num_cubes);
  s.cube_remaining[p][hand_slot] = static_cast<std::int8_t>(card.num_cubes);
  // Refill market: randomly draw from animal_deck if any remains.
  int next_card = 0;
  if (s.animal_deck_count > 0) {
    std::vector<int> candidates;
    candidates.reserve(s.animal_deck_count);
    for (int c = 0; c < kAnimalCards; ++c) {
      if (s.animal_deck[c]) candidates.push_back(c);
    }
    if (!candidates.empty()) {
      const int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(candidates.size()));
      next_card = candidates[pick] + 1;
      s.animal_deck[candidates[pick]] = 0;
    }
    --s.animal_deck_count;
  }
  s.animal_market[slot] = static_cast<std::int8_t>(next_card);  // 0 if deck empty
  s.took_card_this_turn = true;
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::apply_place_cube(State& s, int hex, int card_id) {
  const int p = s.current_player_;
  // Find hand slot for this card.
  int slot = -1;
  for (int i = 0; i < kHandSize; ++i) {
    if (s.held_cards[p][i] == card_id) { slot = i; break; }
  }
  if (slot < 0) throw std::runtime_error("harmonies: PLACE_CUBE card not held");
  if (!card_matches_at(s, p, card_id, hex)) {
    throw std::runtime_error("harmonies: PLACE_CUBE no valid orientation");
  }
  s.has_animal[p][hex] = 1;
  --s.cube_remaining[p][slot];
  const auto& card = kCards[card_id - 1];
  const int cubes_placed = s.cube_initial[p][slot] - s.cube_remaining[p][slot];
  const int idx = cubes_placed - 1;
  int incremental = card.scores[idx];
  if (idx > 0) incremental -= card.scores[idx - 1];
  s.animal_scores[p] += incremental;
  if (s.cube_remaining[p][slot] == 0) {
    s.held_cards[p][slot] = 0;
    s.cube_initial[p][slot] = 0;
    s.completed_cards[card_id - 1] = 1;  // public: the card is out of play
  }
  if (s.space_remaining[p] <= 2) s.end_triggered = true;
}

// ---- Landscape scoring (rules p.8-9) ----
// - Trees (Green on 0/1/2 Brown): 1 / 3 / 7.
// - Mountains (1/2/3 Gray): 1 / 3 / 7, 0 unless adjacent to another mountain.
// - Fields: 5 per connected group of 2+ Yellow.
// - Buildings (Red on Brown/Gray/Red): 5 if the top tokens around show 3+
//   distinct colors.
// - Water, side A: best river = longest shortest path between two blue hexes
//   of one blue group, length k (ends included) -> 0/2/5/8/11/15, +4 per
//   token beyond 6.
// - Water, side B: 5 per island, an island being a connected group of
//   non-blue spaces (empty or not) — "each space or group of spaces separated
//   from each other by blue tokens"; there is always at least 1 island.
template <int NPlayers, int Side>
typename HarmoniesRules<NPlayers, Side>::LandscapeBreakdown
HarmoniesRules<NPlayers, Side>::landscape_breakdown(
    const std::array<std::int8_t, Geo::kHexes>& tt) {
  constexpr int kH = Geo::kHexes;
  LandscapeBreakdown out;

  // ---- Trees: bush(1pt) / tree(3pt) / tall tree(7pt) ----
  for (int h = 0; h < kH; ++h) {
    const int t = tt[h];
    if (t == 4)       out.trees += 1;
    else if (t == 9)  out.trees += 3;
    else if (t == 11) out.trees += 7;
  }

  // ---- Mountains: must be adjacent to another mountain to score ----
  auto is_mountain = [](int t) { return t == 2 || t == 10 || t == 12; };
  for (int h = 0; h < kH; ++h) {
    const int t = tt[h];
    if (!is_mountain(t)) continue;
    bool adj = false;
    for (int d = 0; d < 6; ++d) {
      const int nb = Geo::neighbor(h, d);
      if (nb >= 0 && is_mountain(tt[nb])) { adj = true; break; }
    }
    if (!adj) continue;
    out.mountains += t == 2 ? 1 : t == 10 ? 3 : 7;
  }

  // ---- Buildings: 2-stack red scores 5 if adjacent to 3+ distinct colors ----
  for (int h = 0; h < kH; ++h) {
    if (tt[h] != 7) continue;
    std::uint8_t cset = 0;
    for (int d = 0; d < 6; ++d) {
      const int nb = Geo::neighbor(h, d);
      if (nb < 0) continue;
      const int tc = kTowerTopColor[tt[nb]];
      if (tc > 0) cset |= static_cast<std::uint8_t>(1u << tc);
    }
    int distinct = 0;
    for (auto bits = cset; bits; bits &= (bits - 1)) ++distinct;
    if (distinct >= 3) out.buildings += 5;
  }

  // Connected groups of hexes satisfying `member`; calls visit(size) per group.
  auto groups = [&](auto member, auto visit) {
    std::array<bool, kH> seen{};
    for (int h = 0; h < kH; ++h) {
      if (seen[h] || !member(tt[h])) continue;
      int sz = 0;
      std::array<int, kH> stack{};
      int top = 0;
      stack[top++] = h;
      seen[h] = true;
      while (top > 0) {
        const int cur = stack[--top];
        ++sz;
        for (int d = 0; d < 6; ++d) {
          const int nb = Geo::neighbor(cur, d);
          if (nb < 0 || seen[nb] || !member(tt[nb])) continue;
          seen[nb] = true;
          stack[top++] = nb;
        }
      }
      visit(sz);
    }
  };

  // ---- Fields: each connected group of 2+ Yellow scores 5 ----
  groups([](int t) { return t == 5; }, [&](int sz) { if (sz >= 2) out.fields += 5; });

  if constexpr (Geo::kIslands) {
    int islands = 0;
    groups([](int t) { return t != 1; }, [&](int) { ++islands; });
    out.water = 5 * std::max(1, islands);
  } else {
    // River = longest shortest-path between any two blue hexes.
    int best_river = 0;
    for (int src = 0; src < kH; ++src) {
      if (tt[src] != 1) continue;
      std::array<int, kH> dist;
      dist.fill(-1);
      dist[src] = 0;
      std::queue<int> q;
      q.push(src);
      while (!q.empty()) {
        const int cur = q.front(); q.pop();
        if (dist[cur] > best_river) best_river = dist[cur];
        for (int d = 0; d < 6; ++d) {
          const int nb = Geo::neighbor(cur, d);
          if (nb < 0 || dist[nb] >= 0 || tt[nb] != 1) continue;
          dist[nb] = dist[cur] + 1;
          q.push(nb);
        }
      }
    }
    const int river_len = best_river + 1;
    static constexpr int kWaterScore[] = {0, 0, 2, 5, 8, 11, 15};
    out.water = river_len < 7 ? kWaterScore[river_len] : 15 + 4 * (river_len - 6);
  }
  return out;
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(State& s) {
  for (int p = 0; p < NPlayers; ++p) {
    s.landscape_scores[p] = landscape_breakdown(s.tower_type[p]).total();
  }
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::apply_end_turn(State& s, std::mt19937_64& rng) {
  // Recompute landscape scores before refill (cheap, idempotent).
  recompute_landscape_scores(s);
  // Refill any cleared slot.
  refill_central_board(s, rng);

  // Reset per-turn state.
  s.turn_started = false;
  s.took_card_this_turn = false;
  s.selected_tokens.fill(0);

  // Advance current player.
  const int next = (s.current_player_ + 1) % NPlayers;
  s.current_player_ = next;

  if (s.end_triggered && s.current_player_ == s.game_first_player_) {
    s.terminal = true;
    // Rulebook: most points wins; tie -> most animal cubes placed; still
    // tied -> shared victory (winner_ = -1, see HarmoniesState::winners()).
    const std::vector<int> first = s.first_place();
    s.winner_ = first.size() == 1 ? first[0] : -1;
  }
}

template <int NPlayers, int Side>
std::vector<ActionId> HarmoniesRules<NPlayers, Side>::legal_actions(const IGameState& state) const {
  const auto& s = checked_cast<State>(state);
  std::vector<ActionId> out;
  if (s.terminal) return out;
  out.reserve(64);

  if (!s.turn_started) {
    // SELECT_SLOT: any non-empty slot.
    for (int slot = 0; slot < kSlots; ++slot) {
      int total = 0;
      for (int c = 0; c < kColors; ++c) total += s.central_board_counts[slot][c];
      if (total > 0) {
        out.push_back(static_cast<ActionId>(kActSelectSlotBase + slot));
      }
    }
    return out;
  }

  // turn_started == true.
  int pool_total = 0;
  for (int c = 0; c < kColors; ++c) pool_total += s.selected_tokens[c];

  if (pool_total > 0) {
    const int p = s.current_player_;
    for (int hex = 0; hex < Geo::kHexes; ++hex) {
      if (s.has_animal[p][hex]) continue;
      const int tt = static_cast<int>(s.tower_type[p][hex]);
      for (int color_idx = 0; color_idx < kColors; ++color_idx) {
        if (s.selected_tokens[color_idx] <= 0) continue;
        if (kTowerTransition[tt][color_idx] < 0) continue;
        out.push_back(static_cast<ActionId>(
            kActPlaceTokenBase + hex * kColors + color_idx));
      }
    }
  } else {
    // pool empty AND turn_started => END_TURN allowed.
    out.push_back(static_cast<ActionId>(Geo::kActEndTurn));
  }

  // TAKE_CARD: once per turn, only after turn_started, hand not full,
  // and only for cards currently visible in the market.
  const int p = s.current_player_;
  if (!s.took_card_this_turn) {
    int free_hand = 0;
    for (int i = 0; i < kHandSize; ++i) if (s.held_cards[p][i] == 0) ++free_hand;
    if (free_hand > 0) {
      for (int slot = 0; slot < kSlots; ++slot) {
        const int cid = s.animal_market[slot];
        if (cid > 0) {
          out.push_back(static_cast<ActionId>(kActTakeCardBase + (cid - 1)));
        }
      }
    }
  }

  // PLACE_CUBE: for each held card with cube_remaining > 0, scan anchor hexes
  // for any matching orientation.
  for (int i = 0; i < kHandSize; ++i) {
    const int cid = s.held_cards[p][i];
    if (cid <= 0) continue;
    if (s.cube_remaining[p][i] <= 0) continue;
    for (int hex = 0; hex < Geo::kHexes; ++hex) {
      if (s.has_animal[p][hex]) continue;
      if (card_matches_at(s, p, cid, hex)) {
        out.push_back(static_cast<ActionId>(
            Geo::kActPlaceCubeBase + hex * kAnimalCards + (cid - 1)));
      }
    }
  }

  return out;
}

template <int NPlayers, int Side>
bool HarmoniesRules<NPlayers, Side>::validate_action(const IGameState& state, ActionId action) const {
  if (action < 0 || action >= Geo::kActionSpace) return false;
  const auto legal = legal_actions(state);
  return std::find(legal.begin(), legal.end(), action) != legal.end();
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::do_action_fast_impl(
    IGameState& state, ActionId action, std::mt19937_64& rng) const {
  auto& s = checked_cast<State>(state);
  if (s.terminal) return;
  const int a = static_cast<int>(action);

  if (a >= kActSelectSlotBase && a < kActTakeCardBase) {
    apply_select_slot(s, a - kActSelectSlotBase);
    return;
  }
  if (a >= kActPlaceTokenBase && a < Geo::kActPlaceCubeBase) {
    const int rel = a - kActPlaceTokenBase;
    apply_place_token(s, rel / kColors, rel % kColors + 1);
    return;
  }
  if (a == Geo::kActEndTurn) {
    apply_end_turn(s, rng);
    return;
  }
  if (a >= kActTakeCardBase && a < kActPlaceTokenBase) {
    apply_take_card(s, a - kActTakeCardBase + 1, rng);
    return;
  }
  if (a >= Geo::kActPlaceCubeBase && a < Geo::kActEndTurn) {
    const int rel = a - Geo::kActPlaceCubeBase;
    apply_place_cube(s, rel / kAnimalCards, rel % kAnimalCards + 1);
    return;
  }
  throw std::runtime_error("harmonies: action out of range");
}

template <int NPlayers, int Side>
UndoToken HarmoniesRules<NPlayers, Side>::do_action_deterministic_impl(
    IGameState& /*state*/, ActionId /*action*/) const {
  throw std::runtime_error("harmonies: deterministic/undo path not supported in M1");
}

template <int NPlayers, int Side>
void HarmoniesRules<NPlayers, Side>::undo_action_impl(
    IGameState& /*state*/, const UndoToken& /*token*/) const {
  throw std::runtime_error("harmonies: undo not supported in M1");
}

#define HARMONIES_INST_HarmoniesRules(N, S) template class HarmoniesRules<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesRules)
#undef HARMONIES_INST_HarmoniesRules

}  // namespace board_ai::harmonies
