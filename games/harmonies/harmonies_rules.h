#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "harmonies_state.h"

namespace board_ai::harmonies {

template <int NPlayers, int Side>
class HarmoniesRules final : public IGameRules {
 public:
  using State = HarmoniesState<NPlayers, Side>;
  using Geo = Board<Side>;

  HarmoniesRules() = default;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  // No deterministic / undo path for M1: Harmonies has hidden chance
  // (pouch draws). Tail solver isn't registered.
  UndoToken do_action_deterministic_impl(IGameState& state, ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;

 private:
  static int draw_one_token(State& s, std::mt19937_64& rng);
  static void refill_central_board(State& s, std::mt19937_64& rng);
  static void apply_select_slot(State& s, int slot);
  static void apply_place_token(State& s, int hex, int color);
  static void apply_take_card(State& s, int card_id, std::mt19937_64& rng);
  static void apply_place_cube(State& s, int hex, int card_id);
  static void apply_end_turn(State& s, std::mt19937_64& rng);

  // M2: pattern matching.
  static bool card_matches_at(const State& s, int player, int card_id, int anchor_hex);

 public:
  // Landscape points of one board (tower types per hex), by category:
  // trees, mountains, fields, buildings, water (side A: best river, side B:
  // islands). Public so encoders and heuristics can score hypothetical boards.
  struct LandscapeBreakdown {
    int trees = 0, mountains = 0, fields = 0, buildings = 0, water = 0;
    int total() const { return trees + mountains + fields + buildings + water; }
  };
  static LandscapeBreakdown landscape_breakdown(
      const std::array<std::int8_t, Geo::kHexes>& tower_type);
  // M3: landscape scoring; rewrites s.landscape_scores[]. Public so heuristics
  // can use it to evaluate hypothetical positions mid-turn.
  static void recompute_landscape_scores(State& s);
  // Public version of card_matches_at for heuristics.
  static bool card_matches_at_pub(const State& s, int player, int card_id, int anchor_hex) {
    return card_matches_at(s, player, card_id, anchor_hex);
  }
};

#define HARMONIES_EXTERN_HarmoniesRules(N, S) extern template class HarmoniesRules<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_HarmoniesRules)
#undef HARMONIES_EXTERN_HarmoniesRules

}  // namespace board_ai::harmonies
