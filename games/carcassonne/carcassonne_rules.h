#pragma once

#include <array>
#include <random>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "carcassonne_state.h"

namespace board_ai::carcassonne {

struct FeatureScoreInfo {
  bool complete = false;
  int open_edges = 0;
  int tiles = 0;
  int shields = 0;
  std::array<int, kPlayers> owners{0, 0};
};

struct FarmScoreInfo {
  std::array<int, kPlayers> owners{0, 0};
  int completed_cities = 0;
  int incomplete_cities = 0;
};

// What placing the current tile somewhere does before any meeple is added
// (CarcassonneRules::preview_placement).
struct PlacementPreview {
  // Cities and roads of the placed tile that end up complete, plus the
  // cloisters whose 3x3 block it fills (those holding a monk, and its own).
  int completed = 0;
  // Points each player scores at the end of the turn if no meeple is added.
  std::array<int, kPlayers> score_delta{0, 0};
  // Meeples on the cities / roads the placed tile belongs to, per player.
  std::array<int, kPlayers> meeples{0, 0};
  // Best city / road of the placed tile the mover may put a meeple on:
  // 0 = none, 1 = an open one, 2 = one this placement completes.
  int city_claim = 0;
  int road_claim = 0;
};

class CarcassonneRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

  static ActionId encode_place_action(int row, int col, int rotation);
  static ActionId encode_meeple_pass_action();
  static ActionId encode_meeple_side_action(int side);
  static ActionId encode_meeple_farm_action(int farmer_side);
  static ActionId encode_meeple_center_action();
  static int decode_row(ActionId action);
  static int decode_col(ActionId action);
  static int decode_rotation(ActionId action);
  static int decode_meeple_position(ActionId action);
  static bool can_place_tile_at(const CarcassonneState& s, int tile_type,
                                int row, int col, int rotation);
  static bool can_place_meeple(const CarcassonneState& s, int pos);
  static FeatureScoreInfo collect_city(const CarcassonneState& s, int start_cell,
                                       int start_side);
  static FeatureScoreInfo collect_road(const CarcassonneState& s, int start_cell,
                                      int start_side);
  static FarmScoreInfo collect_farm(const CarcassonneState& s, int start_cell,
                                    int farmer_side);
  static int cloister_occupancy(const CarcassonneState& s, int cell);
  // Lookup the group index containing `side` on the given (tile, rotation)
  // for the given terrain (kCity or kRoad). Used by heuristics to dedup
  // before-state regions when evaluating a tile placement marginally.
  // Returns -1 if no matching group exists (e.g. side terrain is field).
  static int feature_group_for_side(int tile_type, int rotation, Terrain terrain,
                                    int side);

  // Read-only endgame projection: clone the state internally, apply final
  // scoring rules (incomplete cities/roads/cloisters/farms), and return
  // each player's projected final score. Used by the PV value head to
  // turn "current snapshot" into "what would the score be if the game
  // ended right now". In the meeple stage the features the tile just
  // placed completed are scored first, as they will be whichever meeple
  // action follows. The supplied state is not mutated.
  static std::array<int, kPlayers> compute_final_scores(
      const CarcassonneState& s);

  // Outcome of the legal tile action (row, col, rotation) on a scratch copy
  // of `s`, through the same placement / collection / scoring code the real
  // move runs. Throws if the action is not legal.
  static PlacementPreview preview_placement(const CarcassonneState& s, int row,
                                            int col, int rotation);

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state,
                                         ActionId action) const override;
  void undo_action_impl(IGameState& state,
                        const UndoToken& token) const override;

 private:
  static void apply_action(CarcassonneState& s, ActionId action,
                           std::mt19937_64* rng);
  static int draw_next_tile(CarcassonneState& s, std::mt19937_64* rng);
  static bool any_legal_tile_placement(const CarcassonneState& s, int tile_type);
  static void apply_tile_action(CarcassonneState& s, ActionId action);
  static void apply_meeple_action(CarcassonneState& s, ActionId action,
                                  std::mt19937_64* rng);
  static void complete_turn_after_meeple(CarcassonneState& s,
                                         std::mt19937_64* rng);
  static void center_board(CarcassonneState& s);
  static void resolve_completed_features(CarcassonneState& s, int placed_cell);
  static void apply_final_scoring(CarcassonneState& s);
  static void award_to_owner_majority(CarcassonneState& s,
                                      const std::array<int, kPlayers>& owners,
                                      int points);
  static void set_winner(CarcassonneState& s);
};

}  // namespace board_ai::carcassonne
