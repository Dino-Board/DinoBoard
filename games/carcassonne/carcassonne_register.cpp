#include <any>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../heuristic/heuristic_config.h"
#include "../../heuristic/mcts_pv_heuristic.h"
#include "carcassonne_net_adapter.h"
#include "carcassonne_rules.h"
#include "carcassonne_state.h"
#include "heuristic/carcassonne_heuristic_pv_evaluator.h"
#include "heuristic/carcassonne_rule_heuristic.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using namespace board_ai::carcassonne;

std::vector<int> int_vector_from_array(
    const std::array<std::int8_t, kBoardCells>& values) {
  std::vector<int> out(kBoardCells);
  for (int i = 0; i < kBoardCells; ++i) {
    out[static_cast<size_t>(i)] = static_cast<int>(values[static_cast<size_t>(i)]);
  }
  return out;
}

std::vector<int> int_vector_from_array(
    const std::array<int, kBoardCells>& values) {
  std::vector<int> out(kBoardCells);
  for (int i = 0; i < kBoardCells; ++i) {
    out[static_cast<size_t>(i)] = values[static_cast<size_t>(i)];
  }
  return out;
}

AnyMap serialize_carcassonne(const IGameState& state) {
  const auto& s = board_ai::checked_cast<CarcassonneState>(state);
  AnyMap m;
  m["current_player"] = std::any(s.current_player());
  m["is_terminal"] = std::any(s.is_terminal());
  m["winner"] = std::any(s.winner());
  m["shared_victory"] = std::any(s.shared_victory);
  m["num_players"] = std::any(kPlayers);
  m["rows"] = std::any(kBoardRows);
  m["cols"] = std::any(kBoardCols);
  m["plies"] = std::any(s.plies);
  m["stage"] = std::any(static_cast<int>(s.stage));
  m["current_tile"] = std::any(static_cast<int>(s.current_tile));
  m["deck_count"] = std::any(static_cast<int>(s.deck_count));
  m["last_row"] = std::any(static_cast<int>(s.last_row));
  m["last_col"] = std::any(static_cast<int>(s.last_col));
  m["last_rotation"] = std::any(static_cast<int>(s.last_rotation));
  m["last_score_delta"] = std::any(s.last_score_delta);

  std::vector<int> scores(kPlayers);
  for (int p = 0; p < kPlayers; ++p) {
    scores[static_cast<size_t>(p)] = s.scores[static_cast<size_t>(p)];
  }
  m["scores"] = std::any(scores);

  // Scores if the game ended now (the heuristic value head's projection).
  const auto projected = CarcassonneRules::compute_final_scores(s);
  m["projected_final_scores"] = std::any(std::vector<int>(projected.begin(), projected.end()));

  std::vector<int> meeples(kPlayers);
  for (int p = 0; p < kPlayers; ++p) {
    meeples[static_cast<size_t>(p)] =
        static_cast<int>(s.meeples_remaining[static_cast<size_t>(p)]);
  }
  m["meeples_remaining"] = std::any(meeples);

  std::vector<int> counts(kTileTypes);
  for (int i = 0; i < kTileTypes; ++i) {
    int count = tile_def(i).initial_count;
    for (int cell = 0; cell < kBoardCells; ++cell) {
      if (s.board_tile[static_cast<size_t>(cell)] == i) --count;
    }
    if (s.current_tile == i) --count;
    count -= static_cast<int>(s.discarded_tile_counts[static_cast<size_t>(i)]);
    counts[static_cast<size_t>(i)] = count;
  }
  m["remaining_tile_counts"] = std::any(counts);
  m["discarded_tile_counts"] = std::any([&]() {
    std::vector<int> out(kTileTypes);
    for (int i = 0; i < kTileTypes; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<int>(s.discarded_tile_counts[static_cast<size_t>(i)]);
    }
    return out;
  }());
  m["tiles_remaining"] = std::any(static_cast<int>(s.deck_count));

  m["board_tile"] = std::any(int_vector_from_array(s.board_tile));
  m["board_rotation"] = std::any(int_vector_from_array(s.board_rotation));
  m["board_meeple_owner"] =
      std::any(int_vector_from_array(s.board_meeple_owner));
  m["board_meeple_pos"] = std::any(int_vector_from_array(s.board_meeple_pos));
  m["final_meeple_scores"] =
      std::any(int_vector_from_array(s.final_meeple_scores));

  std::vector<int> final_score_deltas(kPlayers);
  for (int p = 0; p < kPlayers; ++p) {
    final_score_deltas[static_cast<size_t>(p)] =
        s.final_score_deltas[static_cast<size_t>(p)];
  }
  m["final_score_deltas"] = std::any(final_score_deltas);

  m["history_tile"] = std::any([&]() {
    std::vector<int> out(kHistorySlots);
    for (int i = 0; i < kHistorySlots; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<int>(s.history_tile[static_cast<size_t>(i)]);
    }
    return out;
  }());
  m["history_rotation"] = std::any([&]() {
    std::vector<int> out(kHistorySlots);
    for (int i = 0; i < kHistorySlots; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<int>(s.history_rotation[static_cast<size_t>(i)]);
    }
    return out;
  }());
  m["history_row"] = std::any([&]() {
    std::vector<int> out(kHistorySlots);
    for (int i = 0; i < kHistorySlots; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<int>(s.history_row[static_cast<size_t>(i)]);
    }
    return out;
  }());
  m["history_col"] = std::any([&]() {
    std::vector<int> out(kHistorySlots);
    for (int i = 0; i < kHistorySlots; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<int>(s.history_col[static_cast<size_t>(i)]);
    }
    return out;
  }());

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
  if (max_row < 0) {
    min_row = max_row = kBoardRows / 2;
    min_col = max_col = kBoardCols / 2;
  }
  min_row = std::max(0, min_row - 1);
  max_row = std::min(kBoardRows - 1, max_row + 1);
  min_col = std::max(0, min_col - 1);
  max_col = std::min(kBoardCols - 1, max_col + 1);
  m["display_min_row"] = std::any(min_row);
  m["display_max_row"] = std::any(max_row);
  m["display_min_col"] = std::any(min_col);
  m["display_max_col"] = std::any(max_col);
  return m;
}

AnyMap describe_carcassonne(ActionId action) {
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  if (action >= 0 && action < kPlaceActionCount) {
    m["type"] = std::any(std::string("place_tile"));
    m["row"] = std::any(CarcassonneRules::decode_row(action));
    m["col"] = std::any(CarcassonneRules::decode_col(action));
    m["rotation"] = std::any(CarcassonneRules::decode_rotation(action));
    m["cell_index"] =
        std::any(CarcassonneRules::decode_row(action) * kBoardCols +
                 CarcassonneRules::decode_col(action));
    return m;
  }
  if (action == kMeeplePassAction) {
    m["type"] = std::any(std::string("meeple_pass"));
    return m;
  }
  if (action >= kMeepleSideActionBase &&
      action < kMeepleSideActionBase + kSides) {
    m["type"] = std::any(std::string("place_meeple"));
    m["position"] = std::any(CarcassonneRules::decode_meeple_position(action));
    m["kind"] = std::any(std::string("feature"));
    return m;
  }
  if (action >= kMeepleFarmActionBase &&
      action < kMeepleFarmActionBase + kFarmSides) {
    m["type"] = std::any(std::string("place_meeple"));
    m["position"] = std::any(CarcassonneRules::decode_meeple_position(action));
    m["kind"] = std::any(std::string("farm"));
    return m;
  }
  if (action == kMeepleCenterAction) {
    m["type"] = std::any(std::string("place_meeple"));
    m["position"] = std::any(kMeepleCenterPos);
    m["kind"] = std::any(std::string("cloister"));
    return m;
  }
  throw std::out_of_range("carcassonne action id out of range");
  return m;
}

enum class EncoderKind { kFlat, kTokens };

// kTokens selects CarcassonneTokenEncoder (transformer) over the flat history
// encoder (MLP). No heuristic here reads an encoder.
board_ai::GameBundle make_carcassonne(const std::string& game_id, std::uint64_t seed,
                                      EncoderKind encoder) {
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<CarcassonneState>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<CarcassonneRules>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  if (encoder == EncoderKind::kTokens) {
    b.encoder = std::make_unique<CarcassonneTokenEncoder>();
  } else {
    b.encoder = std::make_unique<CarcassonneFeatureEncoder>();
  }
  // No belief_tracker registered: every Carcassonne state field is now
  // all_public. The bag is count-only (bag_counts) and counts are
  // publicly derivable. There is no hidden information to track and
  // nothing for randomize_unseen to fill at sim entry. (The previous
  // pre-shuffled `tile_deck` was hidden but its order was leaking into
  // tail_solver TT keys via the hand-written state_hash — see
  // bag-draw.md and the no-deck-order-in-state memory.)
  b.state_serializer = serialize_carcassonne;
  b.action_descriptor = describe_carcassonne;
  b.heuristics.register_named(
      "rule", std::make_unique<CarcassonneRuleHeuristic>(
                  "rule", CarcassonneHeuristicProfile::kSharpPvSoft));
  auto register_rule_mcts =
      [&](const std::string& name, CarcassonneHeuristicProfile profile) {
        auto pv_eval = std::make_unique<CarcassonneHeuristicPVEvaluator>(
            profile);
        auto* pv_eval_ptr = pv_eval.get();
        b.heuristic_pv_evaluators.push_back(std::move(pv_eval));
        const auto hc = board_ai::heuristic::load_mcts_pv_heuristic_config(
            "carcassonne", name, /*require_model_path=*/false);
        b.heuristics.register_named(
            name,
            board_ai::heuristic::MctsPVHeuristic::make_from_evaluator(
                name, pv_eval_ptr, b.value_model.get(),
                hc.simulations, hc.c_puct, hc.world_cap, hc.leaf_batch_size));
      };
  register_rule_mcts("rule_mcts", CarcassonneHeuristicProfile::kSharpPvSoft);
  // rule_mcts_web binds to kSharpOpenEdge — currently the strongest
  // candidate in the post-refactor experiments (40 paired seats vs.
  // sharp_pv_soft = 57.5%, balanced seat distribution; CI ±15pp so
  // statistically still overlapping 50%, but the strongest direction
  // we have data on right now). Revisit after farmpotential is run
  // and the leader is re-confirmed at 80 games.
  register_rule_mcts("rule_mcts_web",
                     CarcassonneHeuristicProfile::kSharpOpenEdge);
  register_rule_mcts("rule_mcts_base", CarcassonneHeuristicProfile::kBaseline);
  register_rule_mcts("rule_mcts_sharp_pv_soft",
                     CarcassonneHeuristicProfile::kSharpPvSoft);
  register_rule_mcts("rule_mcts_sharp_cloister_soft_pv",
                     CarcassonneHeuristicProfile::kSharpCloisterSoftPv);
  register_rule_mcts("rule_mcts_sharp_reserve_soft_pv",
                     CarcassonneHeuristicProfile::kSharpReserveSoftPv);
  register_rule_mcts("rule_mcts_sharp_cloister_reserve_soft_pv",
                     CarcassonneHeuristicProfile::kSharpCloisterReserveSoftPv);
  // 2026-06-22 research profiles — see STRATEGY_NOTES.md.
  register_rule_mcts("rule_mcts_sharefix",
                     CarcassonneHeuristicProfile::kSharpSharefix);
  register_rule_mcts("rule_mcts_sharp_open_edge",
                     CarcassonneHeuristicProfile::kSharpOpenEdge);
  register_rule_mcts("rule_mcts_sharp_cloistercomplete",
                     CarcassonneHeuristicProfile::kSharpCloisterComplete);
  register_rule_mcts("rule_mcts_sharp_farmpotential",
                     CarcassonneHeuristicProfile::kSharpFarmPotential);
  b.install_event_protocol(
      board_ai::viz::no_events_extractor,
      []() -> const board_ai::viz::VisibilitySchema& {
        return CarcassonneState::schema();
      });
  b.adjudicator = [](const board_ai::IGameState& state) -> int {
    const auto& cs = board_ai::checked_cast<CarcassonneState>(state);
    if (cs.scores[0] > cs.scores[1]) return 0;
    if (cs.scores[1] > cs.scores[0]) return 1;
    return -1;
  };
  b.final_score_extractor = [](const board_ai::IGameState& state) -> std::vector<float> {
    const auto& cs = board_ai::checked_cast<CarcassonneState>(state);
    return {static_cast<float>(cs.scores[0]), static_cast<float>(cs.scores[1])};
  };
  return b;
}

board_ai::GameRegistrar reg("carcassonne", [](std::uint64_t seed) {
  return make_carcassonne("carcassonne", seed, EncoderKind::kFlat);
});
board_ai::GameRegistrar reg_tok("carcassonne_tok", [](std::uint64_t seed) {
  return make_carcassonne("carcassonne_tok", seed, EncoderKind::kTokens);
});

}  // namespace
