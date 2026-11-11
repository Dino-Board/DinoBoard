#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/search/tail_solver.h"
#include "../../heuristic/heuristic_config.h"
#include "../../heuristic/mcts_pv_heuristic.h"
#include "azul_state.h"
#include "azul_rules.h"
#include "azul_net_adapter.h"
#include "heuristic/azul_rule_heuristic.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kColorNames[5] = {"blue", "yellow", "red", "black", "white"};
static const char* const kColorLetters[5] = {"B", "Y", "R", "K", "W"};

template <int NPlayers>
int azul_wall_col_for_color(int row, int color_idx) {
  return (color_idx + row) % board_ai::azul::kColors;
}

template <int NPlayers>
AnyMap serialize_azul(const IGameState& state) {
  using Cfg = board_ai::azul::AzulConfig<NPlayers>;
  const auto& s = board_ai::checked_cast<board_ai::azul::AzulState<NPlayers>>(state);

  AnyMap m;
  m["current_player"] = std::any(s.current_player());
  m["is_terminal"] = std::any(s.is_terminal());
  m["winner"] = std::any(s.winner());
  m["num_players"] = std::any(NPlayers);
  m["round_index"] = std::any(s.round_index);
  m["first_player_token_in_center"] = std::any(s.first_player_token_in_center);

  std::vector<int> scores(NPlayers);
  for (int i = 0; i < NPlayers; ++i) scores[i] = s.scores[i];
  m["scores"] = std::any(scores);

  std::vector<std::vector<int>> factories(Cfg::kFactories);
  for (int fi = 0; fi < Cfg::kFactories; ++fi) {
    factories[fi].resize(board_ai::azul::kColors);
    for (int c = 0; c < board_ai::azul::kColors; ++c)
      factories[fi][c] = s.factories[fi][c];
  }
  std::vector<std::any> factories_any;
  for (auto& f : factories) factories_any.push_back(std::any(std::move(f)));
  m["factories"] = std::any(factories_any);

  std::vector<int> center(board_ai::azul::kColors);
  for (int c = 0; c < board_ai::azul::kColors; ++c) center[c] = s.center[c];
  m["center"] = std::any(center);

  std::vector<int> bag_counts(board_ai::azul::kColors, 0);
  int bag_total = 0;
  for (int c = 0; c < board_ai::azul::kColors; ++c) {
    bag_counts[c] = s.bag_counts[c];
    bag_total += s.bag_counts[c];
  }
  m["bag_counts"] = std::any(bag_counts);
  m["bag_total"] = std::any(bag_total);

  // Box lid counts (tiles returned from completed pattern rows / floor
  // overflow). Exposed for tile-conservation invariants in test suites.
  std::vector<int> box_counts(board_ai::azul::kColors, 0);
  for (int c = 0; c < board_ai::azul::kColors; ++c) {
    box_counts[c] = s.box_lid_counts[c];
  }
  m["box_counts"] = std::any(box_counts);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    const auto& ps = s.players[p];
    AnyMap pm;
    pm["score"] = std::any(static_cast<int>(ps.score));

    std::vector<AnyMap> lines;
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      AnyMap line;
      line["length"] = std::any(static_cast<int>(ps.line_len[r]));
      line["color"] = std::any(static_cast<int>(ps.line_color[r]));
      line["capacity"] = std::any(r + 1);
      lines.push_back(std::move(line));
    }
    pm["pattern_lines"] = std::any(lines);

    std::vector<std::vector<int>> wall(board_ai::azul::kRows);
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      wall[r].resize(board_ai::azul::kColors);
      for (int c = 0; c < board_ai::azul::kColors; ++c)
        wall[r][c] = (ps.wall_mask[r] >> c) & 1;
    }
    std::vector<std::any> wall_any;
    for (auto& row : wall) wall_any.push_back(std::any(std::move(row)));
    pm["wall"] = std::any(wall_any);

    std::vector<int> floor;
    for (int fi = 0; fi < static_cast<int>(ps.floor_count); ++fi)
      floor.push_back(static_cast<int>(ps.floor[fi]));
    pm["floor"] = std::any(floor);
    pm["floor_count"] = std::any(static_cast<int>(ps.floor_count));

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  return m;
}

template <int NPlayers>
AnyMap describe_azul(ActionId action) {
  using Cfg = board_ai::azul::AzulConfig<NPlayers>;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  int source = static_cast<int>(action / (board_ai::azul::kColors * board_ai::azul::kTargetsPerColor));
  int color = static_cast<int>((action / board_ai::azul::kTargetsPerColor) % board_ai::azul::kColors);
  int target = static_cast<int>(action % board_ai::azul::kTargetsPerColor);

  m["source"] = std::any(source);
  m["color"] = std::any(color);
  m["target_line"] = std::any(target);

  bool is_center = (source == Cfg::kCenterSource);
  m["is_center"] = std::any(is_center);
  m["color_name"] = std::any(std::string(kColorNames[color]));
  m["color_letter"] = std::any(std::string(kColorLetters[color]));

  if (is_center) {
    m["source_name"] = std::any(std::string("center"));
  } else {
    m["source_name"] = std::any(std::string("factory_") + std::to_string(source));
  }

  if (target < board_ai::azul::kRows) {
    m["target_name"] = std::any(std::string("row_") + std::to_string(target + 1));
  } else {
    m["target_name"] = std::any(std::string("floor"));
  }

  return m;
}

// --- Public-event protocol for AI API belief-equivalence --------------
//
// Azul has one source of hidden randomness: the bag. At game start and at
// every round-end, factories are refilled by drawing from the bag. An AI
// API session with its own seed would produce different draws than ground
// truth, so we need to sync both events (initial factories + per-round
// refills). Azul's belief tracker is stateless (bag contents are fully
// derivable from public state); these events operate purely on state.

namespace azul_events {

using board_ai::azul::AzulConfig;
using board_ai::azul::AzulState;
using board_ai::azul::kColors;

template <int NPlayers>
std::vector<std::any> factories_to_any(const AzulState<NPlayers>& s) {
  std::vector<std::vector<int>> factories(AzulConfig<NPlayers>::kFactories);
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    factories[f].resize(kColors);
    for (int c = 0; c < kColors; ++c) {
      factories[f][c] = static_cast<int>(s.factories[f][c]);
    }
  }
  std::vector<std::any> out;
  out.reserve(factories.size());
  for (auto& f : factories) out.push_back(std::any(std::move(f)));
  return out;
}

template <int NPlayers>
void overwrite_factories_from_any(AzulState<NPlayers>& s, const std::any& val) {
  auto factories_any = std::any_cast<std::vector<std::any>>(val);
  if (factories_any.size() != static_cast<size_t>(AzulConfig<NPlayers>::kFactories)) {
    throw std::runtime_error(
        "azul event: factories count mismatch (expected " +
        std::to_string(AzulConfig<NPlayers>::kFactories) + ")");
  }
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    auto counts = std::any_cast<std::vector<int>>(factories_any[static_cast<size_t>(f)]);
    if (counts.size() != static_cast<size_t>(kColors)) {
      throw std::runtime_error(
          "azul event: color count mismatch for factory " + std::to_string(f));
    }
    for (int c = 0; c < kColors; ++c) {
      s.factories[f][c] = static_cast<std::uint8_t>(counts[static_cast<size_t>(c)]);
    }
  }
}

// After overriding visible tiles, recompute bag = 20-per-color minus all
// visible tiles of that color (factories, center, walls, pattern lines,
// floor, box_lid). This ensures MCTS determinizations drawing from bag
// during simulation will see a consistent pool — even if AI's own earlier
// RNG draws diverged from ground truth.
template <int NPlayers>
void recompute_bag_from_visible(AzulState<NPlayers>& s) {
  std::array<int, kColors> visible{};
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    for (int c = 0; c < kColors; ++c) visible[c] += s.factories[f][c];
  }
  for (int c = 0; c < kColors; ++c) visible[c] += s.center[c];
  for (int p = 0; p < NPlayers; ++p) {
    const auto& ps = s.players[p];
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      for (int c = 0; c < kColors; ++c) {
        if ((ps.wall_mask[r] >> c) & 1U) visible[c]++;
      }
      if (ps.line_color[r] >= 0 && ps.line_color[r] < kColors) {
        visible[ps.line_color[r]] += ps.line_len[r];
      }
    }
    for (int i = 0; i < ps.floor_count; ++i) {
      const int t = ps.floor[i];
      if (t >= 0 && t < kColors) visible[t]++;
    }
  }
  // Box_lid holds tiles already "consumed" out of bag but recyclable. Keep
  // box_lid_counts as-is; put the remainder into bag_counts.
  for (int c = 0; c < kColors; ++c) {
    visible[c] += s.box_lid_counts[c];
  }
  for (int c = 0; c < kColors; ++c) {
    const int remaining = 20 - visible[c];
    s.bag_counts[c] = remaining > 0 ? remaining : 0;
  }
}


template <int NPlayers>
PublicEventTrace extract_events(
    const IGameState& before,
    ActionId /*action*/,
    const IGameState& after,
    int perspective) {
  const auto& sb = board_ai::checked_cast<AzulState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<AzulState<NPlayers>>(after);
  PublicEventTrace out;
  // Round settlement increments round_index and triggers factory refill.
  // If the round advanced AND the game continues (not terminal), emit
  // a factory_refill event describing the newly drawn tiles.
  if (sa.round_index > sb.round_index && !sa.terminal) {
    AnyMap payload;
    payload["factories"] = std::any(factories_to_any(sa));
    out.events.emplace_back("factory_refill", std::move(payload));
  }

  // Public snapshot — walker-driven, single unified call. Azul has no
  // per-perspective hidden info; the wire still ships a per-perspective
  // viz slice (all-1 for every slot) for protocol uniformity.
  // `game_first_player` is fixed at game start and skipped here.
  board_ai::viz::serialize_public_snapshot(
      after, AzulState<NPlayers>::schema(), perspective,
      out.public_snapshot, /*skip=*/{"game_first_player"});

  return out;
}

template <int NPlayers>
void apply_public_state(IGameState& state, const AnyMap& snap,
                        int receiver_seat) {
  board_ai::viz::apply_public_snapshot(
      state, AzulState<NPlayers>::schema(), receiver_seat, snap,
      /*skip=*/{"game_first_player"});
}

}  // namespace azul_events

template <int NPlayers>
board_ai::GameBundle make_azul(const std::string& game_id, std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<board_ai::azul::AzulState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<board_ai::azul::AzulRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<board_ai::azul::AzulFeatureEncoder<NPlayers>>();
  // No belief_tracker registered: every Azul state field is all_public; bag
  // and box_lid live in schema as per-color counts (also all_public). There
  // is no per-perspective hidden information to track and nothing for
  // randomize_unseen to fill at sim entry.
  b.state_serializer = serialize_azul<NPlayers>;
  b.action_descriptor = describe_azul<NPlayers>;
  b.heuristics.register_named(
      "rule_v1", std::make_unique<board_ai::azul::AzulRuleHeuristic<NPlayers>>("rule_v1"));
  // pv_512x4: ONNX-backed PV+MCTS heuristic. MCTS knobs and model path come
  // from heuristic.json::named.pv_512x4.
  {
    const auto hc = board_ai::heuristic::load_mcts_pv_heuristic_config(
        game_id, "pv_512x4", /*require_model_path=*/true);
    b.heuristics.register_named(
        "pv_512x4",
        board_ai::heuristic::MctsPVHeuristic::make_from_onnx(
            "pv_512x4", hc.model_path, b.encoder.get(),
            b.value_model.get(), hc.simulations, hc.c_puct, hc.world_cap,
            hc.leaf_batch_size));
  }
  b.public_event_extractor = azul_events::extract_events<NPlayers>;
  b.public_state_applier = azul_events::apply_public_state<NPlayers>;

  b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();
  // AzulRules::do_action_deterministic forces a draw-terminal (winner=-1)
  // whenever an action would trigger a round-end factory refill (the only
  // source of physical randomness in Azul). Tail solver therefore never
  // consumes hidden chance outcomes — safe even though Azul has no tracker.
  b.stochastic_tail_solve_safe = true;

  // Auxiliary scorer for tail-solve tiebreak among proven wins. Pure margin
  // (no speed bonus): Azul end-game scoring already values "finish patterns"
  // which correlates with longer-game outcomes — preferring faster wins
  // would actively trade away points. Margin normaliser 100 = typical Azul
  // end-game span; tail solver multiplies by `auxiliary_weight` (typ. 0.01)
  // so the bonus stays << 1.0. `depth_remaining` is unused.
  b.auxiliary_scorer = [](const board_ai::IGameState& state, int player,
                          int /*depth_remaining*/) -> float {
    using Cfg = board_ai::azul::AzulConfig<NPlayers>;
    const auto& s = board_ai::checked_cast<board_ai::azul::AzulState<NPlayers>>(state);
    int my = s.scores[static_cast<size_t>(player)];
    int max_opp = std::numeric_limits<int>::min();
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      if (p == player) continue;
      int op = s.scores[static_cast<size_t>(p)];
      if (op > max_opp) max_opp = op;
    }
    return static_cast<float>(my - max_opp) / 100.0f;
  };

  // Trigger: at least one player has ≥4 tiles in some pattern-line row,
  // AND at least 2 factories are empty. Captures "round nearly resolved
  // and a wall placement is imminent" — small enough subgame to solve.
  b.tail_solve_trigger = [](const board_ai::IGameState& state, int /*ply*/) -> bool {
    using Cfg = board_ai::azul::AzulConfig<NPlayers>;
    const auto& s = board_ai::checked_cast<board_ai::azul::AzulState<NPlayers>>(state);
    bool any_line_near_full = false;
    for (int p = 0; p < NPlayers; ++p) {
      const auto& ps = s.players[p];
      for (int r = 0; r < board_ai::azul::kRows; ++r) {
        if (static_cast<int>(ps.line_len[r]) >= 4) {
          any_line_near_full = true;
          break;
        }
      }
      if (any_line_near_full) break;
    }
    if (!any_line_near_full) return false;
    int empty_factories = 0;
    for (int f = 0; f < Cfg::kFactories; ++f) {
      int total = 0;
      for (int c = 0; c < board_ai::azul::kColors; ++c) total += s.factories[f][c];
      if (total == 0) ++empty_factories;
    }
    return empty_factories >= 2;
  };
  return b;
}

board_ai::GameRegistrar reg_azul("azul", [](std::uint64_t seed) {
  return make_azul<2>("azul", seed);
});
board_ai::GameRegistrar reg_azul_2p("azul_2p", [](std::uint64_t seed) {
  return make_azul<2>("azul_2p", seed);
});
board_ai::GameRegistrar reg_azul_3p("azul_3p", [](std::uint64_t seed) {
  return make_azul<3>("azul_3p", seed);
});
board_ai::GameRegistrar reg_azul_4p("azul_4p", [](std::uint64_t seed) {
  return make_azul<4>("azul_4p", seed);
});

}  // namespace
