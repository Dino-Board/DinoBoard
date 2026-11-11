#include <any>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../heuristic/heuristic_config.h"
#include "../../heuristic/mcts_pv_heuristic.h"
#include "../../heuristic/random_heuristic.h"
#include "harmonies_net_adapter.h"
#include "harmonies_rules.h"
#include "harmonies_state.h"
#include "heuristic/harmonies_greedy_heuristic.h"
#include "heuristic/harmonies_greedy_pv_evaluator.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;

template <int NPlayers, int Side>
AnyMap serialize_harmonies(const IGameState& state) {
  using namespace board_ai::harmonies;
  using Geo = Board<Side>;
  const auto& s = board_ai::checked_cast<HarmoniesState<NPlayers, Side>>(state);

  AnyMap m;
  m["current_player"] = std::any(s.current_player());
  m["is_terminal"] = std::any(s.is_terminal());
  m["winner"] = std::any(s.winner());
  m["num_players"] = std::any(NPlayers);
  m["first_player"] = std::any(s.first_player());
  m["turn_started"] = std::any(s.turn_started);
  m["took_card_this_turn"] = std::any(s.took_card_this_turn);
  m["end_triggered"] = std::any(s.end_triggered);

  // Static board description for the web client and tests: the side, hex ids
  // per column (left to right; odd columns sit half a hex lower) and the
  // action layout that follows from the hex count.
  m["board_side"] = std::any(std::string(1, Geo::kName));
  std::vector<std::any> columns;
  for (int q = 0; q < Geo::kColumns; ++q) {
    std::vector<int> col;
    for (int i = 0; i < Geo::kColumnHeight[q]; ++i) col.push_back(Geo::kColumnStart[q] + i);
    columns.push_back(std::any(std::move(col)));
  }
  m["board_columns"] = std::any(columns);
  AnyMap layout;
  layout["select_slot"] = std::any(kActSelectSlotBase);
  layout["take_card"] = std::any(kActTakeCardBase);
  layout["place_token"] = std::any(kActPlaceTokenBase);
  layout["place_cube"] = std::any(Geo::kActPlaceCubeBase);
  layout["end_turn"] = std::any(Geo::kActEndTurn);
  m["action_layout"] = std::any(layout);

  std::vector<int> selected(kColors);
  for (int c = 0; c < kColors; ++c) selected[c] = static_cast<int>(s.selected_tokens[c]);
  m["selected_tokens"] = std::any(selected);

  std::vector<std::any> central_any;
  for (int slot = 0; slot < kSlots; ++slot) {
    std::vector<int> row(kColors);
    for (int c = 0; c < kColors; ++c) row[c] = s.central_board_counts[slot][c];
    central_any.push_back(std::any(std::move(row)));
  }
  m["central_board_counts"] = std::any(central_any);

  std::vector<int> market(kSlots);
  for (int s2 = 0; s2 < kSlots; ++s2) market[s2] = static_cast<int>(s.animal_market[s2]);
  m["animal_market"] = std::any(market);

  std::vector<int> deck(kAnimalCards);
  for (int i = 0; i < kAnimalCards; ++i) deck[i] = static_cast<int>(s.animal_deck[i]);
  m["animal_deck"] = std::any(deck);
  m["animal_deck_count"] = std::any(s.animal_deck_count);
  std::vector<int> completed(kAnimalCards);
  for (int i = 0; i < kAnimalCards; ++i) completed[i] = static_cast<int>(s.completed_cards[i]);
  m["completed_cards"] = std::any(completed);

  std::vector<int> pouch(kColors);
  for (int c = 0; c < kColors; ++c) pouch[c] = s.pouch_counts[c];
  m["pouch_counts"] = std::any(pouch);

  std::vector<std::any> players_any;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> tt(Geo::kHexes), ha(Geo::kHexes);
    for (int h = 0; h < Geo::kHexes; ++h) {
      tt[h] = static_cast<int>(s.tower_type[p][h]);
      ha[h] = static_cast<int>(s.has_animal[p][h]);
    }
    pm["tower_type"] = std::any(tt);
    pm["has_animal"] = std::any(ha);
    std::vector<int> hc(kHandSize), cr(kHandSize), ci(kHandSize);
    for (int i = 0; i < kHandSize; ++i) {
      hc[i] = static_cast<int>(s.held_cards[p][i]);
      cr[i] = static_cast<int>(s.cube_remaining[p][i]);
      ci[i] = static_cast<int>(s.cube_initial[p][i]);
    }
    pm["held_cards"] = std::any(hc);
    pm["cube_remaining"] = std::any(cr);
    pm["cube_initial"] = std::any(ci);
    pm["space_remaining"] = std::any(static_cast<int>(s.space_remaining[p]));
    pm["animal_score"] = std::any(s.animal_scores[p]);
    pm["landscape_score"] = std::any(s.landscape_scores[p]);
    pm["score"] = std::any(s.animal_scores[p] + s.landscape_scores[p]);
    players_any.push_back(std::any(std::move(pm)));
  }
  m["players"] = std::any(players_any);
  return m;
}

template <int Side>
AnyMap describe_harmonies(ActionId a) {
  using namespace board_ai::harmonies;
  using Geo = Board<Side>;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(a));
  if (a >= kActSelectSlotBase && a < kActTakeCardBase) {
    m["type"] = std::any(std::string("select_slot"));
    m["slot"] = std::any(static_cast<int>(a - kActSelectSlotBase));
  } else if (a >= kActTakeCardBase && a < kActPlaceTokenBase) {
    m["type"] = std::any(std::string("take_card"));
    m["card_id"] = std::any(static_cast<int>(a - kActTakeCardBase) + 1);
  } else if (a >= kActPlaceTokenBase && a < Geo::kActPlaceCubeBase) {
    const int rel = static_cast<int>(a - kActPlaceTokenBase);
    m["type"] = std::any(std::string("place_token"));
    m["hex"] = std::any(rel / kColors);
    m["color"] = std::any(rel % kColors + 1);
  } else if (a >= Geo::kActPlaceCubeBase && a < Geo::kActEndTurn) {
    const int rel = static_cast<int>(a - Geo::kActPlaceCubeBase);
    m["type"] = std::any(std::string("place_cube"));
    m["hex"] = std::any(rel / kAnimalCards);
    m["card_id"] = std::any(rel % kAnimalCards + 1);
  } else if (a == Geo::kActEndTurn) {
    m["type"] = std::any(std::string("end_turn"));
  } else {
    m["type"] = std::any(std::string("invalid"));
  }
  return m;
}

enum class EncoderKind { kFlat, kTokens };

template <int NPlayers, int Side>
board_ai::GameBundle make_harmonies(const std::string& game_id, std::uint64_t seed,
                                    EncoderKind encoder = EncoderKind::kFlat) {
  using namespace board_ai::harmonies;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<HarmoniesState<NPlayers, Side>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<HarmoniesRules<NPlayers, Side>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.state_serializer = serialize_harmonies<NPlayers, Side>;
  if (encoder == EncoderKind::kTokens) {
    if constexpr (Side == kSideB) {
      b.encoder = std::make_unique<HarmoniesTokenEncoder<NPlayers, Side>>();
      // Static: the token encoder's feature layout, for tests and documentation.
      b.state_serializer = [](const IGameState& state) {
        static const std::string layout = HarmoniesTokenEncoder<NPlayers, Side>::layout_json();
        AnyMap m = serialize_harmonies<NPlayers, Side>(state);
        m["encoder_layout"] = std::any(layout);
        return m;
      };
    } else {
      throw std::logic_error("harmonies: the token encoder exists for side B only");
    }
  } else {
    b.encoder = std::make_unique<HarmoniesFeatureEncoder<NPlayers, Side>>();
  }
  b.belief_tracker = std::make_unique<HarmoniesBeliefTracker<NPlayers, Side>>();
  b.install_event_protocol(
      board_ai::viz::no_events_extractor,
      []() -> const board_ai::viz::VisibilitySchema& {
        return HarmoniesState<NPlayers, Side>::schema();
      });
  b.action_descriptor = describe_harmonies<Side>;
  b.heuristics.register_named(
      "rule", std::make_unique<board_ai::heuristic::RandomHeuristic>("rule"));
  b.heuristics.register_named(
      "greedy",
      std::make_unique<HarmoniesGreedyHeuristic<NPlayers, Side>>("greedy"));
  auto pv_eval = std::make_unique<HarmoniesGreedyPVEvaluator<NPlayers, Side>>();
  pv_eval->set_rules(b.rules.get());
  auto* pv_eval_ptr = pv_eval.get();
  b.heuristic_pv_evaluators.push_back(std::move(pv_eval));
  // Every variant (side A, side B, token encoder) shares games/harmonies/config;
  // none of these heuristics loads a model, so the base id resolves them.
  const auto hc = board_ai::heuristic::load_mcts_pv_heuristic_config(
      "harmonies", "greedy_mcts", /*require_model_path=*/false);
  b.heuristics.register_named(
      "greedy_mcts",
      board_ai::heuristic::MctsPVHeuristic::make_from_evaluator(
          "greedy_mcts", pv_eval_ptr, b.value_model.get(),
          hc.simulations, hc.c_puct, hc.world_cap, hc.leaf_batch_size));
  const auto hc_train = board_ai::heuristic::load_mcts_pv_heuristic_config(
      "harmonies", "greedy_mcts_train", /*require_model_path=*/false);
  b.heuristics.register_named(
      "greedy_mcts_train",
      board_ai::heuristic::MctsPVHeuristic::make_from_evaluator(
          "greedy_mcts_train", pv_eval_ptr, b.value_model.get(),
          hc_train.simulations, hc_train.c_puct, hc_train.world_cap,
          hc_train.leaf_batch_size));
  return b;
}

using board_ai::harmonies::kSideA;
using board_ai::harmonies::kSideB;

// Side B (islands) is the board of the default ids.
board_ai::GameRegistrar reg_harmonies("harmonies", [](std::uint64_t seed) {
  return make_harmonies<2, kSideB>("harmonies", seed);
});
board_ai::GameRegistrar reg_harmonies_2p("harmonies_2p", [](std::uint64_t seed) {
  return make_harmonies<2, kSideB>("harmonies_2p", seed);
});
board_ai::GameRegistrar reg_harmonies_3p("harmonies_3p", [](std::uint64_t seed) {
  return make_harmonies<3, kSideB>("harmonies_3p", seed);
});
board_ai::GameRegistrar reg_harmonies_4p("harmonies_4p", [](std::uint64_t seed) {
  return make_harmonies<4, kSideB>("harmonies_4p", seed);
});
// Side A (river): the board games/harmonies/model/harmonies_a_2p.onnx was trained on.
board_ai::GameRegistrar reg_harmonies_a("harmonies_a", [](std::uint64_t seed) {
  return make_harmonies<2, kSideA>("harmonies_a", seed);
});
board_ai::GameRegistrar reg_harmonies_a_2p("harmonies_a_2p", [](std::uint64_t seed) {
  return make_harmonies<2, kSideA>("harmonies_a_2p", seed);
});
board_ai::GameRegistrar reg_harmonies_a_3p("harmonies_a_3p", [](std::uint64_t seed) {
  return make_harmonies<3, kSideA>("harmonies_a_3p", seed);
});
board_ai::GameRegistrar reg_harmonies_a_4p("harmonies_a_4p", [](std::uint64_t seed) {
  return make_harmonies<4, kSideA>("harmonies_a_4p", seed);
});
// Side B, 2 players, token encoder for the transformer network.
board_ai::GameRegistrar reg_harmonies_tok("harmonies_tok", [](std::uint64_t seed) {
  return make_harmonies<2, kSideB>("harmonies_tok", seed, EncoderKind::kTokens);
});

}  // namespace
