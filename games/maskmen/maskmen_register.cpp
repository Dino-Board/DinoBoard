#include <memory>
#include <string>
#include <any>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "../../heuristic/random_heuristic.h"
#include "maskmen_net_adapter.h"
#include "maskmen_rules.h"
#include "maskmen_state.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::IGameState;

template <int NPlayers>
AnyMap serialize_maskmen(const IGameState& state) {
  using namespace board_ai::maskmen;
  const auto& s = board_ai::checked_cast<MaskmenState<NPlayers>>(state);
  const auto& d = s.data;
  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["last_play_type"] = std::any(static_cast<int>(d.last_play_type));
  m["last_play_size"] = std::any(static_cast<int>(d.last_play_size));
  m["is_first_play_of_round"] =
      std::any(static_cast<int>(d.is_first_play_of_round));
  m["last_player_to_play"] = std::any(d.last_player_to_play);

  std::vector<int> passed(NPlayers, 0);
  for (int p = 0; p < NPlayers; ++p) {
    passed[static_cast<size_t>(p)] =
        d.passed_this_round[static_cast<size_t>(p)];
  }
  m["passed_this_round"] = std::any(passed);

  std::vector<std::vector<int>> strength(kTypeCount,
                                          std::vector<int>(kTypeCount, 0));
  for (int a = 0; a < kTypeCount; ++a) {
    for (int b = 0; b < kTypeCount; ++b) {
      strength[static_cast<size_t>(a)][static_cast<size_t>(b)] =
          d.strength_above[static_cast<size_t>(a)][static_cast<size_t>(b)];
    }
  }
  m["strength_above"] = std::any(strength);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> hc(kTypeCount, 0);
    for (int t = 0; t < kTypeCount; ++t) {
      hc[static_cast<size_t>(t)] = d.hand_count[static_cast<size_t>(p)]
                                               [static_cast<size_t>(t)];
    }
    pm["hand_count"] = std::any(hc);
    pm["hand_total"] =
        std::any(static_cast<int>(d.hand_size[static_cast<size_t>(p)]));
    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);
  return m;
}

// Public-event extractor. Diffs (before, after) and emits typed events
// the belief tracker reads:
//   - played { type, size, player }      — emitted on every non-pass play
//   - round_end { }                       — emitted when round resets
template <int NPlayers>
std::vector<board_ai::PublicEvent> extract_events_maskmen(
    const board_ai::IGameState& before,
    board_ai::ActionId action,
    const board_ai::IGameState& after,
    int /*perspective*/) {
  using namespace board_ai::maskmen;
  const auto& sb = board_ai::checked_cast<MaskmenState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<MaskmenState<NPlayers>>(after);
  const auto& db = sb.data;
  const auto& da = sa.data;
  std::vector<board_ai::PublicEvent> events;
  if (!is_pass(action)) {
    // Compare per-type hand totals across all seats to find the played
    // (player, type, size). Public — both states are publicly visible at
    // the post-mask layer for the player who just acted (their own row).
    int actor = db.current_player_;
    int played_type = -1;
    int played_size = 0;
    for (int t = 0; t < kTypeCount; ++t) {
      int diff = static_cast<int>(
                     db.hand_count[static_cast<size_t>(actor)]
                                  [static_cast<size_t>(t)]) -
                 static_cast<int>(
                     da.hand_count[static_cast<size_t>(actor)]
                                  [static_cast<size_t>(t)]);
      if (diff > 0) {
        played_type = t;
        played_size = diff;
        break;
      }
    }
    if (played_type < 0) {
      throw std::logic_error(
          "extract_events_maskmen: play action did not consume any cards");
    }
    AnyMap payload;
    payload["player"] = std::any(actor);
    payload["type"] = std::any(played_type);
    payload["size"] = std::any(played_size);
    events.emplace_back("played", std::move(payload));
  }
  // round_end fires when is_first_play_of_round flipped 0→1.
  if (db.is_first_play_of_round == 0 && da.is_first_play_of_round == 1) {
    events.emplace_back("round_end", AnyMap{});
  }
  return events;
}

AnyMap describe_maskmen(ActionId action) {
  using namespace board_ai::maskmen;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  if (is_pass(action)) {
    m["type"] = std::any(std::string("pass"));
  } else {
    m["type"] = std::any(std::string("play"));
    m["wrestler_type"] = std::any(play_type_from_action(action));
  }
  return m;
}

template <int NPlayers>
board_ai::GameBundle make_maskmen(const std::string& game_id,
                                  std::uint64_t seed) {
  using namespace board_ai::maskmen;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<MaskmenState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<MaskmenRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<MaskmenFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<MaskmenBeliefTracker<NPlayers>>();
  b.belief_feature_extractor =
      std::make_unique<MaskmenBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor =
      std::make_unique<MaskmenBeliefLabelExtractor<NPlayers>>();
  b.state_serializer = serialize_maskmen<NPlayers>;
  b.action_descriptor = describe_maskmen;
  b.heuristics.register_named(
      "rule", std::make_unique<board_ai::heuristic::RandomHeuristic>("rule"));

  b.install_event_protocol(
      extract_events_maskmen<NPlayers>,
      []() -> const board_ai::viz::VisibilitySchema& {
        return MaskmenState<NPlayers>::schema();
      });
  return b;
}

// 2p variant intentionally NOT registered: maskmen game design is degenerate
// at 2 players (no third-party dynamics, trivial bluffing surface). Web /
// game.json / README all drop 2p; see commit history for the rationale.
board_ai::GameRegistrar reg_maskmen_3p("maskmen_3p", [](std::uint64_t seed) {
  return make_maskmen<3>("maskmen_3p", seed);
});
board_ai::GameRegistrar reg_maskmen_4p("maskmen_4p", [](std::uint64_t seed) {
  return make_maskmen<4>("maskmen_4p", seed);
});

}  // namespace
