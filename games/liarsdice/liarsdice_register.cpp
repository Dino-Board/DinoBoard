#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "liarsdice_net_adapter.h"
#include "liarsdice_rules.h"
#include "liarsdice_state.h"
#include "heuristic/liarsdice_rule_heuristic.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::IGameState;

template <int NPlayers>
AnyMap serialize_liarsdice(const IGameState& state) {
  using namespace board_ai::liarsdice;
  const auto& s = board_ai::checked_cast<LiarsdiceState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["current_bid_count"] = std::any(static_cast<int>(d.current_bid_count));
  m["current_bid_face"] = std::any(static_cast<int>(d.current_bid_face));

  // Per-player dice. Owner-only schema: viz=1 only for the owner; the
  // rules' challenge resolution flips viz to public for everyone. Frontend
  // reads s["players"][p]["dice"] — when hidden it's an empty list,
  // otherwise the raw face-counts. We expose dice both as a count vector
  // [n_1..n_6] and as a flat ordered list for display convenience.
  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> counts(static_cast<size_t>(kFaceCount), 0);
    std::vector<int> dice_list;
    for (int f = 0; f < kFaceCount; ++f) {
      const int v = d.dice_count[static_cast<size_t>(p)][static_cast<size_t>(f)];
      counts[static_cast<size_t>(f)] = v;
      if (v >= 0) {
        for (int k = 0; k < v; ++k) dice_list.push_back(f + 1);
      }
    }
    pm["dice_count"] = std::any(counts);
    pm["dice"] = std::any(dice_list);
    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  return m;
}

AnyMap describe_liarsdice_2p(ActionId action) {
  using namespace board_ai::liarsdice;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  constexpr int kTotalDice2p = LiarsdiceConfig<2>::kTotalDice;
  if (is_challenge(action, kTotalDice2p)) {
    m["type"] = std::any(std::string("challenge"));
  } else {
    m["type"] = std::any(std::string("bid"));
    m["count"] = std::any(bid_count_from_action(action));
    m["face"] = std::any(bid_face_from_action(action));
  }
  return m;
}
AnyMap describe_liarsdice_3p(ActionId action) {
  using namespace board_ai::liarsdice;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  constexpr int kTotalDice3p = LiarsdiceConfig<3>::kTotalDice;
  if (is_challenge(action, kTotalDice3p)) {
    m["type"] = std::any(std::string("challenge"));
  } else {
    m["type"] = std::any(std::string("bid"));
    m["count"] = std::any(bid_count_from_action(action));
    m["face"] = std::any(bid_face_from_action(action));
  }
  return m;
}
AnyMap describe_liarsdice_4p(ActionId action) {
  using namespace board_ai::liarsdice;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  constexpr int kTotalDice4p = LiarsdiceConfig<4>::kTotalDice;
  if (is_challenge(action, kTotalDice4p)) {
    m["type"] = std::any(std::string("challenge"));
  } else {
    m["type"] = std::any(std::string("bid"));
    m["count"] = std::any(bid_count_from_action(action));
    m["face"] = std::any(bid_face_from_action(action));
  }
  return m;
}

template <int NPlayers>
board_ai::GameBundle make_liarsdice(const std::string& game_id,
                                     std::uint64_t seed) {
  using namespace board_ai::liarsdice;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<LiarsdiceState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<LiarsdiceRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<LiarsdiceFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<LiarsdiceBeliefTracker<NPlayers>>();
  b.belief_feature_extractor =
      std::make_unique<LiarsdiceBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor =
      std::make_unique<LiarsdiceBeliefLabelExtractor<NPlayers>>();
  b.state_serializer = serialize_liarsdice<NPlayers>;
  if constexpr (NPlayers == 2) b.action_descriptor = describe_liarsdice_2p;
  else if constexpr (NPlayers == 3) b.action_descriptor = describe_liarsdice_3p;
  else b.action_descriptor = describe_liarsdice_4p;
  b.heuristics.register_named(
      "rule_v1",
      std::make_unique<board_ai::liarsdice::LiarsdiceRuleHeuristic<NPlayers>>("rule_v1"));

  // Tracker only consumes action_id (no event-stream payload). Use the
  // framework's no-events extractor; the wholesale public_snapshot half
  // is auto-derived by install_event_protocol.
  b.episode_stats_extractor = [](
      const board_ai::IGameState& final_state,
      const board_ai::IGameState& /*initial_state*/,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    using namespace board_ai::liarsdice;
    std::map<std::string, double> out;
    if (samples.empty()) return out;
    const auto& fs = board_ai::checked_cast<LiarsdiceState<NPlayers>>(final_state);
    const auto& d = fs.data;
    // Find the last action — if it's a challenge, compute p_safe from
    // the challenger's perspective (how safe was the bid they challenged).
    const auto& last = samples.back();
    if (is_challenge(last.action_id, LiarsdiceConfig<NPlayers>::kTotalDice)) {
      const int challenger = last.player;
      const int cur_count = d.current_bid_count;
      const int cur_face = d.current_bid_face;
      if (cur_count > 0 && cur_face >= 2) {
        // Count challenger's own matching dice (face + wild face 1).
        int own_match = 0;
        for (int f = 0; f < kFaceCount; ++f) {
          const int v = d.dice_count[static_cast<size_t>(challenger)][static_cast<size_t>(f)];
          if ((f + 1) == cur_face || (f + 1) == 1) own_match += v;
        }
        const int hidden_dice = (NPlayers - 1) * kDicePerPlayer;
        const int needed = cur_count - own_match;
        double p_safe;
        if (needed <= 0) {
          p_safe = 1.0;
        } else if (needed > hidden_dice) {
          p_safe = 0.0;
        } else {
          const double p_match = 1.0 / 3.0;
          const double mu = hidden_dice * p_match;
          const double sigma = std::sqrt(std::max(hidden_dice * p_match * (1.0 - p_match), 1e-9));
          const double z = (static_cast<double>(needed) - 0.5 - mu) / sigma;
          p_safe = std::clamp(1.0 - 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))), 0.0, 1.0);
        }
        // p_safe = prob the bid is true. Low = good time to challenge.
        out["challenge_safety"] = 1.0 - p_safe;
      }
    }
    return out;
  };

  b.install_event_protocol(
      board_ai::viz::no_events_extractor,
      []() -> const board_ai::viz::VisibilitySchema& {
        return LiarsdiceState<NPlayers>::schema();
      });

  return b;
}

board_ai::GameRegistrar reg_liarsdice("liarsdice", [](std::uint64_t seed) {
  return make_liarsdice<2>("liarsdice", seed);
});
board_ai::GameRegistrar reg_liarsdice_2p("liarsdice_2p",
                                          [](std::uint64_t seed) {
  return make_liarsdice<2>("liarsdice_2p", seed);
});
board_ai::GameRegistrar reg_liarsdice_3p("liarsdice_3p",
                                          [](std::uint64_t seed) {
  return make_liarsdice<3>("liarsdice_3p", seed);
});
board_ai::GameRegistrar reg_liarsdice_4p("liarsdice_4p",
                                          [](std::uint64_t seed) {
  return make_liarsdice<4>("liarsdice_4p", seed);
});

}  // namespace
