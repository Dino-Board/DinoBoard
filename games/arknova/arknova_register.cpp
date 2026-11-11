#include <any>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../heuristic/random_heuristic.h"
#include "arknova_actions.h"
#include "arknova_belief.h"
#include "arknova_encoder.h"
#include "arknova_json.h"
#include "arknova_rules.h"
#include "arknova_state.h"
#include "heuristic/arknova_greedy_heuristic.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::IGameState;
using namespace board_ai::arknova;

// UI / replay payload. Runs on truth states, so hidden cards are dropped: the core JSON comes
// from a copy with every player's private cards and the deck contents scrubbed (cards the
// opponent saw enter a hand stay listed).
AnyMap serialize_arknova(const IGameState& state) {
  const auto& st = board_ai::checked_cast<ArknovaState>(state);
  State pub = st.synced_core();
  for (Player& pl : pub.players) {
    for (int c = 0; c < kCardSlots; ++c) {
      if (!pl.shown_in_hand(static_cast<CardIdx>(c))) pl.hand[c] = 0;
    }
    std::fill(pl.final_cards, pl.final_cards + kMaxFinalCards, kNoCard);
    std::fill(pl.tucked, pl.tucked + kMaxTucked, kNoCard);
  }
  for (int i = 0; i < pub.stack_n; ++i) {
    int n_off, list_off;
    private_list_offsets(pub.stack[i].kind, &n_off, &list_off);
    if (list_off < 0) continue;
    for (int j = 0; j < pub.stack[i].d[n_off]; ++j) pub.stack[i].d[list_off + j] = kNoCard;
  }
  std::fill(pub.deck.begin(), pub.deck.end(), kNoCard);
  std::fill(pub.final_deck.begin(), pub.final_deck.end(), kNoCard);

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(kPlayers);
  m["scores"] = std::any(std::vector<int>{st.core.scores[0], st.core.scores[1]});
  std::string prefix;
  for (ActionId a : st.prefix) prefix += (prefix.empty() ? "" : " ") + atom_name(a);
  m["prefix"] = std::any(prefix);
  m["core_json"] = std::any(state_to_json(pub));
  // Static: the token encoder's feature layout, for tests and documentation.
  static const std::string encoder_layout = ArknovaTokenEncoder::layout_json();
  m["encoder_layout"] = std::any(encoder_layout);
  return m;
}

AnyMap describe_arknova(ActionId action) {
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  m["name"] = std::any(atom_name(action));
  for (int base : {atom::kHand, atom::kHandDiscard, atom::kZoo, atom::kZooGive}) {
    const int c = action - base;
    if (c >= 0 && c < num_cards()) m["card"] = std::any(card_id_string(static_cast<CardIdx>(c)));
  }
  return m;
}

// ark-nova-ai interop: its state as exported by games/arknova/tools/diff_test.py export_state, its
// option keys as option_to_json writes them.
void load_ark_nova_ai_state(IGameState& state, const std::string& json) {
  board_ai::checked_cast<ArknovaState>(state).reset_from_json(json);
}

std::vector<std::pair<std::string, std::vector<ActionId>>> ark_nova_ai_options(const IGameState& state) {
  const auto& st = board_ai::checked_cast<ArknovaState>(state);
  const Decision d = current_decision(st);
  const State& s = st.synced_core();
  std::vector<std::pair<std::string, std::vector<ActionId>>> out;
  for (std::size_t i = 0; i < d.options.size(); ++i) {
    if (board_ai::atom_prefix_matches(d.atoms[i], st.prefix)) out.emplace_back(option_to_json(s, d.options[i]), d.atoms[i]);
  }
  return out;
}

board_ai::GameBundle make_arknova(const std::string& game_id, std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<ArknovaState>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<ArknovaRules>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<ArknovaTokenEncoder>();
  b.belief_tracker = std::make_unique<ArknovaBeliefTracker>();
  b.belief_feature_extractor = std::make_unique<ArknovaBeliefFeatureExtractor>();
  b.belief_label_extractor = std::make_unique<ArknovaBeliefLabelExtractor>();
  b.state_serializer = serialize_arknova;
  b.action_descriptor = describe_arknova;
  b.heuristics.register_named("greedy", std::make_unique<ArknovaGreedyHeuristic>("greedy"));
  b.heuristics.register_named("beam_2x3", std::make_unique<ArknovaLookaheadHeuristic>(
                                              "beam_2x3", LookaheadConfig{2, 3, 6, 5}));  // depth, width, variants, follow
  b.heuristics.register_named("beam_3x8", std::make_unique<ArknovaLookaheadHeuristic>(
                                              "beam_3x8", LookaheadConfig{3, 8, 10, 8}));
  b.heuristics.register_named("random", std::make_unique<board_ai::heuristic::RandomHeuristic>("random"));
  b.final_score_extractor = [](const IGameState& state) -> std::vector<float> {
    const State& core = board_ai::checked_cast<ArknovaState>(state).synced_core();
    return {static_cast<float>(core.scores[0]), static_cast<float>(core.scores[1])};
  };
  b.install_event_protocol(board_ai::viz::no_events_extractor,
                           []() -> const board_ai::viz::VisibilitySchema& { return ArknovaState::schema(); });
  b.external_state_loader = load_ark_nova_ai_state;
  b.external_option_lister = ark_nova_ai_options;
  b.external_rules_switch = [](bool enabled) { g_upstream_rules = enabled; };
  return b;
}

board_ai::GameRegistrar reg_arknova("arknova", [](std::uint64_t seed) { return make_arknova("arknova", seed); });

}  // namespace
