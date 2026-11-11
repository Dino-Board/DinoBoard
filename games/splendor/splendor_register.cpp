#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/search/tail_solver.h"
#include "../../heuristic/heuristic_config.h"
#include "../../heuristic/mcts_pv_heuristic.h"
#include "splendor_state.h"
#include "splendor_rules.h"
#include "splendor_net_adapter.h"
#include "heuristic/splendor_rule_heuristic.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kColorNames[6] = {"white", "blue", "green", "red", "black", "gold"};

template <int NPlayers>
AnyMap serialize_splendor(const IGameState& state) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  const auto& s = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state);
  const auto& d = s.persistent.data();

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["plies"] = std::any(d.plies);
  m["stage"] = std::any(static_cast<int>(d.stage));
  m["pending_returns"] = std::any(d.pending_returns);
  m["final_round_remaining"] = std::any(d.final_round_remaining);

  std::vector<int> scores(NPlayers);
  for (int i = 0; i < NPlayers; ++i) scores[i] = d.scores[i];
  m["scores"] = std::any(scores);

  std::vector<int> bank(board_ai::splendor::kTokenTypes);
  for (int i = 0; i < board_ai::splendor::kTokenTypes; ++i) bank[i] = d.bank[i];
  m["bank"] = std::any(bank);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> gems(board_ai::splendor::kTokenTypes);
    for (int j = 0; j < board_ai::splendor::kTokenTypes; ++j) gems[j] = d.player_gems[p][j];
    pm["gems"] = std::any(gems);
    std::vector<int> bonuses(board_ai::splendor::kColorCount);
    for (int j = 0; j < board_ai::splendor::kColorCount; ++j) bonuses[j] = d.player_bonuses[p][j];
    pm["bonuses"] = std::any(bonuses);
    pm["points"] = std::any(static_cast<int>(d.player_points[p]));
    pm["cards_count"] = std::any(static_cast<int>(d.player_cards_count[p]));
    pm["nobles_count"] = std::any(static_cast<int>(d.player_nobles_count[p]));

    int rs = d.reserved_size[p];
    std::vector<AnyMap> reserved;
    const auto& pool = board_ai::splendor::splendor_card_pool();
    for (int ri = 0; ri < rs; ++ri) {
      AnyMap rc;
      int card_id = d.reserved[p][ri];
      bool visible = d.reserved_visible[p][ri] != 0;
      rc["visible"] = std::any(visible);
      // Always serialize full card info whenever card_id is valid, even for
      // deck-reserved (visible=false). The owner always knows their own
      // reserved card, so the web UI must receive its face for rendering.
      // The frontend uses `(visible || isHuman)` to decide whether to show
      // the face or the card back, so opponents' deck-reserved slots still
      // render as a hidden placeholder.
      if (card_id >= 0 && card_id < static_cast<int>(pool.size())) {
        const auto& card = pool[card_id];
        rc["tier"] = std::any(static_cast<int>(card.tier));
        rc["bonus"] = std::any(static_cast<int>(card.bonus));
        rc["points"] = std::any(static_cast<int>(card.points));
        std::vector<int> cost(board_ai::splendor::kColorCount);
        for (int j = 0; j < board_ai::splendor::kColorCount; ++j) cost[j] = card.cost[j];
        rc["cost"] = std::any(cost);
      } else {
        // Card id not available (a state that does not hold this seat's
        // blind reserve — viewer path / observer session). The tier it
        // was drawn from is public: reserved_tier is 0-based, card tiers
        // are 1-based.
        const int rt = d.reserved_tier[p][ri];
        rc["tier"] = std::any(rt >= 0 && rt < 3 ? rt + 1 : -1);
      }
      reserved.push_back(std::move(rc));
    }
    pm["reserved"] = std::any(reserved);
    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  const auto& pool = board_ai::splendor::splendor_card_pool();
  std::vector<std::vector<AnyMap>> tableau(3);
  for (int tier = 0; tier < 3; ++tier) {
    int ts = d.tableau_size[tier];
    for (int slot = 0; slot < ts; ++slot) {
      int card_id = d.tableau[tier][slot];
      AnyMap cm;
      cm["card_id"] = std::any(card_id);
      if (card_id >= 0 && card_id < static_cast<int>(pool.size())) {
        const auto& card = pool[card_id];
        cm["tier"] = std::any(static_cast<int>(card.tier));
        cm["bonus"] = std::any(static_cast<int>(card.bonus));
        cm["points"] = std::any(static_cast<int>(card.points));
        std::vector<int> cost(board_ai::splendor::kColorCount);
        for (int j = 0; j < board_ai::splendor::kColorCount; ++j) cost[j] = card.cost[j];
        cm["cost"] = std::any(cost);
      }
      tableau[tier].push_back(std::move(cm));
    }
  }
  std::vector<std::any> tableau_any;
  for (auto& t : tableau) tableau_any.push_back(std::any(std::move(t)));
  m["tableau"] = std::any(tableau_any);

  std::vector<int> deck_sizes(3);
  // The public per-tier counters, not decks[i].size(): the deck contents
  // are not schema slots, so a state that only follows the public stream
  // (observer session) keeps its own stale decks.
  for (int i = 0; i < 3; ++i) deck_sizes[i] = static_cast<int>(d.deck_sizes[i]);
  m["deck_sizes"] = std::any(deck_sizes);

  std::vector<int> all_deck_ids;
  for (int t = 0; t < 3; ++t) {
    for (auto cid : d.decks[t]) all_deck_ids.push_back(static_cast<int>(cid));
  }
  m["_test_all_deck_ids"] = std::any(all_deck_ids);

  std::vector<int> tableau_ids;
  for (int t = 0; t < 3; ++t) {
    for (int slot = 0; slot < d.tableau_size[t]; ++slot) {
      tableau_ids.push_back(static_cast<int>(d.tableau[t][slot]));
    }
  }
  m["_test_tableau_ids"] = std::any(tableau_ids);

  const auto& noble_reqs = board_ai::splendor::splendor_nobles();
  std::vector<AnyMap> nobles;
  for (int ni = 0; ni < d.nobles_size; ++ni) {
    AnyMap nm;
    int noble_id = d.nobles[ni];
    nm["noble_id"] = std::any(noble_id);
    if (noble_id >= 0 && noble_id < static_cast<int>(noble_reqs.size())) {
      std::vector<int> req(board_ai::splendor::kColorCount);
      for (int j = 0; j < board_ai::splendor::kColorCount; ++j) req[j] = noble_reqs[noble_id][j];
      nm["requirements"] = std::any(req);
    }
    nobles.push_back(std::move(nm));
  }
  m["nobles"] = std::any(nobles);

  return m;
}

template <int NPlayers>
AnyMap describe_splendor(ActionId action) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action == Cfg::kPassAction) {
    m["type"] = std::any(std::string("pass"));
  } else if (action >= Cfg::kReturnTokenOffset && action < Cfg::kReturnTokenOffset + Cfg::kReturnTokenCount) {
    int token = action - Cfg::kReturnTokenOffset;
    m["type"] = std::any(std::string("return_token"));
    m["token"] = std::any(token);
    m["token_name"] = std::any(std::string(kColorNames[token]));
  } else if (action >= Cfg::kChooseNobleOffset && action < Cfg::kChooseNobleOffset + Cfg::kChooseNobleCount) {
    m["type"] = std::any(std::string("choose_noble"));
    m["noble_slot"] = std::any(action - Cfg::kChooseNobleOffset);
  } else if (action >= Cfg::kBuyFaceupOffset && action < Cfg::kBuyFaceupOffset + Cfg::kBuyFaceupCount) {
    int idx = action - Cfg::kBuyFaceupOffset;
    m["type"] = std::any(std::string("buy_faceup"));
    m["tier"] = std::any(idx / 4);
    m["slot"] = std::any(idx % 4);
  } else if (action >= Cfg::kReserveFaceupOffset && action < Cfg::kReserveFaceupOffset + Cfg::kReserveFaceupCount) {
    int idx = action - Cfg::kReserveFaceupOffset;
    m["type"] = std::any(std::string("reserve_faceup"));
    m["tier"] = std::any(idx / 4);
    m["slot"] = std::any(idx % 4);
  } else if (action >= Cfg::kReserveDeckOffset && action < Cfg::kReserveDeckOffset + Cfg::kReserveDeckCount) {
    m["type"] = std::any(std::string("reserve_deck"));
    m["tier"] = std::any(action - Cfg::kReserveDeckOffset);
  } else if (action >= Cfg::kBuyReservedOffset && action < Cfg::kBuyReservedOffset + Cfg::kBuyReservedCount) {
    m["type"] = std::any(std::string("buy_reserved"));
    m["slot"] = std::any(action - Cfg::kBuyReservedOffset);
  } else if (action >= Cfg::kTakeThreeOffset && action < Cfg::kTakeThreeOffset + Cfg::kTakeThreeCount) {
    m["type"] = std::any(std::string("take_three"));
    int combo = action - Cfg::kTakeThreeOffset;
    int idx = 0;
    std::vector<int> colors;
    for (int a = 0; a < 5 && static_cast<int>(colors.size()) < 3; ++a)
      for (int b = a + 1; b < 5 && static_cast<int>(colors.size()) < 3; ++b)
        for (int c = b + 1; c < 5; ++c) {
          if (idx == combo) { colors = {a, b, c}; goto done_t3; }
          ++idx;
        }
    done_t3:
    m["colors"] = std::any(colors);
  } else if (action >= Cfg::kTakeTwoDifferentOffset && action < Cfg::kTakeTwoDifferentOffset + Cfg::kTakeTwoDifferentCount) {
    m["type"] = std::any(std::string("take_two_different"));
    int combo = action - Cfg::kTakeTwoDifferentOffset;
    int idx = 0;
    std::vector<int> colors;
    for (int a = 0; a < 5; ++a)
      for (int b = a + 1; b < 5; ++b) {
        if (idx == combo) { colors = {a, b}; goto done_t2d; }
        ++idx;
      }
    done_t2d:
    m["colors"] = std::any(colors);
  } else if (action >= Cfg::kTakeOneOffset && action < Cfg::kTakeOneOffset + Cfg::kTakeOneCount) {
    m["type"] = std::any(std::string("take_one"));
    m["color"] = std::any(action - Cfg::kTakeOneOffset);
  } else if (action >= Cfg::kTakeTwoSameOffset && action < Cfg::kTakeTwoSameOffset + Cfg::kTakeTwoSameCount) {
    m["type"] = std::any(std::string("take_two_same"));
    m["color"] = std::any(action - Cfg::kTakeTwoSameOffset);
  }
  return m;
}

// --- Public-event protocol -------------------------------------------------
//
// Splendor has two hidden-state touchpoints during play:
//   (1) Buying or reserving a face-up card auto-flips a new card from the
//       corresponding tier's deck onto the same tableau slot. The new card's
//       ID is public — every player at the table sees the flip.
//   (2) Reserving from the top of a deck (a "blind reserve"): the drawer
//       sees their new face-down card; other players don't.
//
// Event schema:
//   post-action "deck_flip": {"tier", "slot", "card_id"} — emitted for each
//     tableau slot whose card changed after the action. card_id = -1 means
//     the slot is now empty (deck exhausted).
//   post-action "self_reserve_deck": {"slot", "card_id"} — emitted only when
//     the acting player is the traced perspective AND the action was a
//     reserve-from-deck. Overrides the AI's randomly-drawn card with ground
//     truth's actual card so the belief tracker's seen_cards agrees.
//   (Opponent reserve-from-deck needs no event — AI's random card in the
//    face-down slot is belief-consistent, since AI never learns what
//    opponent reserved blindly.)

namespace splendor_events {

using board_ai::splendor::SplendorConfig;
using board_ai::splendor::SplendorData;
using board_ai::splendor::SplendorState;
using board_ai::splendor::SplendorPersistentNode;
using board_ai::splendor::SplendorPersistentState;

// Safe helper: materialize the state's persistent data as mutable, mutate
// it via a functor, and reseat state.persistent to point at the new data.
template <int NPlayers>
void mutate_persistent(SplendorState<NPlayers>& s,
                       const std::function<void(SplendorData<NPlayers>&)>& fn) {
  SplendorData<NPlayers> data = s.persistent.data();
  fn(data);
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  s.persistent = SplendorPersistentState<NPlayers>(std::move(node));
  s.undo_stack.clear();
}

template <int NPlayers>
std::vector<PublicEvent> extract_events_only(
    const IGameState& before,
    ActionId action,
    const IGameState& after,
    int perspective) {
  using Cfg = SplendorConfig<NPlayers>;
  const auto& sb = board_ai::checked_cast<SplendorState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<SplendorState<NPlayers>>(after);
  const auto& db = sb.persistent.data();
  const auto& da = sa.persistent.data();
  std::vector<PublicEvent> events;

  const int actor = db.current_player;

  // opp_buy_reserved_reveal (pre): when actor != perspective buys their
  // own hidden reserved card (action 27/28/29 on a reserved_visible==0
  // slot), the card ID becomes public because its cost is paid from the
  // bank. The API side had a sampled placeholder for this card — we must
  // override it to the true card ID BEFORE do_action_fast so that
  // pay_for_card computes the correct payment.
  if (action >= Cfg::kBuyReservedOffset && action < Cfg::kBuyReservedOffset + Cfg::kBuyReservedCount &&
      actor != perspective && actor >= 0 && actor < NPlayers) {
    const int idx = action - Cfg::kBuyReservedOffset;
    if (idx < db.reserved_size[actor] &&
        db.reserved_visible[actor][static_cast<size_t>(idx)] == 0) {
      AnyMap payload;
      payload["player"] = std::any(actor);
      payload["slot"] = std::any(idx);
      payload["card_id"] = std::any(static_cast<int>(db.reserved[actor][static_cast<size_t>(idx)]));
      events.emplace_back("opp_buy_reserved_reveal", std::move(payload));
    }
  }

  // reserved_card_bought: any buy-from-reserve, for every perspective.
  // Buying puts the card face-up in front of the buyer, so its id is public
  // to the whole table. The belief tracker needs it to retire the card from
  // the unseen pool; otherwise a bought blind reserve flows back into the
  // sampled decks.
  if (action >= Cfg::kBuyReservedOffset && action < Cfg::kBuyReservedOffset + Cfg::kBuyReservedCount &&
      actor >= 0 && actor < NPlayers) {
    const int idx = action - Cfg::kBuyReservedOffset;
    if (idx < db.reserved_size[actor]) {
      AnyMap payload;
      payload["player"] = std::any(actor);
      payload["slot"] = std::any(idx);
      payload["card_id"] = std::any(static_cast<int>(db.reserved[actor][static_cast<size_t>(idx)]));
      events.emplace_back("reserved_card_bought", std::move(payload));
    }
  }

  // deck_flip: any tableau slot whose card changed is a public flip.
  for (int t = 0; t < 3; ++t) {
    const int max_slot = std::max<int>(db.tableau_size[t], da.tableau_size[t]);
    for (int slot = 0; slot < max_slot; ++slot) {
      const int before_id = (slot < db.tableau_size[t]) ? static_cast<int>(db.tableau[t][slot]) : -1;
      const int after_id = (slot < da.tableau_size[t]) ? static_cast<int>(da.tableau[t][slot]) : -1;
      if (before_id != after_id) {
        AnyMap payload;
        payload["tier"] = std::any(t);
        payload["slot"] = std::any(slot);
        payload["card_id"] = std::any(after_id);
        events.emplace_back("deck_flip", std::move(payload));
      }
    }
  }

  // self_reserve_deck: emitted only when perspective reserved from deck top.
  const bool is_reserve_deck = action >= Cfg::kReserveDeckOffset &&
      action < Cfg::kReserveDeckOffset + Cfg::kReserveDeckCount;
  if (is_reserve_deck && actor == perspective) {
    // Find the new reserved slot (one whose card changed from -1 to a valid ID).
    for (int slot = 0; slot < 3; ++slot) {
      const int before_id = (slot < db.reserved_size[actor]) ? static_cast<int>(db.reserved[actor][slot]) : -1;
      const int after_id = (slot < da.reserved_size[actor]) ? static_cast<int>(da.reserved[actor][slot]) : -1;
      if (before_id != after_id) {
        AnyMap payload;
        payload["player"] = std::any(actor);
        payload["slot"] = std::any(slot);
        payload["card_id"] = std::any(after_id);
        events.emplace_back("self_reserve_deck", std::move(payload));
      }
    }
  }

  return events;
}

}  // namespace splendor_events


// `token_encoder` selects SplendorTokenEncoder (transformer) over the flat
// SplendorFeatureEncoder (MLP). The pv_512x4 guidance model is an MLP, so
// in token variants it gets its own flat encoder and loads its config
// under the matching flat variant id.
template <int NPlayers>
board_ai::GameBundle make_splendor(const std::string& game_id, std::uint64_t seed,
                                   bool token_encoder = false) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<board_ai::splendor::SplendorState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<board_ai::splendor::SplendorRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  const board_ai::IFeatureEncoder* pv_encoder = nullptr;
  std::string pv_config_id = game_id;
  if (token_encoder) {
    b.encoder = std::make_unique<board_ai::splendor::SplendorTokenEncoder<NPlayers>>();
    b.heuristic_encoders.push_back(
        std::make_unique<board_ai::splendor::SplendorFeatureEncoder<NPlayers>>());
    pv_encoder = b.heuristic_encoders.back().get();
    pv_config_id = NPlayers == 2 ? std::string("splendor")
                                 : "splendor_" + std::to_string(NPlayers) + "p";
  } else {
    b.encoder = std::make_unique<board_ai::splendor::SplendorFeatureEncoder<NPlayers>>();
    pv_encoder = b.encoder.get();
  }
  b.belief_tracker = std::make_unique<board_ai::splendor::SplendorBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_splendor<NPlayers>;
  b.action_descriptor = describe_splendor<NPlayers>;
  b.heuristics.register_named(
      "rule_v1",
      std::make_unique<board_ai::splendor::SplendorRuleHeuristic<NPlayers>>("rule_v1"));
  // pv_512x4: ONNX-backed PV+MCTS heuristic. MCTS knobs and model path come
  // from heuristic.json::named.pv_512x4.
  {
    const auto hc = board_ai::heuristic::load_mcts_pv_heuristic_config(
        pv_config_id, "pv_512x4", /*require_model_path=*/true);
    b.heuristics.register_named(
        "pv_512x4",
        board_ai::heuristic::MctsPVHeuristic::make_from_onnx(
            "pv_512x4", hc.model_path, pv_encoder,
            b.value_model.get(), hc.simulations, hc.c_puct, hc.world_cap,
            hc.leaf_batch_size));
  }
  b.install_event_protocol(
      splendor_events::extract_events_only<NPlayers>,
      []() -> const board_ai::viz::VisibilitySchema& {
        return board_ai::splendor::SplendorState<NPlayers>::schema();
      });

  b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();
  // SplendorRules::do_action_deterministic sets forced_draw_override = -2
  // so replenishing the visible tier skips drawing from the hidden deck.
  // Tail solver never consumes hidden chance outcomes → safe.
  b.stochastic_tail_solve_safe = true;

  b.tail_solve_trigger = [](const board_ai::IGameState& state, int /*ply*/) -> bool {
    const auto& s = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state);
    const auto& d = s.persistent.data();
    for (int p = 0; p < NPlayers; ++p) {
      if (d.player_points[static_cast<size_t>(p)] >= 10) return true;
    }
    return false;
  };

  // Tail-solve tiebreak among proven wins. Standard recipe:
  // `depth_remaining + 0.1 * normalized_margin`.
  //   - depth_remaining as primary: prefer faster wins.
  //   - normalized prestige margin as 1/10 weighting: tiebreak on bigger margin.
  // Tail solver multiplies the whole thing by `auxiliary_weight` (typ. 0.01)
  // so the bonus stays << 1.0 and never lifts a non-terminal cutoff to
  // proven-win range. Margin normaliser 15 = win threshold (prestige cap).
  b.auxiliary_scorer = [](const board_ai::IGameState& state, int player,
                          int depth_remaining) -> float {
    const auto& s = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state);
    const auto& d = s.persistent.data();
    int my = static_cast<int>(d.player_points[static_cast<size_t>(player)]);
    int max_opp = 0;
    for (int p = 0; p < NPlayers; ++p) {
      if (p == player) continue;
      int op = static_cast<int>(d.player_points[static_cast<size_t>(p)]);
      if (op > max_opp) max_opp = op;
    }
    const float margin = static_cast<float>(my - max_opp) / 15.0f;
    return static_cast<float>(depth_remaining) + 0.1f * margin;
  };

  b.episode_stats_extractor = [](
      const board_ai::IGameState& final_state,
      const board_ai::IGameState& /*initial_state*/,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    int main_actions = 0;
    for (const auto& s : samples) {
      if (s.action_id < Cfg::kChooseNobleOffset || s.action_id == Cfg::kPassAction) {
        ++main_actions;
      }
    }
    double turns = static_cast<double>(main_actions) / NPlayers;
    return {{"turns", turns}};
  };

  b.final_score_extractor = [](const board_ai::IGameState& state) -> std::vector<float> {
    const auto& d = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state)
                        .persistent.data();
    std::vector<float> points(NPlayers);
    for (int p = 0; p < NPlayers; ++p) {
      points[static_cast<size_t>(p)] = static_cast<float>(d.player_points[static_cast<size_t>(p)]);
    }
    return points;
  };

  return b;
}

board_ai::GameRegistrar reg_splendor("splendor", [](std::uint64_t seed) {
  return make_splendor<2>("splendor", seed);
});
board_ai::GameRegistrar reg_splendor_2p("splendor_2p", [](std::uint64_t seed) {
  return make_splendor<2>("splendor_2p", seed);
});
board_ai::GameRegistrar reg_splendor_3p("splendor_3p", [](std::uint64_t seed) {
  return make_splendor<3>("splendor_3p", seed);
});
board_ai::GameRegistrar reg_splendor_4p("splendor_4p", [](std::uint64_t seed) {
  return make_splendor<4>("splendor_4p", seed);
});
board_ai::GameRegistrar reg_splendor_tok("splendor_tok", [](std::uint64_t seed) {
  return make_splendor<2>("splendor_tok", seed, /*token_encoder=*/true);
});

}  // namespace
