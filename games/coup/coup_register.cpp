#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../heuristic/random_heuristic.h"
#include "coup_state.h"
#include "coup_rules.h"
#include "coup_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kCharNames[5] = {
    "Duke", "Assassin", "Captain", "Ambassador", "Contessa"};

template <int NPlayers>
AnyMap serialize_coup(const IGameState& state) {
  using namespace board_ai::coup;
  const auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["ply"] = std::any(d.ply);
  m["stage"] = std::any(static_cast<int>(d.stage));
  m["active_player"] = std::any(d.active_player);
  m["declared_action"] = std::any(static_cast<int>(d.declared_action));
  m["action_target"] = std::any(d.action_target);
  m["blocker"] = std::any(d.blocker);
  m["challenger"] = std::any(d.challenger);
  m["claimed_character"] = std::any(static_cast<int>(d.claimed_character));
  m["block_character"] = std::any(static_cast<int>(d.block_character));
  m["deck_size"] = std::any(static_cast<int>(d.deck_size));

  // Exchange-drawn cards (Ambassador). Only exposed during the two
  // exchange-return stages; outside those stages the field is
  // default-zero and should not be confused with an actual drawn Duke.
  // -1 means "slot already emptied during Return1". Frontend renders
  // these as extra cards in the active player's hand during the return
  // stages so the player sees 2 hand + 2 drawn = 4 cards and can click
  // one to return.
  std::vector<int> exchange_drawn;
  if (d.stage == CoupStage::kExchangeReturn1 ||
      d.stage == CoupStage::kExchangeReturn2) {
    exchange_drawn.reserve(2);
    for (int i = 0; i < 2; ++i) {
      exchange_drawn.push_back(static_cast<int>(d.exchange_drawn[i]));
    }
  }
  m["exchange_drawn"] = std::any(exchange_drawn);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    pm["alive"] = std::any(static_cast<bool>(d.alive[p]));
    pm["coins"] = std::any(d.coins[p]);

    std::vector<AnyMap> influences;
    for (int sl = 0; sl < 2; ++sl) {
      AnyMap inf;
      inf["character"] = std::any(static_cast<int>(d.influence[p][sl]));
      inf["character_name"] = std::any(std::string(
          d.influence[p][sl] >= 0 && d.influence[p][sl] < 5
              ? kCharNames[d.influence[p][sl]] : ""));
      inf["revealed"] = std::any(static_cast<bool>(d.revealed[p][sl]));
      influences.push_back(std::move(inf));
    }
    pm["influences"] = std::any(influences);

    // Seat p's certain knowledge of the court deck (CoupData::deck_known)
    // — like deck_count below, exposed for conservation checks in tests;
    // the AI path reads it through the masked state only.
    std::vector<int> deck_known;
    for (int c = 0; c < kCharacterCount; ++c) {
      deck_known.push_back(static_cast<int>(d.deck_known[p][c]));
    }
    pm["deck_known"] = std::any(deck_known);
    std::vector<int> deck_known_taint;
    for (int q = 0; q < NPlayers; ++q) {
      for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
        deck_known_taint.push_back(
            static_cast<int>(d.deck_known_taint[p][q][sl]));
      }
    }
    pm["deck_known_taint"] = std::any(deck_known_taint);

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  // Court deck composition per character. Hidden from every player in
  // normal play; exposed here for card-conservation invariants in tests.
  std::vector<int> deck_count;
  for (int c = 0; c < kCharacterCount; ++c) {
    deck_count.push_back(static_cast<int>(d.deck_count[c]));
  }
  m["deck_count"] = std::any(deck_count);

  return m;
}

AnyMap describe_coup(ActionId action) {
  using namespace board_ai::coup;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action == kIncomeAction) {
    m["type"] = std::any(std::string("income"));
  } else if (action == kForeignAidAction) {
    m["type"] = std::any(std::string("foreign_aid"));
  } else if (action >= kCoupOffset && action < kCoupOffset + kCoupCount) {
    m["type"] = std::any(std::string("coup"));
    m["target"] = std::any(action - kCoupOffset);
  } else if (action == kTaxAction) {
    m["type"] = std::any(std::string("tax"));
    m["claimed"] = std::any(std::string("Duke"));
  } else if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) {
    m["type"] = std::any(std::string("assassinate"));
    m["target"] = std::any(action - kAssassinateOffset);
    m["claimed"] = std::any(std::string("Assassin"));
  } else if (action >= kStealOffset && action < kStealOffset + kStealCount) {
    m["type"] = std::any(std::string("steal"));
    m["target"] = std::any(action - kStealOffset);
    m["claimed"] = std::any(std::string("Captain"));
  } else if (action == kExchangeAction) {
    m["type"] = std::any(std::string("exchange"));
    m["claimed"] = std::any(std::string("Ambassador"));
  } else if (action == kChallengeAction) {
    m["type"] = std::any(std::string("challenge"));
  } else if (action == kAllowAction) {
    m["type"] = std::any(std::string("allow"));
  } else if (action == kBlockDukeAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Duke"));
  } else if (action == kBlockContessaAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Contessa"));
  } else if (action == kBlockAmbassadorAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Ambassador"));
  } else if (action == kBlockCaptainAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Captain"));
  } else if (action == kAllowNoBlockAction) {
    m["type"] = std::any(std::string("allow_no_block"));
  } else if (action == kRevealSlot0 || action == kRevealSlot1) {
    m["type"] = std::any(std::string("reveal"));
    m["slot"] = std::any(action == kRevealSlot0 ? 0 : 1);
  } else if (action == kLoseSlot0 || action == kLoseSlot1) {
    m["type"] = std::any(std::string("lose_influence"));
    m["slot"] = std::any(action == kLoseSlot0 ? 0 : 1);
  } else if (action >= kReturnDuke && action <= kReturnContessa) {
    m["type"] = std::any(std::string("return_card"));
    int char_id = action - kReturnDuke;
    m["character"] = std::any(char_id);
    m["character_name"] = std::any(std::string(
        char_id >= 0 && char_id < 5 ? kCharNames[char_id] : ""));
  } else if (action == kHiddenReturnAction) {
    // What observers receive for an exchange return
    // (CoupRules::public_action): the character stays private.
    m["type"] = std::any(std::string("return_card"));
    m["hidden"] = std::any(true);
  }

  return m;
}

namespace coup_events {

// Events-only extractor — diffs (before, after) and emits a typed event
// stream that CoupBeliefTracker consumes to maintain its claim/role
// signals. Public-state replication on the wire is handled by the
// framework's `viz::serialize_public_snapshot` (auto-derived inside
// `GameBundle::install_event_protocol`); we never serialize a snapshot
// here. Sim path consumes only this events vector — so it's the cheap
// path that keeps sim_tracker in sync with descent.
//
// Events emitted (all payloads are int / bool only; no truth leak):
//   - card_revealed { player, slot, role }
//       Any slot whose `revealed` flag flipped false→true. Covers
//       lose-influence (Coup / failed action / failed block / counter
//       lost) and bluff-caught reveals.
//   - claim_resolved_truthful { claimer, role }
//       In kResolveChallengeAction the claimer revealed the claimed
//       role, so they're holding it for real → card reshuffles back to
//       deck. Detected by none of the claimer's slots turning face up.
//   - block_resolved_truthful { blocker, role }
//       In kResolveChallengeCounter the blocker revealed the block
//       role; mirror of the above.
//   - claim_unchallenged { claimer, role }
//       Challenge phase ended without anyone challenging. Detected by
//       transition out of kChallengeAction without entering
//       kResolveChallengeAction.
//   - block_unchallenged { blocker, role }
//       Counter-challenge phase ended without anyone challenging.
//   - exchange_complete { player }
//       Transition out of kExchangeReturn2 (Ambassador draws done,
//       hand reshuffled — all prior claim signals on this player
//       are stale).
template <int NPlayers>
std::vector<board_ai::PublicEvent> extract_events_only(
    const IGameState& before,
    ActionId /*action*/,
    const IGameState& after,
    int /*perspective*/) {
  using namespace board_ai::coup;
  const auto& sb = board_ai::checked_cast<CoupState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<CoupState<NPlayers>>(after);
  const auto& db = sb.data;
  const auto& da = sa.data;
  std::vector<board_ai::PublicEvent> events;

  // 1. card_revealed: any (p, slot) whose `revealed` flipped false→true.
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!db.revealed[p][sl] && da.revealed[p][sl]) {
        AnyMap payload;
        payload["player"] = std::any(p);
        payload["slot"] = std::any(sl);
        payload["role"] = std::any(static_cast<int>(da.influence[p][sl]));
        events.emplace_back("card_revealed", std::move(payload));
      }
    }
  }

  // A challenged player who did not hold the role loses the revealed card
  // in the same step; one who did shows it and redraws, so none of their
  // slots turns face up. The *_challenge_succeeded flags cannot be used:
  // a caught bluff ends the turn and advance_turn clears them.
  auto lost_a_card = [&](int p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!db.revealed[p][sl] && da.revealed[p][sl]) return true;
    }
    return false;
  };

  // 2. claim_resolved_truthful: kResolveChallengeAction → the claimer
  //    showed the claimed role. before.claimed_character holds the role.
  if (db.stage == CoupStage::kResolveChallengeAction &&
      da.stage != CoupStage::kResolveChallengeAction) {
    if (!lost_a_card(db.active_player) && db.claimed_character >= 0) {
      AnyMap payload;
      payload["claimer"] = std::any(db.active_player);
      payload["role"] = std::any(static_cast<int>(db.claimed_character));
      events.emplace_back("claim_resolved_truthful", std::move(payload));
    }
  }

  // 3. block_resolved_truthful: kResolveChallengeCounter → the blocker
  //    showed the block role.
  if (db.stage == CoupStage::kResolveChallengeCounter &&
      da.stage != CoupStage::kResolveChallengeCounter) {
    if (!lost_a_card(db.blocker) && db.block_character >= 0) {
      AnyMap payload;
      payload["blocker"] = std::any(db.blocker);
      payload["role"] = std::any(static_cast<int>(db.block_character));
      events.emplace_back("block_resolved_truthful", std::move(payload));
    }
  }

  // 4. claim_unchallenged: kChallengeAction → out without entering
  //    kResolveChallengeAction (everyone Allowed).
  if (db.stage == CoupStage::kChallengeAction &&
      da.stage != CoupStage::kChallengeAction &&
      da.stage != CoupStage::kResolveChallengeAction) {
    if (db.claimed_character >= 0) {
      AnyMap payload;
      payload["claimer"] = std::any(db.active_player);
      payload["role"] = std::any(static_cast<int>(db.claimed_character));
      events.emplace_back("claim_unchallenged", std::move(payload));
    }
  }

  // 5. block_unchallenged: kChallengeCounter → out without entering
  //    kResolveChallengeCounter.
  if (db.stage == CoupStage::kChallengeCounter &&
      da.stage != CoupStage::kChallengeCounter &&
      da.stage != CoupStage::kResolveChallengeCounter) {
    if (db.block_character >= 0) {
      AnyMap payload;
      payload["blocker"] = std::any(db.blocker);
      payload["role"] = std::any(static_cast<int>(db.block_character));
      events.emplace_back("block_unchallenged", std::move(payload));
    }
  }

  // 6. exchange_complete: kExchangeReturn2 → out.
  if (db.stage == CoupStage::kExchangeReturn2 &&
      da.stage != CoupStage::kExchangeReturn2) {
    AnyMap payload;
    payload["player"] = std::any(db.active_player);
    events.emplace_back("exchange_complete", std::move(payload));
  }

  return events;
}

}  // namespace coup_events

template <int NPlayers>
board_ai::GameBundle make_coup(const std::string& game_id, std::uint64_t seed) {
  using namespace board_ai::coup;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<CoupState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<CoupRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<CoupFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<CoupBeliefTracker<NPlayers>>();
  b.belief_feature_extractor =
      std::make_unique<CoupBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor =
      std::make_unique<CoupBeliefLabelExtractor<NPlayers>>();
  b.state_serializer = serialize_coup<NPlayers>;
  b.action_descriptor = describe_coup;
  b.heuristics.register_named(
      "rule", std::make_unique<board_ai::heuristic::RandomHeuristic>("rule"));
  b.install_event_protocol(
      coup_events::extract_events_only<NPlayers>,
      []() -> const board_ai::viz::VisibilitySchema& {
        return CoupState<NPlayers>::schema();
      });

  b.adjudicator = [](const board_ai::IGameState&) -> int { return -1; };

  // Per-step training stats. Three opening-Duke probabilities help spot
  // when the network learns coup's central bluff dynamic vs. drifts:
  //   opening_duke_rate           — P(seat0 plays Tax on ply 0)
  //   opening_duke_real_rate      — P(seat0 actually held Duke | played Tax)
  //                                  read from initial_state truth
  //   opening_duke_challenged_rate — P(seat1 challenges | seat0 played Tax)
  // The two conditional rates emit `nan` (silently, by skipping the key) on
  // steps where no episode satisfied the condition; pipeline averages over
  // present values only, so the headline metric stays meaningful.
  b.episode_stats_extractor = [](
      const board_ai::IGameState& /*final_state*/,
      const board_ai::IGameState& initial_state,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    using namespace board_ai::coup;
    std::map<std::string, double> out;
    if (samples.size() < 1) return out;
    const auto& init = board_ai::checked_cast<CoupState<NPlayers>>(initial_state);
    // Opening rate: did seat 0 play Tax (= claim Duke) on ply 0?
    const auto& s0 = samples[0];
    const bool opened_duke = (s0.ply == 0 && s0.player == 0 &&
                              s0.action_id == kTaxAction);
    out["opening_duke_rate"] = opened_duke ? 1.0 : 0.0;
    if (opened_duke) {
      // Conditional: did seat 0's initial hand actually contain Duke?
      bool has_duke = false;
      for (int sl = 0; sl < 2; ++sl) {
        if (init.data.influence[0][sl] == kDuke) {
          has_duke = true;
          break;
        }
      }
      out["opening_duke_real_rate"] = has_duke ? 1.0 : 0.0;
      // Conditional: did seat 1 challenge on ply 1?
      if (samples.size() >= 2) {
        const auto& s1 = samples[1];
        const bool challenged = (s1.ply == 1 && s1.player == 1 &&
                                 s1.action_id == kChallengeAction);
        out["opening_duke_challenged_rate"] = challenged ? 1.0 : 0.0;
      }
    }
    return out;
  };

  return b;
}

board_ai::GameRegistrar reg_coup("coup", [](std::uint64_t seed) {
  return make_coup<2>("coup", seed);
});
board_ai::GameRegistrar reg_coup_2p("coup_2p", [](std::uint64_t seed) {
  return make_coup<2>("coup_2p", seed);
});
board_ai::GameRegistrar reg_coup_3p("coup_3p", [](std::uint64_t seed) {
  return make_coup<3>("coup_3p", seed);
});
board_ai::GameRegistrar reg_coup_4p("coup_4p", [](std::uint64_t seed) {
  return make_coup<4>("coup_4p", seed);
});

}  // namespace
