#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "loveletter_state.h"
#include "loveletter_rules.h"
#include "loveletter_net_adapter.h"
#include "heuristic/loveletter_rule_heuristic.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kCardNames[9] = {
    "", "Guard", "Priest", "Baron", "Handmaid", "Prince", "King", "Countess", "Princess"};

template <int NPlayers>
AnyMap serialize_loveletter(const IGameState& state) {
  using namespace board_ai::loveletter;
  const auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["ply"] = std::any(d.ply);
  // The public counter, not a sum over the hidden per-type deck counts
  // (those are placeholders on a masked / viewer-path state).
  m["deck_size"] = std::any(static_cast<int>(d.deck_size));

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    pm["alive"] = std::any(static_cast<bool>(d.alive[p]));
    pm["protected"] = std::any(static_cast<bool>(d.protected_flags[p]));
    pm["hand"] = std::any(static_cast<int>(d.hand[p]));
    pm["hand_name"] = std::any(std::string(
        d.hand[p] >= 1 && d.hand[p] <= 8 ? kCardNames[d.hand[p]] : ""));
    // Seats that currently see this hand card (owner, Priest peek, Baron
    // tie, King swap, showdown) — the runtime viz row, for tests and UI.
    {
      const auto& hand_viz = board_ai::viz::viz_get(state, "hand");
      const std::size_t base = board_ai::viz::flat_offset_data_only(
          hand_viz.shape, std::vector<int>{p});
      std::vector<int> seen_by;
      for (int v = 0; v < NPlayers; ++v) {
        if (hand_viz.data[base + static_cast<std::size_t>(v)] != 0) {
          seen_by.push_back(v);
        }
      }
      pm["hand_seen_by"] = std::any(seen_by);
    }

    // Discard counts (per card type 1..kCardTypes); UI reconstructs
    // visual order from the action stream. Ship the multiset count
    // vector indexed 0..kCardTypes (slot 0 is unused but kept for
    // alignment with card values).
    std::vector<int> discard_counts;
    discard_counts.reserve(board_ai::loveletter::kCardTypes + 1);
    for (int c = 0; c <= board_ai::loveletter::kCardTypes; ++c) {
      discard_counts.push_back(static_cast<int>(
          d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)]));
    }
    pm["discard_count"] = std::any(discard_counts);

    // Display-side derived list: enumerate every count copy as a flat
    // sequence in card-type order. Tests and any "show me a pile"
    // consumers iterate this; engine-side hash / encoder do not.
    std::vector<int> discards_flat;
    for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
      int n = d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)];
      for (int i = 0; i < n; ++i) discards_flat.push_back(c);
    }
    pm["discards"] = std::any(discards_flat);

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  m["drawn_card"] = std::any(static_cast<int>(d.drawn_card));
  m["drawn_card_name"] = std::any(std::string(
      d.drawn_card >= 1 && d.drawn_card <= 8 ? kCardNames[d.drawn_card] : ""));

  std::vector<int> face_up_count_v;
  face_up_count_v.reserve(board_ai::loveletter::kCardTypes + 1);
  for (int c = 0; c <= board_ai::loveletter::kCardTypes; ++c) {
    face_up_count_v.push_back(static_cast<int>(
        d.face_up_count[static_cast<size_t>(c)]));
  }
  m["face_up_count"] = std::any(face_up_count_v);

  // Display-side derived list (face-up burn pile, 2p only).
  std::vector<int> face_up_flat;
  for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
    int n = d.face_up_count[static_cast<size_t>(c)];
    for (int i = 0; i < n; ++i) face_up_flat.push_back(c);
  }
  m["face_up_removed"] = std::any(face_up_flat);

  // The "set-aside" card is the topmost card removed from the deck at game
  // start (always exactly 1 — Love Letter rule). It's hidden from all
  // players in normal play but exposed here for tile-conservation
  // invariants in test suites. Belief tracker / encoder must NOT read it
  // (they don't — both go through the public API only).
  m["set_aside_card"] = std::any(static_cast<int>(d.set_aside_card));

  return m;
}

AnyMap describe_loveletter(ActionId action) {
  using namespace board_ai::loveletter;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action >= kGuardOffset && action < kGuardOffset + kGuardCount) {
    int idx = action - kGuardOffset;
    int target = idx / 7;
    int guess = idx % 7 + 2;
    m["type"] = std::any(std::string("guard"));
    m["card"] = std::any(1);
    m["card_name"] = std::any(std::string("Guard"));
    m["target"] = std::any(target);
    m["guess"] = std::any(guess);
    m["guess_name"] = std::any(std::string(
        guess >= 1 && guess <= 8 ? kCardNames[guess] : ""));
  } else if (action >= kPriestOffset && action < kPriestOffset + kPriestCount) {
    m["type"] = std::any(std::string("priest"));
    m["card"] = std::any(2);
    m["card_name"] = std::any(std::string("Priest"));
    m["target"] = std::any(action - kPriestOffset);
  } else if (action >= kBaronOffset && action < kBaronOffset + kBaronCount) {
    m["type"] = std::any(std::string("baron"));
    m["card"] = std::any(3);
    m["card_name"] = std::any(std::string("Baron"));
    m["target"] = std::any(action - kBaronOffset);
  } else if (action == kHandmaidAction) {
    m["type"] = std::any(std::string("handmaid"));
    m["card"] = std::any(4);
    m["card_name"] = std::any(std::string("Handmaid"));
  } else if (action >= kPrinceOffset && action < kPrinceOffset + kPrinceCount) {
    m["type"] = std::any(std::string("prince"));
    m["card"] = std::any(5);
    m["card_name"] = std::any(std::string("Prince"));
    m["target"] = std::any(action - kPrinceOffset);
  } else if (action >= kKingOffset && action < kKingOffset + kKingCount) {
    m["type"] = std::any(std::string("king"));
    m["card"] = std::any(6);
    m["card_name"] = std::any(std::string("King"));
    m["target"] = std::any(action - kKingOffset);
  } else if (action == kCountessAction) {
    m["type"] = std::any(std::string("countess"));
    m["card"] = std::any(7);
    m["card_name"] = std::any(std::string("Countess"));
  } else if (action == kPrincessAction) {
    m["type"] = std::any(std::string("princess"));
    m["card"] = std::any(8);
    m["card_name"] = std::any(std::string("Princess"));
  }

  return m;
}


// --- Public-event protocol -------------------------------------------------
//
// GT-side rules (do_action_fast) maintain `state.viz` as they reveal hidden
// slots: Priest → reveal_slot_to(actor), Baron → cross-reveal, King →
// swap_slot_owned, Prince → reset_to_base on the discarded slot, etc. The
// observer side learns those reveals via `public_snapshot` (walker copies
// every viz=1 slot) and via the `events` list below for tracker-only signals
// (discards, eliminations, winner announcement). No private payload rides on
// `events` — every fact the observer learns about hidden cards flows through
// the schema-driven snapshot path.

namespace loveletter_events {

using board_ai::loveletter::LoveLetterConfig;
using board_ai::loveletter::LoveLetterState;
using board_ai::loveletter::LoveLetterData;
using board_ai::loveletter::kGuard;
using board_ai::loveletter::kPriest;
using board_ai::loveletter::kBaron;
using board_ai::loveletter::kHandmaid;
using board_ai::loveletter::kPrince;
using board_ai::loveletter::kKing;
using board_ai::loveletter::kCountess;
using board_ai::loveletter::kPrincess;
using board_ai::loveletter::kGuardOffset;
using board_ai::loveletter::kGuardCount;
using board_ai::loveletter::kPriestOffset;
using board_ai::loveletter::kPriestCount;
using board_ai::loveletter::kBaronOffset;
using board_ai::loveletter::kBaronCount;
using board_ai::loveletter::kHandmaidAction;
using board_ai::loveletter::kPrinceOffset;
using board_ai::loveletter::kPrinceCount;
using board_ai::loveletter::kKingOffset;
using board_ai::loveletter::kKingCount;
using board_ai::loveletter::kCountessAction;
using board_ai::loveletter::kPrincessAction;

// LoveLetter emits no public events at all. Everything the tracker
// learns flows through `init`'s pack_init_payload + state viz=1 reveals
// (Priest peeks etc., handled by rules, not events). The wire-protocol
// extractor and applier are auto-derived by GameBundle::install_event_protocol.

}  // namespace loveletter_events

template <int NPlayers>
board_ai::GameBundle make_loveletter(const std::string& game_id, std::uint64_t seed) {
  using namespace board_ai::loveletter;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<LoveLetterState<NPlayers>>();
  s->reset_with_seed(seed);
  // Apply start-of-game viz reveals (rules are sole viz writer per I1):
  // the seat starting as current_player physically holds the top
  // card just drawn from the deck — reveal `drawn_card` to them only.
  LoveLetterRules<NPlayers>::reveal_starting_draw(*s, s->data.current_player);
  b.state = std::move(s);
  b.rules = std::make_unique<LoveLetterRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<LoveLetterFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<LoveLetterBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_loveletter<NPlayers>;
  b.action_descriptor = describe_loveletter;
  b.heuristics.register_named(
      "rule_v1",
      std::make_unique<board_ai::loveletter::LoveLetterRuleHeuristic<NPlayers>>("rule_v1"));
  b.install_event_protocol(
      board_ai::viz::no_events_extractor,
      []() -> const board_ai::viz::VisibilitySchema& {
        return LoveLetterState<NPlayers>::schema();
      });

  // Episode stats: Guard guess accuracy when the target's hand is visible
  // (viz=1 to the actor — e.g. after a Priest peek) and the actual card is
  // not Guard itself. Reads each Guard sample's encoded features; the
  // perspective is sample.player so target's hand maps to the per-player
  // hand one-hot segment in features.
  //
  // Feature layout for NPlayers=N (kCardTypes=8):
  //   - 13*N public per-player + 12 global = 13N + 12 features
  //   - then per-player hand(8) + drawn(8) = 16N more, ordered by
  //     pi = (target_abs - perspective + N) % N
  // Hand-one-hot start for pi: (13*N + 12) + pi*16, length 8.
  b.episode_stats_extractor = [](
      const board_ai::IGameState& /*final_state*/,
      const board_ai::IGameState& /*initial_state*/,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    using namespace board_ai::loveletter;
    int eligible = 0;  // Guard plays where target hand is viz=1 and != Guard
    int correct = 0;
    const int kHandSegStart = 13 * NPlayers + 12;
    for (const auto& s : samples) {
      const ActionId a = s.action_id;
      if (a < kGuardOffset || a >= kGuardOffset + kGuardCount) continue;
      const int idx = a - kGuardOffset;
      const int target_abs = idx / 7;
      const int guess = (idx % 7) + 2;  // 2..8
      if (target_abs == s.player) continue;  // self-target degenerate slot
      if (!s.features) continue;
      const auto& feats = *s.features;
      const int pi = (target_abs - s.player + NPlayers) % NPlayers;
      const int seg = kHandSegStart + pi * 16;
      if (seg + 8 > static_cast<int>(feats.size())) continue;
      // Find the unique 1.0 in the 8-d one-hot. sum=0 → viz=0 (placeholder),
      // skip. sum=1 → viz=1, card_id = argmax+1.
      int hot_idx = -1;
      for (int c = 0; c < 8; ++c) {
        if (feats[seg + c] > 0.5f) {
          if (hot_idx >= 0) { hot_idx = -2; break; }  // multiple hot bits
          hot_idx = c;
        }
      }
      if (hot_idx < 0) continue;  // viz=0 or malformed
      const int target_card = hot_idx + 1;  // 1..8
      if (target_card == kGuard) continue;  // can't guess Guard
      ++eligible;
      if (guess == target_card) ++correct;
    }
    std::map<std::string, double> out;
    out["guard_known_attempts"] = static_cast<double>(eligible);
    if (eligible > 0) {
      out["guard_known_correct_rate"] =
          static_cast<double>(correct) / static_cast<double>(eligible);
    }
    return out;
  };

  return b;
}

board_ai::GameRegistrar reg_loveletter("loveletter", [](std::uint64_t seed) {
  return make_loveletter<2>("loveletter", seed);
});
board_ai::GameRegistrar reg_loveletter_2p("loveletter_2p", [](std::uint64_t seed) {
  return make_loveletter<2>("loveletter_2p", seed);
});
board_ai::GameRegistrar reg_loveletter_3p("loveletter_3p", [](std::uint64_t seed) {
  return make_loveletter<3>("loveletter_3p", seed);
});
board_ai::GameRegistrar reg_loveletter_4p("loveletter_4p", [](std::uint64_t seed) {
  return make_loveletter<4>("loveletter_4p", seed);
});

}  // namespace
