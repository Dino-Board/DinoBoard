#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "6nimmit_state.h"
#include "6nimmit_rules.h"
#include "6nimmit_net_adapter.h"
#include "heuristic/sixnimmit_rule_heuristic.h"
#include "heuristic/sixnimmit_lkf_heuristic.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::IGameState;
using board_ai::PublicEvent;

// ---------------------------------------------------------------------
// Events-only extractor for the belief tracker. Emits one
// `card_played` event per pending_commits slot that the apply loop
// consumed in this transition — the card has been publicly played onto
// the table (either chained to a row tail, eaten the row, or replaced a
// row via SMALL_CHOOSE_ROW). A commit that stays face down (0 → card)
// and the seal (viz flip, pc unchanged) do not match.
//
// The seat that completes the round commits AND gets its card applied
// inside the same transition, so its slot reads 0 both before and after;
// its card is taken from the action instead. Without this the last seat
// to commit would only be recorded in rounds with a SMALL pause.
// ---------------------------------------------------------------------
template <int NPlayers>
std::vector<PublicEvent> extract_events_only_sixnimmit(
    const IGameState& before, ActionId action, const IGameState& after,
    int /*perspective*/) {
  using namespace board_ai::sixnimmt;
  const auto& sb = board_ai::checked_cast<SixNimmtState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<SixNimmtState<NPlayers>>(after);
  std::vector<PublicEvent> events;
  const bool is_commit = !sb.data.small_phase_active &&
                         action >= kCommitOffset &&
                         action < kCommitOffset + kNumCards;
  for (int p = 0; p < NPlayers; ++p) {
    int before_pc = static_cast<int>(
        sb.data.pending_commits[static_cast<size_t>(p)]);
    if (is_commit && p == sb.data.current_player) {
      before_pc = static_cast<int>(action) - kCommitOffset + 1;
    }
    int after_pc = static_cast<int>(
        sa.data.pending_commits[static_cast<size_t>(p)]);
    if (before_pc >= 1 && before_pc <= kNumCards && after_pc == 0) {
      AnyMap payload;
      payload["player"] = std::any(p);
      payload["card"] = std::any(before_pc);
      events.emplace_back("card_played", std::move(payload));
    }
  }
  return events;
}

// ---------------------------------------------------------------------
// State serializer — UI / replay payload.
// ---------------------------------------------------------------------
template <int NPlayers>
AnyMap serialize_sixnimmit(const IGameState& state) {
  using namespace board_ai::sixnimmt;
  const auto& s = board_ai::checked_cast<SixNimmtState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["ply"] = std::any(d.ply);
  m["current_round"] = std::any(d.current_round);
  m["small_phase_active"] = std::any(static_cast<bool>(d.small_phase_active));
  m["small_player"] = std::any(d.small_player);
  m["apply_cursor"] = std::any(d.apply_cursor);
  // "Seat p has committed this round" is public (its face-down card lies
  // on the table), but its only stored forms are the hidden pending card
  // and the non-schema `committed` mirror, neither of which a masked /
  // observer state carries. Derive it from public slots instead, so the
  // serializer gives the same answer on the truth state and on the
  // viewer-path probes (engine/core/viewer_state.h):
  //   commit phase — the cursor serves seats in ascending order from 0,
  //                  so exactly the seats below it have committed;
  //   SMALL pause  — the round is sealed: every seat has committed.
  // (A sealed round without a SMALL pause resolves inside one transition
  // and is never observed.)
  const bool sealed = !state.is_terminal() && d.small_phase_active;
  auto has_committed = [&](int p) {
    if (state.is_terminal()) return false;
    return sealed || p < d.current_player;
  };
  int num_committed = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (has_committed(p)) ++num_committed;
  }
  m["num_committed_this_round"] = std::any(num_committed);

  // Table: 4 rows × up-to-5 cells, value = card number (0 empty).
  std::vector<std::vector<int>> table_rows;
  table_rows.reserve(kNumRows);
  for (int r = 0; r < kNumRows; ++r) {
    std::vector<int> row;
    row.reserve(kRowCapacity);
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      row.push_back(static_cast<int>(
          d.table[static_cast<size_t>(r)][static_cast<size_t>(s_idx)]));
    }
    table_rows.push_back(std::move(row));
  }
  m["table"] = std::any(table_rows);

  // Per-player snapshots.
  std::vector<AnyMap> players;
  players.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;

    std::vector<int> hand_cards;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.hand[static_cast<size_t>(p)][static_cast<size_t>(c - 1)] == 1) {
        hand_cards.push_back(c);
      }
    }
    pm["hand"] = std::any(hand_cards);
    // Public: every seat starts a round with the same number of cards and
    // has one less once it has committed. Not counted from `hand`, which
    // is hidden from the other seats.
    pm["hand_count"] = std::any(std::max(
        0, kHandSize - d.current_round - (has_committed(p) ? 1 : 0)));

    std::vector<int> bullhead_cards;
    int heads_total = 0;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(p)]
                          [static_cast<size_t>(c - 1)] == 1) {
        bullhead_cards.push_back(c);
        heads_total += head_value(c);
      }
    }
    pm["bullhead_pile"] = std::any(bullhead_cards);
    pm["bullhead_count"] = std::any(heads_total);

    pm["committed"] = std::any(has_committed(p));
    // pending_card: only expose after seal (all players committed and
    // rules called reveal_slot on every pending_commits slot). Before
    // seal, pending_commits is owner_only — the serializer runs on truth
    // state and must not leak opponent cards.
    pm["pending_card"] = std::any(static_cast<int>(
        sealed ? d.pending_commits[static_cast<size_t>(p)] : 0));

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  // Apply queue back-compat shim for the web frontend (PR-1 boundary).
  // Only populated after seal when all pending_commits are public. Before
  // seal, all slots stay 0 to prevent leaking opponent cards through the
  // serializer (which runs on truth state).
  std::vector<std::pair<int, int>> revealed;
  revealed.reserve(NPlayers);
  if (sealed) {
    for (int p = 0; p < NPlayers; ++p) {
      int c = static_cast<int>(d.pending_commits[static_cast<size_t>(p)]);
      if (c >= 1 && c <= kNumCards) revealed.emplace_back(c, p);
    }
  }
  std::sort(revealed.begin(), revealed.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<int> apply_card_v(NPlayers, 0);
  std::vector<int> apply_player_v(NPlayers, 0);
  // Slots [0..apply_cursor) already applied → 0 in card_v (same shape
  // as the old behaviour where run_apply_loop zeroed consumed slots).
  // Slots [apply_cursor..num_committed_this_round) are the still-
  // pending revealed entries in ascending order.
  for (size_t i = 0; i < revealed.size(); ++i) {
    int slot = d.apply_cursor + static_cast<int>(i);
    if (slot < 0 || slot >= NPlayers) break;
    apply_card_v[static_cast<size_t>(slot)] = revealed[i].first;
    apply_player_v[static_cast<size_t>(slot)] = revealed[i].second;
  }
  m["apply_queue_card"] = std::any(apply_card_v);
  m["apply_queue_player"] = std::any(apply_player_v);

  return m;
}

// ---------------------------------------------------------------------
// Action descriptor — one-line summary per action.
// ---------------------------------------------------------------------
AnyMap describe_sixnimmit(ActionId action) {
  using namespace board_ai::sixnimmt;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  if (action >= kCommitOffset && action < kCommitOffset + kNumCards) {
    m["type"] = std::any(std::string("commit"));
    m["card"] = std::any(action - kCommitOffset + 1);
  } else if (action >= kSmallChooseRowOffset &&
             action < kSmallChooseRowOffset + kNumRows) {
    m["type"] = std::any(std::string("small_choose_row"));
    m["row"] = std::any(action - kSmallChooseRowOffset);
  } else if (action == kHiddenCommitAction) {
    // What observers receive for a commit (SixNimmtRules::public_action):
    // the card is face down.
    m["type"] = std::any(std::string("commit"));
    m["hidden"] = std::any(true);
  } else {
    m["type"] = std::any(std::string("invalid"));
  }
  return m;
}

// ---------------------------------------------------------------------
// Heuristic picker — port of 9-year-old games/6nimmit/参考/lyc_pbh_wzq.py
//
// Algorithm (§1.7): segment the perspective player's hand into 5 "intervals"
// by which row each card would land on, flag bad / danger / dead, then run
// a 6-tier commit decision tree. SMALL phase uses findsmall mode 1 (offensive
// evaluate: simulate placing the card as new row head and pick the row
// minimizing future opponent pickup gain).
//
// C++ ports the Python rule semantics directly — no Python subprocess, no
// oracle, no byte-equal assertion. Test layer covers tier-by-tier behavior.
// ---------------------------------------------------------------------

template <int NPlayers>
board_ai::GameBundle make_sixnimmit(const std::string& game_id,
                                    std::uint64_t seed) {
  using namespace board_ai::sixnimmt;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<SixNimmtState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<SixNimmtRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<SixNimmtFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<SixNimmtBeliefTracker<NPlayers>>();
  b.belief_feature_extractor =
      std::make_unique<SixNimmtBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor =
      std::make_unique<SixNimmtBeliefLabelExtractor<NPlayers>>();
  // See .claude/skills/game-rules/simultaneous.md ("What the framework
  // already gives you for free"): tracker no longer touches opp's
  // pending_commits — sim's `current_player()` scan + opponent_selection
  // (Smooth-UCT prior) drives opp's commit decision via standard MCTS
  // descent. No PV injector needed.
  b.state_serializer = serialize_sixnimmit<NPlayers>;
  b.action_descriptor = describe_sixnimmit;
  b.heuristics.register_named(
      "rule_v1",
      std::make_unique<board_ai::sixnimmt::SixNimmtRuleHeuristic<NPlayers>>("rule_v1"));
  b.heuristics.register_named(
      "lkf_v1",
      std::make_unique<board_ai::sixnimmt::SixNimmtLkfHeuristic<NPlayers>>("lkf_v1"));
  // Belief tracker carries per-player public play history (`played_by_`)
  // — the wholesale snapshot already covers table / bullhead_pile, but
  // those are keyed by row / by-eater respectively; the tracker needs
  // to know which seat originally played each card. One `card_played`
  // event fires the moment the apply loop consumes a pending slot.
  b.install_event_protocol(
      extract_events_only_sixnimmit<NPlayers>,
      []() -> const board_ai::viz::VisibilitySchema& {
        return SixNimmtState<NPlayers>::schema();
      });
  return b;
}

board_ai::GameRegistrar reg_sixnimmit("6nimmit", [](std::uint64_t seed) {
  return make_sixnimmit<2>("6nimmit", seed);
});
board_ai::GameRegistrar reg_sixnimmit_2p("6nimmit_2p", [](std::uint64_t seed) {
  return make_sixnimmit<2>("6nimmit_2p", seed);
});
board_ai::GameRegistrar reg_sixnimmit_3p("6nimmit_3p", [](std::uint64_t seed) {
  return make_sixnimmit<3>("6nimmit_3p", seed);
});
board_ai::GameRegistrar reg_sixnimmit_4p("6nimmit_4p", [](std::uint64_t seed) {
  return make_sixnimmit<4>("6nimmit_4p", seed);
});

}  // namespace
