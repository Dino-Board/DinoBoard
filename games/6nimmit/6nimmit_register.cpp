#include <algorithm>
#include <any>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "6nimmit_net_adapter.h"
#include "6nimmit_rules.h"
#include "6nimmit_state.h"
#include "6nimmit_token_encoder.h"
#include "heuristic/sixnimmit_lkf_heuristic.h"
#include "heuristic/sixnimmit_rule_heuristic.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::IGameState;
using namespace board_ai::sixnimmt;

// One `card_played` event per card that was placed in this transition (put
// on a row, or started a row anew after its owner took one): the moment a
// played card stops being a pending card. The UI animates from these; the
// belief tracker remembers them with the rows the round started from (every
// card of a round is placed in one transition, before which no card of the
// round has touched the table), which is what the players saw when they chose.
//
// The seat that completes the round has its card played and placed inside
// the same transition, so its slot reads 0 before and after; its card is
// taken from the action instead.
template <int NPlayers>
std::vector<board_ai::PublicEvent> extract_events_sixnimmit(const IGameState& before, ActionId action,
                                                            const IGameState& after, int /*perspective*/) {
  const auto& b = board_ai::checked_cast<SixNimmtState<NPlayers>>(before).data;
  const auto& a = board_ai::checked_cast<SixNimmtState<NPlayers>>(after).data;
  std::vector<board_ai::PublicEvent> events;
  const bool is_commit = !b.small_phase_active && action >= kCommitOffset && action < kCommitOffset + kNumCards;
  std::vector<int> row_ends, row_lens;  // of the table before the round's cards were placed
  for (const auto& row : b.table) {
    int len = 0;
    while (len < kRowCapacity && row[static_cast<std::size_t>(len)] != 0) ++len;
    row_ends.push_back(len > 0 ? static_cast<int>(row[static_cast<std::size_t>(len - 1)]) : 0);
    row_lens.push_back(len);
  }
  for (int p = 0; p < NPlayers; ++p) {
    int card = static_cast<int>(b.pending_commits[static_cast<std::size_t>(p)]);
    if (is_commit && p == b.current_player) card = static_cast<int>(action) - kCommitOffset + 1;
    if (card >= 1 && card <= kNumCards && a.pending_commits[static_cast<std::size_t>(p)] == 0) {
      AnyMap payload;
      payload["player"] = std::any(p);
      payload["card"] = std::any(card);
      payload["round"] = std::any(b.current_round);
      payload["round_start_ends"] = std::any(row_ends);
      payload["round_start_lens"] = std::any(row_lens);
      events.emplace_back("card_played", std::move(payload));
    }
  }
  return events;
}

// State for the UI and for tests. Hands and face-down cards are written as
// the given state holds them; viewers get a masked state.
template <int NPlayers>
AnyMap serialize_sixnimmit(const IGameState& state) {
  const auto& d = board_ai::checked_cast<SixNimmtState<NPlayers>>(state).data;
  auto at = [](int i) { return static_cast<std::size_t>(i); };
  auto is_card = [](int c) { return c >= 1 && c <= kNumCards; };

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

  // "Seat p has played its card this round" is public (the card lies face
  // down on the table) but stored only as the hidden card itself. Derive it
  // from public slots, so a masked state gives the same answer: in the
  // commit phase the cursor serves the seats in ascending order from 0, so
  // exactly the seats below it have played; while a row is being chosen all
  // have. (A round without such a pause resolves inside one transition.)
  const bool sealed = !state.is_terminal() && d.small_phase_active;
  auto has_committed = [&](int p) { return !state.is_terminal() && (sealed || p < d.current_player); };
  int num_committed = 0;
  for (int p = 0; p < NPlayers; ++p) num_committed += has_committed(p) ? 1 : 0;
  m["num_committed_this_round"] = std::any(num_committed);

  std::vector<std::vector<int>> table;
  for (int r = 0; r < kNumRows; ++r) {
    std::vector<int> row;
    for (int s = 0; s < kRowCapacity; ++s) row.push_back(static_cast<int>(d.table[at(r)][at(s)]));
    table.push_back(std::move(row));
  }
  m["table"] = std::any(table);

  // Cards shown so far: none of them can turn up again.
  std::vector<int> used;
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.used[at(c - 1)] == 1) used.push_back(c);
  }
  m["used"] = std::any(used);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> hand;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.hand[at(p)][at(c - 1)] == 1) hand.push_back(c);
    }
    pm["hand"] = std::any(hand);
    // Public: every seat starts a round with the same number of cards.
    pm["hand_count"] = std::any(std::max(0, kHandSize - d.current_round - (has_committed(p) ? 1 : 0)));
    pm["hand_heads"] = std::any(d.hand_heads[at(p)]);
    pm["committed"] = std::any(has_committed(p));
    // Only once all cards are face up.
    const int pending = d.pending_commits[at(p)];
    pm["pending_card"] = std::any(sealed && is_card(pending) ? pending : 0);
    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  // The face-up cards still to be placed, in the order they will be:
  // slots [apply_cursor, ...) of a queue as long as the table of players.
  std::vector<std::pair<int, int>> face_up;
  if (sealed) {
    for (int p = 0; p < NPlayers; ++p) {
      const int c = d.pending_commits[at(p)];
      if (is_card(c)) face_up.emplace_back(c, p);
    }
  }
  std::sort(face_up.begin(), face_up.end());
  std::vector<int> queue_card(at(NPlayers), 0), queue_player(at(NPlayers), 0);
  for (std::size_t i = 0; i < face_up.size(); ++i) {
    const int slot = d.apply_cursor + static_cast<int>(i);
    if (slot < 0 || slot >= NPlayers) break;
    queue_card[at(slot)] = face_up[i].first;
    queue_player[at(slot)] = face_up[i].second;
  }
  m["apply_queue_card"] = std::any(queue_card);
  m["apply_queue_player"] = std::any(queue_player);
  return m;
}

AnyMap describe_sixnimmit(ActionId action) {
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  if (action >= kCommitOffset && action < kCommitOffset + kNumCards) {
    m["type"] = std::any(std::string("commit"));
    m["card"] = std::any(action - kCommitOffset + 1);
  } else if (action >= kSmallChooseRowOffset && action < kSmallChooseRowOffset + kNumRows) {
    m["type"] = std::any(std::string("small_choose_row"));
    m["row"] = std::any(action - kSmallChooseRowOffset);
  } else if (action == kHiddenCommitAction) {
    m["type"] = std::any(std::string("commit"));  // a card played face down
    m["hidden"] = std::any(true);
  } else {
    m["type"] = std::any(std::string("invalid"));
  }
  return m;
}

template <int NPlayers>
board_ai::GameBundle make_sixnimmit(const std::string& game_id, std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<SixNimmtState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<SixNimmtRules<NPlayers>>();
  b.value_model = std::make_unique<SixNimmtMarginValueModel<NPlayers>>();
  b.encoder = std::make_unique<SixNimmtTokenEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<SixNimmtBeliefTracker<NPlayers>>();
  b.belief_feature_extractor = std::make_unique<SixNimmtBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor = std::make_unique<SixNimmtBeliefLabelExtractor<NPlayers>>();
  b.state_serializer = [](const IGameState& state) {
    // Static: the encoder's feature layout, for tests and documentation.
    static const std::string layout = SixNimmtTokenEncoder<NPlayers>::layout_json();
    AnyMap m = serialize_sixnimmit<NPlayers>(state);
    m["encoder_layout"] = std::any(layout);
    return m;
  };
  b.action_descriptor = describe_sixnimmit;
  b.heuristics.register_named("rule_v1", std::make_unique<SixNimmtRuleHeuristic<NPlayers>>("rule_v1"));
  b.heuristics.register_named("lkf_v1", std::make_unique<SixNimmtLkfHeuristic<NPlayers>>("lkf_v1"));
  // The events are for the UI and for the tracker's memory of the rounds
  // (belief features only; world sampling needs nothing but the state).
  b.install_event_protocol(extract_events_sixnimmit<NPlayers>,
                           []() -> const board_ai::viz::VisibilitySchema& { return SixNimmtState<NPlayers>::schema(); });
  // A game cut off by the ply cap goes to the only player with the fewest bullheads.
  b.adjudicator = [](const IGameState& state) -> int {
    const auto& d = board_ai::checked_cast<SixNimmtState<NPlayers>>(state).data;
    const int best = *std::min_element(d.hand_heads.begin(), d.hand_heads.end());
    int winner = -1;
    for (int p = 0; p < NPlayers; ++p) {
      if (d.hand_heads[static_cast<std::size_t>(p)] != best) continue;
      if (winner >= 0) return -1;
      winner = p;
    }
    return winner;
  };
  // Higher is better: minus the bullheads taken.
  b.final_score_extractor = [](const IGameState& state) -> std::vector<float> {
    const auto& d = board_ai::checked_cast<SixNimmtState<NPlayers>>(state).data;
    std::vector<float> out;
    for (int heads : d.hand_heads) out.push_back(-static_cast<float>(heads));
    return out;
  };
  b.episode_stats_extractor = [](const IGameState& final_state, const IGameState& /*initial_state*/,
                                 const std::vector<board_ai::SelfplaySampleView>& /*samples*/)
      -> std::map<std::string, double> {
    const auto& d = board_ai::checked_cast<SixNimmtState<NPlayers>>(final_state).data;
    return {{"most_heads", static_cast<double>(*std::max_element(d.hand_heads.begin(), d.hand_heads.end()))},
            {"fewest_heads", static_cast<double>(*std::min_element(d.hand_heads.begin(), d.hand_heads.end()))}};
  };
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
