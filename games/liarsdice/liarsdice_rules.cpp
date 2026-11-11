#include "liarsdice_rules.h"

#include <stdexcept>
#include <vector>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::liarsdice {

namespace {

// Strict-monotonic bid order: (count, face) > (cur_count, cur_face) iff
// count > cur_count OR (count == cur_count AND face > cur_face). 1 is wild
// and never a legal bid Y.
inline bool bid_strictly_greater(int new_count, int new_face,
                                 int cur_count, int cur_face) {
  if (new_count > cur_count) return true;
  if (new_count < cur_count) return false;
  return new_face > cur_face;
}

}  // namespace

template <int NPlayers>
bool LiarsdiceRules<NPlayers>::validate_action(
    const IGameState& state, ActionId action) const {
  auto legal = legal_actions(state);
  for (ActionId a : legal) {
    if (a == action) return true;
  }
  return false;
}

template <int NPlayers>
std::vector<ActionId> LiarsdiceRules<NPlayers>::legal_actions(
    const IGameState& state) const {
  using Cfg = LiarsdiceConfig<NPlayers>;
  const auto& s = checked_cast<LiarsdiceState<NPlayers>>(state);
  if (s.data.terminal) return {};

  std::vector<ActionId> out;
  out.reserve(static_cast<size_t>(Cfg::kActionSpace));

  const int cur_count = s.data.current_bid_count;
  const int cur_face = s.data.current_bid_face;
  const bool first_move = (cur_count == 0);

  // Challenge: legal whenever an outstanding bid exists.
  if (!first_move) {
    out.push_back(static_cast<ActionId>(Cfg::kChallengeAction));
  }

  // Bids: X ∈ 1..5N, Y ∈ 2..6, strictly greater than current.
  for (int x = 1; x <= Cfg::kTotalDice; ++x) {
    for (int y = kMinBidFace; y <= kFaceCount; ++y) {
      if (first_move || bid_strictly_greater(x, y, cur_count, cur_face)) {
        out.push_back(encode_bid(x, y));
      }
    }
  }
  return out;
}

template <int NPlayers>
void LiarsdiceRules<NPlayers>::do_action_fast_impl(
    IGameState& state, ActionId action, std::mt19937_64& /*rng*/) const {
  using Cfg = LiarsdiceConfig<NPlayers>;
  auto& s = checked_cast<LiarsdiceState<NPlayers>>(state);
  auto& d = s.data;

  if (is_challenge(action, Cfg::kTotalDice)) {
    // Resolve outstanding bid: count dice with face == current_bid_face,
    // treating face 1 as wild (always counts).
    const int X = d.current_bid_count;
    const int Y = d.current_bid_face;
    if (X == 0 || Y == 0) {
      throw std::logic_error(
          "LiarsdiceRules: challenge before any bid was placed");
    }
    int actual = 0;
    for (int p = 0; p < NPlayers; ++p) {
      actual += d.dice_count[static_cast<size_t>(p)][0];  // face 1 = wild
      actual += d.dice_count[static_cast<size_t>(p)][static_cast<size_t>(Y - 1)];
    }
    const int bidder = (d.current_player_ + NPlayers - 1) % NPlayers;
    const int challenger = d.current_player_;
    d.terminal = true;
    d.winner = (actual >= X) ? bidder : challenger;

    // Reveal every die to every viewer — the challenge resolution makes
    // all hands public.
    for (int p = 0; p < NPlayers; ++p) {
      for (int f = 0; f < kFaceCount; ++f) {
        viz::reveal_slot(s, "dice_count", {p, f});
      }
    }
    return;
  }

  // Bid push.
  const int x = bid_count_from_action(action);
  const int y = bid_face_from_action(action);
  d.current_bid_count = static_cast<std::int8_t>(x);
  d.current_bid_face = static_cast<std::int8_t>(y);
  d.current_player_ = (d.current_player_ + 1) % NPlayers;
}

template <int NPlayers>
void LiarsdiceRules<NPlayers>::undo_action_impl(
    IGameState& /*state*/, const UndoToken& /*token*/) const {
  // Liarsdice runs strictly through do_action_fast (selfplay / arena /
  // API / sim-descent). Tail solver is not wired (single-round game with
  // no undo-friendly hidden draws). Reaching here is a contract violation.
  throw std::logic_error(
      "LiarsdiceRules::undo_action_impl: Liarsdice does not support undo. "
      "do_action_fast is the only supported entry point.");
}

template class LiarsdiceRules<2>;
template class LiarsdiceRules<3>;
template class LiarsdiceRules<4>;

}  // namespace board_ai::liarsdice
