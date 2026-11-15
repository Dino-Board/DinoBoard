#include "6nimmit_net_adapter.h"

#include <any>
#include <array>
#include <cstddef>
#include <stdexcept>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::sixnimmt {

namespace {

std::size_t idx(int i) { return static_cast<std::size_t>(i); }

bool is_card(int c) { return c >= 1 && c <= kNumCards; }

// Whether `observer` can see the played-card slot of `owner`.
bool pending_visible(const IGameState& state, int owner, int observer) {
  const auto& v = viz::viz_get(state, "pending_commits");
  const std::size_t base = viz::flat_offset_data_only(v.shape, std::vector<int>{owner});
  return v.data[base + idx(observer)] != 0;
}

}  // namespace

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::init(IGameState& state, int perspective, const AnyMap& payload) {
  if (perspective < 0 || perspective >= NPlayers) return;
  auto* s = dynamic_cast<SixNimmtState<NPlayers>*>(&state);
  if (!s) return;
  const auto it = payload.find("own_hand");
  if (it == payload.end() || it->second.type() != typeid(std::vector<int>)) return;
  auto& row = s->data.hand[idx(perspective)];
  row.fill(0);
  for (int c : std::any_cast<const std::vector<int>&>(it->second)) {
    if (is_card(c)) row[idx(c - 1)] = 1;
  }
}

template <int NPlayers>
AnyMap SixNimmtBeliefTracker<NPlayers>::pack_init_payload(const IGameState& gt_state, int perspective) const {
  if (perspective < 0 || perspective >= NPlayers) return {};
  const auto* s = dynamic_cast<const SixNimmtState<NPlayers>*>(&gt_state);
  if (!s) return {};
  std::vector<int> cards;
  for (int c = 1; c <= kNumCards; ++c) {
    if (s->data.hand[idx(perspective)][idx(c - 1)] == 1) cards.push_back(c);
  }
  AnyMap out;
  out["own_hand"] = std::any(cards);
  return out;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::observe_public_event(int /*actor*/, ActionId /*action*/,
                                                           const std::vector<PublicEvent>& /*events*/) {}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::randomize_unseen(IGameState& state, int observer,
                                                       std::mt19937_64& rng) const {
  auto* s = dynamic_cast<SixNimmtState<NPlayers>*>(&state);
  if (!s || observer < 0 || observer >= NPlayers) return;
  auto& d = s->data;

  // The cards the others can hold: not shown, not in my hand, not a
  // played card I can see. Hidden slots are not read.
  std::array<bool, kNumCards + 1> seen{};
  for (int c = 1; c <= kNumCards; ++c) {
    seen[idx(c)] = d.used[idx(c - 1)] == 1 || d.hand[idx(observer)][idx(c - 1)] == 1;
  }
  std::array<bool, NPlayers> face_up{};  // a card of this seat lies where I can see it
  for (int p = 0; p < NPlayers; ++p) {
    const int c = d.pending_commits[idx(p)];
    face_up[idx(p)] = (p == observer || pending_visible(state, p, observer)) && is_card(c);
    if (face_up[idx(p)]) seen[idx(c)] = true;
  }
  std::vector<int> pool;
  pool.reserve(idx(kNumCards));
  for (int c = 1; c <= kNumCards; ++c) {
    if (!seen[idx(c)]) pool.push_back(c);
  }

  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer) continue;
    // A face-down card counts as not played yet (slot 0): the search plays
    // it out of the sampled hand itself.
    if (!pending_visible(state, p, observer)) d.pending_commits[idx(p)] = 0;
    d.hand[idx(p)].fill(0);
    if (d.terminal) continue;
    // A card is placed only after all are face up, and a round that pauses
    // does so before any has been placed: a seat without a face-up card
    // still holds this round's card.
    const int draws = kHandSize - d.current_round - (face_up[idx(p)] ? 1 : 0);
    if (static_cast<int>(pool.size()) < draws) {
      throw std::logic_error("6nimmit: fewer unseen cards than the other players hold");
    }
    for (int k = 0; k < draws; ++k) {
      std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
      const std::size_t i = pick(rng);
      d.hand[idx(p)][idx(pool[i] - 1)] = 1;
      pool[i] = pool.back();
      pool.pop_back();
    }
  }
  // The rest of the pool is the deck; no slot stores it.
}

template class SixNimmtBeliefTracker<2>;
template class SixNimmtBeliefTracker<3>;
template class SixNimmtBeliefTracker<4>;

}  // namespace board_ai::sixnimmt
