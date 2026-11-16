#include "6nimmit_net_adapter.h"

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "../../engine/core/belief_sequence.h"
#include "../../engine/core/snapshot_io.h"
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

// The i-th seat after `observer` in turn order.
int other_seat(int observer, int i, int players) { return (observer + 1 + i) % players; }

// Rows as (end, length), ascending by end: the order of the rows on the
// table carries no meaning.
std::array<std::pair<int, int>, kNumRows> sorted_rows(const std::array<std::int8_t, kNumRows>& ends,
                                                      const std::array<std::int8_t, kNumRows>& lens) {
  std::array<std::pair<int, int>, kNumRows> rows;
  for (int r = 0; r < kNumRows; ++r) rows[idx(r)] = {ends[idx(r)], lens[idx(r)]};
  std::sort(rows.begin(), rows.end());
  return rows;
}

}  // namespace

template <int NPlayers>
HiddenCards<NPlayers> hidden_cards(const SixNimmtState<NPlayers>& state, int observer) {
  const auto& d = state.data;
  HiddenCards<NPlayers> h;
  // Not shown, not in my hand, not a played card I can see. Hidden slots
  // are not read.
  for (int c = 1; c <= kNumCards; ++c) {
    h.seen[idx(c)] = d.used[idx(c - 1)] == 1 || d.hand[idx(observer)][idx(c - 1)] == 1;
  }
  for (int p = 0; p < NPlayers; ++p) {
    const int c = d.pending_commits[idx(p)];
    h.face_up[idx(p)] = (p == observer || pending_visible(state, p, observer)) && is_card(c);
    if (h.face_up[idx(p)]) h.seen[idx(c)] = true;
  }
  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer || d.terminal) continue;
    // A card is placed only after all are face up, and a round that pauses
    // does so before any has been placed: a seat without a face-up card
    // still holds this round's card.
    h.hidden[idx(p)] = kHandSize - d.current_round - (h.face_up[idx(p)] ? 1 : 0);
  }
  return h;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::init(IGameState& state, int perspective, const AnyMap& payload) {
  rounds_.clear();
  samples_.reset();
  samples_observer_ = -1;
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
                                                           const std::vector<PublicEvent>& events) {
  for (const auto& [kind, payload] : events) {
    if (kind != "card_played") continue;
    const int round = std::any_cast<int>(payload.at("round"));
    const int player = std::any_cast<int>(payload.at("player"));
    const int card = std::any_cast<int>(payload.at("card"));
    // Lists may arrive as vector<any> after a round trip through Python.
    const std::vector<int> ends = viz::detail::any_to_int_vec(payload.at("round_start_ends"), "6nimmit round_start_ends");
    const std::vector<int> lens = viz::detail::any_to_int_vec(payload.at("round_start_lens"), "6nimmit round_start_lens");
    if (round < 0 || round >= kHandSize || player < 0 || player >= NPlayers || !is_card(card) ||
        ends.size() != idx(kNumRows) || lens.size() != idx(kNumRows)) {
      throw std::invalid_argument("6nimmit tracker: malformed card_played event");
    }
    if (static_cast<int>(rounds_.size()) <= round) rounds_.resize(idx(round + 1));
    Round& r = rounds_[idx(round)];
    for (int i = 0; i < kNumRows; ++i) {
      r.ends[idx(i)] = static_cast<std::int8_t>(ends[idx(i)]);
      r.lens[idx(i)] = static_cast<std::int8_t>(lens[idx(i)]);
    }
    r.cards[idx(player)] = static_cast<std::int8_t>(card);
  }
}

template <int NPlayers>
AnyMap SixNimmtBeliefTracker<NPlayers>::serialize() const {
  std::vector<int> flat;
  for (const Round& r : rounds_) {
    flat.insert(flat.end(), r.ends.begin(), r.ends.end());
    flat.insert(flat.end(), r.lens.begin(), r.lens.end());
    flat.insert(flat.end(), r.cards.begin(), r.cards.end());
  }
  AnyMap m;
  m["rounds"] = std::any(flat);
  return m;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::prepare_for_root(const IGameState& root_state, int root_player,
                                                       const IBeliefFeatureExtractor* extractor,
                                                       const IBeliefEvaluator* evaluator) {
  samples_.reset();
  samples_observer_ = -1;
  if (!extractor || !evaluator) return;
  const BeliefSequenceLayout layout = extractor->sequence_layout();
  if (layout.classes != kNumCards || layout.segments != NPlayers - 1) {
    throw std::logic_error("6nimmit tracker: the belief extractor's sequence layout is not this game's");
  }
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(root_state);
  const HiddenCards<NPlayers> h = hidden_cards(s, root_player);
  belief_sequence::Problem problem;
  problem.pool.assign(idx(kNumCards), 0);
  for (int c = 1; c <= kNumCards; ++c) problem.pool[idx(c - 1)] = h.seen[idx(c)] ? 0 : 1;
  int hidden = 0;
  for (int i = 0; i < NPlayers - 1; ++i) {
    problem.sizes.push_back(h.hidden[idx(other_seat(root_player, i, NPlayers))]);
    hidden += problem.sizes.back();
  }
  if (hidden == 0) return;
  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this, &features)) {
    throw std::logic_error("6nimmit tracker: belief features have the wrong size");
  }
  // The same position always draws the same hands; the worlds' own seeds
  // decide which of them each world gets.
  const StateHash64 hash = root_state.state_hash_for_perspective(root_player);
  std::mt19937_64 rng(static_cast<std::uint64_t>(hash) ^ 0x6e696d6d74ULL);
  samples_ = std::make_shared<const std::vector<std::vector<std::vector<int>>>>(
      belief_sequence::sample(layout, problem, features, *evaluator, kBeliefSamples, rng));
  samples_observer_ = root_player;
  samples_state_ = hash;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::randomize_unseen(IGameState& state, int observer,
                                                       std::mt19937_64& rng) const {
  auto* s = dynamic_cast<SixNimmtState<NPlayers>*>(&state);
  if (!s || observer < 0 || observer >= NPlayers) return;
  auto& d = s->data;

  const HiddenCards<NPlayers> h = hidden_cards(*s, observer);
  // The belief net's hands belong to the position prepare_for_root drew them
  // for (the observer's hash covers the step count too); any other position
  // is sampled uniformly.
  const bool learned =
      samples_observer_ == observer && samples_ && !samples_->empty() &&
      s->state_hash_for_perspective(observer) == samples_state_;
  const std::vector<std::vector<int>>* sample = nullptr;
  if (learned) {
    sample = &(*samples_)[std::uniform_int_distribution<std::size_t>(0, samples_->size() - 1)(rng)];
  }

  std::vector<int> pool;
  pool.reserve(idx(kNumCards));
  for (int c = 1; c <= kNumCards; ++c) {
    if (!h.seen[idx(c)]) pool.push_back(c);
  }

  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer) continue;
    // A face-down card counts as not played yet (slot 0): the search plays
    // it out of the sampled hand itself.
    if (!pending_visible(state, p, observer)) d.pending_commits[idx(p)] = 0;
    d.hand[idx(p)].fill(0);
    const int draws = h.hidden[idx(p)];
    if (sample) {
      // Samples list the seats in turn order after the observer.
      const std::vector<int>& hand = (*sample)[idx((p - observer - 1 + NPlayers) % NPlayers)];
      if (static_cast<int>(hand.size()) != draws) {
        throw std::logic_error("6nimmit tracker: a sampled hand has the wrong size");
      }
      for (int cls : hand) {
        if (cls < 0 || cls >= kNumCards || h.seen[idx(cls + 1)] || d.hand[idx(p)][idx(cls)] == 1) {
          throw std::logic_error("6nimmit tracker: a sampled hand holds a card it cannot");
        }
        d.hand[idx(p)][idx(cls)] = 1;
      }
      continue;
    }
    if (static_cast<int>(pool.size()) < draws) {
      throw std::logic_error("6nimmit: fewer unseen cards than the other players hold");
    }
    for (int k = 0; k < draws; ++k) {
      std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
      const std::size_t j = pick(rng);
      d.hand[idx(p)][idx(pool[j] - 1)] = 1;
      pool[j] = pool.back();
      pool.pop_back();
    }
  }
  // The rest of the pool is the deck; no slot stores it.
}

// ---- belief net features / labels ------------------------------------------------------------

template <int NPlayers>
void SixNimmtBeliefFeatureExtractor<NPlayers>::extract(const IGameState& masked_state, int perspective_player,
                                                       const IBeliefTracker* tracker,
                                                       std::vector<float>* out) const {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(masked_state);
  const auto& d = s.data;
  const int me = perspective_player;
  const HiddenCards<NPlayers> h = hidden_cards(s, me);
  const auto* memory = dynamic_cast<const SixNimmtBeliefTracker<NPlayers>*>(tracker);
  static const std::vector<typename SixNimmtBeliefTracker<NPlayers>::Round> no_rounds;
  const auto& rounds = memory ? memory->rounds() : no_rounds;

  out->assign(idx(kFeatureDim), 0.0f);
  float* x = out->data();
  for (int c = 1; c <= kNumCards; ++c) x[c - 1] = h.seen[idx(c)] ? 0.0f : 1.0f;
  x += kNumCards;
  for (int c = 1; c <= kNumCards; ++c) x[c - 1] = d.hand[idx(me)][idx(c - 1)] == 1 ? 1.0f : 0.0f;
  x += kNumCards;
  for (int i = 0; i < kOthers; ++i) {
    const int p = other_seat(me, i, NPlayers);
    float* played = x;
    float* skipped = x + kNumCards;
    for (const auto& r : rounds) {
      const int card = r.cards[idx(p)];
      if (!is_card(card)) continue;
      played[card - 1] = 1.0f;
      // The row this card was aimed at when it was chosen: the highest end
      // below it. Every card between that end and this one would have gone
      // on the same row with a smaller gap.
      int end = 0;
      for (int e : r.ends) {
        if (e < card) end = std::max(end, e);
      }
      if (end == 0) continue;  // below every row
      for (int c = end + 1; c < card; ++c) skipped[c - 1] += 0.25f;
    }
    x += 2 * kNumCards;
  }
  for (int t = 0; t < kHandSize; ++t) {
    if (t < static_cast<int>(rounds.size())) {
      const auto& r = rounds[idx(t)];
      const auto rows = sorted_rows(r.ends, r.lens);
      for (int k = 0; k < kNumRows; ++k) x[k] = static_cast<float>(rows[idx(k)].first) / kNumCards;
      for (int k = 0; k < kNumRows; ++k) x[kNumRows + k] = static_cast<float>(rows[idx(k)].second) / kRowCapacity;
      x[2 * kNumRows] = static_cast<float>(r.cards[idx(me)]) / kNumCards;
      for (int i = 0; i < kOthers; ++i) {
        x[2 * kNumRows + 1 + i] = static_cast<float>(r.cards[idx(other_seat(me, i, NPlayers))]) / kNumCards;
      }
    }
    x += kRoundDim;
  }
  *x++ = static_cast<float>(d.current_round) / kHandSize;
  for (int i = 0; i < kOthers; ++i) {
    const int p = other_seat(me, i, NPlayers);
    *x++ = static_cast<float>(h.hidden[idx(p)]) / kHandSize;
    *x++ = h.face_up[idx(p)] ? 1.0f : 0.0f;
  }
  *x++ = static_cast<float>(d.hand_heads[idx(me)]) / 30.0f;
  for (int i = 0; i < kOthers; ++i) {
    *x++ = static_cast<float>(d.hand_heads[idx(other_seat(me, i, NPlayers))]) / 30.0f;
  }
  std::array<std::int8_t, kNumRows> ends{}, lens{};
  for (int r = 0; r < kNumRows; ++r) {
    int len = 0;
    while (len < kRowCapacity && d.table[idx(r)][idx(len)] != 0) ++len;
    ends[idx(r)] = len > 0 ? d.table[idx(r)][idx(len - 1)] : 0;
    lens[idx(r)] = static_cast<std::int8_t>(len);
  }
  const auto rows = sorted_rows(ends, lens);
  for (int k = 0; k < kNumRows; ++k) *x++ = static_cast<float>(rows[idx(k)].first) / kNumCards;
  for (int k = 0; k < kNumRows; ++k) *x++ = static_cast<float>(rows[idx(k)].second) / kRowCapacity;
  if (x != out->data() + kFeatureDim) throw std::logic_error("6nimmit belief features: layout mismatch");
}

template <int NPlayers>
void SixNimmtBeliefLabelExtractor<NPlayers>::extract(const IGameState& truth_state, int observer,
                                                     std::vector<std::vector<int>>* out_hand_counts,
                                                     std::vector<int>* out_remaining,
                                                     std::vector<int>* out_alive) const {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(truth_state);
  const auto& d = s.data;
  const HiddenCards<NPlayers> h = hidden_cards(s, observer);
  out_hand_counts->assign(idx(NPlayers - 1), std::vector<int>(idx(kNumCards), 0));
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int p = other_seat(observer, i, NPlayers);
    auto& row = (*out_hand_counts)[idx(i)];
    int held = 0;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.hand[idx(p)][idx(c - 1)] == 1) {
        row[idx(c - 1)] = 1;
        ++held;
      }
    }
    const int pending = d.pending_commits[idx(p)];
    if (is_card(pending) && !h.face_up[idx(p)]) {
      row[idx(pending - 1)] = 1;
      ++held;
    }
    if (held != h.hidden[idx(p)]) {
      throw std::logic_error("6nimmit belief labels: a seat's hidden cards do not match its public hand size");
    }
  }
  out_remaining->assign(idx(kNumCards), 0);
  for (int c = 1; c <= kNumCards; ++c) (*out_remaining)[idx(c - 1)] = h.seen[idx(c)] ? 0 : 1;
  out_alive->assign(idx(NPlayers - 1), 1);
}

template HiddenCards<2> hidden_cards<2>(const SixNimmtState<2>&, int);
template HiddenCards<3> hidden_cards<3>(const SixNimmtState<3>&, int);
template HiddenCards<4> hidden_cards<4>(const SixNimmtState<4>&, int);
template class SixNimmtBeliefTracker<2>;
template class SixNimmtBeliefTracker<3>;
template class SixNimmtBeliefTracker<4>;
template class SixNimmtBeliefFeatureExtractor<2>;
template class SixNimmtBeliefFeatureExtractor<3>;
template class SixNimmtBeliefFeatureExtractor<4>;
template class SixNimmtBeliefLabelExtractor<2>;
template class SixNimmtBeliefLabelExtractor<3>;
template class SixNimmtBeliefLabelExtractor<4>;

}  // namespace board_ai::sixnimmt
