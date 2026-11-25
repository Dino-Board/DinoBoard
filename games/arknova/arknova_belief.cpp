#include "arknova_belief.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../engine/core/belief_sequence.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "arknova_actions.h"
#include "arknova_internal.h"

namespace board_ai::arknova {

namespace {

bool valid_card(int v) { return v >= 0 && v < num_cards(); }
bool is_endgame(int c) { return card(static_cast<CardIdx>(c)).type == CardType::kEndgame; }

int opponent_of(int p) { return 1 - p; }

// Visibility of hidden card slots to the observer: from viz (tracker, truth or world states) or
// from the slot value (masked states, where hidden slots hold kHiddenCard).
class SlotView {
 public:
  SlotView(const ArknovaState& st, int observer, bool by_viz) : st_(st), observer_(observer), by_viz_(by_viz) {}
  bool visible(viz::FieldId fid, const std::vector<int>& idx, int value) const {
    if (!by_viz_) return value != kHiddenCard;
    const viz::VizTensor& v = viz::viz_get_by_id(st_, fid);
    return v.data[viz::flat_offset_data_only(v.shape, idx) + static_cast<std::size_t>(observer_)] != 0;
  }

 private:
  const ArknovaState& st_;
  int observer_;
  bool by_viz_;
};

}  // namespace

// Cards whose location the observer knows: public zones and visible hidden-field slots, and the
// BGA promos when they are not in the game.
std::vector<std::uint8_t> known_cards(const ArknovaState& st, int observer, bool by_viz) {
  const SlotView view(st, observer, by_viz);
  const State& s = st.core;
  std::vector<std::uint8_t> known(static_cast<std::size_t>(num_cards()), 0);
  auto mark = [&](int c) {
    if (valid_card(c)) known[static_cast<std::size_t>(c)] = 1;
  };
  for (int c = 0; c < num_cards(); ++c) {
    if (!s.promos && is_bga_promo(static_cast<CardIdx>(c))) mark(c);
  }
  for (CardIdx c : s.discard) mark(c);
  for (CardIdx c : s.display) mark(c);
  for (int i = 0; i < s.n_base_projects; ++i) mark(s.base_projects[i]);
  for (CardIdx c : s.base_project_deck) mark(c);
  for (int i = 0; i < s.n_projects; ++i) mark(s.projects[i]);
  for (const Player& pl : s.players) {
    for (int i = 0; i < pl.n_zoo_cards; ++i) mark(pl.zoo_cards[i]);
  }
  for (int i = 0; i < s.stack_n; ++i) {
    const Frame& f = s.stack[i];
    if (f.kind != FrameKind::ab_hunter) continue;
    for (int j = 0; j < f.d[fd::cards_n]; ++j) mark(f.d[fd::cards_list + j]);
  }
  for (int p = 0; p < kPlayers; ++p) {
    for (int c = 0; c < num_cards(); ++c) {
      if (st.hand[p][c] == 1 && view.visible(kFieldHand, {p, c}, st.hand[p][c])) mark(c);
    }
    for (int i = 0; i < kMaxFinalCards; ++i) {
      if (view.visible(kFieldFinalCards, {p, i}, st.final_cards[p][i])) mark(st.final_cards[p][i]);
    }
    for (int i = 0; i < kMaxTucked; ++i) {
      if (view.visible(kFieldTucked, {p, i}, st.tucked[p][i])) mark(st.tucked[p][i]);
    }
    for (int k = 0; k < kPrivateFrames; ++k) {
      for (int j = 0; j < kPrivateCards; ++j) {
        if (view.visible(kFieldPrivateCards, {p, k, j}, st.private_cards[p][k][j])) mark(st.private_cards[p][k][j]);
      }
    }
    for (int i = 0; i < kFinalKnown; ++i) {
      if (view.visible(kFieldFinalKnown, {p, i}, st.final_known[p][i])) mark(st.final_known[p][i]);
    }
  }
  return known;
}

HiddenHand hidden_hand(const ArknovaState& st, int observer) {
  const State& s = st.core;
  const SlotView view(st, observer, true);
  const std::vector<std::uint8_t> known = known_cards(st, observer, true);
  const int q = opponent_of(observer);
  // Hand cards the mover already named in the (public) prefix are in its hand.
  std::vector<std::uint8_t> named(static_cast<std::size_t>(num_cards()), 0);
  if (st.current_player() == q) {
    for (ActionId a : st.prefix) {
      for (int base : {atom::kHand, atom::kHandDiscard}) {
        if (a >= base && a < base + num_cards()) named[static_cast<std::size_t>(a - base)] = 1;
      }
    }
  }
  const Player& opp = s.players[q];
  HiddenHand h;
  // Cards seen in the hand before a face-down tuck: each is in the hand or among the tucked cards.
  for (int c = 0; c < num_cards(); ++c) {
    if (opp.maybe_tucked[c] && !named[static_cast<std::size_t>(c)]) h.maybe.push_back(c);
  }
  for (int c = 0; c < num_cards(); ++c) {
    if (!known[static_cast<std::size_t>(c)] && !named[static_cast<std::size_t>(c)] && !opp.maybe_tucked[c]) {
      (is_endgame(c) ? h.endgame : h.regular).push_back(c);
    }
  }
  h.hidden = opp.hand_size;
  for (int c = 0; c < num_cards(); ++c) {
    if (view.visible(kFieldHand, {q, c}, 0)) {
      if (st.hand[q][c] == 1) --h.hidden;
    } else if (named[static_cast<std::size_t>(c)]) {
      if (known[static_cast<std::size_t>(c)]) throw std::logic_error("arknova tracker: a named hand card is elsewhere");
      h.named.push_back(c);
      --h.hidden;
    }
  }
  if (h.hidden < 0) throw std::logic_error("arknova tracker: more visible hand cards than the hand size");
  if (static_cast<int>(h.maybe.size()) > h.hidden + opp.n_tucked) {
    throw std::logic_error("arknova tracker: more maybe-tucked cards than hidden hand and tucked cards");
  }
  return h;
}

namespace {

class Pool {
 public:
  Pool(const char* name, std::vector<int> cards) : name_(name), cards_(std::move(cards)) {}
  std::size_t size() const { return cards_.size(); }
  int draw(std::mt19937_64& rng) {
    if (cards_.empty()) throw std::logic_error(std::string("arknova tracker: ") + name_ + " pool exhausted");
    std::size_t i = std::uniform_int_distribution<std::size_t>(0, cards_.size() - 1)(rng);
    return take(i);
  }
  void remove(int c) {
    auto it = std::find(cards_.begin(), cards_.end(), c);
    if (it == cards_.end()) {
      throw std::logic_error(std::string("arknova tracker: card ") + std::to_string(c) + " is not in the " + name_ +
                             " pool");
    }
    take(static_cast<std::size_t>(it - cards_.begin()));
  }
  std::vector<CardIdx> rest(std::size_t expected, std::mt19937_64& rng) {
    if (cards_.size() != expected) {
      throw std::logic_error(std::string("arknova tracker: ") + name_ + " pool has " + std::to_string(cards_.size()) +
                             " cards for " + std::to_string(expected) + " deck slots");
    }
    std::vector<CardIdx> out(cards_.begin(), cards_.end());
    std::shuffle(out.begin(), out.end(), rng);
    cards_.clear();
    return out;
  }

 private:
  int take(std::size_t i) {
    int c = cards_[i];
    cards_[i] = cards_.back();
    cards_.pop_back();
    return c;
  }
  const char* name_;
  std::vector<int> cards_;
};

// The maybe-tucked cards placed uniformly over the free hand and tucked places.
struct MaybeSplit {
  std::vector<int> hand;
  std::vector<int> tucked;
};
MaybeSplit split_maybe(std::vector<int> maybe, int hidden, int tucked, std::mt19937_64& rng) {
  std::shuffle(maybe.begin(), maybe.end(), rng);
  MaybeSplit out;
  for (int c : maybe) {
    const double p_hand = static_cast<double>(hidden) / (hidden + tucked);
    const bool to_hand = tucked == 0 || (hidden > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p_hand);
    if (to_hand) {
      out.hand.push_back(c);
      --hidden;
    } else {
      out.tucked.push_back(c);
      --tucked;
    }
  }
  return out;
}

std::vector<int> flat_owner_slots(const ArknovaState& st, int p) {
  std::vector<int> out;
  out.insert(out.end(), st.hand[p], st.hand[p] + kCardSlots);
  out.insert(out.end(), st.final_cards[p], st.final_cards[p] + kMaxFinalCards);
  out.insert(out.end(), st.tucked[p], st.tucked[p] + kMaxTucked);
  out.insert(out.end(), &st.private_cards[p][0][0], &st.private_cards[p][0][0] + kPrivateFrames * kPrivateCards);
  out.insert(out.end(), st.final_known[p], st.final_known[p] + kFinalKnown);
  return out;
}

}  // namespace

// ---- tracker ----------------------------------------------------------------------------------

AnyMap ArknovaBeliefTracker::pack_init_payload(const IGameState& gt_state, int perspective) const {
  const auto& st = checked_cast<ArknovaState>(gt_state);
  AnyMap out;
  out["own_slots"] = std::any(flat_owner_slots(st, perspective));
  return out;
}

void ArknovaBeliefTracker::init(IGameState& state, int perspective, const AnyMap& payload) {
  auto it = payload.find("own_slots");
  if (it == payload.end()) return;
  auto& st = checked_cast<ArknovaState>(state);
  std::vector<int> v = viz::detail::any_to_int_vec(it->second, "arknova own_slots");
  std::vector<int> layout = flat_owner_slots(st, perspective);
  if (v.size() != layout.size()) throw std::invalid_argument("arknova tracker: own_slots has the wrong size");
  auto put = [&](std::int16_t* first, int n, std::size_t& at) {
    for (int i = 0; i < n; ++i) first[i] = static_cast<std::int16_t>(v[at++]);
  };
  std::size_t at = 0;
  int p = perspective;
  put(st.hand[p], kCardSlots, at);
  put(st.final_cards[p], kMaxFinalCards, at);
  put(st.tucked[p], kMaxTucked, at);
  put(&st.private_cards[p][0][0], kPrivateFrames * kPrivateCards, at);
  put(st.final_known[p], kFinalKnown, at);
  st.mark_core_stale();
}

void ArknovaBeliefTracker::randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const {
  auto& st = checked_cast<ArknovaState>(state);
  // The belief net's hands belong to the position prepare_for_root drew them for (the observer's
  // hash covers the step count too). Any other position is sampled uniformly, e.g. a later
  // heuristic decision of the same seat, for which the cached hands are stale.
  const bool learned = samples_observer_ == observer && samples_ && !samples_->empty() &&
                       st.state_hash_for_perspective(observer) == samples_state_;
  constexpr int kMaxTries = 1000;
  for (int t = 0; t < kMaxTries; ++t) {
    const HandSample* sample = nullptr;
    if (learned) sample = &(*samples_)[std::uniform_int_distribution<std::size_t>(0, samples_->size() - 1)(rng)];
    sample_world(st, observer, sample, rng);
    if (st.current_player() == observer || prefix_consistent(st)) return;
  }
  throw std::runtime_error("arknova tracker: no sampled world fits the mover's chosen atoms");
}

void ArknovaBeliefTracker::sample_world(ArknovaState& st, int observer, const HandSample* sample,
                                        std::mt19937_64& rng) const {
  const State& s = st.core;
  SlotView view(st, observer, true);
  const int q = opponent_of(observer);
  const Player& opp = s.players[q];
  HiddenHand h = hidden_hand(st, observer);
  Pool regular("regular", std::move(h.regular)), endgame("endgame", std::move(h.endgame));

  for (int c = 0; c < kCardSlots; ++c) {
    if (!(c < num_cards() && view.visible(kFieldHand, {q, c}, 0))) st.hand[q][c] = 0;
  }
  for (int c : h.named) st.hand[q][c] = 1;
  int hidden_in_hand = h.hidden;
  int tucked_left = opp.n_tucked;
  MaybeSplit split;
  if (sample) {
    split.hand = sample->maybe_in_hand;
    for (int c : h.maybe) {
      if (std::find(split.hand.begin(), split.hand.end(), c) == split.hand.end()) split.tucked.push_back(c);
    }
    if (split.hand.size() + split.tucked.size() != h.maybe.size()) {
      throw std::logic_error("arknova tracker: a sampled hand places a card that is not maybe-tucked");
    }
  } else {
    split = split_maybe(h.maybe, hidden_in_hand, tucked_left, rng);
  }
  for (int c : split.hand) st.hand[q][c] = 1;
  hidden_in_hand -= static_cast<int>(split.hand.size());
  tucked_left -= static_cast<int>(split.tucked.size());
  if (hidden_in_hand < 0 || tucked_left < 0) throw std::logic_error("arknova tracker: maybe-tucked cards overfill");
  if (sample) {
    if (static_cast<int>(sample->drawn.size()) != hidden_in_hand) {
      throw std::logic_error("arknova tracker: a sampled hand has the wrong size");
    }
    for (int c : sample->drawn) {
      regular.remove(c);
      st.hand[q][c] = 1;
    }
  } else {
    for (int i = 0; i < hidden_in_hand; ++i) st.hand[q][regular.draw(rng)] = 1;
  }
  std::vector<int> tucked_cards = std::move(split.tucked);
  for (int i = 0; i < tucked_left; ++i) tucked_cards.push_back(regular.draw(rng));
  std::shuffle(tucked_cards.begin(), tucked_cards.end(), rng);
  for (int i = 0; i < kMaxTucked; ++i) {
    st.tucked[q][i] = static_cast<std::int16_t>(i < opp.n_tucked ? tucked_cards[static_cast<std::size_t>(i)] : kEmptySlot);
  }
  for (int i = 0; i < kMaxFinalCards; ++i) {
    st.final_cards[q][i] = static_cast<std::int16_t>(i < opp.n_final ? endgame.draw(rng) : kEmptySlot);
  }
  std::fill(&st.private_cards[q][0][0], &st.private_cards[q][0][0] + kPrivateFrames * kPrivateCards, kEmptySlot);
  std::vector<int> frames = private_frames_of(s, q);
  for (std::size_t k = 0; k < frames.size(); ++k) {
    const Frame& f = s.stack[frames[k]];
    int n_off, list_off;
    private_list_offsets(f.kind, &n_off, &list_off);
    Pool& pool = f.kind == FrameKind::ab_resist ? endgame : regular;
    for (int j = 0; j < f.d[n_off]; ++j) st.private_cards[q][k][j] = static_cast<std::int16_t>(pool.draw(rng));
  }
  std::fill(st.final_known[q], st.final_known[q] + kFinalKnown, kEmptySlot);

  st.core.deck = regular.rest(s.deck.size(), rng);
  std::vector<CardIdx> own_returned;
  for (int i = 0; i < kFinalKnown; ++i) {
    if (valid_card(st.final_known[observer][i])) own_returned.push_back(st.final_known[observer][i]);
  }
  if (own_returned.size() > s.final_deck.size()) throw std::logic_error("arknova tracker: final deck too small");
  std::vector<CardIdx> final_deck = endgame.rest(s.final_deck.size() - own_returned.size(), rng);
  final_deck.insert(final_deck.begin(), own_returned.begin(), own_returned.end());
  st.core.final_deck = std::move(final_deck);
  st.mark_core_stale();
}

void ArknovaBeliefTracker::prepare_for_root(const IGameState& root_state, int root_player,
                                            const IBeliefFeatureExtractor* extractor,
                                            const IBeliefEvaluator* evaluator) {
  samples_.reset();
  samples_observer_ = -1;
  if (!extractor || !evaluator) return;
  const BeliefSequenceLayout layout = extractor->sequence_layout();
  if (layout.classes != kBeliefClasses || layout.segments != kPlayers - 1) {
    throw std::logic_error("arknova tracker: the belief extractor's sequence layout is not this game's");
  }
  const auto& st = checked_cast<ArknovaState>(root_state);
  const HiddenHand h = hidden_hand(st, root_player);
  if (h.hidden == 0) return;
  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this, &features)) {
    throw std::logic_error("arknova tracker: belief features have the wrong size");
  }
  // The same position always draws the same hands; the worlds' own seeds decide which of them
  // each world gets.
  const StateHash64 hash = root_state.state_hash_for_perspective(root_player);
  std::mt19937_64 rng(static_cast<std::uint64_t>(hash) ^ 0x61726b6e6f7661ULL);
  // The maybe-tucked cards take their places uniformly, as without the net; the net writes the
  // rest of the hand, one batch of samples per number of cards left to write.
  const int tucked = st.core.players[opponent_of(root_player)].n_tucked;
  auto samples = std::make_shared<std::vector<HandSample>>(static_cast<std::size_t>(kBeliefSamples));
  std::map<int, std::vector<std::size_t>> by_size;
  for (std::size_t i = 0; i < samples->size(); ++i) {
    (*samples)[i].maybe_in_hand = split_maybe(h.maybe, h.hidden, tucked, rng).hand;
    by_size[h.hidden - static_cast<int>((*samples)[i].maybe_in_hand.size())].push_back(i);
  }
  belief_sequence::Problem problem;
  problem.pool.assign(static_cast<std::size_t>(kBeliefClasses), 0);
  for (int c : h.regular) problem.pool[static_cast<std::size_t>(c)] = 1;
  for (const auto& [size, members] : by_size) {
    if (size == 0) continue;
    problem.sizes = {size};
    auto hands = belief_sequence::sample(layout, problem, features, *evaluator, static_cast<int>(members.size()), rng);
    for (std::size_t j = 0; j < members.size(); ++j) (*samples)[members[j]].drawn = std::move(hands[j][0]);
  }
  samples_ = std::move(samples);
  samples_observer_ = root_player;
  samples_state_ = hash;
}

// ---- belief net features / labels ---------------------------------------------------------------

void ArknovaBeliefFeatureExtractor::extract(const IGameState& masked_state, int perspective_player,
                                            const IBeliefTracker*, std::vector<float>* out) const {
  const auto& st = checked_cast<ArknovaState>(masked_state);
  const State& s = st.core;
  int me = perspective_player, q = opponent_of(me);
  const Player& opp = s.players[q];
  const Player& mine = s.players[me];
  out->assign(static_cast<std::size_t>(kFeatureDim), 0.0f);
  float* unseen = out->data();
  float* opp_known = unseen + kBeliefClasses;
  float* opp_maybe = opp_known + kBeliefClasses;
  float* opp_zoo = opp_maybe + kBeliefClasses;
  float* discard = opp_zoo + kBeliefClasses;
  float* scalars = discard + kBeliefClasses;

  std::vector<std::uint8_t> known = known_cards(st, me, false);
  for (int c = 0; c < num_cards(); ++c) {
    if (opp.maybe_tucked[c]) {
      opp_maybe[c] = 1.0f;
    } else if (!known[static_cast<std::size_t>(c)] && !is_endgame(c)) {
      unseen[c] = 1.0f;
    }
  }
  int opp_known_n = 0;
  for (int c = 0; c < num_cards(); ++c) {
    if (st.hand[q][c] == 1) {
      opp_known[c] = 1.0f;
      ++opp_known_n;
    }
  }
  for (int i = 0; i < opp.n_zoo_cards; ++i) opp_zoo[opp.zoo_cards[i]] = 1.0f;
  for (CardIdx c : s.discard) discard[c] = 1.0f;

  float* x = scalars;
  *x++ = opp.hand_size / 20.0f;
  *x++ = (opp.hand_size - opp_known_n) / 20.0f;
  *x++ = opp.n_tucked / 10.0f;
  *x++ = mine.hand_size / 20.0f;
  *x++ = static_cast<float>(s.deck.size()) / 200.0f;
  *x++ = static_cast<float>(s.discard.size()) / 200.0f;
  *x++ = s.break_pos / 12.0f;
  *x++ = opp.money / 40.0f;
  *x++ = opp.appeal / 60.0f;
  *x++ = opp.conservation / 30.0f;
  *x++ = opp.reputation / 15.0f;
  *x++ = opp.xtokens / 5.0f;
  *x++ = opp.n_zoo_cards / 30.0f;
  for (int a = 0; a < kNumActions; ++a) *x++ = opp.is_upgraded(a) ? 1.0f : 0.0f;
  if (x != out->data() + kFeatureDim) throw std::logic_error("arknova belief features: layout mismatch");
}

void ArknovaBeliefLabelExtractor::extract(const IGameState& truth_state, int observer,
                                          std::vector<std::vector<int>>* out_hand_counts,
                                          std::vector<int>* out_remaining, std::vector<int>* out_alive) const {
  const auto& st = checked_cast<ArknovaState>(truth_state);
  const int q = opponent_of(observer);
  const HiddenHand h = hidden_hand(st, observer);
  out_remaining->assign(static_cast<std::size_t>(kBeliefClasses), 0);
  for (int c : h.regular) (*out_remaining)[static_cast<std::size_t>(c)] = 1;
  out_hand_counts->assign(1, std::vector<int>(static_cast<std::size_t>(kBeliefClasses), 0));
  auto& hand = (*out_hand_counts)[0];
  const SlotView view(st, observer, true);
  const Player& opp = st.core.players[q];
  for (int c = 0; c < num_cards(); ++c) {
    const bool named = std::find(h.named.begin(), h.named.end(), c) != h.named.end();
    if (st.hand[q][c] != 1 || view.visible(kFieldHand, {q, c}, 1) || opp.maybe_tucked[c] || named) continue;
    if (!(*out_remaining)[static_cast<std::size_t>(c)]) {
      throw std::logic_error("arknova belief labels: a hidden hand card is outside the regular pool");
    }
    hand[static_cast<std::size_t>(c)] = 1;
  }
  out_alive->assign(1, 1);
}

}  // namespace board_ai::arknova
