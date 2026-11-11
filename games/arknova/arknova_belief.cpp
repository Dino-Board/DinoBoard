#include "arknova_belief.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

// Cards whose location the observer knows: public zones and visible hidden-field slots.
std::vector<std::uint8_t> known_cards(const ArknovaState& st, int observer, bool by_viz) {
  const SlotView view(st, observer, by_viz);
  const State& s = st.core;
  std::vector<std::uint8_t> known(static_cast<std::size_t>(num_cards()), 0);
  auto mark = [&](int c) {
    if (valid_card(c)) known[static_cast<std::size_t>(c)] = 1;
  };
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

namespace {

class Pool {
 public:
  explicit Pool(const char* name) : name_(name) {}
  void add(int c) { cards_.push_back(c); }
  std::size_t size() const { return cards_.size(); }
  int draw(std::mt19937_64& rng) {
    if (cards_.empty()) throw std::logic_error(std::string("arknova tracker: ") + name_ + " pool exhausted");
    std::size_t i = std::uniform_int_distribution<std::size_t>(0, cards_.size() - 1)(rng);
    return take(i);
  }
  // Weighted draw without replacement; uniform when the weights of the remaining cards vanish.
  int draw_weighted(const std::array<float, kBeliefClasses>& w, std::mt19937_64& rng) {
    double total = 0.0;
    for (int c : cards_) total += w[static_cast<std::size_t>(c)];
    if (!(total > 0.0) || !std::isfinite(total)) return draw(rng);
    double r = std::uniform_real_distribution<double>(0.0, total)(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < cards_.size(); ++i) {
      acc += w[static_cast<std::size_t>(cards_[i])];
      if (r <= acc) return take(i);
    }
    return take(cards_.size() - 1);
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
  // The posterior belongs to the position prepare_for_root computed it for (the observer's hash
  // covers the step count too). Any other position is sampled uniformly, e.g. a later heuristic
  // decision of the same seat, for which the cached numbers are stale.
  const bool weighted = probs_observer_ == observer && st.state_hash_for_perspective(observer) == probs_state_;
  constexpr int kMaxTries = 1000;
  for (int t = 0; t < kMaxTries; ++t) {
    sample_world(st, observer, weighted, rng);
    if (st.current_player() == observer || prefix_consistent(st)) return;
  }
  throw std::runtime_error("arknova tracker: no sampled world fits the mover's chosen atoms");
}

void ArknovaBeliefTracker::sample_world(ArknovaState& st, int observer, bool weighted, std::mt19937_64& rng) const {
  const State& s = st.core;
  SlotView view(st, observer, true);
  std::vector<std::uint8_t> known = known_cards(st, observer, true);
  int q = opponent_of(observer);
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
  // Cards seen in the hand before a face-down tuck: each is in the hand or among the tucked cards.
  std::vector<int> maybe;
  for (int c = 0; c < num_cards(); ++c) {
    if (opp.maybe_tucked[c] && !named[static_cast<std::size_t>(c)]) maybe.push_back(c);
  }
  Pool regular("regular"), endgame("endgame");
  for (int c = 0; c < num_cards(); ++c) {
    if (!known[static_cast<std::size_t>(c)] && !named[static_cast<std::size_t>(c)] && !opp.maybe_tucked[c]) {
      (is_endgame(c) ? endgame : regular).add(c);
    }
  }

  int hidden_in_hand = opp.hand_size;
  for (int c = 0; c < kCardSlots; ++c) {
    if (c < num_cards() && view.visible(kFieldHand, {q, c}, 0)) {
      if (st.hand[q][c] == 1) --hidden_in_hand;
    } else if (c < num_cards() && named[static_cast<std::size_t>(c)]) {
      if (known[static_cast<std::size_t>(c)]) throw std::logic_error("arknova tracker: a named hand card is elsewhere");
      st.hand[q][c] = 1;
      --hidden_in_hand;
    } else {
      st.hand[q][c] = 0;
    }
  }
  if (hidden_in_hand < 0) throw std::logic_error("arknova tracker: more visible hand cards than the hand size");
  int tucked_left = opp.n_tucked;
  if (static_cast<int>(maybe.size()) > hidden_in_hand + tucked_left) {
    throw std::logic_error("arknova tracker: more maybe-tucked cards than hidden hand and tucked cards");
  }
  std::shuffle(maybe.begin(), maybe.end(), rng);
  std::vector<int> tucked_cards;
  for (int c : maybe) {
    // Uniform over the free hand and tucked places, or the belief net's hand probability.
    double p_hand = weighted ? hand_probs_[static_cast<std::size_t>(c)]
                             : static_cast<double>(hidden_in_hand) / (hidden_in_hand + tucked_left);
    bool to_hand = tucked_left == 0 || (hidden_in_hand > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p_hand);
    if (to_hand) {
      st.hand[q][c] = 1;
      --hidden_in_hand;
    } else {
      tucked_cards.push_back(c);
      --tucked_left;
    }
  }
  for (int i = 0; i < hidden_in_hand; ++i) {
    st.hand[q][weighted ? regular.draw_weighted(hand_probs_, rng) : regular.draw(rng)] = 1;
  }
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
  probs_observer_ = -1;
  hand_probs_.fill(0.0f);
  if (!extractor || !evaluator) return;
  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this, &features)) {
    throw std::logic_error("arknova tracker: belief features have the wrong size");
  }
  std::vector<float> logits;
  if (!evaluator->evaluate(features, &logits)) throw std::runtime_error("arknova tracker: belief evaluation failed");
  if (static_cast<int>(logits.size()) != kBeliefClasses) {
    throw std::logic_error("arknova tracker: belief net returned the wrong logit count");
  }
  for (int c = 0; c < kBeliefClasses; ++c) {
    double z = static_cast<double>(logits[static_cast<std::size_t>(c)]) / kBeliefTemperature;
    hand_probs_[static_cast<std::size_t>(c)] = static_cast<float>(1.0 / (1.0 + std::exp(-z)));
  }
  probs_observer_ = root_player;
  probs_state_ = root_state.state_hash_for_perspective(root_player);
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
  int q = opponent_of(observer);
  SlotView view(st, observer, true);
  out_hand_counts->assign(1, std::vector<int>(kBeliefClasses, 0));
  for (int c = 0; c < num_cards(); ++c) {
    if (st.hand[q][c] == 1 && !view.visible(kFieldHand, {q, c}, 1)) (*out_hand_counts)[0][static_cast<std::size_t>(c)] = 1;
  }
  out_remaining->assign(kBeliefClasses, 0);
  out_alive->assign(1, 1);
}

}  // namespace board_ai::arknova
