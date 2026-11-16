#include "belief_sequence.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace board_ai::belief_sequence {

namespace {

std::size_t at(int i) { return static_cast<std::size_t>(i); }

constexpr float kImpossible = -std::numeric_limits<float>::infinity();

// log n!
double log_factorial(int n) {
  static const std::vector<double> table = [] {
    std::vector<double> t(1025, 0.0);
    for (std::size_t i = 1; i < t.size(); ++i) t[i] = t[i - 1] + std::log(static_cast<double>(i));
    return t;
  }();
  return at(n) < table.size() ? table[at(n)] : std::lgamma(n + 1.0);
}

// log C(n, k), -inf when k > n.
double log_choose(int n, int k) {
  if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
  return log_factorial(n) - log_factorial(k) - log_factorial(n - k);
}

}  // namespace

Cursor::Cursor(const BeliefSequenceLayout& layout, const Problem& problem)
    : layout_(layout), sizes_(problem.sizes), left_(problem.pool) {
  if (layout.classes <= 0 || layout.segments <= 0) {
    throw std::invalid_argument("belief_sequence: the layout has no classes or segments");
  }
  if (static_cast<int>(left_.size()) != layout.classes || static_cast<int>(sizes_.size()) != layout.segments) {
    throw std::invalid_argument("belief_sequence: pool / sizes do not match the layout");
  }
  for (int v : left_) {
    if (v != 0 && v != 1) throw std::invalid_argument("belief_sequence: the pool holds each element at most once");
  }
  for (int v : sizes_) {
    if (v < 0) throw std::invalid_argument("belief_sequence: negative segment size");
  }
  const int have = std::accumulate(left_.begin(), left_.end(), 0);
  const int need = std::accumulate(sizes_.begin(), sizes_.end(), 0);
  if (have < need) {
    throw std::invalid_argument("belief_sequence: the pool holds " + std::to_string(have) +
                                " elements for hands of " + std::to_string(need));
  }
  chosen_.assign(at(layout.segments * layout.classes), 0);
  hands_.assign(at(layout.segments), {});
  skip_filled_segments();
}

void Cursor::skip_filled_segments() {
  while (segment_ < layout_.segments && static_cast<int>(hands_[at(segment_)].size()) >= sizes_[at(segment_)]) {
    ++segment_;
    last_ = -1;
  }
}

void Cursor::write_prefix(float* out) const {
  if (done()) throw std::logic_error("belief_sequence: prefix of a finished assignment");
  const int k = layout_.classes, s = layout_.segments;
  for (int i = 0; i < s * k; ++i) out[i] = static_cast<float>(chosen_[at(i)]);
  out += s * k;
  for (int i = 0; i < s; ++i) out[i] = i == segment_ ? 1.0f : 0.0f;
  out += s;
  for (int i = 0; i < k; ++i) out[i] = i == last_ ? 1.0f : 0.0f;
}

void Cursor::write_log_prior(float* out) const {
  if (done()) throw std::logic_error("belief_sequence: prior of a finished assignment");
  // Given what is written, the rest of this segment is a uniformly random
  // set of `need` of the pool elements above the previous one. Its smallest
  // is c with probability C(above(c), need - 1) / C(above(last), need),
  // where above(x) counts the pool elements greater than x.
  const int need = sizes_[at(segment_)] - static_cast<int>(hands_[at(segment_)].size());
  int above = 0;  // pool elements greater than c, as c walks down
  for (int c = layout_.classes - 1; c > last_; --c) {
    out[c] = left_[at(c)] ? static_cast<float>(log_choose(above, need - 1)) : kImpossible;
    above += left_[at(c)];
  }
  const double total = log_choose(above, need);  // above == above(last) now
  for (int c = layout_.classes - 1; c > last_; --c) {
    if (out[c] != kImpossible) out[c] = static_cast<float>(out[c] - total);
  }
  for (int c = 0; c <= last_; ++c) out[c] = kImpossible;
}

void Cursor::choose(int cls) {
  if (done()) throw std::logic_error("belief_sequence: choice after the assignment is finished");
  if (cls < 0 || cls >= layout_.classes) throw std::invalid_argument("belief_sequence: class out of range");
  const int need = sizes_[at(segment_)] - static_cast<int>(hands_[at(segment_)].size());
  int above = 0;
  for (int c = cls + 1; c < layout_.classes; ++c) above += left_[at(c)];
  if (!left_[at(cls)] || cls <= last_ || above < need - 1) {
    throw std::invalid_argument("belief_sequence: element " + std::to_string(cls) + " cannot come next in segment " +
                                std::to_string(segment_));
  }
  left_[at(cls)] = 0;
  chosen_[at(segment_ * layout_.classes + cls)] = 1;
  hands_[at(segment_)].push_back(cls);
  last_ = cls;
  skip_filled_segments();
}

void append_training_rows(const BeliefSequenceLayout& layout, const std::vector<int>& pool,
                          const std::vector<std::vector<int>>& hands, TrainingRows* out) {
  if (static_cast<int>(hands.size()) != layout.segments) {
    throw std::invalid_argument("belief_sequence: hands do not match the layout's segments");
  }
  Problem problem;
  problem.pool = pool;
  for (const auto& hand : hands) {
    if (static_cast<int>(hand.size()) != layout.classes) {
      throw std::invalid_argument("belief_sequence: a hand does not match the layout's classes");
    }
    for (int v : hand) {
      if (v != 0 && v != 1) throw std::invalid_argument("belief_sequence: a hand holds each element at most once");
    }
    problem.sizes.push_back(std::accumulate(hand.begin(), hand.end(), 0));
  }
  Cursor cur(layout, problem);
  const std::size_t p = at(prefix_dim(layout)), k = at(layout.classes);
  for (int s = 0; s < layout.segments; ++s) {
    for (int c = 0; c < layout.classes; ++c) {
      if (!hands[at(s)][at(c)]) continue;
      out->prefix.resize(out->prefix.size() + p);
      cur.write_prefix(out->prefix.data() + out->prefix.size() - p);
      out->log_prior.resize(out->log_prior.size() + k);
      cur.write_log_prior(out->log_prior.data() + out->log_prior.size() - k);
      out->target.push_back(c);
      cur.choose(c);
    }
  }
}

std::vector<std::vector<std::vector<int>>> sample(const BeliefSequenceLayout& layout, const Problem& problem,
                                                  const std::vector<float>& features,
                                                  const IBeliefEvaluator& evaluator, int n, std::mt19937_64& rng) {
  if (n <= 0) throw std::invalid_argument("belief_sequence: sample count must be positive");
  std::vector<Cursor> cursors(at(n), Cursor(layout, problem));
  const std::size_t p = at(prefix_dim(layout)), k = at(layout.classes);
  // Samples that made the same choices so far are in the same state: the net
  // is asked once per state, not once per sample. (All start in one state,
  // and every sample fills the same segments in the same order, so all are at
  // the same step.)
  std::vector<int> state(at(n), 0);  // of each sample
  std::vector<int> member{0};        // one sample of each state
  std::vector<int> next_member, last_possible;
  std::unordered_map<std::uint64_t, int> successor;  // (state, choice) -> next state
  std::vector<float> prefixes, logits, prior(k);
  std::vector<double> cumulative;  // [states x classes] running sums of the weights
  while (!cursors[0].done()) {
    const std::size_t states = member.size();
    prefixes.resize(states * p);
    for (std::size_t s = 0; s < states; ++s) cursors[at(member[s])].write_prefix(prefixes.data() + s * p);
    evaluator.evaluate_steps(features, prefixes, static_cast<int>(states), &logits);
    if (logits.size() != states * k) {
      throw std::logic_error("belief_sequence: the belief net returned " + std::to_string(logits.size()) +
                             " logits for " + std::to_string(states) + " rows of " + std::to_string(k) + " classes");
    }
    cumulative.assign(states * k, 0.0);
    last_possible.assign(states, -1);
    for (std::size_t s = 0; s < states; ++s) {
      cursors[at(member[s])].write_log_prior(prior.data());
      const float* z = logits.data() + s * k;
      double top = -std::numeric_limits<double>::infinity();
      for (std::size_t c = 0; c < k; ++c) {
        if (prior[c] != kImpossible) top = std::max(top, static_cast<double>(z[c]) + prior[c]);
      }
      if (!std::isfinite(top)) throw std::logic_error("belief_sequence: no element can come next");
      double total = 0.0;
      for (std::size_t c = 0; c < k; ++c) {
        if (prior[c] != kImpossible) {
          total += std::exp((static_cast<double>(z[c]) + prior[c] - top) / kBeliefTemperature);
          last_possible[s] = static_cast<int>(c);
        }
        cumulative[s * k + c] = total;
      }
    }
    successor.clear();
    next_member.clear();
    for (int i = 0; i < n; ++i) {
      const std::size_t s = at(state[at(i)]);
      const double* sums = cumulative.data() + s * k;
      const double r = std::uniform_real_distribution<double>(0.0, sums[k - 1])(rng);
      // The first class whose running sum exceeds r (an impossible class
      // adds nothing, so it is never the first); the last possible class
      // absorbs rounding.
      int pick = static_cast<int>(std::upper_bound(sums, sums + k, r) - sums);
      if (pick > last_possible[s]) pick = last_possible[s];
      cursors[at(i)].choose(pick);
      const std::uint64_t key = (static_cast<std::uint64_t>(s) << 32) | static_cast<std::uint64_t>(pick);
      const auto [it, fresh] = successor.emplace(key, static_cast<int>(next_member.size()));
      if (fresh) next_member.push_back(i);
      state[at(i)] = it->second;
    }
    member.swap(next_member);
  }
  std::vector<std::vector<std::vector<int>>> out;
  out.reserve(at(n));
  for (const Cursor& c : cursors) out.push_back(c.hands());
  return out;
}

}  // namespace board_ai::belief_sequence
