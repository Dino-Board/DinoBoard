#pragma once

// Autoregressive belief over sorted hands (belief.json "type": "autoregressive").
//
// The hidden information is, per segment (an opponent), a hand of distinct
// elements drawn from a pool the observer cannot place (the unseen cards). A
// hand is written smallest class first, segment after segment, and the belief
// net scores the next element given the position features and the elements
// written so far (the prefix). The net has two inputs, so that its part that
// reads the position runs once per position and only a small head runs per
// prefix. Its logits are added to the log-probabilities of the next
// element under a uniform deal (the prior), which are -inf for an element
// that cannot come next: one the pool no longer has, one not above the
// segment's previous element, or one so high that the rest of the segment
// could not be filled. So every sample is a possible assignment, a net that
// outputs zeros deals uniformly, and what the net learns is how the hands
// differ from a uniform deal. Training walks the true assignment through the
// same prefix and prior (teacher forcing): both sides share this one
// implementation.
//
// A game opts in with IBeliefFeatureExtractor::sequence_layout(); its tracker
// calls sample() from prepare_for_root and hands the samples to
// randomize_unseen; its label extractor's hand_counts / remaining are the
// true hands and the pool.

#include <cstdint>
#include <random>
#include <vector>

#include "belief_evaluator.h"
#include "belief_feature_extractor.h"

namespace board_ai::belief_sequence {

// Width of the prefix the net reads after the position features:
//   [segments x classes]  1 for the elements given to each segment so far,
//   [segments]            one-hot of the segment being filled,
//   [classes]             one-hot of that segment's previous element (zeros
//                         at its start).
inline int prefix_dim(const BeliefSequenceLayout& l) { return l.segments * l.classes + l.segments + l.classes; }

struct Problem {
  std::vector<int> pool;   // [classes] 1 for the elements the observer cannot place, else 0
  std::vector<int> sizes;  // [segments] hidden elements of each segment
};

// A partly written assignment.
class Cursor {
 public:
  // Throws when the pool holds an element twice or fewer elements than the
  // segments need.
  Cursor(const BeliefSequenceLayout& layout, const Problem& problem);
  bool done() const { return segment_ >= layout_.segments; }
  void write_prefix(float* out) const;  // prefix_dim(layout) floats
  // layout.classes floats: log-probability of each class coming next under a
  // uniform deal, -inf for one that cannot.
  void write_log_prior(float* out) const;
  void choose(int cls);  // must be possible next
  // [segments][elements], each ascending.
  const std::vector<std::vector<int>>& hands() const { return hands_; }

 private:
  void skip_filled_segments();

  BeliefSequenceLayout layout_;
  std::vector<int> sizes_;
  std::vector<int> left_;    // [classes] pool not yet handed out
  std::vector<int> chosen_;  // [segments x classes]
  std::vector<std::vector<int>> hands_;
  int segment_ = 0;
  int last_ = -1;  // the current segment's previous element
};

// Every step of a true assignment, appended to `out`: the prefix and prior
// before the step and the class chosen at it. `hands[s][c]` = 1 when segment
// s holds element c; `pool` includes those elements. Throws when the
// assignment is not a possible one.
struct TrainingRows {
  std::vector<float> prefix;     // [rows x prefix_dim]
  std::vector<float> log_prior;  // [rows x classes]
  std::vector<int> target;       // [rows]
};
void append_training_rows(const BeliefSequenceLayout& layout, const std::vector<int>& pool,
                          const std::vector<std::vector<int>>& hands, TrainingRows* out);

// `n` assignments sampled from prior x net at kBeliefTemperature: result[i][s]
// is segment s of sample i, ascending. `features` are the position features
// (the extractor's output); the net is run with
// IBeliefEvaluator::evaluate_steps(features, prefixes), one call per hidden
// element with one row per distinct prefix among the samples.
std::vector<std::vector<std::vector<int>>> sample(const BeliefSequenceLayout& layout, const Problem& problem,
                                                  const std::vector<float>& features,
                                                  const IBeliefEvaluator& evaluator, int n, std::mt19937_64& rng);

}  // namespace board_ai::belief_sequence
