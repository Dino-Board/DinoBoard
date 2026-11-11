#pragma once

#include <array>
#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"

namespace board_ai {

class IBeliefTracker;

// Describes an encoder whose flat feature vector is `num_tokens` blocks of
// `token_dim` floats, consumed by a set-style (transformer) network that
// reshapes `[B, num_tokens * token_dim]` to `[B, num_tokens, token_dim]`.
// Per token: [0] = valid flag (0 → masked out of attention), [1] = entity
// id (0 = none, else in [1, num_entity_ids)), [2..] = type-specific
// features. Token types are fixed per slot and carry no slot ordinal, so
// interchangeable slots (e.g. cards within a tier) stay permutable.
// Policy: action a is scored by pointer head `action_head[a]` on token
// `action_token[a]`, or by the global head when both are -1.
//
// Anchors (optional, `num_anchor_ids > 0`) bind actions to whichever token
// currently sits at a position, for games whose slots move between states
// (e.g. frontier cells of a tile-laying board). Each token then carries an
// anchor id at [2] (0 = none, else in [1, num_anchor_ids]) and its
// type-specific features start at [3]; the network adds a learned embedding
// per anchor id. Pointer head h on a token anchored at id k scores action
// `anchor_action[(k - 1) * num_pointer_heads + h]` (-1 = no action). An
// action is bound statically or by anchor, never both, and at most one
// valid token per state may carry a given anchor id.
//
// Variable-length layouts (`variable_tokens`) have no slots: each encoding
// is n * token_dim floats for its own n >= 1, so collections of any size
// (a hand, a zoo) need no capacity. `feature_dim()` is 0, `num_tokens` and
// `token_types` are unused, every action is bound by anchor or the global
// head (`action_token` / `action_head` all -1), and each token carries
// [0] = 1, [1] = entity id, [2] = anchor id, [3] = token type, features from
// [4]. Batches are padded with all-zero tokens to `padded_token_count` of
// their longest encoding; padding tokens are invalid and never read.
//
// Token positions (`token_positions`, variable layouts only) put tokens on a
// 2D grid (square or axial hex coordinates): each token carries [4] = 1 if it
// has a position, [5] = x, [6] = y (integers), and features start at [7]. The
// network biases attention between two positioned tokens by their offset
// (dx, dy), so it can learn local relations (neighbours, short paths)
// independent of where on the board they are.
struct TokenLayout {
  int num_tokens = 0;
  int token_dim = 0;
  int num_token_types = 0;
  int num_entity_ids = 0;
  int num_pointer_heads = 0;
  std::vector<int> token_types;
  std::vector<int> action_token;
  std::vector<int> action_head;
  int num_anchor_ids = 0;
  std::vector<int> anchor_action;
  bool variable_tokens = false;
  bool token_positions = false;
};

// A board symmetry for data augmentation (train.json
// `training.symmetry_augmentation`). Symmetry g maps a state s to an
// equivalent state g(s): legal actions correspond through `action_map` and
// the outcome is the same. The encoder states how its encoding of s becomes
// its encoding of g(s), so training transforms stored samples instead of
// re-encoding:
//   flat:    encode(g s)[feature_map[i]] = encode(s)[i]
//   variable token layouts: every token of encode(s) is a token of
//            encode(g s) (token order does not matter to the network) with
//            anchor id anchor_map[anchor] (anchor_map[0] = 0), position
//            (x, y) -> (a x + b y + e, c x + d y + f), position_affine =
//            {a, b, c, d, e, f}, if it has one, and column c moved to
//            column_map[type][c] (empty = unchanged); the header columns
//            [0, 7) always map to themselves.
//   policy:  a target over the actions of s, with action a moved to
//            action_map[a], is the target over the actions of g(s).
// symmetries()[0] is the identity; no symmetries = no augmentation.
struct EncoderSymmetry {
  std::vector<int> feature_map;
  std::vector<int> anchor_map;
  std::array<int, 6> position_affine{1, 0, 0, 1, 0, 0};
  std::vector<std::vector<int>> column_map;
  std::vector<int> action_map;
};

// Token count a batch of variable-length encodings is padded to: at least
// n, with n + 1 (the network's CLS token) a multiple of 8, which
// onnxruntime's fused fp16 attention needs for its additive bias.
constexpr int padded_token_count(int n) { return (n + 1 + 7) / 8 * 8 - 1; }


// Feature encoder reading a perspective-masked state (golden standard
// §2.5 / I13).
//
// Input surface = a state with hidden slots overwritten by kPlaceholder*
// + tracker (ALGORITHM_OVERVIEW §6). The masked state carries every
// state field the perspective can legally see; the optional `tracker`
// carries public derived statistics the network wants but the rules
// engine doesn't (e.g. claim history, multiset summaries). Games that
// don't need tracker-sourced features ignore the parameter; the tracker
// pointer may be null when no tracker is registered for the game.
//
// `encode` materializes the masked state once via `make_masked_state(
// state, schema, perspective)` and feeds it to the encoder. Encoders
// read the masked state by field name; any slot whose runtime viz
// hides it from `perspective_player` shows up as `kPlaceholder*` and
// the encoder branches on the placeholder value, not on a viz query.
// The signature is `const IGameState&` because there is no distinct
// MaskedState type — the placeholder write is the structural barrier,
// enforced by `make_masked_state` always being the producer of states
// fed to `encode_features` (and by `test_encoder_respects_hash_scope`).
class IFeatureEncoder {
 public:
  virtual ~IFeatureEncoder() = default;
  virtual int action_space() const = 0;
  // 0 for variable-length token encoders.
  virtual int feature_dim() const = 0;
  // Non-null only for token-structured encoders (see TokenLayout).
  virtual const TokenLayout* token_layout() const { return nullptr; }
  // Board symmetries of the encoding (see EncoderSymmetry); empty if none.
  virtual std::vector<EncoderSymmetry> symmetries() const { return {}; }

  bool variable_tokens() const {
    const TokenLayout* layout = token_layout();
    return layout && layout->variable_tokens;
  }
  // Whether `n` floats is a whole encoding: feature_dim(), or a positive
  // multiple of token_dim for variable-length token encoders.
  bool valid_feature_length(std::size_t n) const {
    if (!variable_tokens()) return n == static_cast<std::size_t>(feature_dim());
    const auto token_dim = static_cast<std::size_t>(token_layout()->token_dim);
    return n > 0 && n % token_dim == 0;
  }

  // Encode the full feature vector for `perspective_player`, reading
  // from a perspective-masked state. Slots hidden from the perspective
  // arrive as kPlaceholder* — encoder must branch on placeholder, never
  // query viz. `tracker` is the perspective's belief tracker (or null
  // if the game didn't register one); tracker reads are public-derived
  // statistics only, perspective-private knowledge does not live there.
  virtual void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;

  // Composes the perspective's features + legal mask into the flat
  // feature vector consumed by the network. Materializes the masked
  // state once. Games override `encode_features`, not this.
  bool encode(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const {
    auto masked = make_masked_state(state, state.schema_ref(),
                                    perspective_player);
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    encode_features(*masked, perspective_player, tracker, features);
    fill_legal_mask_impl(legal_actions, legal_mask);
    return valid_feature_length(features->size());
  }

  // Caller already has a masked state (e.g. MCTS descent that masked
  // once for hashing). Skips the second clone+mask.
  bool encode_with_masked(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const {
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    encode_features(masked_state, perspective_player, tracker, features);
    fill_legal_mask_impl(legal_actions, legal_mask);
    return valid_feature_length(features->size());
  }

 private:
  void fill_legal_mask_impl(
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* legal_mask) const {
    const int a_space = action_space();
    legal_mask->assign(static_cast<size_t>(a_space), 0.0f);
    for (ActionId a : legal_actions) {
      if (a >= 0 && a < a_space) {
        (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
      }
    }
  }
};

inline void fill_legal_mask(
    int action_space,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* legal_mask) {
  legal_mask->assign(static_cast<size_t>(action_space), 0.0f);
  for (ActionId a : legal_actions) {
    if (a >= 0 && a < action_space) {
      (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
    }
  }
}

}  // namespace board_ai
