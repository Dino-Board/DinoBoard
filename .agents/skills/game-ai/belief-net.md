# Learned Belief Network Reference

## When to use

Uniform `randomize_unseen` degenerates when opponents' hidden state is strongly
conditioned on the public action history. Coup opponents misrepresent roles,
Liar's Dice opponents bluff about die faces, and 6 nimmt! opponents' remaining
hands are constrained by public play history. A learned belief net replaces the
uniform prior with a posterior conditioned on public observations (claims,
challenges, reveals, bid sequences, played-card history).

## GameBundle fields (3 belief-related)

```cpp
// engine/core/game_registry.h — inside GameBundle
std::unique_ptr<IBeliefFeatureExtractor> belief_feature_extractor;
std::string                              belief_model_path;
std::unique_ptr<IBeliefLabelExtractor>   belief_label_extractor;
```

All three are optional at the framework level. Runtime learned belief needs a
feature extractor, a non-empty `belief_model_path`, and a tracker whose
`prepare_for_root` consumes the evaluator output. The label extractor is GT-side
training plumbing: it is needed to emit belief samples, but it is not read by
MCTS or AI sessions. When no model path/evaluator is available,
`prepare_for_root` behaves like a no-op and `randomize_unseen` falls back to the
tracker's uniform / multiset-derived sampling.

## IBeliefFeatureExtractor (engine/core/belief_feature_extractor.h)

```cpp
class IBeliefFeatureExtractor {
  virtual int feature_dim() const = 0;
  virtual int output_logit_count() const = 0;   // (N-1) x K flat, game-specific K
  virtual void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;
};
```

Input contract identical to IFeatureEncoder: reads a perspective-masked state
(viz=0 slots = kPlaceholder). `tracker` carries public-derived statistics
(claim history, reshuffle counts). May be null.

Convenience: `extract_from_state()` materializes the masked clone and forwards.

`K` is the per-opponent output width, not necessarily a mutually-exclusive
class count. Coup uses two categorical branches, Liar's Dice uses per-face
categoricals that are reconstructed into a 252-outcome joint, and 6 nimmt! uses
104 independent sigmoid logits.

## IBeliefLabelExtractor (engine/core/belief_label_extractor.h)

GT-side only. Reads truth IGameState; never reachable from AI session. Only
selfplay_runner consults it (emit path). Output per call uses the framework's
common tensor shape (sized to observer, N-1, K):

- `hand_counts[i][R]` = unrevealed R-cards in opp i's hand
- `remaining[R]` = total_R - observer's own unrevealed R - publicly revealed R
- `alive_per_opp[i]` = 1 iff opp i still has unrevealed influence (row mask in loss)

The framework transports these tensors but does not impose one universal loss.
Coup uses alive-selected categorical CE, Liar's Dice uses per-face categorical
targets, and 6 nimmt! uses multi-hot BCE over 104 cards.

```cpp
class IBeliefLabelExtractor {
  virtual int label_class_count() const = 0;  // K (Coup: 5, LL: 8)
  virtual void extract(
      const IGameState& truth_state, int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const = 0;
};
```

## IBeliefEvaluator (engine/core/belief_evaluator.h)

Runtime evaluator. Single input (flat feature vector), single output (flat
(N-1) x K logits). ONNX-backed implementation: `OnnxBeliefEvaluator`
(`engine/infer/onnx_belief_evaluator.h`). Logits returned uncalibrated; the
tracker applies per-game post-processing (softmax branches, sigmoid marginals,
joint reconstruction, feasibility masks) before `randomize_unseen` samples a
world.

## belief.json config (games/<id>/config/belief.json)

```json
{
  "architecture": [256, 256, 256, 256, 256, 256],
  "activation": "relu",
  "batch_norm": true,
  "residual": true,
  "block_size": 2,
  "training": { "loss": "ce" }
}
```

Separate from PV net. Independent ONNX session, independent cache slot.
Per-variant independent training (e.g. coup_belief_2p.onnx).
Presence of `belief.json` is the enable signal when the engine registers belief
features; there is no `enabled` field. `activation` is currently validated as
`"relu"` because no other activation is implemented.

## Belief temperature invariant

Belief logit temperature is framework-fixed at `1.0`
(`board_ai::kBeliefTemperature`). It is not a game knob and must not appear in
`belief.json`, game registration, or tracker-local constants. If
`belief.json` contains `temperature`, the training config loader rejects it.

Trackers may still use per-game post-processing shapes (softmax branches,
sigmoid marginals, joint reconstruction, feasibility masks), but the logit
temperature inside those transforms stays `1.0`. Calibration belongs in the
trained model, not a hidden runtime transform.

Current training losses are game-specific (`ce` for categorical heads, `bce`
for 6 nimmt!'s multi-hot card ownership head).

## Tracker integration: prepare_for_root

```cpp
// IBeliefTracker (engine/core/belief_tracker.h)
virtual void prepare_for_root(
    const IGameState& root_state,
    int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator);
```

Called once at MCTS root before any simulation. Flow:
1. Extractor produces feature vector from masked root state + tracker
2. Evaluator returns (N-1) x K logits
3. Tracker converts logits into a per-game posterior cache
4. World trackers cloned from this root tracker inherit the cache
5. `randomize_unseen` at MCTS world entry samples hidden slots from the cache;
   inner sims clone the already-randomized world template

## Sampling shape (per-game)

| Game | Output shape | Post-processing | `randomize_unseen` sampling |
|------|--------------|-----------------|------------------------------|
| Coup | Per opp 20 logits: `[15 two-card multiset | 5 single-card]` | Pick branch by current unknown alive count, softmax within that branch at fixed `T=1.0`, cache `pi_hand_[opp]` | Shuffle opponent order. For each opponent, mask infeasible hands whose required role counts exceed remaining deck counts, renormalize, sample one whole hand, subtract it from `remaining`, then write the selected role multiset into viz=0 influence/exchange slots. |
| Liar's Dice | Per opp 36 logits: `6 faces x 6 count bins` | Softmax each face's count distribution at fixed `T=1.0`, build the 252 legal `sum=5` outcome joint by product and IPF renormalization, cache `pi_outcome_[opp]` | For each opponent, sample one of the 252 cached dice-count outcomes and write it into that opponent's hidden `dice_count`. |
| 6 nimmt! | Per opp 104 logits, one per card | Apply sigmoid at fixed `T=1.0` to get marginal `P(opp holds card c)`, cache `belief_probs_[opp][card]` | Build the unseen card pool from public played cards and observer-known hand/pending cards. For each opponent, sample the needed number of cards without replacement by Plackett-Luce / weighted multinomial using cached marginals; fall back to uniform if the residual weight mass is degenerate. |

These shapes are deliberately per-game. Do not assume every belief net is a
single categorical distribution or that every tracker consumes `pi[opp][R]`
directly.

## Decision record

DEC-004: belief net is independent of PV net; separate ONNX, separate training
loop, separate config. The feature extractor mirrors IFeatureEncoder's masked-state
contract so no truth can leak. Labels are GT-side only (emit-file-train pipeline,
same as PV samples).
