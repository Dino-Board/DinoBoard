#pragma once

#include <any>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"

namespace board_ai {

class IBeliefEvaluator;
class IBeliefFeatureExtractor;

// Read an int list out of an AnyMap payload/event value. Two shapes reach a
// tracker: std::vector<int> (py_engine's py_to_any for int-only Python
// lists, and viz slices) and std::vector<std::any> (framework serializers).
// Shared by the standard `seen_cards` restore contract below.
inline std::vector<int> any_int_list(const std::any& value, const char* ctx) {
  if (value.type() == typeid(std::vector<int>)) {
    return std::any_cast<const std::vector<int>&>(value);
  }
  if (value.type() == typeid(std::vector<std::any>)) {
    std::vector<int> out;
    for (const auto& entry : std::any_cast<const std::vector<std::any>&>(value)) {
      if (entry.type() != typeid(int)) {
        throw std::invalid_argument(std::string(ctx) + ": non-int element");
      }
      out.push_back(std::any_cast<int>(entry));
    }
    return out;
  }
  throw std::invalid_argument(std::string(ctx) + ": payload is not an int list");
}

// Observer's memory of a game in progress, derived purely from the
// public-event stream. perspective-agnostic: the same belief content is
// shared by every seat in the AI session — what differs per seat is what
// is publicly observable to *someone*, not who that someone is.
//
// Concretely, this means:
//   - `init` receives the independent session state plus the AnyMap
//     initial-observation payload. It may write perspective-private
//     bootstrap facts into that session state, but tracker memory itself
//     remains public-derived and perspective-agnostic.
//   - `observe_public_event` MUST update internal belief from the given
//     event payloads only — no peeking at truth or any other state.
//   - Per-perspective private knowledge (own hand, peeked opp hand,
//     exchange-seen cards, etc.) lives on state with `viz=1` for the
//     owner — NOT on the tracker. The tracker holds only public-derivable
//     aggregates (claim history, public discard pile composition, …).
//
// Usage:
//   - Each AI session holds its own tracker. They all converge to the
//     same belief content from the same observation stream — independent
//     storage exists only so each session can clone its tracker into
//     each MCTS sim (`clone()` + sim-local `observe_public_event`).
//   - GT does NOT hold a tracker — GT runs `do_action_fast` on truth,
//     which already encodes everything `randomize_unseen` would invent.
//
// Enforced by tests:
//   - test_api_belief_matches_selfplay — tracker initialized with a
//     different seed than ground truth must converge to the same belief
//     after replaying the observation stream. Any hidden-state read
//     would diverge.
//   - test_api_mcts_policy_invariance — MCTS policy depending on tracker
//     belief must be identical across selfplay and API paths.
class IBeliefTracker {
 public:
  virtual ~IBeliefTracker() = default;

  // Initialize at game start.
  //
  // Two-step initialization protocol (the per-ply path's parallel):
  //   - per-ply: walker `apply_public_snapshot` overwrites every viz=1
  //     slot and the perspective's `__viz__` slice, then
  //     `observe_public_event` updates tracker memory from the event
  //     stream;
  //   - opening: walker `apply_public_snapshot` does the same wholesale
  //     replace from the opening snapshot, then `init` is called with a
  //     `payload` AnyMap carrying any perspective-private bootstrap that
  //     the public broadcast can't reach (e.g. Love Letter's
  //     `hand[perspective]`, Coup's two own influence cids). The tracker
  //     (a) writes those private values into `state` (and toggles
  //     `state.viz_` if needed), and (b) seeds its own internal memory
  //     by reading public facts from `state`.
  //
  // payload's shape is game-specific and produced by `pack_init_payload`
  // on the GT side. For fully-public / symmetric-random games and games
  // whose perspective-private bootstrap is empty, payload is empty and
  // the default impl does nothing beyond initializing tracker memory.
  //
  // In selfplay / arena / heuristic paths the per-seat session state is
  // initialized with the same seed as truth and already carries correct
  // data + correct viz, so payload is empty and `init` is a no-op-on-state
  // (just seeds tracker memory). In the API / web session path the session
  // state was seeded independently and its perspective-private slots hold
  // wrong values; payload carries the correct ones.
  //
  // The payload is an authoritative restore, not merely an opening-time
  // hint. `pack_init_payload` may be called on a GT state at ANY ply, so a
  // peer session can be bootstrapped from a mid-game `initial_observation`
  // (external clients that only see the current position, e.g. a BGA/table
  // adapter, cannot stream the full action history). When the payload
  // carries tracker memory, `init` MUST restore it instead of re-deriving
  // memory from `state` — the re-derivation only knows public fields, not
  // what has already left the hidden pool.
  virtual void init(IGameState& state, int perspective,
                    const AnyMap& payload) = 0;

  // GT-side: package the bootstrap that the walker's public broadcast
  // cannot reach. Called on the GT state when building the wire
  // `initial_observation` (paired with `viz::serialize_public_snapshot`
  // for the public part). Default: empty — override in games whose
  // opening reveals owner-private slots (LL hand, Coup own influences) or
  // whose tracker memory is not fully re-derivable from the current public
  // state (Splendor's seen-card multiset).
  //
  // Contract for mid-game bootstrapping: the payload must be a COMPLETE
  // restore of everything the peer's tracker needs, i.e. a fresh session
  // that applies `initial_observation` taken at ply N must end up with the
  // same belief content as a session that streamed every action from ply 0.
  // Shared convention for card/entity-pool games:
  //   payload["seen_cards"] = std::vector<int> of entity ids the tracker
  //                           has retired from the hidden pool
  //                           (publicly revealed / consumed / known).
  // Games are free to define additional keys. The output is fed back as
  // `payload` to a peer session's `init` after walker
  // `apply_public_snapshot` has run.
  virtual AnyMap pack_init_payload(
      const IGameState& /*gt_state*/, int /*perspective*/) const {
    return {};
  }

  // Update after each action using ONLY the public event stream.
  //
  //   actor: the player whose action was taken
  //   action: the ActionId taken by `actor`
  //   events: public observations the perspective player can derive from
  //       this transition (e.g. Splendor deck flip, Coup card_revealed).
  //       List order is the producer's emission order; the tracker treats
  //       it as an observation log.
  //
  // Tracker must update its belief state solely from these payloads.
  // Payload kinds and keys are defined by the game in
  // public_event_extractor.
  virtual void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& events) = 0;

  // Randomize all unseen information in-place, producing a world
  // consistent with `observer`'s information set.
  //
  // `observer` is the perspective for whom slots are filled: any slot
  // where `state.viz_[..., observer] == 0` is considered unseen and gets
  // a fresh sample; slots with viz=1 are observer-known truth and stay.
  //
  // Information sources:
  //   - state's `viz=1` slots (own hand, public revealed cards, scores,
  //     discards, deck sizes visible by counts, …) — the observer's
  //     current concrete knowledge.
  //   - tracker's own derived public history — public-event aggregates
  //     not present as state fields (claim sequences, publicly-revealed
  //     card multisets, …). Tracker content is perspective-agnostic:
  //     two trackers seeded differently but fed the same public event
  //     stream produce equal aggregates.
  //
  // Unseen-pool formula (canonical):
  //     unseen_pool = full_pool − tracker.public_seen − state.viz=1[observer]
  // Then unseen_pool is shuffled and consumed to fill viz=0 slots.
  //
  // Contract: the result must satisfy every public invariant —
  // `state_hash_for_perspective(observer)` on the output is byte-equal
  // across any two trackers with the same observation history, regardless
  // of the input state's hidden contents or the caller's RNG.
  //
  // Called in exactly one place (per DEC-003): at MCTS world entry on a
  // cloned world_tracker. With world_cap > 0, multiple inner sims clone
  // that already-randomized world template. The session itself does NOT
  // call randomize_unseen — its viz=0 slots are unread bytes that the
  // framework structurally hides from the hash (kHiddenHashSentinel), the
  // encoder (masked-state placeholder), and from world entry (search
  // clones and resamples independently).
  virtual void randomize_unseen(IGameState& state, int observer,
                                std::mt19937_64& rng) const = 0;

  // Deep-copy this tracker. Search first clones the session tracker into a
  // world_tracker, randomizes that world once, then each inner sim clones
  // the world_tracker so descent-time `observe_public_event` calls don't
  // pollute either the session tracker or the world template.
  virtual std::unique_ptr<IBeliefTracker> clone() const = 0;

  // Per-decision belief-net hook. Called once at MCTS root before any
  // simulation; the tracker (a) extracts a feature vector via the
  // extractor, (b) runs the belief-net evaluator to obtain (N-1) × K
  // logits, (c) softmaxes with its own temperature and caches the
  // resulting `pi[opp][R]` posterior internally. Subsequent world_tracker
  // clones inherit that posterior; their `randomize_unseen` calls draw
  // determinization samples from it. The exact sampling shape is per-game —
  // current implementations (Coup multiset categorical over the alive-
  // count branch, LiarsDice categorical over enumerated Σ=K outcomes)
  // both consume `pi[opp][R]` directly without per-slot Wallenius
  // weighting; a game whose unseen pool is per-slot exchangeable could
  // alternatively wire Wallenius hypergeometric on top of `pi[opp][R]`.
  //
  // Default implementation is a no-op — games with a flat-uniform
  // determinization policy (Splendor multiset, Love Letter discard
  // tracking) override neither this nor extractor; their
  // `randomize_unseen` does not consult `pi`.
  //
  // `extractor` and `evaluator` may be null. When null, the tracker
  // MUST behave as if no belief net is installed (uniform sampling).
  // The pointers are held only for the duration of the call; the
  // tracker may store the resulting posterior but not the pointers.
  virtual void prepare_for_root(
      const IGameState& /*root_state*/,
      int /*root_player*/,
      const IBeliefFeatureExtractor* /*extractor*/,
      const IBeliefEvaluator* /*evaluator*/) {}

  // Serialize the tracker's internal belief to a canonical, comparable form.
  // Used by the AI API belief-equivalence tests: a self-play session and an
  // API session with different seeds must agree on belief after the same
  // action + event sequence. The returned map must be deterministic across
  // identical internal states — sort sets, use stable keys. The default
  // returns an empty map for trackers that hold no explicit state.
  virtual AnyMap serialize() const { return {}; }
};

}  // namespace board_ai
