#pragma once

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "types.h"
#include "visibility_schema.h"

namespace board_ai {

// ---- Event protocol primitives (used by IBeliefTracker and game bundle) ----
// AnyMap: game-specific payload for events and observations. Keys are
// stable string names chosen by the game; values are wrapped in std::any
// so different games can carry different payload types.
using AnyMap = std::map<std::string, std::any>;

// A single public event: (kind, payload). Kind is a stable string chosen by
// the game (e.g. "deck_flip", "opp_buy_reserved_reveal"). Payload keys/values
// are game-defined.
using PublicEvent = std::pair<std::string, AnyMap>;

// Events + public snapshot for a single action.
//
// `events` — public observations the perspective player can derive from this
// transition. Consumed only by the belief tracker (observer sessions never
// run `do_action_fast`, so events have no in-rules timing to ride). The list
// order matches the producer's emission order; the tracker treats it as an
// observation log.
//
// `public_snapshot` — truth-side serialization of the post-action visible
// fields for this perspective. Games that register a `public_state_applier`
// populate this in their extractor; observer sessions then overwrite their
// state_'s public fields from it at the end of apply_observation. Empty map
// for fully-public games (tictactoe, quoridor) that do not register an
// applier.
struct PublicEventTrace {
  std::vector<PublicEvent> events{};
  AnyMap public_snapshot{};
};

class IGameState {
 public:
  virtual ~IGameState() = default;
  virtual std::unique_ptr<IGameState> clone_state() const = 0;
  virtual void copy_from(const IGameState& other) = 0;

  // Legacy full-state hash. Retained for non-ISMCTS paths (tail solver,
  // transposition debug). New code paths use state_hash_for_perspective.
  virtual StateHash64 state_hash() const = 0;

  // ========== Schema-driven hash API ==========
  //
  // `state_hash_for_perspective(player)` walks the FULL schema (every
  // slot, visible and hidden) in declaration order × row-major idx.
  // For each slot the framework first mixes the structural position
  // (field index in schema + idx[]), then dispatches the value mix:
  //   - visible to `player` (viz=1): call `hash_field_slot`, which
  //     mixes only the slot's value.
  //   - hidden from `player` (viz=0): mix `kHiddenHashSentinel`.
  //
  // Every piece of game state that participates in DAG node identity
  // MUST be a schema slot. There is no off-schema escape hatch — if
  // you find yourself wanting one, declare a fixed-shape count array
  // and treat the variable-length structure as a (multiset, ordering)
  // pair where ordering is purely a presentation concern reconstructed
  // from the action stream. (Why: any off-schema state silently breaks
  // the framework's "two states with the same observation history
  // produce the same digest" guarantee — see BUG-037 postmortem.)

  // Typed value emission for one schema slot. The framework walker
  // visits every visible (fid, idx) and dispatches here; games answer
  // with `h.add(this->myfield[idx...])`. Walker has already filtered
  // on viz, so this must NOT re-consult viz_.
  //
  // `fid` is the field's index in `schema_ref().fields` — games define
  // a per-game enum at the top of `<game>_state.cpp` matching the
  // schema declaration order, then dispatch with `switch (fid)`. This
  // replaces the previous string-compare chain on `name`, which was the
  // dominant cost in the hash hot path for fields with many slots.
  virtual void hash_field_slot(Hasher& /*h*/, viz::FieldId /*fid*/,
                               const std::vector<int>& /*idx*/) const {}

  // Bulk hash for one field. The framework hash loop dispatches one
  // call per FIELD (not per slot) to amortize per-slot virtual + switch
  // dispatch on fields with many slots (6nimmt's bullhead_pile / hand).
  // Implementations MUST replicate the framework per-slot semantics
  // exactly: for every data-axis slot in row-major order, mix the
  // structural position (fid + idx axes) and then the value mix —
  // visible (viz=1) → game-specific value, hidden (viz=0) →
  // kHiddenHashSentinel.
  //
  // The default body in `mask_state_impl.cpp` walks `v` and dispatches
  // per slot to `hash_field_slot` / kHiddenHashSentinel — bit-equal to
  // the per-slot loop, so games that don't override pay the same cost
  // they paid before (no behavior change). Games override only for
  // hot fields where a tight inlined loop wins.
  virtual void hash_field_bulk(Hasher& h, viz::FieldId fid,
                               const viz::VizTensor& v,
                               int perspective) const;

  // Bulk hidden-slot masking for one field. Mirrors hash_field_bulk:
  // framework calls one method per FIELD; game writes placeholder into
  // every slot whose viz[..., perspective] == 0. Default in
  // `mask_state_impl.cpp` walks hidden slots and dispatches per slot
  // to `mask_field_slot`. Games override for hot all-hidden / many-
  // slot fields where a memset / direct-array-fill is cheaper.
  virtual void mask_field_bulk(viz::FieldId fid,
                               const viz::VizTensor& v,
                               int perspective);

  // Typed slot read/write for walker-driven snapshot wire I/O
  // (`viz::serialize_public_snapshot` / `viz::apply_public_snapshot`).
  // Game returns a typed std::any for the slot, or accepts one back.
  // Mirrors hash_field_slot dispatch shape. Wire format remains
  // string-keyed (I15) — the snapshot path translates string ↔ FieldId
  // via `schema_lookup` in `snapshot_io.h`.
  virtual std::any read_field_slot(viz::FieldId /*fid*/,
                                   const std::vector<int>& /*idx*/) const {
    return {};
  }
  virtual void write_field_slot(viz::FieldId /*fid*/,
                                const std::vector<int>& /*idx*/,
                                const std::any& /*value*/) {}

  // Polymorphic accessor for the game's static VisibilitySchema.
  // Framework helpers (`make_masked_state`, MCTS descent hoist) call
  // this to drive the walker without knowing the concrete state type.
  virtual const viz::VisibilitySchema& schema_ref() const = 0;

  // Framework-provided perspective hash. Walks the FULL schema, mixes
  // (field_pos, idx[]) for every slot, then dispatches the value mix:
  // visible → `hash_field_slot`, hidden → `kHiddenHashSentinel`.
  // NOT virtual — games extend only via `hash_field_slot`. Definition
  // lives in schema_hash.h to break the include cycle.
  StateHash64 state_hash_for_perspective(int player) const;

  // ========== Framework-provided step counter ==========
  //
  // Monotonically increasing across do_action_fast calls. Guarantees DAG
  // acyclicity in MCTS: any two states along a sim path have distinct
  // step_count → distinct hash (state_hash_for_perspective folds it in
  // automatically).
  //
  // **Game rules code never calls these directly.** The framework wrappers
  // around IGameRules::do_action_fast / do_action_deterministic /
  // undo_action are the only callers from the rules path. The session /
  // runner code that applies a public snapshot to per-seat state (without
  // running rules) calls begin_step_for_session_observe() to keep the seat
  // state's step_count_ in sync with truth. Game `reset_with_seed`
  // overrides call reset_step_count_base() to zero the counter on reset.
  std::uint32_t step_count() const { return step_count_; }

  // Framework-internal: bump step_count when applying a public snapshot to
  // a per-seat session state (rules are not run on the session). Named
  // verbosely to discourage accidental use from game code; the only legit
  // callers are py_engine.cpp's apply_observation, the AI-view advance
  // helper, and the runners' advance_per_seat_states. This stays public on
  // the IGameState surface so framework callers (which don't share a
  // friend relationship with IGameState) can reach it.
  void begin_step_for_session_observe() { ++step_count_; }

  // Single canonical base setup for reset_with_seed overrides. Resets the
  // framework-managed step counter; games call this as the first line of
  // their reset_with_seed override. RNG is NOT framework state — each
  // game's reset_with_seed constructs a one-shot std::mt19937_64 from the
  // seed for its initial deal/shuffle and drops it on return.
  void reset_step_count_base() {
    step_count_ = 0;
  }

  virtual int current_player() const = 0;
  virtual int first_player() const { return 0; }
  virtual bool is_terminal() const = 0;
  virtual int num_players() const = 0;
  virtual int winner() const = 0;
  virtual bool is_turn_start() const { return true; }

  virtual std::vector<int> winners() const {
    const int w = winner();
    if (w < 0) return {};
    return {w};
  }

  virtual std::vector<int> losers() const {
    const int n = num_players();
    const int w = winner();
    if (w < 0) return {};
    std::vector<int> result;
    for (int p = 0; p < n; ++p) {
      if (p != w) result.push_back(p);
    }
    return result;
  }
  virtual void reset_with_seed(std::uint64_t seed) = 0;

 protected:
  // Counter for DAG acyclicity. Framework-managed. Reset to 0 in
  // reset_with_seed implementations; bumped by IGameRules wrappers around
  // do_action_fast / do_action_deterministic; rolled back by undo_action
  // wrapper. Also bumped by begin_step_for_session_observe() on the
  // session-side public-snapshot path.
  std::uint32_t step_count_ = 0;

  // IGameRules wrappers are the rules-path callers that bump / roll back
  // step_count_. Friended so the helpers stay invisible to game code.
  friend class IGameRules;

 public:
  // Per-state visibility tensor, indexed by FieldId (= position in
  // schema_ref().fields). Initialized by viz::init_viz at
  // reset_with_seed by push_back-ing each FieldDecl's base_viz in
  // declaration order, so viz_[fid] is the viz tensor for that field.
  //
  // GT-side: mutated ONLY by rules inside do_action_fast via reveal_slot /
  // reveal_slot_to / reset_to_base (golden standard I1 — rules are the
  // sole *game-side* writer; enforced by test_rules_sole_viz_writer).
  // Receiver / session path: the framework helper `viz::apply_full_slice`
  // (invoked from `viz::apply_public_snapshot`) overwrites
  // `viz_[fid][..., perspective]` byte-for-byte from the perspective's
  // viz slice carried on the wire under the `__viz__` key — receiver-side
  // reconstruction of GT-side writes, not a second game-side writer.
  // Framework readers (hash walker, snapshot serializer,
  // mask_all_hidden_slots) read it directly via FieldId.
  std::vector<viz::VizTensor> viz_;

  // Per-slot placeholder write for hidden slots. Walker calls this for
  // each (fid, idx) where viz[idx, perspective] == 0. Game writes
  // kPlaceholderInt32 / kPlaceholderInt8 / kPlaceholderBool into the
  // matching typed slot. Must be idempotent. Default no-op covers
  // fully-public games whose walker never visits a hidden slot.
  virtual void mask_field_slot(viz::FieldId /*fid*/,
                               const std::vector<int>& /*idx*/) {}

  // Drives the walker over hidden slots and dispatches each to
  // `mask_field_slot`. Virtual so games with COW-shared state (e.g.
  // Splendor's shared_ptr<const SplendorData>) can detach a writable
  // copy once before running the walker, instead of reseating per slot.
  virtual void mask_all_hidden_slots(
      const viz::VisibilitySchema& schema, int perspective);
};

template <typename Derived>
class CloneableState : public IGameState {
 public:
  std::unique_ptr<IGameState> clone_state() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
  void copy_from(const IGameState& other) override {
    static_cast<Derived&>(*this) = static_cast<const Derived&>(other);
  }
};

template <typename ConcreteState>
const ConcreteState& checked_cast(const IGameState& state) {
  const auto* p = dynamic_cast<const ConcreteState*>(&state);
  if (!p) throw std::invalid_argument("unexpected IGameState subtype");
  return *p;
}

template <typename ConcreteState>
ConcreteState& checked_cast(IGameState& state) {
  auto* p = dynamic_cast<ConcreteState*>(&state);
  if (!p) throw std::invalid_argument("unexpected IGameState subtype");
  return *p;
}

class IGameRules {
 public:
  virtual ~IGameRules() = default;
  virtual bool validate_action(const IGameState& state, ActionId action) const = 0;
  virtual std::vector<ActionId> legal_actions(const IGameState& state) const = 0;

  // The id every player observes when `action` is taken in `state_before`.
  // Action ids are public: belief trackers, observation traces and AI
  // sessions only ever receive this projection. Games whose action ids can
  // name a hidden object staying hidden (a card tucked face down) return a
  // stand-in id that is never legal; it must depend on public information
  // only. Default: the action itself.
  virtual ActionId public_action(const IGameState& /*state_before*/, ActionId action) const {
    return action;
  }

  // ===== Framework wrappers =====
  //
  // These are the public entry points callers (MCTS, runner, session, tail
  // solver) use. They are NOT virtual: the framework forces step_count
  // bookkeeping around the game-defined impl, so step_count_ stays an
  // invariant the game author cannot break by forgetting to call begin_step
  // / end_step (and cannot accidentally double-bump).
  //
  // do_action_fast: caller-owned RNG threads through hidden draws (deck
  // pulls, bag draws). The state itself holds no rng; the runner / session
  // owns one and refreshes it between sims. Returns void — the fast path
  // does NOT support undo (MCTS / selfplay / web all discard their state).
  //
  // do_action_deterministic: tail-solver path; same as fast but never draws
  // from hidden sources. Games with physical randomness in fast should
  // override the impl — e.g. Splendor uses `forced_draw_override = -2`
  // sentinel to mean "freeze the random source". Returns an UndoToken
  // pairable with undo_action.
  //
  // undo_action: paired with do_action_deterministic only. Must restore
  // the state byte-equally so alpha-beta on a single state object works.
  void do_action_fast(IGameState& state, ActionId action,
                      std::mt19937_64& rng) const {
    state.step_count_ += 1;
    do_action_fast_impl(state, action, rng);
  }
  UndoToken do_action_deterministic(IGameState& state, ActionId action) const {
    state.step_count_ += 1;
    return do_action_deterministic_impl(state, action);
  }
  void undo_action(IGameState& state, const UndoToken& token) const {
    undo_action_impl(state, token);
    state.step_count_ -= 1;
  }

 protected:
  // ===== Game-defined impls =====
  //
  // Game authors override these. step_count is already bumped before the
  // fast / deterministic impls run, and rolled back after undo runs — the
  // game cannot see step_count and cannot break the DAG-acyclicity
  // invariant by forgetting / double-calling.
  virtual void do_action_fast_impl(IGameState& state, ActionId action,
                                   std::mt19937_64& rng) const = 0;
  virtual void undo_action_impl(IGameState& state,
                                const UndoToken& token) const = 0;

  // Default impl forwards to do_action_fast_impl with a throwaway rng,
  // returning a bare token. Subclasses that have hidden draws MUST override
  // this to suppress them.
  virtual UndoToken do_action_deterministic_impl(IGameState& state,
                                                 ActionId action) const {
    std::mt19937_64 unused_rng(0);
    do_action_fast_impl(state, action, unused_rng);
    return UndoToken{};
  }

  // Static dispatch helpers for derived classes that need to forward into
  // another IGameRules instance's impl (e.g. FilteredRulesWrapper, which
  // wraps a delegate while replacing legal_actions). Without these, a
  // derived class' access to *_impl is limited to its own object — C++
  // protected access does not extend across sibling instances. Marking
  // these `protected static` lets any IGameRules subclass call into any
  // other IGameRules' impl, while keeping the impl invisible to non-
  // IGameRules code (game authors, runners, etc.).
  static void invoke_do_action_fast_impl(const IGameRules& r, IGameState& s,
                                         ActionId a, std::mt19937_64& rng) {
    r.do_action_fast_impl(s, a, rng);
  }
  static void invoke_undo_action_impl(const IGameRules& r, IGameState& s,
                                      const UndoToken& t) {
    r.undo_action_impl(s, t);
  }
  static UndoToken invoke_do_action_deterministic_impl(const IGameRules& r,
                                                       IGameState& s,
                                                       ActionId a) {
    return r.do_action_deterministic_impl(s, a);
  }
};

class IStateValueModel {
 public:
  virtual ~IStateValueModel() = default;
  virtual float terminal_value_for_player(const IGameState& state, int player) const = 0;

  virtual std::vector<float> terminal_values(const IGameState& state) const {
    const int n = state.num_players();
    std::vector<float> v(static_cast<size_t>(n));
    for (int p = 0; p < n; ++p) {
      v[static_cast<size_t>(p)] = terminal_value_for_player(state, p);
    }
    return v;
  }
};

class DefaultStateValueModel final : public IStateValueModel {
 public:
  float terminal_value_for_player(const IGameState& state, int player) const override {
    if (!state.is_terminal()) return 0.0f;
    const int n = state.num_players();
    if (n <= 0) return 0.0f;
    const auto w = state.winners();
    const auto l = state.losers();
    const int nw = static_cast<int>(w.size());
    const int nl = static_cast<int>(l.size());
    if (nw == 0 && nl == 0) return 0.0f;
    if (nw == n || nl == n) return 0.0f;
    bool is_winner = false;
    for (int x : w) { if (x == player) { is_winner = true; break; } }
    bool is_loser = false;
    for (int x : l) { if (x == player) { is_loser = true; break; } }
    if (is_winner && nw > 0) return 1.0f / static_cast<float>(nw);
    if (is_loser && nl > 0) return -1.0f / static_cast<float>(nl);
    return 0.0f;
  }
};

}  // namespace board_ai
