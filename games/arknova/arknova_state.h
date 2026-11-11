#pragma once

// DinoBoard adapter of the Ark Nova rules core (2 players, Map 0).
//
// Hidden information:
//   - hand: per-player membership by card (hand[p][c] = 1 when card c is in p's hand), so a hand
//     of any size needs no slots. Owners see their row; the opponent sees the entries of cards it
//     saw enter the hand (Player::hand_known). Hand sizes are public (part of `public`).
//   - final_cards, tucked (Pouch), private_cards (the cards of the owner's ab_pick / ab_resist
//     frames) and final_known (endgame cards the owner returned to the final deck): owner only.
//   - deck and final_deck contents are not schema slots; their sizes are public. The tracker
//     resamples them (randomize_unseen).
// Everything else is public and travels as one packed integer vector (field `public`).
//
// The hidden fields above are the source of truth for the matching core members (hand bitmap,
// final cards, tucked cards, private frame cards); `sync_core` writes them into the core state
// before the rules read it. A snapshot apply or a resample only marks the core stale.
//
// Decisions are factored (engine/core/factored_action.h): the atoms chosen so far for the
// pending decision are `prefix`, which is public.

#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"
#include "arknova_core.h"

namespace board_ai::arknova {

constexpr int kPrivateFrames = 4;  // private frames of one player on the stack at once
constexpr int kPrivateCards = 6;
constexpr int kFinalKnown = 4;
constexpr int kMaxPrefix = 6;
constexpr std::int16_t kEmptySlot = -1;
// Placeholder written into masked hidden card slots (card ids are int16).
constexpr std::int16_t kHiddenCard = std::numeric_limits<std::int16_t>::min();

// Schema field ids, in declaration order.
enum ArknovaField : viz::FieldId {
  kFieldPublic = 0,
  kFieldHand,
  kFieldFinalCards,
  kFieldTucked,
  kFieldPrivateCards,
  kFieldFinalKnown,
};

class ArknovaState final : public CloneableState<ArknovaState> {
 public:
  ArknovaState();

  static const viz::VisibilitySchema& schema();
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }

  void reset_with_seed(std::uint64_t seed) override;
  // Reset to a core state in the Python engine's JSON layout (arknova_json.h), at its pending decision.
  void reset_from_json(const std::string& json);
  int current_player() const override;
  bool is_terminal() const override { return core.over; }
  int num_players() const override { return kPlayers; }
  int winner() const override;
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;
  bool is_turn_start() const override;

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid, const std::vector<int>& idx) const override;
  std::any read_field_slot(viz::FieldId fid, const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid, const std::vector<int>& idx, const std::any& value) override;
  void mask_field_slot(viz::FieldId fid, const std::vector<int>& idx) override;
  void mask_all_hidden_slots(const viz::VisibilitySchema& schema, int perspective) override;

  // Core state with the hidden fields written in (no-op unless stale).
  State& synced_core() const;
  // After the core changed through the rules: refresh the hidden fields from the core.
  // Visibility is the rules' job (absorb_step).
  void absorb_core();
  void mark_core_stale() { core_stale_ = true; }

  // Public part of the core plus the public adapter fields (prefix).
  void pack_public(std::vector<int>& out) const;
  void unpack_public(const std::vector<int>& in);

  mutable State core;
  std::int16_t hand[kPlayers][kCardSlots];
  std::int16_t final_cards[kPlayers][kMaxFinalCards];
  std::int16_t tucked[kPlayers][kMaxTucked];
  std::int16_t private_cards[kPlayers][kPrivateFrames][kPrivateCards];
  std::int16_t final_known[kPlayers][kFinalKnown];
  std::vector<ActionId> prefix;

 private:
  mutable bool core_stale_ = false;
};

// Private frames: ab_pick and ab_resist hold cards only their player sees. Returns the stack
// indices of player p's private frames, bottom first, and the frame's card list offsets.
bool is_private_frame(FrameKind k);
void private_list_offsets(FrameKind k, int* n_off, int* list_off);
std::vector<int> private_frames_of(const State& s, int p);

}  // namespace board_ai::arknova
