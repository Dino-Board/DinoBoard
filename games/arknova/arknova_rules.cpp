#include "arknova_rules.h"

#include <algorithm>
#include <stdexcept>

#include "../../engine/core/viz_runtime.h"
#include "arknova_actions.h"

namespace board_ai::arknova {

namespace {

class RngChance final : public Chance {
 public:
  explicit RngChance(std::mt19937_64& rng) : rng_(rng) {}
  void shuffle(std::vector<CardIdx>& cards) override {
    for (int i = static_cast<int>(cards.size()) - 1; i > 0; --i) {
      std::swap(cards[i], cards[std::uniform_int_distribution<int>(0, i)(rng_)]);
    }
  }
  CardIdx pick(const std::vector<CardIdx>& candidates) override {
    if (candidates.empty()) throw std::logic_error("arknova: pick from no candidates");
    return candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng_)];
  }

 private:
  std::mt19937_64& rng_;
};

// Extends the prefix through forced atoms; applies the option when complete. Returns whether an
// option was applied.
bool settle_prefix(ArknovaState& st, std::mt19937_64& rng) {
  Decision d = current_decision(st);
  int done = factored_extend_forced(d.atoms, st.prefix);
  if (done < 0) {
    canonicalize_prefix(st);
    return false;
  }
  RngChance chance(rng);
  step(st.synced_core(), d.options[done], chance);
  st.prefix.clear();
  absorb_step(st);
  return true;
}

}  // namespace

void absorb_step(ArknovaState& st) {
  st.absorb_core();
  const viz::VizTensor& v = viz::viz_get_by_id(st, kFieldHand);
  for (int p = 0; p < kPlayers; ++p) {
    int opp = 1 - p;
    for (int c = 0; c < num_cards(); ++c) {
      bool want = st.hand[p][c] == 1 && st.core.players[p].shown_in_hand(static_cast<CardIdx>(c));
      bool now = v.data[viz::flat_offset_data_only(v.shape, {p, c}) + static_cast<std::size_t>(opp)] != 0;
      if (want == now) continue;
      if (want) {
        viz::reveal_slot_to(st, "hand", {p, c}, opp);
      } else {
        viz::reset_to_base(st, "hand", ArknovaState::schema(), {p, c});
      }
    }
  }
}

void advance_forced(ArknovaState& st, std::mt19937_64& rng) {
  while (!st.core.over && settle_prefix(st, rng)) {
  }
}

void apply_atom(ArknovaState& st, ActionId a, std::mt19937_64& rng) {
  if (static_cast<int>(st.prefix.size()) >= kMaxPrefix) throw std::logic_error("arknova: prefix overflow");
  st.prefix.push_back(a);
  advance_forced(st, rng);
}

std::vector<ActionId> ArknovaRules::legal_actions(const IGameState& state) const {
  const auto& st = checked_cast<ArknovaState>(state);
  if (st.is_terminal()) return {};
  Decision d = current_decision(st);
  return factored_next_atoms(d.atoms, st.prefix);
}

ActionId ArknovaRules::public_action(const IGameState& state_before, ActionId action) const {
  const auto& st = checked_cast<ArknovaState>(state_before);
  const State& s = st.core;
  const bool tuck_atom = action >= atom::kHandDiscard && action < atom::kHandDiscard + kCardSlots;
  // After `harbor` the card goes face up to the discard pile, whatever decision is pending.
  const bool harbor = !st.prefix.empty() && st.prefix.back() == atom::kHarbor;
  if (tuck_atom && !harbor && s.stack_n && s.top().kind == FrameKind::ab_pouch) return atom::kHiddenHandCard;
  return action;
}

bool ArknovaRules::validate_action(const IGameState& state, ActionId action) const {
  std::vector<ActionId> legal = legal_actions(state);
  return std::binary_search(legal.begin(), legal.end(), action);
}

void ArknovaRules::do_action_fast_impl(IGameState& state, ActionId action, std::mt19937_64& rng) const {
  auto& st = checked_cast<ArknovaState>(state);
  if (st.is_terminal()) throw std::logic_error("arknova: action on a finished game");
  apply_atom(st, action, rng);
}

UndoToken ArknovaRules::do_action_deterministic_impl(IGameState&, ActionId) const {
  throw std::logic_error("arknova: deterministic application is not supported (no tail solver)");
}

void ArknovaRules::undo_action_impl(IGameState&, const UndoToken&) const {
  throw std::logic_error("arknova: undo_action is not supported (no tail solver)");
}

}  // namespace board_ai::arknova
