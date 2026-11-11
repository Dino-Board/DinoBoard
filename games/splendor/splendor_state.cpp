#include "splendor_state.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "splendor_rules.h"
#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"
#include "../../engine/core/viz_walker.h"

namespace board_ai::splendor {

namespace {

// Caller-owned rng. Modulo on a 64-bit output is unbiased to within
// 90/2^64 ≈ 5e-18 for all decks here.
template <typename T>
T take_random_from_vector(std::vector<T>& deck, std::mt19937_64& rng) {
  if (deck.empty()) return T{};
  const size_t idx = static_cast<size_t>(rng() % deck.size());
  const T picked = deck[idx];
  if (idx + 1 < deck.size()) {
    deck[idx] = deck.back();
  }
  deck.pop_back();
  return picked;
}

std::vector<SplendorCard> build_card_pool() {
  struct CardDef {
    std::int8_t tier;
    std::int8_t bonus;
    std::int8_t points;
    std::array<std::int8_t, 5> cost;
  };
  static const std::array<CardDef, 90> defs{{
      {1, 1, 0, {0, 0, 0, 0, 3}}, {1, 1, 0, {1, 0, 0, 0, 2}}, {1, 1, 0, {0, 0, 2, 0, 2}},
      {1, 1, 0, {1, 0, 2, 2, 0}}, {1, 1, 0, {0, 1, 3, 1, 0}}, {1, 1, 0, {1, 0, 1, 1, 1}},
      {1, 1, 0, {1, 0, 1, 2, 1}}, {1, 1, 1, {0, 0, 0, 4, 0}}, {1, 3, 0, {3, 0, 0, 0, 0}},
      {1, 3, 0, {0, 2, 1, 0, 0}}, {1, 3, 0, {2, 0, 0, 2, 0}}, {1, 3, 0, {2, 0, 1, 0, 2}},
      {1, 3, 0, {1, 0, 0, 1, 3}}, {1, 3, 0, {1, 1, 1, 0, 1}}, {1, 3, 0, {2, 1, 1, 0, 1}},
      {1, 3, 1, {4, 0, 0, 0, 0}}, {1, 4, 0, {0, 0, 3, 0, 0}}, {1, 4, 0, {0, 0, 2, 1, 0}},
      {1, 4, 0, {2, 0, 2, 0, 0}}, {1, 4, 0, {2, 2, 0, 1, 0}}, {1, 4, 0, {0, 0, 1, 3, 1}},
      {1, 4, 0, {1, 1, 1, 1, 0}}, {1, 4, 0, {1, 2, 1, 1, 0}}, {1, 4, 1, {0, 4, 0, 0, 0}},
      {1, 0, 0, {0, 3, 0, 0, 0}}, {1, 0, 0, {0, 0, 0, 2, 1}}, {1, 0, 0, {0, 2, 0, 0, 2}},
      {1, 0, 0, {0, 2, 2, 0, 1}}, {1, 0, 0, {3, 1, 0, 0, 1}}, {1, 0, 0, {0, 1, 1, 1, 1}},
      {1, 0, 0, {0, 1, 2, 1, 1}}, {1, 0, 1, {0, 0, 4, 0, 0}}, {1, 2, 0, {0, 0, 0, 3, 0}},
      {1, 2, 0, {2, 1, 0, 0, 0}}, {1, 2, 0, {0, 2, 0, 2, 0}}, {1, 2, 0, {0, 1, 0, 2, 2}},
      {1, 2, 0, {1, 3, 1, 0, 0}}, {1, 2, 0, {1, 1, 0, 1, 1}}, {1, 2, 0, {1, 1, 0, 1, 2}},
      {1, 2, 1, {0, 0, 0, 0, 4}}, {2, 1, 1, {0, 2, 2, 3, 0}}, {2, 1, 1, {0, 2, 3, 0, 3}},
      {2, 1, 2, {0, 5, 0, 0, 0}}, {2, 1, 2, {5, 3, 0, 0, 0}}, {2, 1, 2, {2, 0, 0, 1, 4}},
      {2, 1, 3, {0, 6, 0, 0, 0}}, {2, 3, 1, {2, 0, 0, 2, 3}}, {2, 3, 1, {0, 3, 0, 2, 3}},
      {2, 3, 2, {0, 0, 0, 0, 5}}, {2, 3, 2, {3, 0, 0, 0, 5}}, {2, 3, 2, {1, 4, 2, 0, 0}},
      {2, 3, 3, {0, 0, 0, 6, 0}}, {2, 4, 1, {3, 2, 2, 0, 0}}, {2, 4, 1, {3, 0, 3, 0, 2}},
      {2, 4, 2, {5, 0, 0, 0, 0}}, {2, 4, 2, {0, 0, 5, 3, 0}}, {2, 4, 2, {0, 1, 4, 2, 0}},
      {2, 4, 3, {0, 0, 0, 0, 6}}, {2, 0, 1, {0, 0, 3, 2, 2}}, {2, 0, 1, {2, 3, 0, 3, 0}},
      {2, 0, 2, {0, 0, 0, 5, 0}}, {2, 0, 2, {0, 0, 0, 5, 3}}, {2, 0, 2, {0, 0, 1, 4, 2}},
      {2, 0, 3, {6, 0, 0, 0, 0}}, {2, 2, 1, {2, 3, 0, 0, 2}}, {2, 2, 1, {3, 0, 2, 3, 0}},
      {2, 2, 2, {0, 0, 5, 0, 0}}, {2, 2, 2, {0, 5, 3, 0, 0}}, {2, 2, 2, {4, 2, 0, 0, 1}},
      {2, 2, 3, {0, 0, 6, 0, 0}}, {3, 1, 3, {3, 0, 3, 3, 5}}, {3, 1, 4, {7, 0, 0, 0, 0}},
      {3, 1, 4, {6, 3, 0, 0, 3}}, {3, 1, 5, {7, 3, 0, 0, 0}}, {3, 3, 3, {3, 5, 3, 0, 3}},
      {3, 3, 4, {0, 0, 7, 0, 0}}, {3, 3, 4, {0, 3, 6, 3, 0}}, {3, 3, 5, {0, 0, 7, 3, 0}},
      {3, 4, 3, {3, 3, 5, 3, 0}}, {3, 4, 4, {0, 0, 0, 7, 0}}, {3, 4, 4, {0, 0, 3, 6, 3}},
      {3, 4, 5, {0, 0, 0, 7, 3}}, {3, 0, 3, {0, 3, 3, 5, 3}}, {3, 0, 4, {0, 0, 0, 0, 7}},
      {3, 0, 4, {3, 0, 0, 3, 6}}, {3, 0, 5, {3, 0, 0, 0, 7}}, {3, 2, 3, {5, 3, 0, 3, 3}},
      {3, 2, 4, {0, 7, 0, 0, 0}}, {3, 2, 4, {3, 6, 3, 0, 0}}, {3, 2, 5, {0, 7, 3, 0, 0}},
  }};
  std::vector<SplendorCard> cards;
  cards.reserve(defs.size());
  for (const auto& def : defs) {
    SplendorCard c;
    c.tier = def.tier;
    c.bonus = def.bonus;
    c.points = def.points;
    for (int i = 0; i < kColorCount; ++i) {
      c.cost[static_cast<size_t>(i)] = def.cost[static_cast<size_t>(i)];
    }
    cards.push_back(c);
  }
  return cards;
}

}  // namespace

const std::vector<SplendorCard>& splendor_card_pool() {
  static const std::vector<SplendorCard> cards = build_card_pool();
  return cards;
}

const std::array<std::array<std::int8_t, kColorCount>, 12>& splendor_nobles() {
  static const std::array<std::array<std::int8_t, kColorCount>, 12> nobles{{
      {{0, 0, 4, 4, 0}}, {{0, 0, 0, 4, 4}}, {{0, 4, 4, 0, 0}},
      {{4, 0, 0, 0, 4}}, {{4, 4, 0, 0, 0}}, {{3, 0, 0, 3, 3}},
      {{3, 3, 3, 0, 0}}, {{0, 0, 3, 3, 3}}, {{0, 3, 3, 3, 0}},
      {{3, 3, 0, 0, 3}}, {{4, 0, 0, 4, 0}}, {{0, 3, 3, 0, 3}},
  }};
  return nobles;
}

template <int NPlayers>
SplendorPersistentState<NPlayers>::SplendorPersistentState(
    std::shared_ptr<const SplendorPersistentNode<NPlayers>> node)
    : node_(std::move(node)) {}

template <int NPlayers>
SplendorPersistentState<NPlayers> SplendorPersistentState<NPlayers>::root_from_state(std::mt19937_64& rng) {
  using Cfg = SplendorConfig<NPlayers>;
  auto data = std::make_shared<SplendorData<NPlayers>>();
  data->current_player = 0;
  data->first_player = 0;
  data->plies = 0;
  data->final_round_remaining = -1;
  data->stage = static_cast<std::int8_t>(SplendorTurnStage::kNormal);
  data->pending_returns = 0;
  data->pending_noble_slots.fill(-1);
  data->pending_nobles_size = 0;
  data->winner = -1;
  data->terminal = false;
  data->shared_victory = false;
  data->scores = {};
  for (int c = 0; c < kColorCount; ++c) {
    data->bank[static_cast<size_t>(c)] = static_cast<std::int8_t>(Cfg::kGemCount);
  }
  data->bank[5] = static_cast<std::int8_t>(Cfg::kGoldCount);
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    data->player_gems[p].fill(0);
    data->player_bonuses[p].fill(0);
    data->player_points[p] = 0;
    data->player_cards_count[p] = 0;
    data->player_nobles_count[p] = 0;
    data->reserved[p] = {{-1, -1, -1}};
    data->reserved_visible[p] = {{0, 0, 0}};
    data->reserved_tier[p] = {{-1, -1, -1}};
    data->reserved_size[p] = 0;
  }
  for (int t = 0; t < 3; ++t) {
    data->tableau[t] = {{-1, -1, -1, -1}};
    data->tableau_size[t] = 0;
  }

  const auto& cards = splendor_card_pool();
  for (int id = 0; id < static_cast<int>(cards.size()); ++id) {
    const int tier = static_cast<int>(cards[static_cast<size_t>(id)].tier);
    if (tier >= 1 && tier <= 3) {
      data->decks[static_cast<size_t>(tier - 1)].push_back(static_cast<std::int16_t>(id));
    }
  }
  for (int t = 0; t < 3; ++t) {
    auto& deck = data->decks[static_cast<size_t>(t)];
    for (int k = 0; k < 4 && !deck.empty(); ++k) {
      data->tableau[static_cast<size_t>(t)][static_cast<size_t>(k)] =
          take_random_from_vector(deck, rng);
      data->tableau_size[static_cast<size_t>(t)] += 1;
    }
    // First-class public per-tier size — sync to decks[t] once at root.
    // Subsequent updates flow through draw_random_from_deck on the truth
    // side; observers receive deck_sizes via apply_public_snapshot from the wire.
    data->deck_sizes[static_cast<size_t>(t)] =
        static_cast<std::int16_t>(deck.size());
  }

  std::vector<int> noble_ids(12);
  for (int i = 0; i < 12; ++i) noble_ids[static_cast<size_t>(i)] = i;
  data->nobles_size = static_cast<std::int8_t>(Cfg::kNobleCount);
  data->nobles.fill(-1);
  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    data->nobles[static_cast<size_t>(i)] = static_cast<std::int16_t>(
        take_random_from_vector(noble_ids, rng));
  }

  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::move(data);
  return SplendorPersistentState<NPlayers>(std::move(node));
}

template <int NPlayers>
const SplendorData<NPlayers>& SplendorPersistentState<NPlayers>::data() const {
  if (!node_) {
    throw std::runtime_error("SplendorPersistentState is not initialized");
  }
  // Materialization is eager. Every node is created with
  // materialized!=nullptr by `advance` (which runs apply_action_copy
  // with the caller's rng) or by `root_from_state`.
  if (!node_->materialized) {
    throw std::runtime_error(
        "SplendorPersistentState node has no materialized data");
  }
  return *node_->materialized;
}

template <int NPlayers>
SplendorPersistentState<NPlayers> SplendorPersistentState<NPlayers>::advance(
    ActionId action, std::mt19937_64& rng) const {
  // Eagerly materialize the child by running apply_action_copy with
  // the caller's rng. This is the only place rule transitions consume
  // randomness (other than initial setup).
  const SplendorData<NPlayers>& parent_data = data();
  auto child = std::make_shared<SplendorData<NPlayers>>(
      SplendorRules<NPlayers>::apply_action_copy(parent_data, action, rng));
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->parent = node_;
  node->action_from_parent = action;
  node->materialized = std::move(child);
  return SplendorPersistentState<NPlayers>(std::move(node));
}

template <int NPlayers>
StateHash64 SplendorPersistentState<NPlayers>::state_hash() const {
  using Cfg = SplendorConfig<NPlayers>;
  const SplendorData<NPlayers>& d = data();
  std::size_t h = 0;
  hash_combine(h,static_cast<std::size_t>(d.current_player + 3));
  hash_combine(h,static_cast<std::size_t>(d.first_player + 5));
  hash_combine(h,static_cast<std::size_t>(d.plies + 17));
  hash_combine(h,static_cast<std::size_t>(d.final_round_remaining + 9));
  hash_combine(h,static_cast<std::size_t>(d.stage + 21));
  hash_combine(h,static_cast<std::size_t>(d.pending_returns + 25));
  hash_combine(h,static_cast<std::size_t>(d.pending_nobles_size + 27));
  for (auto slot : d.pending_noble_slots) hash_combine(h,static_cast<std::size_t>(slot + 29));
  hash_combine(h,static_cast<std::size_t>(d.winner + 11));
  hash_combine(h,static_cast<std::size_t>(d.terminal ? 1 : 0));
  for (int v : d.scores) hash_combine(h,static_cast<std::size_t>(v + 101));
  for (auto v : d.bank) hash_combine(h,static_cast<std::size_t>(v + 7));
  const int actor = d.current_player;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (auto v : d.player_gems[p]) hash_combine(h,static_cast<std::size_t>(v + 13));
    for (auto v : d.player_bonuses[p]) hash_combine(h,static_cast<std::size_t>(v + 19));
    hash_combine(h,static_cast<std::size_t>(d.player_points[p] + 23));
    hash_combine(h,static_cast<std::size_t>(d.player_cards_count[p] + 29));
    hash_combine(h,static_cast<std::size_t>(d.player_nobles_count[p] + 31));
    hash_combine(h,static_cast<std::size_t>(d.reserved_size[p] + 37));
    for (int i = 0; i < 3; ++i) {
      const std::int16_t cid = d.reserved[p][static_cast<size_t>(i)];
      const bool visible_to_actor = d.reserved_visible[p][static_cast<size_t>(i)] != 0;
      if (p == actor || visible_to_actor) {
        hash_combine(h,static_cast<std::size_t>(cid + 41));
      } else {
        hash_combine(h,static_cast<std::size_t>(-1 + 41));
      }
      hash_combine(h,static_cast<std::size_t>((visible_to_actor ? 1 : 0) + 43));
      // reserved_tier: public-to-all, mixed unconditionally.
      hash_combine(h,static_cast<std::size_t>(d.reserved_tier[p][static_cast<size_t>(i)] + 45));
    }
  }
  for (int t = 0; t < 3; ++t) {
    hash_combine(h,static_cast<std::size_t>(d.tableau_size[t] + 47));
    for (auto cid : d.tableau[t]) hash_combine(h,static_cast<std::size_t>(cid + 53));
    hash_combine(h,static_cast<std::size_t>(d.decks[t].size() + 59));
  }
  hash_combine(h,static_cast<std::size_t>(d.nobles_size + 67));
  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    hash_combine(h,static_cast<std::size_t>(d.nobles[static_cast<size_t>(i)] + 71));
  }
  return static_cast<StateHash64>(h);
}

template <int NPlayers>
SplendorState<NPlayers>::SplendorState() { reset_with_seed(0xC0FFEEu); }

template <int NPlayers>
const viz::VisibilitySchema& SplendorState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "plies", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "final_round_remaining",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "stage", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "pending_returns",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "pending_nobles_size",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "shared_victory",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "nobles_size",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public 1D ----
    viz::declare_field(
        schema, "pending_noble_slots",
        viz::all_public({Cfg::kNobleCount}, Cfg::kPlayers));
    viz::declare_field(schema, "scores",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "bank",
                       viz::all_public({kTokenTypes}, Cfg::kPlayers));
    viz::declare_field(schema, "player_points",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_cards_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_nobles_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "reserved_size",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "tableau_size",
                       viz::all_public({3}, Cfg::kPlayers));
    // deck_sizes[3]: per-tier deck size. Public count, hidden contents.
    // Size is a derived public quantity off `decks[t].size()` —
    // read/write_field_slot project to / from `decks[t].resize(...)`;
    // contents are filled by randomize_unseen.
    viz::declare_field(schema, "deck_sizes",
                       viz::all_public({3}, Cfg::kPlayers));
    viz::declare_field(schema, "nobles",
                       viz::all_public({Cfg::kNobleCount}, Cfg::kPlayers));

    // ---- public 2D ----
    viz::declare_field(
        schema, "player_gems",
        viz::all_public({Cfg::kPlayers, kTokenTypes}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_bonuses",
        viz::all_public({Cfg::kPlayers, kColorCount}, Cfg::kPlayers));
    viz::declare_field(schema, "tableau",
                       viz::all_public({3, 4}, Cfg::kPlayers));
    // reserved_visible[p][i]: public face-up flag per reserve slot.
    viz::declare_field(schema, "reserved_visible",
                       viz::all_public({Cfg::kPlayers, 3}, Cfg::kPlayers));
    // reserved_tier[p][i]: which tier (0/1/2) the reserved card came from,
    // -1 for empty slots. Public to all observers — randomize_unseen reads
    // this to consume the right tier from the unseen pool, instead of
    // reverse-looking-up from the (potentially stale across sims) reserved
    // card id.
    viz::declare_field(schema, "reserved_tier",
                       viz::all_public({Cfg::kPlayers, 3}, Cfg::kPlayers));

    // ---- private (per-owner) ----
    // reserved[p][i]: card id of player p's reserve slot i. Base
    // owner-only; rules call viz::reveal_slot(reserved, {p, i}) when a
    // reserve becomes face-up, reset_to_base when bought/discarded.
    // The encoder reads the masked clone's `reserved` slot directly —
    // face-down opp slots arrive as kHiddenCardId.
    viz::declare_field(
        schema, "reserved",
        viz::owner_only_first_axis({Cfg::kPlayers, 3}, Cfg::kPlayers),
        // who sees which slot == owner + the public reserved_visible flag
        viz::VizSync::kAllViewers);

    return schema;
  }();
  return s;
}

template <int NPlayers>
void SplendorState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  // One-shot rng for the initial deck/noble setup. RNG is not stored
  // on state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness (tableau refills after buys).
  std::mt19937_64 init_rng(seed);
  persistent = SplendorPersistentState<NPlayers>::root_from_state(init_rng);
  undo_stack.clear();
  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 SplendorState<NPlayers>::state_hash() const {
  return persistent.state_hash();
}

// Field IDs match schema declaration order in schema().
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kFirstPlayer,
  kPlies,
  kFinalRoundRemaining,
  kStage,
  kPendingReturns,
  kPendingNoblesSize,
  kWinner,
  kTerminal,
  kSharedVictory,
  kNoblesSize,
  kPendingNobleSlots,
  kScores,
  kBank,
  kPlayerPoints,
  kPlayerCardsCount,
  kPlayerNoblesCount,
  kReservedSize,
  kTableauSize,
  kDeckSizes,
  kNobles,
  kPlayerGems,
  kPlayerBonuses,
  kTableau,
  kReservedVisible,
  kReservedTier,
  kReserved,
};
}  // namespace

// Per-field bulk hash override for Splendor's largest array fields.
// Splendor has one hidden field (reserved, owner_only_first_axis) and
// many public array fields. The override eliminates per-slot virtual
// dispatch + idx vector construction overhead for the largest fields.
template <int NPlayers>
void SplendorState<NPlayers>::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  const SplendorData<NPlayers>& d = persistent.data();
  switch (fid) {
    case kPlayerGems: {
      // 2D: [kPlayers][kTokenTypes], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int t = 0; t < kTokenTypes; ++t) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(t);
          h.add(d.player_gems[static_cast<size_t>(p)]
                              [static_cast<size_t>(t)] + 13);
        }
      }
      return;
    }
    case kPlayerBonuses: {
      // 2D: [kPlayers][kColorCount], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int c = 0; c < kColorCount; ++c) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(c);
          h.add(d.player_bonuses[static_cast<size_t>(p)]
                                 [static_cast<size_t>(c)] + 19);
        }
      }
      return;
    }
    case kTableau: {
      // 2D: [3][4], all_public.
      for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
          h.add(static_cast<std::size_t>(fid));
          h.add(t);
          h.add(s);
          h.add(d.tableau[static_cast<size_t>(t)]
                          [static_cast<size_t>(s)] + 53);
        }
      }
      return;
    }
    case kReservedVisible: {
      // 2D: [kPlayers][3], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int i = 0; i < 3; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          const std::int8_t rv =
              d.reserved_visible[static_cast<size_t>(p)]
                                [static_cast<size_t>(i)];
          h.add((rv != 0 ? 1 : 0) + 43);
        }
      }
      return;
    }
    case kReservedTier: {
      // 2D: [kPlayers][3], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int i = 0; i < 3; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          h.add(d.reserved_tier[static_cast<size_t>(p)]
                                [static_cast<size_t>(i)] + 45);
        }
      }
      return;
    }
    case kReserved: {
      // 2D: [kPlayers][3], owner_only_first_axis — owner == perspective
      // sees own row; other rows are hidden.
      // viz shape: [kPlayers, 3, kPlayers]. Row stride = 3 * kPlayers.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int i = 0; i < 3; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          const std::size_t flat =
              (static_cast<std::size_t>(p) * 3 +
               static_cast<std::size_t>(i)) *
                  static_cast<std::size_t>(Cfg::kPlayers) +
              static_cast<std::size_t>(perspective);
          if (v.data[flat]) {
            h.add(d.reserved[static_cast<size_t>(p)]
                             [static_cast<size_t>(i)] + 41);
          } else {
            h.combine(kHiddenHashSentinel);
          }
        }
      }
      return;
    }
    case kNobles: {
      // 1D: [kNobleCount], all_public.
      for (int i = 0; i < Cfg::kNobleCount; ++i) {
        h.add(static_cast<std::size_t>(fid));
        h.add(i);
        h.add(d.nobles[static_cast<size_t>(i)] + 71);
      }
      return;
    }
    case kScores: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(d.scores[static_cast<size_t>(p)] + 101);
      }
      return;
    }
    case kBank: {
      // 1D: [kTokenTypes], all_public.
      for (int t = 0; t < kTokenTypes; ++t) {
        h.add(static_cast<std::size_t>(fid));
        h.add(t);
        h.add(d.bank[static_cast<size_t>(t)] + 7);
      }
      return;
    }
    case kPendingNobleSlots: {
      // 1D: [kNobleCount], all_public.
      for (int i = 0; i < Cfg::kNobleCount; ++i) {
        h.add(static_cast<std::size_t>(fid));
        h.add(i);
        h.add(d.pending_noble_slots[static_cast<size_t>(i)] + 29);
      }
      return;
    }
    default:
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers>
void SplendorState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const SplendorData<NPlayers>& d = persistent.data();
  switch (fid) {
    case kCurrentPlayer:        h.add(d.current_player + 3); return;
    case kFirstPlayer:          h.add(d.first_player + 5); return;
    case kPlies:                h.add(d.plies + 17); return;
    case kFinalRoundRemaining:  h.add(d.final_round_remaining + 9); return;
    case kStage:                h.add(d.stage + 21); return;
    case kPendingReturns:       h.add(d.pending_returns + 25); return;
    case kPendingNoblesSize:    h.add(d.pending_nobles_size + 27); return;
    case kWinner:               h.add(d.winner + 11); return;
    case kTerminal:             h.add(d.terminal ? 1 : 0); return;
    case kSharedVictory:        h.add(d.shared_victory ? 1 : 0); return;
    case kNoblesSize:           h.add(d.nobles_size + 67); return;
    case kPendingNobleSlots:
      h.add(d.pending_noble_slots[static_cast<size_t>(idx[0])] + 29); return;
    case kScores:
      h.add(d.scores[static_cast<size_t>(idx[0])] + 101); return;
    case kBank:
      h.add(d.bank[static_cast<size_t>(idx[0])] + 7); return;
    case kPlayerPoints:
      h.add(d.player_points[static_cast<size_t>(idx[0])] + 23); return;
    case kPlayerCardsCount:
      h.add(d.player_cards_count[static_cast<size_t>(idx[0])] + 29); return;
    case kPlayerNoblesCount:
      h.add(d.player_nobles_count[static_cast<size_t>(idx[0])] + 31); return;
    case kReservedSize:
      h.add(d.reserved_size[static_cast<size_t>(idx[0])] + 37); return;
    case kTableauSize:
      h.add(d.tableau_size[static_cast<size_t>(idx[0])] + 47); return;
    // First-class deck_sizes — read d.deck_sizes[t] directly. Rules
    // maintain it at draw_random_from_deck; observers receive it via
    // apply_public_snapshot. Do NOT read d.decks[t].size() here: decks are
    // hidden contents, and tying the hash to a hidden container's size
    // forces observers to re-randomize every ply just to recover this
    // value (the bug Plan A surfaced).
    case kDeckSizes:
      h.add(static_cast<std::int64_t>(d.deck_sizes[static_cast<size_t>(idx[0])]) + 59);
      return;
    case kNobles:
      h.add(d.nobles[static_cast<size_t>(idx[0])] + 71); return;
    case kPlayerGems:
      h.add(d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 13);
      return;
    case kPlayerBonuses:
      h.add(d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 19);
      return;
    case kTableau:
      h.add(d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 53);
      return;
    case kReservedVisible: {
      const std::int8_t v =
          d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])];
      h.add((v != 0 ? 1 : 0) + 43);
      return;
    }
    case kReservedTier:
      h.add(d.reserved_tier[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 45);
      return;
    case kReserved:
      h.add(d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 41);
      return;
  }
}

// Walker-driven snapshot wire I/O dispatchers. `read_field_slot`
// returns std::any of int / bool. `write_field_slot` is gated behind
// the COW persistent — register.cpp opens a `mutate_persistent` block
// around `viz::apply_public_snapshot(...)` so all per-slot writes land
// on the detached writable copy.

template <int NPlayers>
std::any SplendorState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const SplendorData<NPlayers>& d = persistent.data();
  switch (fid) {
    case kCurrentPlayer:        return std::any(static_cast<int>(d.current_player));
    case kFirstPlayer:          return std::any(static_cast<int>(d.first_player));
    case kPlies:                return std::any(static_cast<int>(d.plies));
    case kFinalRoundRemaining:  return std::any(static_cast<int>(d.final_round_remaining));
    case kStage:                return std::any(static_cast<int>(d.stage));
    case kPendingReturns:       return std::any(static_cast<int>(d.pending_returns));
    case kPendingNoblesSize:    return std::any(static_cast<int>(d.pending_nobles_size));
    case kWinner:               return std::any(static_cast<int>(d.winner));
    case kTerminal:             return std::any(static_cast<bool>(d.terminal));
    case kSharedVictory:        return std::any(static_cast<bool>(d.shared_victory));
    case kNoblesSize:           return std::any(static_cast<int>(d.nobles_size));
    case kPendingNobleSlots:
      return std::any(static_cast<int>(d.pending_noble_slots[static_cast<size_t>(idx[0])]));
    case kScores:
      return std::any(static_cast<int>(d.scores[static_cast<size_t>(idx[0])]));
    case kBank:
      return std::any(static_cast<int>(d.bank[static_cast<size_t>(idx[0])]));
    case kPlayerPoints:
      return std::any(static_cast<int>(d.player_points[static_cast<size_t>(idx[0])]));
    case kPlayerCardsCount:
      return std::any(static_cast<int>(d.player_cards_count[static_cast<size_t>(idx[0])]));
    case kPlayerNoblesCount:
      return std::any(static_cast<int>(d.player_nobles_count[static_cast<size_t>(idx[0])]));
    case kReservedSize:
      return std::any(static_cast<int>(d.reserved_size[static_cast<size_t>(idx[0])]));
    case kTableauSize:
      return std::any(static_cast<int>(d.tableau_size[static_cast<size_t>(idx[0])]));
    case kDeckSizes:
      return std::any(static_cast<int>(d.deck_sizes[static_cast<size_t>(idx[0])]));
    case kNobles:
      return std::any(static_cast<int>(d.nobles[static_cast<size_t>(idx[0])]));
    case kPlayerGems:
      return std::any(static_cast<int>(
          d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kPlayerBonuses:
      return std::any(static_cast<int>(
          d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kTableau:
      return std::any(static_cast<int>(
          d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kReservedVisible:
      return std::any(
          d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] != 0
              ? 1 : 0);
    case kReservedTier:
      return std::any(static_cast<int>(
          d.reserved_tier[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    // reserved: walker only visits the slot iff its viewer bit is 1 for
    // perspective. Rules maintain viz_["reserved"] in sync with
    // reserved_visible via reveal_slot / reset_to_base; the receiver
    // picks up the same bits via the wire's `__viz__` slice.
    case kReserved:
      return std::any(static_cast<int>(
          d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  return {};
}

template <int NPlayers>
void SplendorState<NPlayers>::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx,
    const std::any& value) {
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    return 0;
  };
  auto as_bool = [&]() -> bool {
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value);
    if (value.type() == typeid(int)) return std::any_cast<int>(value) != 0;
    return false;
  };

  // Detach a writable SplendorData and reseat once per write. (register.cpp
  // wraps the whole apply_public_snapshot call in a single mutate_persistent
  // so typical N-write batch is still O(1) reseats; this code path is only
  // hit when called outside a wrapping mutate_persistent — kept correct
  // either way.)
  SplendorData<NPlayers> data = persistent.data();
  auto& d = data;

  bool handled = true;
  switch (fid) {
    case kCurrentPlayer:        d.current_player = static_cast<std::int8_t>(as_int()); break;
    case kFirstPlayer:          d.first_player = static_cast<std::int8_t>(as_int()); break;
    case kPlies:                d.plies = static_cast<std::int16_t>(as_int()); break;
    case kFinalRoundRemaining:  d.final_round_remaining = static_cast<std::int8_t>(as_int()); break;
    case kStage:                d.stage = static_cast<std::int8_t>(as_int()); break;
    case kPendingReturns:       d.pending_returns = static_cast<std::int8_t>(as_int()); break;
    case kPendingNoblesSize:    d.pending_nobles_size = static_cast<std::int8_t>(as_int()); break;
    case kWinner:               d.winner = static_cast<std::int8_t>(as_int()); break;
    case kTerminal:             d.terminal = as_bool(); break;
    case kSharedVictory:        d.shared_victory = as_bool(); break;
    case kNoblesSize:           d.nobles_size = static_cast<std::int8_t>(as_int()); break;
    case kPendingNobleSlots:
      d.pending_noble_slots[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      break;
    case kScores:
      d.scores[static_cast<size_t>(idx[0])] = as_int(); break;
    case kBank:
      d.bank[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int()); break;
    case kPlayerPoints:
      d.player_points[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
      break;
    case kPlayerCardsCount:
      d.player_cards_count[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
      break;
    case kPlayerNoblesCount:
      d.player_nobles_count[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
      break;
    case kReservedSize:
      d.reserved_size[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      break;
    case kTableauSize:
      d.tableau_size[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      break;
    // First-class public deck_sizes. Write d.deck_sizes[t] directly;
    // the hidden d.decks[t] multiset is tracker territory (sim-entry
    // randomize_unseen fills it from belief). We still resize the
    // observer's d.decks[t] to match so the legacy full-state
    // state_hash() (used by tail solver / transposition debug) reads a
    // consistent length — but the canonical schema-walker hash source
    // is d.deck_sizes[t].
    case kDeckSizes: {
      const int target = as_int();
      if (target >= 0) {
        d.deck_sizes[static_cast<size_t>(idx[0])] =
            static_cast<std::int16_t>(target);
        auto& deck = d.decks[static_cast<size_t>(idx[0])];
        if (static_cast<int>(deck.size()) > target) {
          deck.resize(static_cast<size_t>(target));
        } else {
          while (static_cast<int>(deck.size()) < target) {
            deck.push_back(-1);
          }
        }
      }
      break;
    }
    case kNobles:
      d.nobles[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int()); break;
    case kPlayerGems:
      d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      break;
    case kPlayerBonuses:
      d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      break;
    case kTableau:
      d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int16_t>(as_int());
      break;
    case kReservedVisible:
      d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      break;
    case kReservedTier:
      d.reserved_tier[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      break;
    case kReserved:
      d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int16_t>(as_int());
      break;
    default:
      handled = false;
      break;
  }

  if (!handled) return;
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

// mask_field_slot writes kPlaceholder into the only hidden field
// Splendor declares — `reserved` (face-down id, owner_only_first_axis
// base). All other fields are all_public, so the walker never visits
// them as hidden.
template <int NPlayers>
void SplendorState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  if (fid != kReserved) return;
  SplendorData<NPlayers> data = persistent.data();
  data.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
      kHiddenCardId;
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

// COW-aware mask_all_hidden_slots. Detach once into a single writable
// SplendorData, run the walker dispatch on it, reseat once at the end.
// Avoids one shared_ptr reseat per masked slot.
//
// In addition to schema-declared hidden slots (`reserved`), this also
// clears the off-schema hidden state `decks[t]`. `decks[t]` is a
// variable-length per-tier vector whose contents are hidden multiset —
// only its size (`deck_sizes[t]`, all_public) is on schema. Without an
// explicit clear here, mask leaves whatever bytes were last written to
// the persistent struct: the seed-time initial deck on a session that
// has never run randomize_unseen, or the previous sim's belief-sampled
// deck. Both are leak-shaped (not necessarily a leak today, since
// eval_position / encoder don't read decks and do_action_deterministic
// freezes draws via forced_draw_override=-2; but the only structural
// guarantee is "mask wipes the field"). Clearing here makes that
// guarantee real.
//
// Safety w.r.t. callers:
//   - heuristic path: do_action_deterministic sets forced_draw_override=-2
//     so draw_random_from_deck early-returns -1 on `if (deck.empty())`,
//     same behavior as on a non-empty frozen deck. eval_position never
//     reads decks.
//   - sim/MCTS path: make_masked_state runs AFTER randomize_unseen has
//     written sim_state.decks. mask_all_hidden_slots operates on the
//     clone, not sim_state itself; sim_state retains its sampled deck
//     for do_action_fast draws during descent. The encoder reads the
//     clone, but encode_features doesn't touch decks.
template <int NPlayers>
void SplendorState<NPlayers>::mask_all_hidden_slots(
    const viz::VisibilitySchema& schema, int perspective) {
  SplendorData<NPlayers> data = persistent.data();
  viz::for_each_hidden_slot(
      *this, schema, perspective,
      [&](viz::FieldId fid, const std::vector<int>& idx,
          const viz::VizTensor& /*v*/) {
        if (fid == kReserved) {
          data.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
              kHiddenCardId;
        }
      });
  // Off-schema hidden: replace deck multiset contents with placeholder
  // entries while preserving the public deck size. Two reasons we don't
  // just clear():
  //   1. legal_actions() (splendor_rules.cpp `legal_for_player`) checks
  //      `!d.decks[t].empty()` to decide whether "reserve from deck tier t"
  //      is legal. Clearing here would silently strip those actions on the
  //      masked clone, breaking heuristic_pick / lookahead callers.
  //   2. The wholesale snapshot apply path resizes `decks[t]` to match
  //      `deck_sizes[t]` and fills with placeholder; preserving the same
  //      size on the mask path keeps both paths interchangeable from the
  //      consumer's perspective (legal-action set, hash via deck_sizes).
  // Placeholder content is never read: do_action_deterministic freezes
  // draws via forced_draw_override=-2; eval_position / encoder don't read
  // decks; the sim path runs randomize_unseen BEFORE mask, which writes
  // belief-consistent contents that survive into sim_state — the masked
  // clone the encoder sees is separate from sim_state.
  for (int t = 0; t < 3; ++t) {
    auto& deck = data.decks[static_cast<size_t>(t)];
    const int sz = static_cast<int>(deck.size());
    deck.assign(static_cast<size_t>(sz), kHiddenCardId);
  }
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

template <int NPlayers>
int SplendorState<NPlayers>::current_player() const {
  return persistent.data().current_player;
}

template <int NPlayers>
int SplendorState<NPlayers>::first_player() const {
  return persistent.data().first_player;
}

template <int NPlayers>
bool SplendorState<NPlayers>::is_terminal() const {
  return persistent.data().terminal;
}

template <int NPlayers>
bool SplendorState<NPlayers>::is_turn_start() const {
  return static_cast<SplendorTurnStage>(persistent.data().stage) == SplendorTurnStage::kNormal;
}

template <int NPlayers>
int SplendorState<NPlayers>::winner() const {
  return persistent.data().winner;
}

template <int NPlayers>
std::vector<int> SplendorState<NPlayers>::winners() const {
  const auto& d = persistent.data();
  if (!d.terminal) return {};
  if (d.winner >= 0) return {d.winner};
  if (!d.shared_victory) return {};
  // Shared victory: everyone tied on the rules' ranking key (most points,
  // then fewest cards) — read-only mirror of update_terminal's ranking.
  int best_pts = -1;
  int best_cards = 0;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    const int pts = d.player_points[static_cast<size_t>(p)];
    const int cards = d.player_cards_count[static_cast<size_t>(p)];
    if (pts > best_pts || (pts == best_pts && cards < best_cards)) {
      best_pts = pts;
      best_cards = cards;
    }
  }
  std::vector<int> out;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    if (d.player_points[static_cast<size_t>(p)] == best_pts &&
        d.player_cards_count[static_cast<size_t>(p)] == best_cards) {
      out.push_back(p);
    }
  }
  if (static_cast<int>(out.size()) == Cfg::kPlayers) return {};  // everyone tied: draw
  return out;
}

template <int NPlayers>
std::vector<int> SplendorState<NPlayers>::losers() const {
  const std::vector<int> w = winners();
  if (w.empty()) return {};
  std::vector<int> out;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    if (std::find(w.begin(), w.end(), p) == w.end()) out.push_back(p);
  }
  return out;
}

template struct SplendorData<2>;
template struct SplendorData<3>;
template struct SplendorData<4>;
template class SplendorPersistentState<2>;
template class SplendorPersistentState<3>;
template class SplendorPersistentState<4>;
template struct SplendorState<2>;
template struct SplendorState<3>;
template struct SplendorState<4>;

}  // namespace board_ai::splendor
