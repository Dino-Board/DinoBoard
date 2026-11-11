# Ark Nova API

Ark Nova is the 2-player base game on map 0. The rules core
(`arknova_core.h`, `arknova_engine.cpp`, effect / action modules) is a C++ port
of the Python engine in [ark-nova-ai](https://github.com/collinfeng/ark-nova-ai); card data is
generated from that repo by `games/arknova/tools/gen_cards.py` into
`arknova_cards_data.inc` (committed).

## Correctness

`games/arknova/tools/diff_test.py` plays random games with the Python engine
and loads every Python state into `arknova_diff_cli`. Legal option lists and
the state after each step (with Python's random outcomes replayed) must be
identical. 2000 games (about 1.14 million steps) agree.

Checked against the rulebook (`games/arknova/Arche_Nova_Rules_EN_Low_2022_01.pdf`),
ark-nova-ai departs from it in three places, which the engine fixes:

- whether a second animal can be played is decided after the first one's
  effects (rulebook p. 12: the Eagle-Owl draws the Lynx, which is then played);
- Animals II at strength 5+ is judged with the +1 reputation it gains first (p. 13);
- an animal leaving a standard enclosure (release, move into a new Reptile House
  or Aviary) flips the smallest occupied standard enclosure meeting its needs (pp. 12, 15).

`arknova_diff_cli` runs with `g_upstream_rules` (ark-nova-ai's behaviour
there), so the differential test still covers everything else;
`games/arknova/tools/rule_scenarios.py` (same `--ref`/`--cli` arguments)
plays the three rulebook situations through `arknova_diff_cli --rulebook`
and checks both behaviours. Not modelled:
the player choosing the order of simultaneous effects and of "after finishing"
effects (fixed order here).

```bash
python games/arknova/tools/diff_test.py --ref ~/ref/ark-nova-ai \
    --cli build/<dir>/arknova_diff_cli --games 200
```

A frame's card list (a Hunter reveal, the base project pile) holds up to 31
cards (`kFrameData` 32). The longest reveal the cards allow is one card per
predator icon in a zoo (Spotted Hyena Compound); the game has 29 predator
icons, and the engine refuses to start if a card table ever needs more than a
frame can hold.

## Action Encoding (factored atoms)

A core decision has up to thousands of options (card × building × cell ×
rotation). Each option is a short sequence of **atoms**; one action id is one
atom. `legal_actions` returns the atoms that extend the current prefix
(`state.prefix`, public). Once the prefix spells out an option, the rules apply
it; decisions with a single continuation are applied without asking. So one
Python step is one or more framework plies.

Action space: 1569 atoms. Atoms 0–131 are **global** (scored by the network's
global head from the CLS token); atoms 132–1568 are **pointer** atoms, each
scored by one pointer head of the token that carries its anchor (see
[anchors](#anchors)).

**One atom, one meaning.** An atom never stands for gaining something in one
decision and giving it up in another:

- a card has a *take* atom (play it, take it, keep it: pointer head 0) and a
  separate *give* atom (discard, sell, tuck, release it: pointer head 1);
- `money` is "take 5 money" at a conservation space, `give_money` is the
  Pilfering victim paying; `give_card` is the victim giving a card;
- `zoo:c` moves an animal into a new special enclosure, `zoo_give:c` releases
  it into the wild (or spends a token of a breeding card).

`tests/arknova/test_arknova_actions.py` lists, per decision frame, every atom
class that can be legal there with its meaning (take / give / neither), checks
that no class has both, that the list is exactly what random games produce, and
that the cards really go where the atom's class says.

Typical sequences:

- play animal: `[hand:c or display:i, enclosure:b]`
- support project: `[project:i / hand:c / display:i, position:k, map_token:t, no_extra or zoo_give:c]`
- build: `[build:<type>, cell:<lowest cell>/o<rotation>]`
- setup: the 4 cards not kept, as 4 `hand_discard:c` atoms **in any order**.
  Every hand card not yet chosen is legal at each step. The prefix is kept sorted, so the same discards named in another
  order are the same state (same hash, same encoding), and the cards chosen so
  far carry the `chosen_discard` feature. The network therefore never has to
  know the card index order.

Hand and zoo atoms name the card itself, so neither collection has a size cap.
A public list has 31 slots, as many as a frame can hold, so no list is too long
to encode or to choose from.

`describe_action(a)["name"]` gives the atom names used below.

### atoms

| first | count | name | meaning |
|---|---|---|---|
| 0 | 1 | `skip` | decline |
| 1 | 1 | `done` | stop here |
| 2 | 1 | `no` | Okapi Stable: no |
| 3 | 1 | `yes` | Okapi Stable: play a sponsor |
| 4 | 1 | `money` | conservation space: take 5 money |
| 5 | 1 | `deck` | draw from the deck |
| 6 | 1 | `donate` | make the donation |
| 7 | 1 | `reputation` | association task: reputation |
| 8 | 1 | `repeat` | Multiplier: repeat the action |
| 9 | 1 | `worker` | conservation 2: take a worker |
| 10 | 1 | `replenish` | Snapping: replenish the display |
| 11 | 1 | `give_card` | Pilfering victim: give a random hand card |
| 12 | 1 | `break` | Sponsors: advance the break and take money |
| 13 | 1 | `no_extra` | project support without a released animal / breeding token (never a choice of its own) |
| 14 | 1 | `hidden_hand_card` | never legal: what observers see for a Pouch tuck |
| 15 | 1 | `give_money` | Pilfering victim: pay 5 money |
| 16 | 30 | `act:.+` | use action card with X X-tokens (`act:<card>+<X>`, 5 cards x 6) |
| 46 | 5 | `xtoken:.+` | X-token action moving this card |
| 51 | 5 | `upgrade:.+` | flip this action card |
| 56 | 5 | `clever:.+` | Clever: move this card to slot 1 |
| 61 | 5 | `multiplier:.+` | put a Multiplier token on this card |
| 66 | 2 | `boost:.+` | Boost: move the card to slot 1 / 5 |
| 68 | 5 | `partner_zoo:.+` | take this partner zoo |
| 73 | 3 | `university:.+` | take this university |
| 76 | 9 | `tile:.+` | take this bonus tile |
| 85 | 2 | `kind:.+` | Waza Special Assignment: small / large |
| 87 | 2 | `target:.+` | target player (unused with 2 players) |
| 89 | 11 | `build:.+` | Build action: this building type |
| 100 | 11 | `engineer:.+` | Engineer's extra build: this type |
| 111 | 11 | `place:.+` | free building (effect / sponsor): this type |
| 122 | 3 | `position:.+` | project level 1-3 |
| 125 | 7 | `map_token:.+` | map token 0-6 to place on the project |
| 132 | 240 | `hand:.+` | play this hand card (animal, sponsor, project to support) |
| 372 | 240 | `hand_discard:.+` | give up this hand card: setup discard, hand-limit discard, Sun Bathing sale, Pouch tuck, Digging discard |
| 612 | 6 | `display:.+` | take this display card into the hand, or play it from there (animal, sponsor, project) |
| 618 | 6 | `display_discard:.+` | discard this display card (Digging) |
| 624 | 6 | `snap:.+` | Cards action: snap this display card |
| 630 | 10 | `project:.+` | support this project in play |
| 640 | 240 | `zoo:.+` | move this animal into the new special enclosure |
| 880 | 240 | `zoo_give:.+` | release this animal / spend a token of this breeding card |
| 1120 | 31 | `list:.+` | take this card of the public list |
| 1151 | 6 | `private:.+` | keep this card of my private list |
| 1157 | 6 | `final_discard:.+` | discard this final scoring card |
| 1163 | 58 | `enclosure:.+` | house the animal in this enclosure (building id) |
| 1221 | 348 | `cell:.+` | place the building with this lowest cell in orientation o (`cell:<c>/o<k>`, 58 cells x 6) |

## Hidden Information

| Field | Visibility |
|---|---|
| `public` (packed core) | public |
| `hand` [2 × 240] | membership by card id (1 = in the hand); owner, and the opponent for each card that entered the hand publicly (`hand_known`) |
| `final_cards`, `tucked`, `private_cards`, `final_known` | owner only |
| deck, final deck | off-schema (contents follow from every other zone) |

Hand sizes are public. The draw order is never part of `state_hash`.

Pouch tucks a hand card face down. Its `hand_discard` atom would name the card, so
`ArknovaRules::public_action` reports it to trackers, traces and the opponent's
view as `hidden_hand_card`, an atom that is never legal. The opponent also can
no longer tell which of the cards it saw in that hand went: they become
`Player::maybe_tucked` (public, "in the hand or tucked") and stop being shown
in the hand until one leaves publicly. The flag is information only; the rules
and the JSON (hence the reference diff) never read it.

The tracker (`ArknovaBeliefTracker`) is stateless: the unseen pool is derived
from public zones plus the observer's visible entries. `randomize_unseen` clears
the opponent's hidden hand entries and draws as many cards as the hand size
leaves unseen, then
tucked cards and private draws from the regular pool and opponent finals from
the endgame pool; the rest is the shuffled deck. Hand cards the opponent named
in the public prefix go into its hand first, and each maybe-tucked card goes
into its hidden hand or among its tucked cards (uniform over the free places,
or the belief net's hand probability); if it is still mid-decision, worlds are
resampled until the prefix is legal. The belief net predicts, per card, whether
it is a hidden opponent hand card (240 sigmoid logits, BCE; feature dim 1218,
including the maybe-tucked cards and the tucked count).

The belief net's probabilities are computed by `prepare_for_root` for one
observer and one position; the tracker stores that position's masked state hash
with them. `randomize_unseen` weights its draws by them only for that observer
at exactly that position and draws uniformly everywhere else (a heuristic
player's sampled world, a later position): probabilities from an earlier
search are never reused.

## Token Encoder

`ArknovaTokenEncoder` is variable-length (`TokenLayout.variable_tokens`,
`feature_dim` 0) with positions (`TokenLayout.token_positions`): 173 floats per
token, as many tokens as the state has (89–177 over 100 random and 30 greedy
games, median 117).

Token header (7 floats): `valid` (1), `entity` (see [entities](#entities); 0 =
none), `anchor` (1 + anchor index, 0 = no pointer atoms), `type` (0–7 in the
order of the tables below), `has_position`, `q`, `r`. Features start at float
7; a token type narrower than 166 features (the `card` token) leaves the rest 0.

Token order: `global`, `decision`, `player` (me, opponent), 5 `action_card`
tokens of mine and 5 of the opponent, then `card` / `project` tokens (a
conservation project card gets a `project` token wherever it is, any other card
a `card` token) for: my hand, the opponent's hand cards I saw enter it, one
`unknown` token per other opponent hand card, its maybe-tucked cards, the
display, my zoo, the opponent's zoo, the pending frame's public list, my
pending private list, my final scoring cards and the projects in play; then my
58 `cell` tokens, my `enclosure` tokens and the opponent's `enclosure` tokens.
`card` and `project` tokens share their first 10 location features; only a
`card` token can be `unknown`, in a zoo, a final card or `pending` (zoos hold
animals and sponsors, final cards come from the final scoring deck), only a
`project` token can be `in_play`. The encoder refuses a state that breaks this.

Everything is encoded from the perspective player's point of view ("me"). The
encoder reads only what that player may see: the opponent's hidden hand cards
appear as `unknown` tokens without any card feature, its final scoring cards,
tucked cards and private draws not at all. Features about the opponent
(`can_opp`, `count_opp`, `yield` of its zoo cards, its `placeable`, …) are
computed from its public zoo. Card features about playing a card now are for the
card's *holder*: the opponent for its seen hand cards, maybe-tucked cards and
zoo cards, me for everything else (my hand, the display, lists I choose from).
So a seen opponent hand card says whether the opponent could play it now.

**Positions.** My `cell` tokens carry their axial hex coordinates (q 0–8, r
−4–6), my `enclosure` tokens those of their lowest cell. With
`network.relative_position_radius` 4 (`config/train.json`) the model adds a
learned attention bias per pair of positioned tokens, indexed by the offset
(Δq, Δr) with each axis clipped to ±4 (81 buckets), and one shared bias for
every pair in which a token (or CLS) has no position. So the six neighbours of
a cell — offsets (±1, 0), (0, ±1), (1, −1), (−1, 1) — and every pair of cells
of one building (largest offset on an axis: 3) have a bucket of their own;
cells further apart share the bucket of the clipped offset, which still tells
the direction. Tokens without a position (cards, players, the opponent's
enclosures) relate to the map through features instead.

**Action cards (one token per card and player).** Slot, strength in use,
Venom / Constriction / Multiplier tokens and the upgrade are properties of one
action card, and the atoms `act:`, `xtoken:`, `upgrade:`, `clever:`,
`multiplier:` each name one. A token per card puts these facts in one place
where attention can relate them to the cards and projects that need a strength;
the alternative, 5 × 9 more features on the player token, would save 10 tokens
but leave the model to pick five interleaved groups apart. The tokens are not
pointer targets: the action-card atoms stay global, because one card has up to
6 `act` atoms (X-tokens spent) plus four other atom classes (ten pointer heads
instead of six on every token), and under Hypnosis the same `act:` atoms name
the opponent's cards (`decision.extra_from_opponent`).

**Maps.** Only my map has cell tokens, since only my buildings are placed by
my atoms and only there do the cell-level facts (terrain, bonus, adjacency)
decide what I can do. The opponent's map is summarised by its `enclosure`
tokens (what it can still house: kind, size, free room, rock / water) and the
`player` features `buildings` (count per kind), `covered_cells` and
`kiosk_income`; its 58 cell tokens of the previous layout are gone. An
`enclosure` token exists per standard enclosure and special enclosure (the
buildings animals live in); mine are positioned and are the pointer targets of
`enclosure:b`. It aggregates what belongs to the whole building: rock and
water next to any of its cells, kiosks next to it, room left. Kiosks,
pavilions and unique buildings have no token; they show in the cell features
(`kind`, `anchor`, `anchor_dq` / `anchor_dr`, `kiosk_income`) and the player
counts.

**Card semantics.** `card` and `project` tokens keep the entity embedding (card
index) and add what the card table says: `ability` (keyword and number), `req`,
`special`, `building`, `scoring`, the project's `category` / `icon` / per-level
requirements and rewards. They are read from the same `CardDef` table the
rules use (no second copy); the encoder checks at start-up that the table has
the shape these features assume (a keyword is always or never numbered, numbers
≤ 5, every project has a category, an icon and three levels). Sponsor effects
are implemented in code per printed card (`arknova_card_effects.cpp`), not in
a data table. So their sizes are measured with that code itself: `immediate`,
`income` and `yield` run the card's immediate / income / end-of-game effect on
a copy of the (masked) state and record what changed; `trigger_*` come from
`icon_trigger_spec`, the same icon / gain mapping `icon_trigger` applies. The
`effect` flags say which kinds of effect the card has.

**Card pools.** The network has no history, so the `global` token says what is
gone and what can still turn up, as two summaries of the same shape: the
discard pile (public; also what Scavenging draws from) and the *unseen* regular
cards, those whose place I do not know: the deck plus the opponent's hidden
hand, tucked cards and private draws, less its maybe-tucked cards (they have
tokens). "Known" is `known_cards` of `arknova_belief.h` on my masked view, the
function the belief tracker builds its sampling pools from, so the encoder and
the tracker agree on what is unseen and nothing hidden is read. Each column is
the pool's count (or sum) over the same count among all 235 − 11 = 224 regular
cards of the game (128 animals, 64 sponsors, 32 projects), read from `CardDef`:
1 means the whole bucket is in that pool. Per pool cards rather than per-card
tokens: the discard pile passes 100 cards in a typical game, which would double
the token count. `unseen_final` flags the final scoring cards I have not seen
(in the final deck or among the opponent's final cards): 11 columns, one per
card, since which ones the opponent may hold is what matters there.

Not encoded: which of the legal placements of a building gives which bonus
(the cells' `bonus` / `buildable` features and positions carry that), the
order of cards in the deck or discard pile, and the identity of single cards
in the pools (only their composition).

### `global` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `break_pos` | break token position / 9 (9 = break after this turn) |
| 1-2 | 2 | `break_by` | who triggered the pending break: [me, opponent] |
| 3 | 1 | `deck` | deck size / 200 |
| 4 | 1 | `discard` | discard pile size / 200 |
| 5 | 1 | `final_deck` | final scoring deck size / 12 |
| 6 | 1 | `donations` | donations made / 5 |
| 7 | 1 | `donation_cost` | money the next donation costs / 12 |
| 8-9 | 2 | `end_by` | who triggered the end of the game: [me, opponent] |
| 10 | 1 | `final_turns` | turns still to play after the trigger / 2 (0 before it) |
| 11 | 1 | `final_card_discarded` | the final scoring cards were already cut down to one each |
| 12 | 1 | `my_turn` | it is my turn (I may also decide during the opponent's turn: Pilfering, break discards) |
| 13-17 | 5 | `partner_zoo_available` | partner zoos on the association board, by continent (africa, americas, asia, australia, europe) |
| 18-20 | 3 | `university_available` | universities on the association board (hand, science2, science_rep) |
| 21-38 | 18 | `bonus_tiles` | bonus tiles left at conservation 5 (first 9) and 8 (last 9), by tile (money10, xtoken3, reputation2, university, partner_zoo, enclosure3, cards3, multiplier, sponsor) |
| 39 | 1 | `base_pile` | unused base projects (Assertion / Dominance) / 10 |
| 40 | 1 | `stack_depth` | frames on the stack, min(n, 10) / 10 |
| 41-82 | 42 | `pending` | frames below the pending decision, by frame kind (same order as `decision.frame`): min(count, 4) / 4 |
| 83 | 1 | `pending_opponent` | a decision of the opponent waits below mine |
| 84 | 1 | `after_clever` | queued "after finishing" Clever effects, min(n, 4) / 4 |
| 85 | 1 | `after_boost` | queued "after finishing" Boost effects, min(n, 4) / 4 |
| 86 | 1 | `after_extra_action` | queued "after finishing" extra actions (Action, Determination, Hypnosis), min(n, 4) / 4 |
| 87-89 | 3 | `discard_type` | discard pile (see Card pools; every pool column is count in the pool / count among all regular cards): animals / 128, sponsors / 64, projects / 32 |
| 90-94 | 5 | `discard_animal_size` | animals of size 1..5 |
| 95-109 | 15 | `discard_icons` | icons the cards add to a zoo (tags plus rock / water requirement icons, as `player.icons`), by icon (same order as `card.tags`), summed |
| 110-113 | 4 | `discard_sponsor_level` | sponsors of level ≤3, 4, 5, ≥6 |
| 114-117 | 4 | `discard_project_category` | projects by category (base, release, breed, normal) |
| 118 | 1 | `discard_appeal` | animals' appeal, summed |
| 119 | 1 | `discard_conservation` | animals' conservation points, summed |
| 120-122 | 3 | `unseen_type` | unseen regular cards (deck, the opponent's hidden hand / tucked / private draws), same columns as `discard_type` |
| 123-127 | 5 | `unseen_animal_size` | as `discard_animal_size` |
| 128-142 | 15 | `unseen_icons` | as `discard_icons` |
| 143-146 | 4 | `unseen_sponsor_level` | as `discard_sponsor_level` |
| 147-150 | 4 | `unseen_project_category` | as `discard_project_category` |
| 151 | 1 | `unseen_appeal` | as `discard_appeal` |
| 152 | 1 | `unseen_conservation` | as `discard_conservation` |
| 153-163 | 11 | `unseen_final` | final scoring cards 1..11 I have not seen (final deck, the opponent's final cards and Resistance draws) |

### `decision` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0-41 | 42 | `frame` | kind of the pending decision frame, one-hot in the order of `ARKNOVA_FRAME_KINDS` (arknova_core.h) |
| 42 | 1 | `to_move` | the perspective is the player to decide (always 1 when the policy is queried) |
| 43 | 1 | `prefix_len` | atoms already chosen for this decision / 6 |
| 44-48 | 5 | `turn_action` | action card of the (innermost) action in progress: animals, build, cards, association, sponsors |
| 49 | 1 | `turn_strength` | its strength including X-tokens / 10 (0 for an X-token turn) |
| 50 | 1 | `turn_xtoken` | the turn is an X-token action |
| 51 | 1 | `turn_depth` | nesting of actions in progress (extra actions) / 4 |
| 52 | 1 | `turn_borrowed` | the action uses the opponent's action card (Hypnosis) |
| 53 | 1 | `budget` | act_build: size left / 10; act_association: task value left / 10; act_sponsors: level budget left / 10 |
| 54 | 1 | `remaining` | picks left / 5: act_animals animals, act_cards draws, ab_sun / ab_dig / ab_pouch / ab_snap uses, ab_pick cards to keep, discard_hand cards to discard |
| 55 | 1 | `done_count` | done so far / 5: animals played, sponsors played, cards drawn, buildings built |
| 56 | 1 | `side2` | the action is performed on side II of its card (the card in use, also when borrowed) |
| 57 | 1 | `discard_after` | act_cards: cards to discard after drawing (0 or 1) |
| 58 | 1 | `may_snap` | act_cards: snapping is still allowed |
| 59 | 1 | `normal_done` | act_build: the regular builds are over (only the Engineer's extra build remains) |
| 60 | 1 | `engineer_used` | act_build: the Engineer's extra build was used |
| 61-63 | 3 | `animals_phase` | act_animals: play, bonus play (Waza Small Animals Program), take a small animal |
| 64 | 1 | `all_small` | act_animals: every animal played so far was small |
| 65-68 | 4 | `tasks_used` | act_association: tasks already done this action (reputation, partner zoo, university, project) |
| 69 | 1 | `donated` | act_association: the donation was made |
| 70 | 1 | `optional` | extra_action: it may be skipped |
| 71-75 | 5 | `extra_actions` | extra_action: the action cards it may use |
| 76 | 1 | `extra_from_opponent` | extra_action: it would use the opponent's cards (Hypnosis) |
| 77-81 | 5 | `boost_action` | ab_boost: the action card to move |
| 82-83 | 2 | `tile_space` | conservation_tile: the space reached, [5, 8] |
| 84 | 1 | `can_replenish` | ab_snap: the display may be replenished now |
| 85 | 1 | `replenish_between` | ab_snap: a replenish between the two snaps is still available |
| 86 | 1 | `only_one` | ab_project: only one of the listed base projects may be taken (Dominance) |
| 87-97 | 11 | `built` | act_build: building types already built this action (same order as `chosen_type`) |
| 98-108 | 11 | `offered` | free_build: building types on offer |
| 109 | 1 | `list_len` | cards in the frame's list / 10: Hunter reveal, Perception / Scavenging / Resistance draw, base project pile |
| 110-112 | 3 | `chosen_verb` | prefix: build, engineer, place |
| 113-123 | 11 | `chosen_type` | prefix: building type (enclosure 1-5, kiosk, pavilion, petting zoo, reptile house, aviary, unique) |
| 124-126 | 3 | `chosen_position` | prefix: project level 1-3 |
| 127-133 | 7 | `chosen_map_token` | prefix: map token 0-6 |

### `player` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `me` | this is my token |
| 1 | 1 | `opponent` | this is the opponent's token |
| 2 | 1 | `to_move` | this player decides now |
| 3 | 1 | `appeal` | appeal / 60 |
| 4 | 1 | `money` | money / 50 |
| 5 | 1 | `conservation` | conservation / 30 |
| 6 | 1 | `reputation` | reputation / 15 |
| 7 | 1 | `xtokens` | X-tokens / 5 |
| 8 | 1 | `workers` | association workers owned / 4 |
| 9 | 1 | `active_workers` | workers not on the association board / 4 |
| 10-13 | 4 | `workers_out` | workers placed per task (reputation, partner zoo, university, project) / 4 |
| 14 | 1 | `hand_size` | hand cards / 20 |
| 15 | 1 | `hand_limit` | hand limit / 5 |
| 16 | 1 | `zoo_cards` | cards in the zoo / 30 |
| 17 | 1 | `tucked` | cards tucked face down (Pouch) / 10 |
| 18 | 1 | `final_cards` | final scoring cards held / 6 |
| 19-23 | 5 | `partner_zoos` | partner zoos, by continent |
| 24-26 | 3 | `universities` | universities (hand, science2, science_rep) |
| 27-33 | 7 | `map_tokens` | the 7 tokens still on the map's left edge (a token leaves when a project is supported) |
| 34-48 | 15 | `icons` | icons of the zoo / 10: bird, herbivore, predator, primate, reptile, bear, pet, africa, americas, asia, australia, europe, science, water, rock |
| 49 | 1 | `victory_points` | appeal minus the target appeal of the conservation space / 30 |
| 50 | 1 | `poisoned` | a Venom token lies on one of the action cards |
| 51 | 1 | `covered_cells` | map cells covered by buildings / 50 |
| 52-58 | 7 | `buildings` | buildings by kind / 10: enclosure, kiosk, pavilion, petting zoo, reptile house, aviary, unique |
| 59 | 1 | `kiosk_income` | money all kiosks earn at a break / 10 |
| 60 | 1 | `appeal_income` | money the appeal earns at a break / 40 |
| 61 | 1 | `display_range` | display folders within reputation range / 6 |
| 62-72 | 11 | `placeable` | per building type (enclosure 1-5, kiosk, pavilion, petting zoo, reptile house, aviary, unique): this player's own Build card offers it and its map has a legal spot for it now, money and strength aside (unique: always 0) |

### `action_card` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `mine` | my card |
| 1 | 1 | `opponent` | the opponent's card |
| 2-6 | 5 | `action` | animals, build, cards, association, sponsors (also the entity id) |
| 7-11 | 5 | `slot` | slot 1-5, one-hot |
| 12 | 1 | `strength` | slot / 5 |
| 13 | 1 | `effective_strength` | strength it would be used with now / 5: slot minus 2 under a Constriction token (can be <= 0) |
| 14 | 1 | `upgraded` | side II |
| 15 | 1 | `venom` | a Venom token lies on it |
| 16 | 1 | `constriction` | a Constriction token lies on it |
| 17 | 1 | `multiplier` | Multiplier tokens on it / 4 |
| 18 | 1 | `venom_cost` | using it now costs 2 money (the owner is poisoned and this card has no Venom token) |
| 19 | 1 | `in_use` | it is the card of the action in progress |

### `card` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `my_hand` | in my hand |
| 1 | 1 | `opp_hand` | in the opponent's hand (seen entering it, or unknown) |
| 2 | 1 | `display` | on the display |
| 3 | 1 | `public_list` | in the pending frame's public list (Hunter reveal, base project pile) |
| 4 | 1 | `private_list` | in my pending private list (Perception / Scavenging / Resistance draw) |
| 5 | 1 | `opp_maybe_tucked` | seen in the opponent's hand before a face-down tuck: in its hand or tucked |
| 6 | 1 | `chosen` | named by a take atom of the prefix (hand / display / project) |
| 7 | 1 | `chosen_discard` | named by a discard atom of the prefix (setup discards so far) |
| 8 | 1 | `display_pos` | display folder / 6 (also the extra money to play it from there) |
| 9 | 1 | `in_range` | display card within my reputation range |
| 10 | 1 | `unknown` | an unseen hand card of the opponent: entity 0, no other feature |
| 11 | 1 | `my_zoo` | in my zoo |
| 12 | 1 | `opp_zoo` | in the opponent's zoo |
| 13 | 1 | `my_final` | one of my final scoring cards |
| 14 | 1 | `pending` | the sponsor whose unique building is being placed |
| 15-17 | 3 | `type` | animal, sponsor, final scoring card |
| 18 | 1 | `cost` | printed cost / 36 |
| 19 | 1 | `money_now` | money for its holder to play it now / 36: animals cost less the holder's partner zoo and expert discounts, plus the display folder (cards not yet in a zoo) |
| 20 | 1 | `affordable` | the holder has that money |
| 21 | 1 | `size` | animal size / 5 |
| 22 | 1 | `appeal` | appeal / 10 |
| 23 | 1 | `conservation` | conservation / 5 |
| 24 | 1 | `reputation` | reputation / 5 |
| 25 | 1 | `level` | sponsor level / 6 |
| 26 | 1 | `rock` | rock spaces required next to the enclosure / 2 |
| 27 | 1 | `water` | water spaces required / 2 |
| 28 | 1 | `no_standard` | animal that cannot live in a standard enclosure (petting zoo animal) |
| 29-34 | 6 | `special` | per special enclosure (reptile house, aviary, petting zoo): [may live there, spaces it takes / 5] |
| 35-49 | 15 | `tags` | icons printed top right / 2 (same order as `player.icons`) |
| 50-69 | 20 | `req` | conditions: 15 icons (count / 2), partner zoo (count / 2), Animals II, Sponsors II (0.5), maximum appeal / 50, minimum reputation / 15 |
| 70 | 1 | `unmet` | conditions the holder does not meet / 3 |
| 71 | 1 | `req_ok` | the holder meets the conditions to play it (cards not yet in a zoo) |
| 72 | 1 | `has_place` | the holder's zoo has room for it: an enclosure for the animal, a spot for the sponsor's building |
| 73-116 | 44 | `ability` | ability keywords in the order of `Keyword` (arknova_cards.h): 1, or (X + 1) / 6 for "keyword X" |
| 117-121 | 5 | `effect` | the card has: an immediate effect, an icon-triggered effect, a break income, an end-of-game effect, any lasting effect |
| 122-127 | 6 | `building` | sponsor's unique building: [present, size / 5, rock next to it / 2, water / 2, border spaces / 2, must touch a building] |
| 128-135 | 8 | `scoring` | final scoring card: 4 x [threshold / 24, conservation / 4] |
| 136-137 | 2 | `yield` | what its end-of-game effect would score now for its holder: [conservation / 4, appeal / 10] (final scoring cards: my final cards and offered ones; sponsors: in a zoo for its owner, in my private list for me) |
| 138 | 1 | `tokens` | tokens on the card / 5 (breeding, Okapi; Waza: 1 small, 2 large) |
| 139-142 | 4 | `immediate` | sponsor: what its immediate effect would give its holder if played now (cards not yet in a zoo; the rules' effect run on a copy of the state, with the card counted in the holder's zoo): [money / 20, appeal / 10, conservation / 5, cards to take + workers + free buildings / 2], each capped at 1 |
| 143-145 | 3 | `income` | sponsor: what its break income pays its holder now (as if in the holder's zoo when not yet there): [money / 10, conservation / 2, X-tokens + cards to take / 2], each capped at 1 |
| 146-160 | 15 | `trigger_icon` | sponsor with an icon trigger: the icon whose plays trigger it, one-hot (same order as `player.icons`; none for Explorer, which reacts to icons new to the zoo) |
| 161 | 1 | `trigger_any` | the trigger counts icons played into any zoo (else only into its owner's) |
| 162-164 | 3 | `trigger_gain` | per triggering icon: [money / 3, appeal / 2, something else: a track step, an X-token, a free building, a card, an ability] |
| 165 | 1 | `strength_short` | animal / sponsor not yet in a zoo: strength points the holder's Animals / Sponsors card lacks to play it now, X-tokens aside, / 5 (capped at 1). Animals I plays from strength 2, Animals II from 1; Sponsors plays levels up to its strength, +1 on side II |

### `project` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `my_hand` | in my hand |
| 1 | 1 | `opp_hand` | in the opponent's hand (seen entering it, or unknown) |
| 2 | 1 | `display` | on the display |
| 3 | 1 | `public_list` | in the pending frame's public list (Hunter reveal, base project pile) |
| 4 | 1 | `private_list` | in my pending private list (Perception / Scavenging / Resistance draw) |
| 5 | 1 | `opp_maybe_tucked` | seen in the opponent's hand before a face-down tuck: in its hand or tucked |
| 6 | 1 | `chosen` | named by a take atom of the prefix (hand / display / project) |
| 7 | 1 | `chosen_discard` | named by a discard atom of the prefix (setup discards so far) |
| 8 | 1 | `display_pos` | display folder / 6 (also the extra money to play it from there) |
| 9 | 1 | `in_range` | display card within my reputation range |
| 10 | 1 | `in_play` | a conservation project in play |
| 11-14 | 4 | `category` | base, release, breed, normal |
| 15-33 | 19 | `icon` | what it counts, in the order of `ProjIcon`: the 15 icons, all animal categories, all continents, small animals, large animals |
| 34-36 | 3 | `level_required` | per level 1-3: icons required / 8 (0: none) |
| 37-39 | 3 | `level_size_min` | per level: smallest animal size to release / 5 |
| 40-42 | 3 | `level_size_max` | per level: largest animal size to release / 5 |
| 43-45 | 3 | `level_conservation` | per level: conservation reward / 5 |
| 46-48 | 3 | `level_reputation` | per level: reputation reward / 5 |
| 49-52 | 4 | `play_bonus` | bonus for playing the card: [money / 10, appeal / 10, conservation / 5, reputation / 5] |
| 53-55 | 3 | `occupied_free` | in play, per level: free |
| 56-58 | 3 | `occupied_me` | in play, per level: mine |
| 59-61 | 3 | `occupied_opp` | in play, per level: the opponent's |
| 62-64 | 3 | `occupied_blocked` | in play, per level: blocked (2 players) |
| 65-67 | 3 | `can_me` | per level: I could support it there now (level free, not yet supported by me, condition met, a map token left) |
| 68-70 | 3 | `can_opp` | per level: the opponent could |
| 71 | 1 | `count_me` | how many of its icon my zoo has / 8 |
| 72 | 1 | `count_opp` | the opponent's count / 8 |
| 73 | 1 | `affordable` | I have the money for its display folder (1 when not on the display) |
| 74 | 1 | `strength_short` | strength points the holder's Association card lacks to support a project, X-tokens aside, / 5 (needs 5, 4 with the Veterinarian) |

### `cell` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0-2 | 3 | `terrain` | plain, water, rock |
| 3 | 1 | `build2_only` | needs Build II |
| 4 | 1 | `border` | on the border |
| 5-11 | 7 | `bonus` | placement bonus: none, money5, money10, xtoken, rep2, card, clever |
| 12 | 1 | `covered` | covered by a building |
| 13 | 1 | `adjacent` | next to a covered cell |
| 14 | 1 | `buildable` | free and allowed for me now (terrain, Build II) |
| 15-21 | 7 | `kind` | kind of the building on it: enclosure, kiosk, pavilion, petting zoo, reptile house, aviary, unique |
| 22 | 1 | `anchor` | the lowest cell of its building (where an enclosure's token sits) |
| 23 | 1 | `anchor_dq` | q offset to that cell / 4 |
| 24 | 1 | `anchor_dr` | r offset to that cell / 4 |
| 25 | 1 | `rock_neighbors` | rock cells next to it / 6 |
| 26 | 1 | `water_neighbors` | water cells next to it / 6 |
| 27 | 1 | `kiosk_income` | a kiosk: money it earns at a break / 6 |
| 28 | 1 | `kiosks_adjacent` | my kiosks next to it / 3 |

### `enclosure` token

| offset | width | feature | meaning |
|---|---|---|---|
| 0 | 1 | `mine` | in my zoo (positioned, pointer target) |
| 1 | 1 | `opponent` | in the opponent's zoo |
| 2-8 | 7 | `kind` | enclosure, kiosk, pavilion, petting zoo, reptile house, aviary, unique (only housing kinds occur) |
| 9 | 1 | `size` | size / 5 |
| 10 | 1 | `capacity` | animal size it holds (standard) or spaces (special) / 5 |
| 11 | 1 | `occupied` | standard enclosure with an animal |
| 12 | 1 | `used` | special enclosure: spaces used / 5 |
| 13 | 1 | `free` | room left / 5: capacity of an empty standard enclosure, free spaces of a special one |
| 14 | 1 | `rock` | rock cells next to the whole enclosure / 6 |
| 15 | 1 | `water` | water cells next to it / 6 |
| 16 | 1 | `kiosks_adjacent` | kiosks next to it / 3 |
| 17 | 1 | `animals` | animals housed / 4 |
| 18 | 1 | `target` | the new special enclosure animals may move into (move_animals) |

### anchors

| anchor | first id | count | token | head 0 | head 1 | head 2 |
|---|---|---|---|---|---|---|
| `hand` | 1 | 240 | my hand card, by card index | `hand:c` | `hand_discard:c` |  |
| `zoo` | 241 | 240 | my zoo card, by card index | `zoo:c` | `zoo_give:c` |  |
| `display` | 481 | 6 | display card, by folder | `display:i` | `display_discard:i` | `snap:i` |
| `project` | 487 | 10 | project in play, by slot (0-2 base, 3+ above the board) | `project:i` |  |  |
| `public_list` | 497 | 31 | card of the public list, by position | `list:j` |  |  |
| `private_list` | 528 | 6 | card of my private list, by position | `private:j` |  |  |
| `final` | 534 | 6 | my final scoring card, by position |  | `final_discard:i` |  |
| `enclosure` | 540 | 58 | my enclosure, by building id | `enclosure:b` |  |  |
| `cell` | 598 | 58 | my map cell | `cell:c/o0` | `cell:c/o1` | `cell:c/o2` (heads 3-5: `o3`-`o5`) |

Pointer heads: 6 per token (`TokenLayout.pointer_heads`). Head 0 = take / pick,
head 1 = give, head 2 = snap; on `cell` tokens heads 0–5 are the six rotations
of the building whose lowest cell this is. The logit of pointer atom `a` is
read from the token whose header names the atom's anchor, on the atom's head;
an atom whose token is absent is never legal. Atoms 0–131 have no anchor.
`tests/arknova/test_arknova_actions.py` checks on a model built from
`config/train.json` that each legal atom's logit comes from exactly that token
and head, and that no two legal atoms share one.

### entities

| entity | first id | count |
|---|---|---|
| `card` | 1 | 240 |
| `cell` | 241 | 58 |
| `action_card` | 299 | 5 |

Tokens of other types have entity 0, as has an `unknown` card.

### Verification

`tests/arknova/arknova_reference.py` re-derives the facts from sources other
than the encoder: the enum orders from the headers, the card table from
`arknova_cards_data.inc`, map 0 from `kMap0Columns`, the state from
`get_state_dict()["core_json"]`, rulebook formulas (strength, income,
conditions, project support, the map facts end-of-game cards count) and the
sponsor effects as printed on the cards (typed in the test from the card
texts, since the card table has no effect data). `tests/arknova/arknova_checks.py` decodes every
token of an encoding back into facts with the layout the engine reports
(`get_state_dict()["encoder_layout"]`) and compares them, for both players'
views, along random and greedy games (`test_arknova_encoding.py`). The same
check also compares the selfplay sample (encoded from the per-seat state, whose
hidden slots are stale) bit for bit with the encoding of the ground-truth
state. Over 100 random and 30 greedy full games (71,433 encodings, layout
before the sponsor effect features) every feature was compared, all but
`decision.only_one` at least once with a non-zero value; with the current
layout, 30 random and 10 greedy full games (21,494 encodings) gave no mismatch
either (the sponsor features and `yield` non-zero in 19k-77k card tokens each).
`decision.only_one` (Dominance with the primates base project still in the
pile) is checked in greedy game 47 by
`test_dominance_offers_only_the_primates_project`. The feature tables above
are checked against the engine's layout by
`test_docs_list_every_feature_and_atom`.

## State Fields (`get_state_dict`)

- `core_json`: the core state as JSON in the Python engine's layout, with every
  player's private cards and the deck contents removed (hand cards stay only
  when listed in `public_hand`).
- `prefix`: the atoms chosen so far in the pending decision, space-separated.
- `encoder_layout`: JSON description of the token layout (token types with
  their features' offsets and widths, anchors, entities, header fields), the
  same for every state.
- `scores`, `winner`, `current_player`, `is_terminal`, `num_players`.

## Heuristics

- `greedy`: C++ port of the reference `GreedyBot` (valuation in
  appeal-equivalents plus a search to the end of its own turn). It runs on one
  world sampled by the tracker from the masked state. Self-play scores match
  the Python bot (about 14 points per player) at about 0.5 s per game.
- `beam_2x3`, `beam_3x8`: the reference `LookaheadBot` (depth x width, opponents
  passive, no `reply` / `open_info`), on one sampled world. Being stateless, it
  re-runs the beam at every decision of its own turn instead of replaying the
  plan made at the turn decision. Over 100 games per pair: beam_2x3 beats greedy
  64%, beam_3x8 beats greedy 74% and beam_2x3 64% (Python bots: 67 / 75 / 75%).
  beam_2x3 guides selfplay; beam_3x8 is the `eval_vs_heuristic` benchmark.
- `random`: uniform over legal atoms.

## Playing on ark-nova-ai's Board

`scripts/arknova_board.py` runs ark-nova-ai's server (its page, engine and logins) with
two opponents, `dinoboard` and its `beam_3x8`, on Map 0. The bundle's external-engine
hooks carry it:

- `external_state_loader`: an ark-nova-ai state, as `tools/diff_test.py` `export_state`
  writes it, becomes the truth state (`GameSession.load_external_state`). The AI views
  are rebuilt from it masked, so the search never reads the other player's cards;
  cards in `public_hand` stay visible to them.
- `external_option_lister`: the pending decision as (ark-nova-ai option key, atom
  sequence) pairs consistent with the atoms applied so far
  (`GameSession.external_options`). The bridge searches one atom at a time until
  the atoms name a single option.
- `external_rules_switch`: `g_upstream_rules` for the whole process
  (`set_external_rules("arknova", True)`), so both engines list the same options.

`tests/arknova/test_ark_nova_ai_interop.py` checks the option lists, the atom
narrowing and that the search ignores the opponent's hidden cards and the deck.

## Configs

`config/game.json` (feature 0 = variable tokens, actions 1569, belief 1218 / 240, max 1500
plies), `mcts.json`, `train.json` (token transformer with
`relative_position_radius` 4, beam_2x3 guidance),
`belief.json`, `heuristic.json`, `web.json`.
