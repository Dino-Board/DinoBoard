#include "arknova_greedy_heuristic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../../engine/core/belief_tracker.h"
#include "../arknova_actions.h"
#include "../arknova_internal.h"
#include "../arknova_state.h"

namespace board_ai::arknova {

namespace {

constexpr double kUpgradeWeight[kNumActions] = {1.2, 0.85, 1.0, 1.1, 0.7};  // ActionCard order
constexpr double kSlotWeight[kNumActions] = {1.2, 1.0, 0.7, 1.1, 0.6};
constexpr int kBreaksPerGame = 7;
constexpr int kMoneyWorking = 25;  // priced in full: enough for an animal and its enclosure
constexpr int kMoneySpare = 25;    // priced low; a larger hoard adds nothing
constexpr int kProjectStrength = 5;
constexpr int kPlacementsPerBuilding = 3;
constexpr std::size_t kFollowWidth = 15;
constexpr int kPassiveLimit = 80;
constexpr int kTrackStart = -14;

class SearchChance final : public Chance {
 public:
  explicit SearchChance(std::mt19937_64& rng) : rng_(rng) {}
  void shuffle(std::vector<CardIdx>& cards) override {
    for (int i = static_cast<int>(cards.size()) - 1; i > 0; --i) {
      std::swap(cards[i], cards[std::uniform_int_distribution<int>(0, i)(rng_)]);
    }
  }
  CardIdx pick(const std::vector<CardIdx>& candidates) override {
    if (candidates.empty()) throw std::logic_error("arknova greedy: pick from no candidates");
    return candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng_)];
  }

 private:
  std::mt19937_64& rng_;
};

// ---- valuation ------------------------------------------------------------------------

int track_value(const Player& pl) { return pl.appeal + 100 - target_appeal(pl.conservation); }

double progress(const State& s) {
  double best = -std::numeric_limits<double>::infinity();
  for (const Player& pl : s.players) {
    best = std::max(best, static_cast<double>(track_value(pl) - kTrackStart) / (100 - kTrackStart));
  }
  return std::clamp(best, 0.0, 1.0);
}

// (appeal, conservation) the player's sponsors and final scoring cards would score right now.
std::pair<int, int> endgame_points(const State& s, int p) {
  State copy = s;
  const Player& pl = s.players[p];
  for (int i = 0; i < pl.n_zoo_cards; ++i) {
    if (has_endgame(pl.zoo_cards[i])) endgame_effect(copy, pl.zoo_cards[i], p);
  }
  for (int i = 0; i < pl.n_final; ++i) {
    if (has_endgame(pl.final_cards[i])) endgame_effect(copy, pl.final_cards[i], p);
  }
  return {copy.players[p].appeal - pl.appeal, copy.players[p].conservation - pl.conservation};
}

double sponsor_lasting(CardIdx c, const GreedyWeights& w) {
  return card(c).lasting_effect ? w.sponsor_level * card(c).level : 0.0;
}

struct PlayGain {
  double gain;
  int cost;
  bool playable;
};

PlayGain play_gain(const Player& pl, CardIdx c, double left, const GreedyWeights& w) {
  const CardDef& d = card(c);
  double printed = d.appeal + w.conservation * d.conservation + w.reputation * d.reputation;
  if (d.type == CardType::kAnimal) {
    int cost = animal_cost(pl, c);
    if (housings(pl, c).empty() && d.enc_standard > 0) cost += kBuildCostPerSpace * d.enc_standard;
    double gain = printed + (d.n_abilities ? 1.0 : 0.0) - cost * w.money;
    return {gain, cost, unmet_requirements(pl, c) == 0};
  }
  if (d.type == CardType::kSponsor) {
    return {printed + sponsor_lasting(c, w) * left + 1.0, 0, unmet_requirements(pl, c) == 0};
  }
  return {1.5, 0, false};
}

double hand_value(const Player& pl, double left, const GreedyWeights& w) {
  std::vector<PlayGain> rated;
  for (CardIdx c : pl.hand_cards()) rated.push_back(play_gain(pl, c, left, w));
  std::stable_sort(rated.begin(), rated.end(), [](const PlayGain& a, const PlayGain& b) {
    return std::tie(a.playable, a.gain) > std::tie(b.playable, b.gain);
  });
  double value = 0.0;
  int limit = std::min<int>(pl.hand_limit(), static_cast<int>(rated.size()));  // the rest is discarded
  for (int i = 0; i < limit; ++i) {
    value += rated[i].playable && rated[i].gain > 0 ? std::min(3.0, w.hold * rated[i].gain) + 0.3 : 0.4;
  }
  return value;
}

double diminishing(double first, int count, double decay) {
  double v = 0.0;
  for (int i = 0; i < count; ++i) v += first * std::pow(decay, i);
  return v;
}

bool houses_standard(const Player& pl, CardIdx c, int building) {
  for (const Housing& h : housings(pl, c)) {
    if (h.building == building && h.spaces == 0) return true;
  }
  return false;
}

// ---- option handling --------------------------------------------------------------------

double placement_score(const Zoo& zoo, CellMask cells, bool enclosure) {
  double score = 0.0;
  for (CellMask m = cells; m; m &= m - 1) {
    const int c = __builtin_ctzll(m);
    if (zoo.cell(c).bonus != kPBNone) score += 2.0;
    for (int n : map_cell(c).nbr) {
      if (n < 0) continue;
      if (zoo.cell_owner[n] != kNoBuilding) {
        score += 0.15;  // keep the zoo compact
      } else if (enclosure && zoo.cell(n).terrain != Terrain::kPlain) {
        score += 0.3;  // rock and water neighbours keep animal options open
      }
    }
  }
  return score;
}

int card_strength(const State& s, int owner, int action) {
  int penalty = has_token(s, owner, action, TokenKind::kConstriction) ? kConstrictionPenalty : 0;
  return s.players[owner].strength(action) - penalty;
}

// Indices of `options` worth searching: a few good spots per building, and each action card
// with no X-token, one, or exactly enough to reach strength 5.
std::vector<int> shortlist(const State& s, const std::vector<Option>& options) {
  const Frame& f = s.top();
  std::vector<int> out;
  if (f.kind == FrameKind::act_build || f.kind == FrameKind::free_build || f.kind == FrameKind::sponsor_building) {
    const Zoo& zoo = s.players[f.player].zoo;
    std::vector<std::tuple<int, int, int>> keys;
    std::map<std::tuple<int, int, int>, std::vector<int>> groups;
    std::vector<int> rest;
    for (int i = 0; i < static_cast<int>(options.size()); ++i) {
      const Option& o = options[i];
      bool placement = o.head == Head::build || o.head == Head::engineer || o.head == Head::place;
      if (!placement || !o.cells) {
        rest.push_back(i);
        continue;
      }
      auto key = o.n >= 2 ? std::make_tuple(static_cast<int>(o.head), int{o.a[0]}, int{o.a[1]})
                          : std::make_tuple(static_cast<int>(o.head), -1, -1);
      if (!groups.count(key)) keys.push_back(key);
      groups[key].push_back(i);
    }
    for (const auto& key : keys) {
      std::vector<int>& g = groups[key];
      bool enclosure = std::get<1>(key) == static_cast<int>(BuildingKind::kEnclosure);
      std::vector<double> score(options.size());
      for (int i : g) score[i] = placement_score(zoo, options[i].cells, enclosure);
      std::stable_sort(g.begin(), g.end(), [&](int a, int b) { return score[a] > score[b]; });
      out.insert(out.end(), g.begin(), g.begin() + std::min<std::size_t>(g.size(), kPlacementsPerBuilding));
    }
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
  }
  bool turn = f.kind == FrameKind::turn || f.kind == FrameKind::extra_action;
  int owner = f.kind == FrameKind::extra_action ? f.d[fd::extra_owner] : f.player;
  for (int i = 0; i < static_cast<int>(options.size()); ++i) {
    const Option& o = options[i];
    if (turn && o.head == Head::act && o.a[1] > 1 && card_strength(s, owner, o.a[0]) + o.a[1] != kProjectStrength) {
      continue;
    }
    out.push_back(i);
  }
  return out;
}

const Option& passive_option(const std::vector<Option>& options) {
  for (const Option& o : options) {
    if (o.head == Head::done || o.head == Head::skip || o.head == Head::no) return o;
  }
  return options.front();
}

void finish_passively(State& s, Chance& ch) {
  for (int limit = kPassiveLimit; !s.over && s.top().kind != FrameKind::turn && limit; --limit) {
    step(s, passive_option(legal_options(s)), ch);
  }
}

double value_after(const State& s, const Option& o, int p, int depth, const GreedyWeights& w, Chance& ch) {
  State clone = s;
  step(clone, o, ch);
  if (depth && !clone.over && clone.top().kind != FrameKind::turn && clone.top().player == p) {
    std::vector<Option> follow = legal_options(clone);
    std::vector<int> keep = shortlist(clone, follow);
    if (keep.size() > kFollowWidth) keep.resize(kFollowWidth);
    if (keep.size() > 1) {
      double best = -std::numeric_limits<double>::infinity();
      for (int i : keep) best = std::max(best, value_after(clone, follow[i], p, depth - 1, w, ch));
      return best;
    }
  }
  finish_passively(clone, ch);
  return greedy_evaluate(clone, p, w);
}

double keep_value(const Player& pl, const Option& o, const GreedyWeights& w) {
  double v = 0.0;
  for (int i = 0; i < o.n; ++i) {
    PlayGain g = play_gain(pl, o.a[i], 1.0, w);
    v += std::max(0.0, g.gain) * (g.playable ? 1.0 : 0.3);
  }
  return v;
}

}  // namespace

double greedy_evaluate(const State& s, int p, const GreedyWeights& w) {
  const Player& pl = s.players[p];
  double left = 1.0 - progress(s);
  double value = track_value(pl);
  auto [appeal, conservation] = endgame_points(s, p);
  value += (appeal + w.conservation * conservation) * (1.0 - 0.5 * left);
  value += (appeal_income(pl.appeal) + pl.zoo.kiosk_income()) * kBreaksPerGame * left * w.income;

  value += hand_value(pl, left, w) + w.money * std::min<int>(pl.money, kMoneyWorking);
  value += w.spare_money * std::min(std::max(pl.money - kMoneyWorking, 0), kMoneySpare);

  value += pl.xtokens * w.xtoken;
  value += (w.reputation * std::min(pl.reputation - 1, 8) + w.reputation_high * std::max(pl.reputation - 9, 0)) * left;
  double upgraded = 0.0;
  for (int a = 0; a < kNumActions; ++a) {
    if (pl.is_upgraded(a)) upgraded += kUpgradeWeight[a];
  }
  value += upgraded * (w.upgrade * left + 0.5);
  value += (diminishing(w.worker, pl.workers - 1, w.decay) + diminishing(w.partner, pl.n_partner_zoos, w.decay) +
            diminishing(w.university, pl.n_universities, w.decay)) *
           left;
  double slots = 0.0;
  for (int i = 0; i < kNumActions; ++i) slots += kSlotWeight[pl.slots[i]] * i;
  value += slots * w.slot;
  double lasting = 0.0;
  for (int i = 0; i < pl.n_zoo_cards; ++i) {
    if (is_sponsor(pl.zoo_cards[i])) lasting += sponsor_lasting(pl.zoo_cards[i], w);
  }
  value += lasting * left;

  std::vector<CardIdx> playable_animals;
  for (CardIdx c : pl.hand_cards()) {
    if (is_animal(c) && unmet_requirements(pl, c) == 0) playable_animals.push_back(c);
  }
  for (int b = 0; b < pl.zoo.n_buildings; ++b) {
    const Building& bd = pl.zoo.buildings[b];
    if (bd.kind == BuildingKind::kEnclosure && !bd.occupied) {
      bool fits = std::any_of(playable_animals.begin(), playable_animals.end(),
                              [&](CardIdx c) { return houses_standard(pl, c, b); });
      double share = fits ? 1.0 : w.idle_enclosure;
      value += kBuildCostPerSpace * bd.size * w.money * share + (fits ? 0.3 : 0.0);
    } else if (is_special(bd.kind)) {
      value += kBuildCostPerSpace * (bd.size - bd.used) * w.money * w.idle_enclosure;
    }
  }

  if (s.end_by == p || (s.end_by < 0 && pl.victory_points() >= 0)) {
    int best_other = std::numeric_limits<int>::min();
    for (int q = 0; q < kPlayers; ++q) {
      if (q != p) best_other = std::max(best_other, s.players[q].victory_points());
    }
    value += pl.victory_points() >= best_other ? 3 : -4;  // only end the game when ahead
  }
  return value;
}

namespace {

// bot.GreedyBot.choose's valuation of each option (-inf for options it does not search).
std::vector<double> greedy_values(const State& s, int p, const std::vector<Option>& options, const GreedyWeights& w,
                                  std::mt19937_64& rng) {
  std::vector<double> value(options.size(), -std::numeric_limits<double>::infinity());
  const Frame& f = s.top();
  if (f.kind == FrameKind::setup_keep) {
    for (std::size_t i = 0; i < options.size(); ++i) value[i] = keep_value(s.players[p], options[i], w);
    return value;
  }
  SearchChance chance(rng);
  std::uniform_real_distribution<double> jitter(0.0, 0.01);
  int depth = f.kind == FrameKind::turn || f.kind == FrameKind::extra_action ? 2 : 1;
  std::vector<int> keep = shortlist(s, options);
  if (keep.empty()) throw std::logic_error("arknova greedy: shortlist dropped every option");
  for (int i : keep) value[i] = value_after(s, options[i], p, depth, w, chance) + jitter(rng);
  return value;
}

// ---- LookaheadBot ---------------------------------------------------------------------

// bot._one_of_each groups options by their Python key minus a trailing int or cell tuple.
std::string kind_key(const State& s, const Option& o) {
  std::string key = option_to_json(s, o);
  std::size_t last_comma = std::string::npos;
  int depth = 0, elements = 1;
  bool in_string = false;
  for (std::size_t i = 0; i < key.size(); ++i) {
    char c = key[i];
    if (in_string) {
      if (c == '\\') ++i;
      else if (c == '"') in_string = false;
    } else if (c == '"') {
      in_string = true;
    } else if (c == '[') {
      ++depth;
    } else if (c == ']') {
      --depth;
    } else if (c == ',' && depth == 1) {
      last_comma = i;
      ++elements;
    }
  }
  if (elements < 2) return key;
  char first = key[last_comma + 1];
  bool drop = first == '[' || first == '-' || (first >= '0' && first <= '9');
  return drop ? key.substr(0, last_comma) + "]" : key;
}

// _turn_options as indices into `options`: each action card at base strength, boosted to strength
// 5 with X-tokens when it has them (one X-token also when that is the only way to use the card),
// one X-token action. All of them when the filter keeps none.
std::vector<int> turn_option_indices(const State& s, const std::vector<Option>& options) {
  const Frame& f = s.top();
  int owner = f.kind == FrameKind::extra_action ? f.d[fd::extra_owner] : f.player;
  std::vector<bool> usable(kNumActions, false);
  for (const Option& o : options) {
    if (o.head == Head::act && o.a[1] == 0) usable[o.a[0]] = true;
  }
  std::vector<int> kept;
  bool x_taken = false;
  for (int i = 0; i < static_cast<int>(options.size()); ++i) {
    const Option& o = options[i];
    if (o.head == Head::act) {
      int spent = o.a[1];
      bool reaches_five = spent && card_strength(s, owner, o.a[0]) + spent == kProjectStrength;
      if (spent == 0 || reaches_five || (spent == 1 && !usable[o.a[0]])) kept.push_back(i);
    } else if (o.head == Head::xtoken) {
      if (!x_taken) kept.push_back(i);
      x_taken = true;
    } else {
      kept.push_back(i);
    }
  }
  if (kept.empty()) {
    for (int i = 0; i < static_cast<int>(options.size()); ++i) kept.push_back(i);
  }
  return kept;
}

class Lookahead {
 public:
  Lookahead(const LookaheadConfig& cfg, const GreedyWeights& w, Chance& ch) : cfg_(cfg), w_(w), ch_(ch) {}

  // Value of each candidate: the best final beam line starting with it (-inf for candidates the
  // filter or the beam dropped). At a turn decision the filter is _turn_options; at a later step
  // of the mover's own turn it is the shortlist, and the line first completes the current turn.
  std::vector<double> plan_candidates(const State& base, int p, const std::vector<Option>& candidates) {
    FrameKind k = base.top().kind;
    std::vector<int> kept = k == FrameKind::turn || k == FrameKind::extra_action ? turn_option_indices(base, candidates)
                                                                               : shortlist(base, candidates);
    if (kept.empty()) throw std::logic_error("arknova lookahead: no candidate survived the filter");
    std::vector<Option> roots;
    for (int i : kept) roots.push_back(candidates[i]);
    std::vector<double> best = plan(base, p, roots);
    std::vector<double> out(candidates.size(), -std::numeric_limits<double>::infinity());
    for (std::size_t r = 0; r < kept.size(); ++r) out[kept[r]] = best[r];
    return out;
  }

 private:
  // Best final beam value per root option (-inf for roots the beam dropped).
  std::vector<double> plan(const State& base, int p, const std::vector<Option>& roots) {
    struct Line {
      double value;
      long long order;
      State s;
      int root;
    };
    auto by_value = [](const Line& a, const Line& b) {
      return a.value != b.value ? a.value > b.value : a.order < b.order;
    };
    std::vector<Line> beam;
    {
      State start = base;
      auto lines = turns_from(start, p, &roots);
      for (std::size_t i = 0; i < lines.size(); ++i) {
        beam.push_back({greedy_evaluate(lines[i].first, p, w_), static_cast<long long>(i), std::move(lines[i].first),
                        lines[i].second});
      }
    }
    std::stable_sort(beam.begin(), beam.end(), by_value);
    if (static_cast<int>(beam.size()) > cfg_.width) beam.resize(cfg_.width);
    for (int d = 1; d < cfg_.depth; ++d) {
      std::vector<Line> reached;
      for (Line& line : beam) {
        if (line.s.over) {
          reached.push_back(std::move(line));
          continue;
        }
        if (line.s.top().player != p) {  // opponents are passive in the search
          line.s.stack[line.s.stack_n - 1].player = static_cast<std::int8_t>(p);
          line.s.current = static_cast<std::int8_t>(p);
        }
        auto lines = turns_from(line.s, p, nullptr);
        for (std::size_t j = 0; j < lines.size(); ++j) {
          reached.push_back({greedy_evaluate(lines[j].first, p, w_), line.order * 1000 + static_cast<long long>(j),
                             std::move(lines[j].first), line.root});
        }
      }
      std::stable_sort(reached.begin(), reached.end(), by_value);
      if (static_cast<int>(reached.size()) > cfg_.width) reached.resize(cfg_.width);
      beam = std::move(reached);
    }
    std::vector<double> best(roots.size(), -std::numeric_limits<double>::infinity());
    for (const Line& line : beam) best[line.root] = std::max(best[line.root], line.value);
    return best;
  }

  std::vector<Option> turn_options(State& s) {
    std::vector<Option> options = legal_options(s);
    std::vector<Option> kept;
    for (int i : turn_option_indices(s, options)) kept.push_back(options[i]);
    return kept;
  }

  // _one_of_each: the best option of each kind, the `variants` best kinds.
  std::vector<Option> one_of_each(const State& s, int p, const std::vector<Option>& options) {
    std::vector<std::string> order;
    std::map<std::string, std::vector<int>> groups;
    for (int i = 0; i < static_cast<int>(options.size()); ++i) {
      std::string k = kind_key(s, options[i]);
      auto [it, fresh] = groups.try_emplace(k);
      if (fresh) order.push_back(k);
      it->second.push_back(i);
    }
    std::vector<std::pair<double, int>> scored;
    for (const std::string& k : order) {
      const std::vector<int>& g = groups[k];
      double best = -std::numeric_limits<double>::infinity();
      int best_i = -1;
      for (std::size_t j = 0; j < std::min<std::size_t>(g.size(), 6); ++j) {
        double v = value_after(s, options[g[j]], p, 0, w_, ch_);
        if (best_i < 0 || v > best) best = v, best_i = g[j];
      }
      scored.push_back({best, best_i});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<Option> out;
    for (std::size_t i = 0; i < scored.size() && static_cast<int>(i) < cfg_.variants; ++i) {
      out.push_back(options[scored[i].second]);
    }
    return out;
  }

  // _finish_turn: p's later steps by the best of `follow` shortlisted options, others passive.
  void finish_turn(State& s, int p) {
    for (int limit = kPassiveLimit; !s.over && s.top().kind != FrameKind::turn && limit; --limit) {
      std::vector<Option> options = legal_options(s);
      if (s.top().player == p && options.size() > 1) {
        std::vector<int> keep = shortlist(s, options);
        if (keep.empty()) throw std::logic_error("arknova lookahead: shortlist dropped every option");
        if (static_cast<int>(keep.size()) > cfg_.follow) keep.resize(cfg_.follow);
        int best_i = -1;
        double best = 0.0;
        for (int i : keep) {
          double v = value_after(s, options[i], p, 0, w_, ch_);
          if (best_i < 0 || v > best) best = v, best_i = i;
        }
        step(s, options[best_i], ch_);
      } else {
        step(s, passive_option(options), ch_);
      }
    }
  }

  // _turns_from: ways to play p's whole turn, as (position after it, index of its first option
  // in `roots`). With `roots` null the first options are _turn_options and the index is unused.
  std::vector<std::pair<State, int>> turns_from(State& s, int p, const std::vector<Option>* roots) {
    std::vector<Option> firsts = roots ? *roots : turn_options(s);
    std::vector<std::pair<State, int>> lines;
    for (int r = 0; r < static_cast<int>(firsts.size()); ++r) {
      State first = s;
      step(first, firsts[r], ch_);
      std::vector<State> branches;
      if (!first.over && first.top().kind != FrameKind::turn && first.top().player == p) {
        std::vector<Option> all = legal_options(first);
        std::vector<int> keep = shortlist(first, all);
        if (keep.size() > 1) {
          std::vector<Option> follow;
          for (int i : keep) follow.push_back(all[i]);
          for (const Option& x : one_of_each(first, p, follow)) {
            branches.push_back(first);
            step(branches.back(), x, ch_);
          }
        }
      }
      if (branches.empty()) branches.push_back(std::move(first));
      for (State& b : branches) {
        finish_turn(b, p);
        lines.emplace_back(std::move(b), r);
      }
    }
    return lines;
  }

  const LookaheadConfig& cfg_;
  const GreedyWeights& w_;
  Chance& ch_;
};

// Samples one world for the mover, values the options under the current atom prefix with
// `values(state, mover, options, rng)`, and scores each legal atom by its best option.
template <class Values>
HeuristicResult score_atoms(IGameState& masked_state, const IBeliefTracker* tracker, const IGameRules& rules,
                            std::uint64_t rng_seed, const char* who, Values values) {
  if (!tracker) throw std::logic_error(std::string(who) + ": needs the belief tracker to sample a world");
  std::mt19937_64 rng(rng_seed);
  auto world_ptr = masked_state.clone_state();
  auto& world = checked_cast<ArknovaState>(*world_ptr);
  const int p = world.current_player();
  tracker->randomize_unseen(world, p, rng);

  HeuristicResult out;
  out.actions = rules.legal_actions(world);
  if (out.actions.empty()) throw std::logic_error(std::string(who) + ": no legal actions");
  out.scores.assign(out.actions.size(), 0.0);
  if (out.actions.size() == 1) return out;

  Decision d = current_decision(world);
  const std::size_t k = world.prefix.size();
  std::vector<Option> options;
  std::vector<ActionId> next_atom;
  for (std::size_t i = 0; i < d.options.size(); ++i) {
    if (d.atoms[i].size() > k && atom_prefix_matches(d.atoms[i], world.prefix)) {
      options.push_back(d.options[i]);
      next_atom.push_back(d.atoms[i][k]);
    }
  }
  std::vector<double> value = values(world.synced_core(), p, options, rng);

  double lowest = std::numeric_limits<double>::infinity();
  for (double v : value) {
    if (std::isfinite(v)) lowest = std::min(lowest, v);
  }
  if (!std::isfinite(lowest)) throw std::logic_error(std::string(who) + ": no option was valued");
  for (std::size_t j = 0; j < out.actions.size(); ++j) {
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < options.size(); ++i) {
      if (next_atom[i] == out.actions[j]) best = std::max(best, value[i]);
    }
    out.scores[j] = std::isfinite(best) ? best : lowest - 1.0;
  }
  return out;
}

}  // namespace

HeuristicResult ArknovaGreedyHeuristic::pick(IGameState& masked_state, const IBeliefTracker* tracker,
                                             const IGameRules& rules, std::uint64_t rng_seed) const {
  return score_atoms(masked_state, tracker, rules, rng_seed, "arknova greedy",
                     [this](const State& s, int p, const std::vector<Option>& options, std::mt19937_64& rng) {
                       return greedy_values(s, p, options, w_, rng);
                     });
}

HeuristicResult ArknovaLookaheadHeuristic::pick(IGameState& masked_state, const IBeliefTracker* tracker,
                                                const IGameRules& rules, std::uint64_t rng_seed) const {
  return score_atoms(masked_state, tracker, rules, rng_seed, "arknova lookahead",
                     [this](const State& s, int p, const std::vector<Option>& options, std::mt19937_64& rng) {
                       const Frame& f = s.top();
                       bool own_turn = f.kind == FrameKind::turn || (s.current == p && f.player == p);
                       if (f.kind == FrameKind::setup_keep || !own_turn) return greedy_values(s, p, options, w_, rng);
                       SearchChance chance(rng);
                       return Lookahead(cfg_, w_, chance).plan_candidates(s, p, options);
                     });
}

}  // namespace board_ai::arknova
