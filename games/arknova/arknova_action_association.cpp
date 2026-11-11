// Association action: reputation, partner zoos, universities, conservation projects and
// donations (ark-nova-ai arknova/actions/association.py).
//
// Side I performs one task. Side II performs several tasks of different types whose values sum to
// at most the strength, then may donate once.

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

int workers_needed(const Player& pl, int task) {
  int out = pl.workers_out[task];
  int need = out == 0 ? 1 : out < 3 ? 2 : -1;
  if (need < 0) return -1;
  return pl.active_workers() >= need ? need : -1;
}

// ---- projects ---------------------------------------------------------------------

int project_count(const Player& pl, ProjIcon icon) {
  IconCounts have = icons(pl);
  if (icon == ProjIcon::kPIconAllAnimals) {
    int n = 0;
    for (int k = 0; k < kNumAnimalCategories; ++k) n += have[k] > 0;
    return n;
  }
  if (icon == ProjIcon::kPIconAllContinents) {
    int n = 0;
    for (int k = 0; k < kNumContinents; ++k) n += have[kContinentIcon0 + k] > 0;
    return n;
  }
  if (icon == ProjIcon::kPIconAnimalSize2 || icon == ProjIcon::kPIconAnimalSize4) {
    int n = 0;
    for (int i = 0; i < pl.n_zoo_cards; ++i) {
      CardIdx c = pl.zoo_cards[i];
      if (!is_animal(c)) continue;
      int size = card(c).size;
      n += icon == ProjIcon::kPIconAnimalSize2 ? size <= 2 : size >= 4;
    }
    return n;
  }
  return have[project_icon(icon)];
}

bool breed_ok(const Player& pl, ProjIcon icon) {
  Icon want = project_icon(icon);
  for (int i = 0; i < pl.n_zoo_cards; ++i) {
    CardIdx c = pl.zoo_cards[i];
    if (!is_animal(c) || !card(c).tags[want]) continue;
    for (int k = 0; k < kNumContinents; ++k) {
      if ((card(c).continents >> k & 1) && pl.has_partner_zoo(k)) return true;
    }
  }
  return false;
}

struct Extra {
  int kind;
  CardIdx card;
};

std::vector<Extra> slot_extras(const Player& pl, CardIdx project, int slot_index) {
  const CardDef& pj = card(project);
  const ProjectSlot& slot = pj.slots[slot_index];
  std::vector<Extra> out;
  switch (pj.proj_category) {
    case ProjCategory::kProjRelease:
      for (CardIdx a : sr::releasable(pl, project, slot_index)) out.push_back({kExtraRelease, a});
      return out;
    case ProjCategory::kProjBreed:
      if (breed_ok(pl, pj.proj_icon)) out.push_back({kExtraNone, kNoCard});
      return out;
    case ProjCategory::kProjBase:
    case ProjCategory::kProjNormal:
      if (slot.required < 0) return out;
      {
        int count = project_count(pl, pj.proj_icon);
        if (count >= slot.required) {
          out.push_back({kExtraNone, kNoCard});
          return out;
        }
        if (pj.proj_category == ProjCategory::kProjBase && count + 1 >= slot.required) {
          for (CardIdx b : sr::breeding_cards(pl)) out.push_back({kExtraToken, b});
        }
      }
      return out;
    default:
      return out;
  }
}

struct SupportChoice {
  int position;
  int token;
  Extra extra;
};

std::vector<SupportChoice> support_choices(const State& s, int p, CardIdx project) {
  const Player& pl = s.players[p];
  const ProjectTokens* taken = s.tokens_of(project);
  const CardDef& pj = card(project);
  std::vector<SupportChoice> out;
  if (pj.proj_category == ProjCategory::kNone) return out;
  bool mine = false;
  if (taken) {
    for (int k = 0; k < 3; ++k) mine |= taken->slot[k] == p;
  }
  if (mine && !(pj.proj_category == ProjCategory::kProjRelease && sr::multi_support_release(pl))) return out;
  for (int k = 0; k < pj.n_slots; ++k) {
    if (taken && taken->slot[k] != -1) continue;
    for (const Extra& e : slot_extras(pl, project, k)) {
      for (int t = 0; t < kNumMapTokens; ++t) {
        if (pl.tokens[t]) out.push_back({k + 1, t, e});
      }
    }
  }
  return out;
}

Option support_key(Head h, int ref, const SupportChoice& c) {
  Option o = opt(h, ref, c.position, c.token);
  o.n = 5;
  o.a[3] = static_cast<std::int16_t>(c.extra.kind);
  o.a[4] = c.extra.card;
  return o;
}

void project_options(const State& s, int p, bool side2, std::vector<Option>& out) {
  const Player& pl = s.players[p];
  std::vector<CardIdx> in_play(s.base_projects, s.base_projects + s.n_base_projects);
  in_play.insert(in_play.end(), s.projects, s.projects + s.n_projects);
  for (CardIdx pid : in_play) {
    Head h = card(pid).proj_category == ProjCategory::kProjRelease ? Head::release : Head::support;
    for (const SupportChoice& c : support_choices(s, p, pid)) out.push_back(support_key(h, pid, c));
  }
  for (CardIdx cid : pl.hand_cards()) {
    if (!is_project(cid)) continue;
    Head h = card(cid).proj_category == ProjCategory::kProjRelease ? Head::play_release : Head::play_project;
    for (const SupportChoice& c : support_choices(s, p, cid)) out.push_back(support_key(h, cid, c));
  }
  if (side2) {
    for (int index : cards_in_range(s, p)) {
      CardIdx cid = s.display[index];
      if (!is_project(cid) || pl.money < index + 1) continue;
      Head h = card(cid).proj_category == ProjCategory::kProjRelease ? Head::play_release_display : Head::play_project_display;
      for (const SupportChoice& c : support_choices(s, p, cid)) out.push_back(support_key(h, index, c));
    }
  }
}

// ---- tasks ------------------------------------------------------------------------

void task_options(const State& s, int p, int remaining, int used, bool side2, std::vector<Option>& out) {
  const Player& pl = s.players[p];
  for (int task = 0; task < kNumTasks; ++task) {
    if ((used >> task & 1) || sr::task_value(pl, task) > remaining || workers_needed(pl, task) < 0) continue;
    if (task == kTaskReputation) {
      out.push_back(opt(Head::reputation));
    } else if (task == kTaskPartnerZoo) {
      for (int c = 0; c < kNumContinents; ++c) {
        if (can_take_partner_zoo(s, p, c)) out.push_back(opt(Head::partner_zoo, c));
      }
    } else if (task == kTaskUniversity) {
      for (int u = 0; u < kNumUniversities; ++u) {
        if (can_take_university(s, p, u)) out.push_back(opt(Head::university, u));
      }
    } else {
      project_options(s, p, side2, out);
    }
  }
}

int task_of(Head h) {
  switch (h) {
    case Head::reputation: return kTaskReputation;
    case Head::partner_zoo: return kTaskPartnerZoo;
    case Head::university: return kTaskUniversity;
    case Head::support: case Head::play_project: case Head::play_project_display:
    case Head::release: case Head::play_release: case Head::play_release_display:
      return kTaskProject;
    default: engine_error("act_association: not a task");
  }
}

int donation_cost(const State& s) {
  return s.donations < kNumDonationCosts ? kDonationCosts[s.donations] : kDonationFinalCost;
}

bool can_donate(const State& s, int p, const Frame& f) {
  return f.d[fd::assoc_side2] && f.d[fd::assoc_used] && !f.d[fd::assoc_donated] && s.players[p].money >= donation_cost(s);
}

bool anything_possible(const State& s, int p, const Frame& f) {
  std::vector<Option> opts;
  task_options(s, p, f.d[fd::assoc_remaining], f.d[fd::assoc_used], f.d[fd::assoc_side2], opts);
  return !opts.empty() || can_donate(s, p, f);
}

bool association_feasible(State& s, int p, int strength) {
  std::vector<Option> opts;
  task_options(s, p, strength, 0, is_upgraded(s, p, kAssociation), opts);
  return !opts.empty();
}

void association_start(State& s, int p, int strength) {
  int fi = push(s, FrameKind::act_association, p);
  Frame& f = frame(s, fi);
  f.d[fd::assoc_strength] = static_cast<std::int16_t>(strength);
  f.d[fd::assoc_remaining] = static_cast<std::int16_t>(strength);
  f.d[fd::assoc_used] = 0;
  f.d[fd::assoc_donated] = 0;
  f.d[fd::assoc_side2] = is_upgraded(s, p, kAssociation);
}

void association_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  int p = f.player;
  task_options(s, p, f.d[fd::assoc_remaining], f.d[fd::assoc_used], f.d[fd::assoc_side2], out);
  if (can_donate(s, p, f)) out.push_back(opt(Head::donate));
  if (f.d[fd::assoc_used]) out.push_back(opt(Head::done));
}

void put_project_in_play(State& s, CardIdx project) {
  std::copy_backward(s.projects, s.projects + s.n_projects, s.projects + s.n_projects + 1);
  s.projects[0] = project;
  ++s.n_projects;
  ProjectTokens& t = s.project_tokens[s.n_project_tokens++];
  t = ProjectTokens{};
  t.project = project;
  while (s.n_projects > kProjectsAboveBoard) {
    CardIdx gone = s.projects[--s.n_projects];
    for (int i = 0; i < s.n_project_tokens; ++i) {
      if (s.project_tokens[i].project != gone) continue;
      for (int j = i + 1; j < s.n_project_tokens; ++j) s.project_tokens[j - 1] = s.project_tokens[j];
      --s.n_project_tokens;
      break;
    }
    s.discard.push_back(gone);
  }
}

// The task's immediate part (paying, taking the card) happens here; the returned closure-like
// struct holds what happens after the frame bookkeeping, as in Python's _task_effects.
struct TaskEffects {
  Head head = Head::reputation;
  int arg = 0;
  bool played = false;
  CardIdx project = kNoCard;
  int position = 0, token = 0;
  CardIdx release = kNoCard;
};

TaskEffects task_effects(State& s, int p, const Option& o) {
  TaskEffects e;
  e.head = o.head;
  if (o.head == Head::reputation) return e;
  if (o.head == Head::partner_zoo || o.head == Head::university) {
    e.arg = o.a[0];
    return e;
  }
  Player& pl = s.players[p];
  e.played = o.head == Head::play_project || o.head == Head::play_release || o.head == Head::play_project_display ||
             o.head == Head::play_release_display;
  if (o.head == Head::support || o.head == Head::release) {
    e.project = o.a[0];
  } else if (o.head == Head::play_project || o.head == Head::play_release) {
    e.project = o.a[0];
    remove_from_hand(s, p, e.project);
    put_project_in_play(s, e.project);
  } else {
    pl.money -= o.a[0] + 1;
    e.project = take_from_display(s, o.a[0]);
    put_project_in_play(s, e.project);
  }
  e.position = o.a[1];
  e.token = o.a[2];
  if (o.a[3] == kExtraToken) {
    sr::spend_breeding_token(pl, o.a[4]);
  } else if (o.a[3] == kExtraRelease) {
    e.release = o.a[4];
  }
  return e;
}

void run_task_effects(State& s, int p, const TaskEffects& e, Chance& ch) {
  if (e.head == Head::reputation) {
    gain(s, p, EffectKind::kEffReputation, kTaskReputationGain);
    return;
  }
  if (e.head == Head::partner_zoo) {
    take_partner_zoo(s, p, e.arg, ch);
    return;
  }
  if (e.head == Head::university) {
    take_university(s, p, e.arg, ch);
    return;
  }
  const CardDef& pj = card(e.project);
  if (e.played) {
    for (int i = 0; i < pj.n_play_bonus; ++i) gain(s, p, pj.play_bonus[i]);
  }
  if (e.release != kNoCard) release_animal(s, p, e.release);
  ProjectTokens* t = s.tokens_of(e.project);
  if (!t) engine_error("project not in play");
  t->slot[e.position - 1] = static_cast<std::int8_t>(p);
  take_map_token(s, p, e.token);
  const ProjectSlot& slot = pj.slots[e.position - 1];
  for (int i = 0; i < slot.n_rewards; ++i) {
    EffectKind k = slot.rewards[i].reward == Reward::kRewardConservation ? EffectKind::kEffConservation : EffectKind::kEffReputation;
    gain(s, p, k, slot.rewards[i].amount);
  }
  if (pj.proj_category == ProjCategory::kProjRelease) sr::release_bonus(s, p);
}

void association_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  int p = f.player;
  std::uint16_t uid = f.uid;
  Player& pl = s.players[p];
  if (o.head == Head::done) {
    pop(s, fi);
    return;
  }
  bool side2 = f.d[fd::assoc_side2];
  TaskEffects effects;
  bool donate = o.head == Head::donate;
  if (donate) {
    pl.money -= donation_cost(s);
    ++s.donations;
    f.d[fd::assoc_donated] = 1;
  } else {
    int task = task_of(o.head);
    pl.workers_out[task] += workers_needed(pl, task);
    f.d[fd::assoc_remaining] -= sr::task_value(pl, task);
    f.d[fd::assoc_used] |= 1 << task;
    effects = task_effects(s, p, o);
  }
  auto run = [&] {
    if (donate) {
      gain(s, p, EffectKind::kEffConservation, 1);
    } else {
      run_task_effects(s, p, effects, ch);
    }
  };
  if (!side2) {
    pop(s, find_frame(s, uid));
    run();
    return;
  }
  run();
  int cur = find_frame(s, uid);
  if (cur >= 0 && cur == s.stack_n - 1 && !anything_possible(s, p, frame(s, cur))) pop(s, cur);
}

}  // namespace

int project_icon_count(const Player& pl, ProjIcon icon) { return project_count(pl, icon); }

int next_donation_cost(const State& s) { return donation_cost(s); }

int project_strength_needed(const Player& pl) { return sr::task_value(pl, kTaskProject); }

std::array<bool, 3> supportable_levels(const State& s, int p, CardIdx project) {
  std::array<bool, 3> levels{};
  for (const SupportChoice& c : support_choices(s, p, project)) levels[c.position - 1] = true;
  return levels;
}

void register_association_action() {
  register_action(kAssociation, association_feasible, association_start);
  register_choice(FrameKind::act_association, association_options, association_apply);
}

}  // namespace board_ai::arknova
