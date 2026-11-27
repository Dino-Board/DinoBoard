#pragma once

#include <algorithm>
#include <cmath>

namespace board_ai::search {

struct TemperatureSchedule {
  bool enabled = false;
  bool has_initial = false;
  bool has_final = false;
  double initial = 0.0;
  double final_ = 0.0;
  double halflife_plies = 0.0;
};

// KataGo-style exponential decay:
//   T(ply) = final + (initial - final) * 0.5^(ply / halflife_plies)
// When halflife_plies <= 0 (or initial == final), returns final immediately.
inline double resolve_exponential_temperature(
    const TemperatureSchedule& schedule,
    double base_temperature,
    int ply_index) {
  const double base = std::max(0.0, base_temperature);
  if (!schedule.enabled) return base;

  const double t0 = schedule.has_initial ? std::max(0.0, schedule.initial) : base;
  const double t1 = schedule.has_final ? std::max(0.0, schedule.final_) : base;
  if (!(std::isfinite(schedule.halflife_plies) && schedule.halflife_plies > 0.0) || t0 == t1) {
    return t1;
  }

  const double clipped_ply = static_cast<double>(std::max(0, ply_index));
  return t1 + (t0 - t1) * std::pow(0.5, clipped_ply / schedule.halflife_plies);
}

}  // namespace board_ai::search
