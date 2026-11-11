#pragma once

#include <cstdio>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <direct.h>
#define board_ai_getcwd _getcwd
#else
#include <unistd.h>
#define board_ai_getcwd getcwd
#endif

namespace board_ai::heuristic {

struct MctsPvHeuristicConfig {
  int simulations = 0;
  // Caps how many distinct world seeds the MCTS budget is split across.
  // 0 = uncapped (legacy: K=simulations, M=1, every sim its own seed).
  // > 0 = K=clamp(world_cap, 1, simulations), M=ceil(sims/K), so the
  // M sims inside each world share one world_seed and accumulate
  // visits along whatever do_action_fast draws produce. For
  // fully-public games with physical randomness in do_action_fast (e.g.
  // Carcassonne's bag draws), this is the lever that prevents every
  // sim from being its own deterministic line. Optional in
  // heuristic.json (defaults to 0 for backwards compatibility).
  int world_cap = 0;
  // Forwarded to NetMctsConfig.leaf_batch_size. Optional in heuristic.json
  // (defaults to 1 = unbatched search).
  int leaf_batch_size = 1;
  float c_puct = 1.4f;
  std::string model_path;
};

inline std::string base_game_id(std::string game_id) {
  // `_tok` marks a token-encoder variant sharing the base game's configs.
  static const std::regex suffix_re(R"((_v?tok)?(_\d+p)?$)");
  return std::regex_replace(game_id, suffix_re, "");
}

namespace detail {

inline std::string get_cwd() {
  char buf[4096];
  if (!board_ai_getcwd(buf, sizeof(buf))) return ".";
  return std::string(buf);
}

inline bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

inline std::string parent_dir(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return path.substr(0, pos);
}

}  // namespace detail

inline std::optional<std::string> find_repo_relative_config(
    const std::string& base_id) {
  std::string cur = detail::get_cwd();
  for (int depth = 0; depth < 8; ++depth) {
    std::string candidate = cur + "/games/" + base_id + "/config/heuristic.json";
    if (detail::file_exists(candidate)) return candidate;
    std::string parent = detail::parent_dir(cur);
    if (parent == cur) break;
    cur = parent;
  }
  return std::nullopt;
}

inline std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline std::string extract_named_object(
    const std::string& text,
    const std::string& name,
    const std::string& path) {
  const std::regex entry_re(
      "\"" + name + R"("\s*:\s*\{([^{}]*)\})");
  std::smatch m;
  if (!std::regex_search(text, m, entry_re)) {
    throw std::runtime_error(
        path + ": missing heuristic.json::named." + name);
  }
  return m[1].str();
}

inline int extract_int_field(
    const std::string& object,
    const std::string& field,
    const std::string& path,
    const std::string& name) {
  const std::regex re("\"" + field + R"("\s*:\s*([0-9]+))");
  std::smatch m;
  if (!std::regex_search(object, m, re)) {
    throw std::runtime_error(
        path + ": missing named." + name + "." + field);
  }
  return std::stoi(m[1].str());
}

inline float extract_float_field(
    const std::string& object,
    const std::string& field,
    const std::string& path,
    const std::string& name) {
  const std::regex re(
      "\"" + field + R"("\s*:\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))");
  std::smatch m;
  if (!std::regex_search(object, m, re)) {
    throw std::runtime_error(
        path + ": missing named." + name + "." + field);
  }
  return std::stof(m[1].str());
}

inline std::string extract_optional_string_field(
    const std::string& object,
    const std::string& field) {
  const std::regex re("\"" + field + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch m;
  if (!std::regex_search(object, m, re)) return "";
  return m[1].str();
}

inline int extract_optional_int_field(
    const std::string& object,
    const std::string& field,
    int default_value) {
  const std::regex re("\"" + field + R"("\s*:\s*([0-9]+))");
  std::smatch m;
  if (!std::regex_search(object, m, re)) return default_value;
  return std::stoi(m[1].str());
}

inline std::string resolve_variant_model_path(
    std::string model_path,
    const std::string& game_id,
    const std::string& base_id) {
  if (model_path.empty() || game_id == base_id || game_id == base_id + "_2p") {
    return model_path;
  }
  const std::string from = base_id + "_2p.onnx";
  const std::string to = game_id + ".onnx";
  const auto pos = model_path.rfind(from);
  if (pos != std::string::npos) {
    model_path.replace(pos, from.size(), to);
  }
  return model_path;
}

inline MctsPvHeuristicConfig load_mcts_pv_heuristic_config(
    const std::string& game_id,
    const std::string& name,
    bool require_model_path) {
  const std::string base_id = base_game_id(game_id);
  const auto path_opt = find_repo_relative_config(base_id);
  if (!path_opt) {
    throw std::runtime_error(
        "missing games/" + base_id + "/config/heuristic.json");
  }
  const auto& path = *path_opt;
  // Derive repo root from the config path:
  // path = <repo>/games/<base>/config/heuristic.json → repo = 4 dirs up
  std::string repo_root = detail::parent_dir(
      detail::parent_dir(detail::parent_dir(detail::parent_dir(path))));
  const std::string text = read_text_file(path);
  const std::string object = extract_named_object(text, name, path);

  MctsPvHeuristicConfig cfg;
  cfg.simulations = extract_int_field(object, "simulations", path, name);
  cfg.c_puct = extract_float_field(object, "c_puct", path, name);
  cfg.world_cap = extract_optional_int_field(object, "world_cap", 0);
  cfg.leaf_batch_size = extract_optional_int_field(object, "leaf_batch_size", 1);
  if (cfg.leaf_batch_size < 1) {
    throw std::runtime_error(
        path + ": named." + name + ".leaf_batch_size must be >= 1");
  }
  std::string raw_model = resolve_variant_model_path(
      extract_optional_string_field(object, "model_path"), game_id, base_id);
  // model_path in heuristic.json is repo-relative (e.g. "games/azul/model/azul_2p.onnx").
  // Resolve to absolute using the repo root derived from the config file path.
  if (!raw_model.empty() && raw_model[0] != '/') {
    cfg.model_path = repo_root + "/" + raw_model;
  } else {
    cfg.model_path = raw_model;
  }
  if (require_model_path && cfg.model_path.empty()) {
    throw std::runtime_error(
        path + ": missing named." + name + ".model_path");
  }
  return cfg;
}

}  // namespace board_ai::heuristic
