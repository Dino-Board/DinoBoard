"""MCTS profile resolver — single source of truth for MCTS callsite parameters.

Three config files per game, one shared baseline plus two profile sets:
  - `games/<g>/config/mcts.json`   — flat baseline (no profile names).
                                     Shared MCTS knobs every callsite agrees on.
  - `games/<g>/config/train.json`  — `mcts.<selfplay|eval>`: profile deltas
                                     applied on top of the baseline. Selfplay
                                     carries the dirichlet root-noise override
                                     (the only place noise lives); eval is
                                     used both for benchmark eval (vs heuristic /
                                     vs ONNX) and for gating (latest vs best),
                                     as those are the same MCTS behavior — a
                                     pair-wise tournament for win-rate.
  - `games/<g>/config/web.json`    — `<web_expert|web_casual|analysis>`: profile
                                     deltas applied on top of the baseline. The
                                     web/REST AI never injects noise.

`resolve_profile(game_id, profile_name)` dispatches: `{selfplay, eval}` →
train.json's `mcts` block; `{web_expert, web_casual, analysis}` → web.json. All
profile keys override baseline keys field-by-field (no `inherits` chain — there
is exactly one parent, the baseline).

Dirichlet noise defaults to 0/0/0 in the baseline (= "off"). Arena / eval / all
web profiles inherit that default and don't have to write zeros explicitly.
Only selfplay's profile delta in train.json sets non-zero values.

The `analysis` profile is a deliberate exception: only its `simulations` and
`world_cap` fields are consumed by the web analysis path. The drop-score /
smart-hint path hard-codes `temperature=0.0`, `cover_root_edges=True`, and
`opponent_selection="puct"` regardless of profile contents — those are
architectural constraints (see CLAUDE.md "Opponent-node selection" and
docs/guide/CONFIG_REFERENCE.md "分析 pipeline 的特殊行为"), not knobs.

Per CLAUDE.md "no silent fallbacks": every resolver branch raises on missing
keys, unknown profile names, unknown fields, illegal opponent_selection values,
or `tail_solve_enabled=true` without a registered trigger. There is no default
profile name; callsite picks the right one.
"""
from __future__ import annotations

import dataclasses
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dinoboard_engine

from .game_ids import base_game_id as _base_game_id

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_VALID_OPPONENT_SELECTION = {"puct", "prior"}

_TRAIN_PROFILE_NAMES = ("selfplay", "eval")
_WEB_PROFILE_NAMES = ("web_expert", "web_casual", "analysis")
_ALL_PROFILE_NAMES = _TRAIN_PROFILE_NAMES + _WEB_PROFILE_NAMES

_PROFILE_FIELDS = {
    "simulations",
    "world_cap",
    "c_puct",
    "fpu_reduction",
    "uct2_arrival_visits",
    "temperature",
    "temperature_schedule",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "dirichlet_on_first_n_plies",
    "opponent_selection",
    "tail_solve_enabled",
    "tail_solve_depth_limit",
    "tail_solve_node_budget",
    "tail_solve_auxiliary_weight",
    "ai_use_action_filter",
    "cover_root_edges",
    "top_k",
}

# Fields only train.json's `mcts.<selfplay|eval>` profiles may set. The web
# platform resolves the web profiles (baseline included) but does not pass
# these on to the engine, so a value in mcts.json or web.json would be
# silently ignored there — rejected instead.
_TRAIN_PROFILE_ONLY_FIELDS = {"uct2_arrival_visits"}

_TEMPERATURE_SCHEDULE_FIELDS = {"enabled", "initial", "final", "decay_plies"}


@dataclass(frozen=True)
class MctsProfile:
    """Resolved MCTS parameters for a (game, profile_name) pair."""

    name: str

    simulations: int
    world_cap: int
    c_puct: float
    # None = legacy Q=0 for unvisited edges; r >= 0 = parent Q minus
    # r * sqrt(visited prior mass). See NetMctsConfig::fpu_reduction.
    fpu_reduction: float | None
    # UCT2 parent count. True (default): traversals of the incoming edge that
    # arrived at the node; False: the edge's visits over all its outcomes (the
    # search before this switch existed). See NetMctsConfig::uct2_arrival_visits.
    uct2_arrival_visits: bool
    temperature: float
    temperature_schedule_enabled: bool
    temperature_initial: float
    temperature_final: float
    temperature_decay_plies: int

    dirichlet_alpha: float
    dirichlet_epsilon: float
    dirichlet_on_first_n_plies: int

    opponent_selection: str

    tail_solve_enabled: bool
    tail_solve_depth_limit: int
    tail_solve_node_budget: int
    tail_solve_auxiliary_weight: float

    ai_use_action_filter: bool
    cover_root_edges: bool
    top_k: int


# Implicit defaults applied when neither baseline nor profile specifies a field.
# The baseline mcts.json is expected to fully specify the shared knobs; only
# dirichlet (off-by-default) and tail_solve_*_weight (rarely set) really lean
# on these defaults in practice.
_BASE_DICT: dict[str, Any] = {
    "simulations": 200,
    # 0 = legacy K=simulations / M=1 (every sim its own world).
    # >0 caps the number of distinct sampled worlds and reuses each
    # world for ceil(simulations / K) sims. Drives the inner-M visit
    # amortization that 6 nimmt!-style very-wide-action games need.
    "world_cap": 0,
    "c_puct": 1.4,
    "fpu_reduction": None,
    "uct2_arrival_visits": True,
    "temperature": 0.0,
    "temperature_schedule_enabled": False,
    "temperature_initial": 0.0,
    "temperature_final": 0.0,
    "temperature_decay_plies": 0,
    # Dirichlet root noise is off by default. Only train.json's selfplay
    # profile (and only that one) is expected to switch it on.
    "dirichlet_alpha": 0.0,
    "dirichlet_epsilon": 0.0,
    # Sentinel -1 = "use the game's max_game_plies", resolved at resolve time.
    # Explicit 0 still means "don't add dirichlet at any ply"; explicit positive
    # int overrides. This way games don't have to repeat max_game_plies in
    # train.json::mcts.selfplay.
    "dirichlet_on_first_n_plies": -1,
    "opponent_selection": "puct",
    "tail_solve_enabled": False,
    "tail_solve_depth_limit": 10,
    "tail_solve_node_budget": 200_000,
    "tail_solve_auxiliary_weight": 0.0,
    "ai_use_action_filter": False,
    "cover_root_edges": False,
    "top_k": 0,
}


_PROFILE_CACHE: dict[tuple[str, str], MctsProfile] = {}
_BASELINE_CACHE: dict[str, dict[str, Any]] = {}
_TRAIN_PROFILES_CACHE: dict[str, dict[str, dict]] = {}
_WEB_PROFILES_CACHE: dict[str, dict[str, dict]] = {}
_MAX_PLIES_CACHE: dict[str, int] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop all caches. Tests use this between fixtures."""
    with _CACHE_LOCK:
        _PROFILE_CACHE.clear()
        _BASELINE_CACHE.clear()
        _TRAIN_PROFILES_CACHE.clear()
        _WEB_PROFILES_CACHE.clear()
        _MAX_PLIES_CACHE.clear()


def _resolve_config_path(game_id: str, filename: str) -> Path:
    """Find `games/<id>/config/<filename>`, falling back to base id for variants."""
    base = _base_game_id(game_id)
    candidates = [game_id, base] if game_id != base else [game_id]
    for cand in candidates:
        path = _PROJECT_ROOT / "games" / cand / "config" / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{filename} not found for {game_id!r} (tried "
        f"{', '.join(str(_PROJECT_ROOT / 'games' / c / 'config' / filename) for c in candidates)})")


def _reject_train_profile_only_fields(node: dict, source: str) -> None:
    misplaced = set(node.keys()) & _TRAIN_PROFILE_ONLY_FIELDS
    if misplaced:
        raise ValueError(
            f"{source}: field(s) {sorted(misplaced)} can only be set in train.json's "
            f"`mcts.selfplay` / `mcts.eval` profiles. The web platform does not "
            f"forward them, so a value here would be silently ignored for web play.")


def _validate_profile_node(name: str, game_id: str, node: dict, source: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(
            f"profile {name!r} for game {game_id!r} in {source} must be a JSON "
            f"object, got {type(node).__name__}.")
    unknown = set(node.keys()) - _PROFILE_FIELDS
    if unknown:
        raise ValueError(
            f"profile {name!r} for game {game_id!r} in {source} has unknown "
            f"field(s): {sorted(unknown)}. Allowed: {sorted(_PROFILE_FIELDS)}.")


def _load_baseline(game_id: str) -> dict[str, Any]:
    with _CACHE_LOCK:
        cached = _BASELINE_CACHE.get(game_id)
        if cached is not None:
            return cached

    path = _resolve_config_path(game_id, "mcts.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a JSON object whose keys are MCTS knob names "
            f"(flat baseline, no profile names); got {type(raw).__name__}.")
    unknown = set(raw.keys()) - _PROFILE_FIELDS
    if unknown:
        raise ValueError(
            f"{path} has unknown field(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_PROFILE_FIELDS)}. Note: mcts.json is the flat "
            f"baseline (no profile names); profile-specific knobs live in "
            f"train.json's `mcts` block or web.json.")
    _reject_train_profile_only_fields(raw, str(path))

    with _CACHE_LOCK:
        _BASELINE_CACHE[game_id] = raw
    return raw


def _load_train_profiles(game_id: str) -> dict[str, dict]:
    with _CACHE_LOCK:
        cached = _TRAIN_PROFILES_CACHE.get(game_id)
        if cached is not None:
            return cached

    path = _resolve_config_path(game_id, "train.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "mcts" not in cfg:
        raise KeyError(
            f"{path} is missing required 'mcts' block — must contain at least "
            f"'selfplay' (with dirichlet override). See docs/guide/CONFIG_REFERENCE.md.")
    raw = cfg["mcts"]
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}['mcts'] must be a JSON object whose keys are profile "
            f"names {sorted(_TRAIN_PROFILE_NAMES)}, got {type(raw).__name__}.")
    extras = set(raw.keys()) - set(_TRAIN_PROFILE_NAMES)
    if extras:
        raise ValueError(
            f"{path}['mcts'] has profile name(s) {sorted(extras)} that don't "
            f"belong in train.json (web profiles live in web.json). "
            f"Allowed: {sorted(_TRAIN_PROFILE_NAMES)}.")
    for name in _TRAIN_PROFILE_NAMES:
        if name not in raw:
            raise KeyError(
                f"{path}['mcts'] missing required profile {name!r}.")
        _validate_profile_node(name, game_id, raw[name], f"{path}['mcts']")

    with _CACHE_LOCK:
        _TRAIN_PROFILES_CACHE[game_id] = raw
    return raw


def _load_web_profiles(game_id: str) -> dict[str, dict]:
    with _CACHE_LOCK:
        cached = _WEB_PROFILES_CACHE.get(game_id)
        if cached is not None:
            return cached

    path = _resolve_config_path(game_id, "web.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must be a JSON object whose keys are profile names "
            f"{sorted(_WEB_PROFILE_NAMES)}, got {type(raw).__name__}.")
    extras = set(raw.keys()) - set(_WEB_PROFILE_NAMES)
    if extras:
        raise ValueError(
            f"{path} has profile name(s) {sorted(extras)} that don't belong "
            f"in web.json (train profiles live in train.json). "
            f"Allowed: {sorted(_WEB_PROFILE_NAMES)}.")
    for name in _WEB_PROFILE_NAMES:
        if name not in raw:
            raise KeyError(
                f"{path} missing required profile {name!r}.")
        _validate_profile_node(name, game_id, raw[name], str(path))
        _reject_train_profile_only_fields(raw[name], f"{path} profile {name!r}")

    with _CACHE_LOCK:
        _WEB_PROFILES_CACHE[game_id] = raw
    return raw


def _flatten_temperature_schedule(layer: dict[str, Any]) -> dict[str, Any]:
    """Expand `temperature_schedule: {...}` into flat profile fields."""
    if "temperature_schedule" not in layer:
        return layer
    schedule = layer["temperature_schedule"]
    if not isinstance(schedule, dict):
        raise ValueError(
            f"temperature_schedule must be a JSON object with keys "
            f"{sorted(_TEMPERATURE_SCHEDULE_FIELDS)}, got {type(schedule).__name__}.")
    missing = _TEMPERATURE_SCHEDULE_FIELDS - set(schedule.keys())
    if missing:
        raise ValueError(
            f"temperature_schedule missing required keys: {sorted(missing)}.")
    extra = set(schedule.keys()) - _TEMPERATURE_SCHEDULE_FIELDS
    if extra:
        raise ValueError(
            f"temperature_schedule has unknown keys: {sorted(extra)}. "
            f"Allowed: {sorted(_TEMPERATURE_SCHEDULE_FIELDS)}.")
    out = {k: v for k, v in layer.items() if k != "temperature_schedule"}
    out["temperature_schedule_enabled"] = bool(schedule["enabled"])
    out["temperature_initial"] = float(schedule["initial"])
    out["temperature_final"] = float(schedule["final"])
    out["temperature_decay_plies"] = int(schedule["decay_plies"])
    return out


def _validate_resolved(name: str, game_id: str, merged: dict[str, Any]) -> None:
    if int(merged["world_cap"]) < 0:
        raise ValueError(
            f"profile {name!r} for game {game_id!r}: world_cap must be >= 0 "
            f"(0 disables, >0 sets K outer determinizations), got "
            f"{merged['world_cap']!r}.")
    fpu = merged["fpu_reduction"]
    if fpu is not None and (isinstance(fpu, bool) or not isinstance(fpu, (int, float))
                            or not math.isfinite(fpu) or fpu < 0):
        raise ValueError(
            f"profile {name!r} for game {game_id!r}: fpu_reduction must be null "
            f"(legacy Q=0) or a finite number >= 0, got {fpu!r}.")
    if not isinstance(merged["uct2_arrival_visits"], bool):
        raise ValueError(
            f"profile {name!r} for game {game_id!r}: uct2_arrival_visits must be "
            f"true or false, got {merged['uct2_arrival_visits']!r}.")
    if merged["opponent_selection"] not in _VALID_OPPONENT_SELECTION:
        raise ValueError(
            f"profile {name!r} for game {game_id!r}: opponent_selection must be "
            f"one of {sorted(_VALID_OPPONENT_SELECTION)}, got "
            f"{merged['opponent_selection']!r}.")
    if merged["tail_solve_enabled"]:
        meta = dinoboard_engine.game_metadata(game_id)
        if not meta["has_tail_solve_trigger"]:
            raise ValueError(
                f"profile {name!r} for game {game_id!r}: tail_solve_enabled=true "
                f"but no tail_solve_trigger is registered in the GameBundle.")


def resolve_profile(game_id: str, profile_name: str) -> MctsProfile:
    """Return the fully-resolved MctsProfile for (game_id, profile_name).

    Layering: BASE (in-code defaults) → mcts.json (baseline) → profile delta
    (train.json's `mcts.<name>` for train profiles, web.json's `<name>` for web
    profiles).
    """
    if profile_name not in _ALL_PROFILE_NAMES:
        raise KeyError(
            f"unknown MCTS profile {profile_name!r}. Allowed: "
            f"{sorted(_ALL_PROFILE_NAMES)}.")

    key = (game_id, profile_name)
    with _CACHE_LOCK:
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            return cached

    baseline = _load_baseline(game_id)
    if profile_name in _TRAIN_PROFILE_NAMES:
        delta = _load_train_profiles(game_id)[profile_name]
    else:
        delta = _load_web_profiles(game_id)[profile_name]

    merged = dict(_BASE_DICT)
    merged.update(_flatten_temperature_schedule(baseline))
    merged.update(_flatten_temperature_schedule(delta))

    # dirichlet_on_first_n_plies sentinel: -1 means "the whole game"
    # (= game.json::max_game_plies). Lets each game omit the field
    # rather than repeat the max-plies value in mcts.selfplay.
    if int(merged["dirichlet_on_first_n_plies"]) < 0:
        merged["dirichlet_on_first_n_plies"] = max_game_plies(game_id)

    _validate_resolved(profile_name, game_id, merged)

    profile = MctsProfile(
        name=profile_name,
        simulations=int(merged["simulations"]),
        world_cap=int(merged["world_cap"]),
        c_puct=float(merged["c_puct"]),
        fpu_reduction=None if merged["fpu_reduction"] is None else float(merged["fpu_reduction"]),
        uct2_arrival_visits=merged["uct2_arrival_visits"],
        temperature=float(merged["temperature"]),
        temperature_schedule_enabled=bool(merged["temperature_schedule_enabled"]),
        temperature_initial=float(merged["temperature_initial"]),
        temperature_final=float(merged["temperature_final"]),
        temperature_decay_plies=int(merged["temperature_decay_plies"]),
        dirichlet_alpha=float(merged["dirichlet_alpha"]),
        dirichlet_epsilon=float(merged["dirichlet_epsilon"]),
        dirichlet_on_first_n_plies=int(merged["dirichlet_on_first_n_plies"]),
        opponent_selection=str(merged["opponent_selection"]),
        tail_solve_enabled=bool(merged["tail_solve_enabled"]),
        tail_solve_depth_limit=int(merged["tail_solve_depth_limit"]),
        tail_solve_node_budget=int(merged["tail_solve_node_budget"]),
        tail_solve_auxiliary_weight=float(merged["tail_solve_auxiliary_weight"]),
        ai_use_action_filter=bool(merged["ai_use_action_filter"]),
        cover_root_edges=bool(merged["cover_root_edges"]),
        top_k=int(merged["top_k"]),
    )

    with _CACHE_LOCK:
        _PROFILE_CACHE[key] = profile
    return profile


def max_game_plies(game_id: str) -> int:
    """Top-level `max_game_plies` from game.json (game property, all profiles share)."""
    with _CACHE_LOCK:
        cached = _MAX_PLIES_CACHE.get(game_id)
        if cached is not None:
            return cached

    game_path = _resolve_config_path(game_id, "game.json")
    with open(game_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "max_game_plies" not in cfg:
        raise KeyError(
            f"{game_path} is missing required top-level key 'max_game_plies'.")
    value = int(cfg["max_game_plies"])

    with _CACHE_LOCK:
        _MAX_PLIES_CACHE[game_id] = value
    return value


def profile_as_dict(profile: MctsProfile) -> dict[str, Any]:
    """Serialize a profile for cross-process worker handoff (ProcessPoolExecutor)."""
    return dataclasses.asdict(profile)


def profile_to_seat_config(
    profile: MctsProfile,
    *,
    model_path: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a resolved MctsProfile into a per-seat dict for run_arena_match.

    Field set is the union of "MCTS knobs the C++ binding's seat_configs path
    actually consumes" — currently `model_path`, `simulations`, `world_cap`,
    `temperature`, `fpu_reduction`, `uct2_arrival_visits`, `opponent_selection`,
    plus tail_solve_*. Items the binding
    ignores (dirichlet_*, c_puct, ai_use_action_filter, cover_root_edges) are
    deliberately left out so passing a stale field surfaces as a bind-time
    schema error rather than silently being dropped.

    `overrides` are merged last and let the caller flip individual knobs (e.g.
    `tail_solve_enabled=False` for an A/B test) without mutating the cached
    profile object.
    """
    if not model_path:
        raise ValueError("profile_to_seat_config: model_path must not be empty")
    cfg: dict[str, Any] = {
        "model_path": model_path,
        "simulations": profile.simulations,
        "world_cap": profile.world_cap,
        "temperature": profile.temperature,
        "fpu_reduction": profile.fpu_reduction,
        "uct2_arrival_visits": profile.uct2_arrival_visits,
        "opponent_selection": profile.opponent_selection,
        "tail_solve_enabled": profile.tail_solve_enabled,
        "tail_solve_depth_limit": profile.tail_solve_depth_limit,
        "tail_solve_node_budget": profile.tail_solve_node_budget,
        "tail_solve_auxiliary_weight": profile.tail_solve_auxiliary_weight,
        "top_k": profile.top_k,
    }
    if overrides:
        cfg.update(overrides)
    return cfg
